import os
import textwrap
import warnings
from collections import defaultdict, deque
from collections.abc import Sized
from contextlib import nullcontext
from typing import Any, Callable, Optional, Union

import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_liger_kernel_available, is_peft_available

from ..data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from ..extras.profiling import profiling_context, profiling_decorator
from ..extras.vllm_client import VLLMClient
from ..import_utils import is_deepspeed_available, is_rich_available, is_vllm_available
from ..models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from .callbacks import SyncRefModelCallback
from .grpo_config import GRPOConfig
from .utils import (
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)


if is_deepspeed_available():
    import deepspeed

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss

if is_wandb_available():
    import wandb

#我们称之为奖励函数的是一个可调用函数，它接受一系列提示和完成，并返回一个奖励。当它是字符串时，它是一个模型ID，因此它被加载为预训练模型。
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RepeatRandomSampler(Sampler):
    """
    以结构化方式重复数据集索引的采样器。

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, batch_size=3, repeat_count=4)
    >>> list(sampler)
    [4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,

     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6]
    ```

    ```txt
    mini_repeat_count = 3
          -   -   -
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                repeat_count = 2
          0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
          ---------   ---------   ---------   ---------
           ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                         batch_size = 12
    ```
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()  # Create a local random generator
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    计算张量的标准偏差，忽略NaN。此函数仅支持1D张量。

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)


class GRPOTrainer(Trainer):
    """
    组相对策略优化（GRPO）方法训练。该算法最初是在 paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    def reward_func(completions, **kwargs):
        # Dummy reward function that rewards completions with more unique letters.
        return [float(len(set(completion))) for completion in completions]

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_func,
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
要训练的模型。
可以是：
-一个字符串，是huggingface.co上模型仓库内托管的预训练模型的*model id*，或者是包含使用[`~transformers.PreTrainingModel.save_presetrained`]保存的模型权重的*目录*的路径，例如“”/
my_model_目录/“”。
使用[`~transformers.AutoModelForCausalLM.from_petrained`]加载模型，并在`args.model_init_kwargs`中包含关键字参数。
-[`~transformers.PreTrainingModel`]对象。
仅支持因果语言模型。
reward_funcs（`Union[RewardFunc，list[RewardFunction]]）：用于计算奖励的奖励函数。
为了计算奖励，我们调用所有带有提示和完成的奖励函数，并将奖励相加。
可以是：
-单一奖励功能，例如：
-字符串：huggingface.co上模型仓库内托管的预训练模型的*model ID*，或包含使用[`~transformers.PreTrainedModel.save_presetrained`]保存的模型权重的*目录*的路径，例如“”/
my_model_目录/“”。
使用[`~transformers.AutoModelForSequenceClassification.from_petrained`]加载模型，其中`num_labels=1 `和`args.model_init_kwargs`中的关键字参数。
-[`~transformers.PreTrainingModel`]对象：仅支持序列分类模型。
-自定义奖励函数：该函数提供提示和生成的补全，以及数据集中的任何其他列。
它应该返回一个奖励列表。
当奖励不适用于这些样本时，自定义奖励函数也可以返回None。
这对于多任务训练非常有用，其中不同的奖励函数适用于不同类型的样本。
当奖励函数为样本返回None时，该奖励函数将被排除在该样本的奖励计算之外。
有关更多详细信息，请参阅[使用自定义奖励功能]（#使用自定义奖励函数）。
-奖励函数列表，其中每个项目可以独立地是上述任何类型。
允许在列表中混合不同类型（例如，字符串模型ID和自定义奖励函数）。
args（[`GRPOConfig`]，*可选*，默认为`None `）：此培训器的配置。
如果为“无”，则使用默认配置。
train_dataset（[`~datas.dataset`]或[`~dataset.IterableDataset`）：
用于训练的数据集。
它必须包含一列“提示”。
数据集中的任何其他列都将被忽略。
样本的格式可以是：

-[Standard]（dataset_formats#Standard）：每个样本都包含纯文本。
-[会话]（dataset_formats#conventional）：每个样本都包含结构化消息（例如，角色和内容）。
eval_adset（[`~datas.dataset`]，[`~dataset.IterableDataset`]]或`dict[str，Union[dataset，IterableDataset]]）：用于计算的数据集。
它必须满足与“train_dataset”相同的要求。
processing_class（[`~transformers.PreTrainingTokenBase`]，*可选*，默认为“无”）：用于处理数据的处理类。
衬垫侧必须设置为“左侧”。
如果为“None”，则使用[`~transformers.AutoTokenizer.from_petrained`从模型名称加载处理类。
reward_processing_classes（`Union[PreTrainedTokenizerBase，list[PreTraindTokenizerBase]]，*可选*，默认为`None’）：与`reward_funcs'中指定的奖励函数对应的处理类。
可以是：
-单个处理类：当`reward_funcs`只包含一个奖励函数时使用。
-处理类列表：必须与“reward_funcs”中奖励函数的顺序和长度匹配。
如果设置为“None”，或者如果列表中与[`~transformers.PreTrainedModel`]对应的元素为“None”，则使用[`~transformers.AutoTokenizer.from_petrained']自动加载模型的标记器。
对于“reward_funcs”中属于自定义奖励函数（不是[`~transformers.PreTrainingModel`]）的元素，将忽略“reward_processing_classes”中的相应条目。
callbacks（[`~transformers.TrainerCallback`]列表，*可选*，默认为“无”）：用于自定义训练循环的回调列表。
将这些添加到[此处]中详细说明的默认回调列表中(
https://huggingface.co/docs/transformers/main_classes/callback
). 
如果要删除使用的默认回调之一，请使用[`~transformers.Trainer.remove_callback`]方法。
优化器（“tuple[torch.optimi.Optimizer，torch.ooptim.lr_scheduler.LambdaLR]”，*可选*，默认为“（None，None）”）：包含要使用的优化器和调度器的元组。
将默认为模型上的[`AdamW`]实例和由`args `控制的[`get_liner_schedule_with_warmup`]给出的调度器。
peft_config（[`~peft.PeftConfig `]，*可选*，默认为“无”）：用于包装模型的peft配置。
如果为“无”，则模型未被包装。
    """

    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # 如果启用了梯度检查点，则禁用缓存（不支持）
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            if not is_peft_available():
                raise ImportError("PEFT is required to use `peft_config`. Run `pip install peft`.")
            model = get_peft_model(model, peft_config)

        # 如果需要，启用梯度检查点
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # 如果beta为0.0，则不需要参考模型
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif is_peft_model(model):
            # 如果使用PEFT，则不需要参考模型，因为可以禁用适配器以恢复到初始模型。
            self.ref_model = None
        else:
            # 如果没有提供PEFT配置，请基于初始模型创建参考模型。
            self.ref_model = create_reference_model(model)

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # 奖励模型计算输入序列中最新非填充令牌的奖励。
                # 因此，将填充令牌ID设置为处理类的填充令牌ID非常重要。
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.use_vllm = args.use_vllm
        self.use_liger_loss = args.use_liger_loss

        # Datasets
        if (
            isinstance(train_dataset, IterableDataset)
            or isinstance(eval_dataset, IterableDataset)
            or (
                isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
            )
        ):
            # See https://github.com/huggingface/trl/issues/3213
            raise NotImplementedError(
                "Iterable datasets are not yet supported in GRPOTrainer. Please use a standard dataset instead."
            )

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        # 跟踪迭代次数（正向+反向传递），包括一个梯度累积周期内的迭代次数
        self._step = 0
        # 缓冲批处理，以便在多个更新中重用生成的输出。有关更多详细信息，请参阅
        # #“_get_train_sampler”和“_prepare_inputs”。
        self._buffered_inputs = [None] * args.gradient_accumulation_steps

        # 训练使用中的元素数量来估计FLOP（浮点运算）的数量
        # #与键“input_ids”关联的输入张量。
        # 然而，在GRPO中，采样数据不包括
        # #“input_ids”键。
        # 相反，可用的密钥是“提示”。
        # 因此，培训师发出警告：
        # #无法估计输入的令牌数，将不计算浮点运算
        # #取消此警告，我们将模型的“warnings_issued”字典中的“estimate_tokens”键设置为True。
        # #这相当于一个标志，表示警告已经发出。
        model.warnings_issued["estimate_tokens"] = True

        if self.use_liger_loss:
            if not is_liger_kernel_available():
                raise ImportError(
                    "Liger is required to use `liger_loss` as the GRPO loss. Run `pip install liger-kernel`."
                )
            if is_peft_model(model):
                raise ValueError("Liger loss is not supported with a PEFT model.")

            self.liger_grpo_loss = LigerFusedLinearGRPOLoss(
                beta=self.beta,
                epsilon_low=self.epsilon_low,
                epsilon_high=self.epsilon_high,
                temperature=self.temperature,
                use_ref_model=self.ref_model is not None,
            )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # 初始化指标
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.num_completions_to_print = args.num_completions_to_print
        # maxlen设置为每一步向前传递的总次数。`maxlen`的值确保我们只记录最后的优化步骤。
        maxlen = self.accelerator.num_processes * args.per_device_train_batch_size * args.gradient_accumulation_steps
        self._textual_logs = {
            "prompt": deque(maxlen=maxlen),
            "completion": deque(maxlen=maxlen),
            "rewards": defaultdict(lambda: deque(maxlen=maxlen)),
        }

        # 检查per_device_train/eval_batch_size*num进程是否可以除以代数
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations < 2:
            raise ValueError(
                f"GRPO requires at least 2 generations per prompt to calculate the advantages. "
                f"You provided {self.num_generations}, which is less than the minimum required."
            )
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        # 确保每个进程都收到一个唯一的种子，以防止在生成时重复完成
        # #如果num_generation超过per_device_train_batch_size，则使用变压器。
        # 如果我们使用vLLM，我们可以跳过它，但是
        # #在所有情况下设置它都更安全。
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            if self.accelerator.is_main_process:
                self.vllm_client = VLLMClient(
                    args.vllm_server_host, args.vllm_server_port, connection_timeout=args.vllm_server_timeout
                )

            # vLLM特定的采样参数
            self.guided_decoding_regex = args.vllm_guided_decoding_regex

            self._last_loaded_step = -1  # 标记以避免在梯度累积过程中无用的加载

            # 使用vLLM时，主进程负责加载模型权重。
            # 这可能会导致进程去同步，似乎会导致初始化期间DeepSpeed挂起。
            # 为了防止这种情况，我们在vLLM完全初始化后同步所有进程。
            self.accelerator.wait_for_everyone()
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                pad_token_id=processing_class.pad_token_id,
                bos_token_id=processing_class.bos_token_id,
                eos_token_id=processing_class.eos_token_id,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                repetition_penalty=self.repetition_penalty,
                cache_implementation=args.cache_implementation,
            )

        # 梯度累积需要按比例损失。
        # 通常，父类中的损失缩放取决于模型接受与损失相关的kwargs。由于我们计算自己的损失，因此此检查无关紧要。
        # 我们设置self.model_accepts_loss_kwargs设置为False以启用缩放。
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # 如果`self.args.remove_unused_columns`为True，则删除非签名列。
        # #默认情况下，此方法设置`self_signature_columns”到模型的预期输入。
        # #在GRPOTrainer中，我们对数据进行预处理，因此使用模型的签名列是行不通的。相反，我们将它们设置为“training_step”方法所期望的列，因此进行了重写。
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _get_train_sampler(self) -> Sampler:
        # 返回一个采样器
        #1.确保每个提示在多个进程中重复。这保证了相同的提示
        # 分布到不同的GPU，允许在每个提示内正确计算和规范奖励
        # 集团。跨流程使用相同的种子可确保一致的及时分配，防止差异
        # 在群体形成中。
        #2.多次重复该批处理，以允许在多个更新中重用代。提到
        # _prepare_inputs，查看如何存储和重用这些代。

        # 在下图中，这些值是提示索引。
        第一行显示了第一个采样批次
        # 第二行显示第二个采样批次，以此类推。
        #
        #                                     |     GPU 0     |     GPU 1     |     GPU 2    |
        #
        #               global_step   step     <───────>  num_generations=3
        #                                      <───────────> per_device_train_batch_size=4
        #                ▲   0          0      0   0   0   1   1   1   2   2   2   3   3   3  │
        #  grad_accum=3  │   0          1      4   4   4   5   5   5   6   6   6   7   7   7  │ Generate completions for each prompt
        #                ▼   0          2      8   8   8   9   9   9  10  10  10  11  11  11  │
        #
        #                    1          3      0   0   0   1   1   1   2   2   2   3   3   3  │ The sampled prompts are the same as in the first iteration
        #                    1          4      4   4   4   5   5   5   6   6   6   7   7   7  │ Reuse the completions (here, once, because num_iterations=2)
        #                    1          5      8   8   8   9   9   9  10  10  10  11  11  11  │
        #
        #                    2          6     12  12  12  13  13  13  14  14  14  15  15  15
        #                    2          7     16  16  16  17  17  17  18  18  18  19  19  19
        #                    2          8     20  20  20  21  21  21  22  22  22  23  23  23
        #                                          ...
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        return RepeatRandomSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # 有关采样器的说明，请参阅_get_train_sampler。
        return RepeatRandomSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # 确保已禁用use_cache
        model.config.use_cache = False

        # 在PEFT的基础模型上启用梯度检查点
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # 为非PEFT模型启用梯度检查点
        else:
            model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model

    @profiling_decorator
    def _get_last_hidden_state(self, model, input_ids, attention_mask, logits_to_keep=None):
        # 打开模型以访问model.model
        unwrapped_model = self.accelerator.unwrap_model(model)
        last_hidden_state = unwrapped_model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        if logits_to_keep is not None:
            last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
        return last_hidden_state

    # 获取模型和参考模型完成的每个令牌日志概率
    @profiling_decorator
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        # 我们在`logits_to_cheet`中添加1，因为序列的最后一个logits稍后会被排除在外
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, -logits_to_keep:]
        # 对于<=4.48的变压器，logits_to_head参数不受支持，因此在这里我们自己删除logits。
        # See https://github.com/huggingface/trl/issues/2770
        logits = logits[:, -logits_to_keep:]
        # 将logits除以采样温度。
        # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
        logits = logits / self.temperature
        return selective_log_softmax(logits, input_ids)  # compute logprobs for the input tokens

    @profiling_decorator
    def _move_model_to_vllm(self):
        # 对于DeepSpeed ZeRO-3，我们需要在操作前收集所有参数
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        gather_if_zero3 = deepspeed.zero.GatheredParameters if zero_stage_3 else nullcontext

        if is_peft_model(self.model):
            # 使用PEFT和DeepSpeed ZeRO Stage 3，我们必须在合并之前立即收集完整的模型，因为合并
            # #不支持分片方式的适配器。
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                # 在收集参数时更新vLLM权重
                for name, param in self.model.named_parameters():
                    # 使用PEFT时，我们需要恢复原始参数名称并丢弃一些参数
                    name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                    if self.model.prefix in name:
                        continue
                    # 保存模块时，删除其前缀并丢弃原始模块
                    if "original_module" in name:
                        continue
                    name = name.replace("modules_to_save.default.", "")

                    if self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(name, param.data)

                # 在仍在收集参数时取消合并适配器
                self.model.unmerge_adapter()
                # 退出上下文时，参数将自动重新分区
        else:
            # 对于非PEFT模型，只需单独收集和更新每个参数。
            for name, param in self.model.named_parameters():
                with gather_if_zero3([param]):
                    if self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(name, param.data)

        # 重置主进程上的缓存
        if self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()

    @profiling_decorator
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            buffer_index = self._step % self.args.gradient_accumulation_steps
            buffered_inputs = self._buffered_inputs[buffer_index]
            if self.state.global_step % self.num_iterations == 0 or buffered_inputs is None:
                # buffered_inputs=从检查点恢复时不会发生任何情况
                inputs = self._generate_and_score_completions(inputs)
                self._buffered_inputs[buffer_index] = inputs
            else:
                inputs = buffered_inputs
            self._step += 1
        else:
            # 在评估中，我们不会在多个更新之间重用补全，因此我们不需要缓冲输入。
            inputs = self._generate_and_score_completions(inputs)
        return inputs

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # 使用vLLM或常规生成生成完成
        if self.args.use_vllm:
            # #首先，如有需要，确定主要加载权重
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # 使用vLLM生成补全：收集所有提示并在主进程中的单个调用中使用它们
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                # 由于“promises”包含“num_generations”重复项，我们首先采用唯一的提示，并生成num_generation为每个输出。
                # 这比为每个副本生成输出更快单独提示。
                ordered_set_of_prompts = all_prompts_text[:: self.num_generations]
                with profiling_context(self, "vLLM.generate"):
                    completion_ids = self.vllm_client.generate(
                        prompts=ordered_set_of_prompts,
                        n=self.num_generations,
                        repetition_penalty=self.repetition_penalty,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=-1 if self.top_k is None else self.top_k,
                        min_p=0.0 if self.min_p is None else self.min_p,
                        max_tokens=self.max_completion_length,
                        guided_decoding_regex=self.guided_decoding_regex,
                    )
            else:
                completion_ids = [None] * len(all_prompts_text)
            # 将主流程的完成情况广播给所有流程，确保每个流程都收到其对应切片。
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]

            # 填充补全，并将其与提示连接起来
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # 常规生成路径
            with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                )

            # 计算提示长度并提取完成ID
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # 在第一个EOS代币之后屏蔽所有内容
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # 将prompt_mask与completion_mask连接起来进行logit计算
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # 我们只需要计算完成令牌的logits

        with torch.no_grad():
            # 当使用num_iterations==1时，old_per_token_logps==per_token_logs，所以我们可以跳过它
            # #在这里进行计算，并改用per_token_logs.detach（）。
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # 对生成的补全进行解码
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
            else:
                reward_func_name = reward_func.__name__
            with profiling_context(self, reward_func_name):
                if isinstance(
                    reward_func, nn.Module
                ):  #模块而不是预训练模型，以与编译后的模型兼容
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    # 重复所有输入列（但“提示”和“完成”除外）以匹配代数
                    keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                    reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                    output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                    # Convert None values to NaN
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # 如果给定行的所有奖励函数都返回None，则发出详细警告
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        # 收集每个功能的奖励：这部分至关重要，因为每个组的奖励都是标准化的
        # #完工可能分布在各个流程中
        rewards_per_func = gather(rewards_per_func)

        # 对每个奖励函数的输出和总和应用权重
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # 计算分组智能奖励
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # 将奖励标准化以计算优势
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.args.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # 切片仅保留数据的本地部分
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # log completion lengths, mean, min, max
        agg_completion_mask = self.accelerator.gather_for_metrics(completion_mask.sum(1))
        self._metrics[mode]["completions/mean_length"].append(agg_completion_mask.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_mask.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_mask.float().max().item())

        # 识别以EOS结尾的序列并记录其长度
        agg_terminated_with_eos = self.accelerator.gather_for_metrics(is_eos.any(dim=1))
        term_completion_mask = agg_completion_mask[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_mask) / len(agg_completion_mask)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_mask) == 0:
            # 未找到完整序列的边缘情况
            term_completion_mask = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_mask.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_mask.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_mask.float().max().item())

        # 获取奖励函数的名称
        reward_func_names = []
        for reward_func in self.reward_funcs:
            if isinstance(reward_func, nn.Module):  # 模块而不是预训练模型，以与编译后的模型兼容
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            reward_func_names.append(reward_func_name)

        # 计算每个函数的平均奖励，但仅适用于应用该函数的样本（非NaN值）
        for i, reward_func_name in enumerate(reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        # 日志提示和完成文本
        self._textual_logs["prompt"].extend(gather_object(prompts_text))
        self._textual_logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
        }

    def compute_liger_loss(self, model, inputs):
        # 计算模型的每个令牌的日志概率
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # 我们只需要计算完成令牌的logits

        # 获取模型的最后一个隐藏状态
        last_hidden_state = self._get_last_hidden_state(model, input_ids, attention_mask, logits_to_keep)
        unwrapped_model = self.accelerator.unwrap_model(model)
        # 使用liger grpo损失计算损失和度量
        loss, metrics = self.liger_grpo_loss(
            _input=last_hidden_state,
            lin_weight=unwrapped_model.lm_head.weight,
            selected_token_ids=completion_ids,
            attention_mask=completion_mask,
            advantages=inputs["advantages"],
            bias=unwrapped_model.lm_head.bias,
            ref_per_token_logps=inputs["ref_per_token_logps"],
            old_per_token_logps=inputs["old_per_token_logps"],
        )
        # 从liger_grpo_loss输出中提取指标
        # #当贝塔为非零时，KL散度是第一个度量
        mean_kl = metrics[0] if self.beta != 0.0 else None
        clip_ratio = metrics[-1]

        mode = "eval" if self.control.should_evaluate else "train"
        if self.beta != 0.0:
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        return loss

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        if self.use_liger_loss:
            # 用liger grpo损耗计算损耗
            return self.compute_liger_loss(model, inputs)
        else:
            return self._compute_loss(model, inputs)

    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # 我们只需要计算完成令牌的logits

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # 计算模型和参考模型之间的KL散度
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # 当使用num_iterations==1时，old_per_token_logps==per_token_logs，因此我们可以跳过它的计算（参见
        # #_generate_and_score_collections）并使用per_token_logps.detach（）代替。
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        # Compute the clip ratio
        is_clipped = ((coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)) | (
            (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        )
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "eval" if self.control.should_evaluate else "train"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # 这种方法既可以在训练中使用，也可以在评估中使用。
        # 在评估中调用时，`logs中的键`
        # #以“eval”开头。
        # 我们需要在“metrics”中的键前添加前缀“eval_”以匹配格式。
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process:
            if is_rich_available():
                print_prompt_completions_sample(
                    self._textual_logs["prompt"],
                    self._textual_logs["completion"],
                    self._textual_logs["rewards"],
                    self.state.global_step,
                    self.num_completions_to_print,
                )

            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd

                table = {
                    "step": [str(self.state.global_step)] * len(self._textual_logs["prompt"]),
                    "prompt": self._textual_logs["prompt"],
                    "completion": self._textual_logs["completion"],
                    **self._textual_logs["rewards"],
                }
                df = pd.DataFrame(table)
                if self.args.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])
                wandb.log({"completions": wandb.Table(dataframe=df)})

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))