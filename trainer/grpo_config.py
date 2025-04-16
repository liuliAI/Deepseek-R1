import warnings
from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class GRPOConfig(TrainingArguments):
    r"""
[`GRPOTrainer`]的配置类。

此处仅列出了GRPO训练的特定参数。
有关其他参数的详细信息，请参阅[`~transformers.TrainingArguments`]文档。

使用[`~transformers.HfArgumentParser`]，我们可以将这个类转换为[argparse](
https://docs.python.org/3/library/argparse#module
-argparse）参数，可以在命令行上指定。

参数：
>控制模型和参考模型的参数

model_init_kwargs（`dict[str，Any]`或`None `，*可选*，默认为`None `）：
[`~transformers.AutoModelForCausalLM.from_petrained`]的关键字参数，当[`GRPOTrainer`]的`model`参数以字符串形式提供时使用。

>控制数据预处理的参数

remove_unused_columns（`bool`，*可选*，默认为`False `）：
是否只保留数据集中的“prompt”列。如果您使用的自定义奖励函数需要除“提示”和“完成”之外的任何列，则应将其设置为“False”。

max_prompt_length（“int”或“None”，*可选*，默认为“512”）：
提示的最大长度。如果提示长于此值，它将被截断为左。

num_generations（`int`或`None `，*可选*，默认为`8`）：
每个提示采样的代数。

全局批大小（num_processs*per_device_batch_size）必须能被此值整除。

max_coompletion_length（`int`或`None `，*可选*，默认为`256`）：
生成的完成的最大长度。

ds3_gather_for_generation（'bool'，*可选*，默认为'True'）：
此设置适用于DeepSpeed ZeRO-3。如果启用，将收集策略模型权重以进行生成，从而提高生成速度。然而，禁用此选项允许训练模型超过单个GPU的VRAM容量，尽管代价是生成速度较慢。禁用此选项与vLLM生成不兼容。

>控制生成的参数

temperature（“浮动”，默认为“0.9”）：
取样温度。温度越高，完井越随机。

top_p（`float`，*可选*，默认为`1.0`）：
浮点数，控制要考虑的顶级令牌的累积概率。必须在（0，1]中。设置为“1.0”以考虑所有令牌。

top_k（“int”或“None”，*可选*，默认为“50”）：
保留用于top-k过滤的最高概率词汇标记的数量。如果为“无”，则禁用top-k筛选。

min_p（“float”或“None”，*可选*，默认为“None”）：
最小令牌概率，将根据最可能令牌的概率进行缩放。它必须是介于“0.0”和“1.0”之间的值。典型值在“0.01-0.2”范围内。

repetition_ppenalty（'float'，*可选*，默认为'1.0'）：
浮点数，根据新标记是否出现在提示中以及到目前为止生成的文本来惩罚新标记。值>“1.0”鼓励模型使用新的令牌，而值<“1.0”则鼓励模型重复令牌。

cache_implementation（`str`或`None `，*可选*，默认为`None `）：
当use_vllm设置为False时，实现缓存方法以加快生成速度。

>控制vLLM驱动的生成加速的参数

use_vllm（`bool `，*可选*，默认为`False `）：
是否使用vLLM生成完井。如果设置为“True”，请确保GPU不用于训练，因为vLLM需要一个GPU来生成。必须安装vLLM（`pip install vLLM`）。

vllm_server_host（`str`，*可选*，默认为`“0.0.0.0”`）：
要连接的vLLM服务器的主机。

vllm_server_port（`int`，*可选*，默认为`8000`）：
要连接到的vLLM服务器的端口。

vllm_server_timeout（`float`，*可选*，默认为`120.0`）：
等待vLLM服务器启动的总超时时间（秒）。如果服务器在超时后未启动，则会引发“ConnectionError”。

vllm_guided_decoding_regex（“str”或“None”，*可选*，默认为“None”）：
用于vLLM引导解码的正则表达式。如果“无”（默认），则禁用引导解码。

    """

    # 控制模型和参考模型的参数
    model_init_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when the `model` " #“transformers”的关键字参数。AutoModelForCausalLM.from_petrained，用于“模型”`
            "argument of the `GRPOTrainer` is provided as a string." #“GRPOTrainer”的参数以字符串形式提供。
        },
    )

#控制数据预处理的参数
#默认值remove_unused_columns会从父类中覆盖，因为在GRPO中我们通常依赖于
#用于计算奖励的附加列
    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to only keep the column 'prompt' in the dataset. If you use a custom reward function "  #是否仅在数据集中保留列“prompt”。如果您使用自定义奖励功能
            "that requires any column other than 'prompts' and 'completions', you should keep this to `False`."      #这需要除“提示”和“补全”之外的任何列，您应该将其保持为“False”。
        },
    )
    max_prompt_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left."  #提示的最大长度。如果提示长于此值，它将被截断为左。
        },
    )
    num_generations: Optional[int] = field(
        default=8,
        metadata={
            "help": "Number of generations to sample. The global batch size (num_processes * per_device_batch_size) "   #要采样的数。全局批大小（num_processs*per_device_batch_size） 必须能被这个值整除。
            "must be divisible by this value."
        },
    )
    max_completion_length: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum length of the generated completion."},  #生成的完成的最大长度。
    )
    ds3_gather_for_generation: bool = field(
        default=True,
        metadata={
            "help": "This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for "
            "generation, improving generation speed. However, disabling this option allows training models that "
            "exceed the VRAM capacity of a single GPU, albeit at the cost of slower generation. Disabling this option "
            "is not compatible with vLLM generation."
            #此设置适用于DeepSpeed ZeRO-3。如果启用，将收集策略模型权重以进行生成，从而提高生成速度。然而，禁用此选项允许训练模型超过单个GPU的VRAM容量，尽管代价是生成速度较慢。禁用此选项与vLLM生成不兼容。
        },
    )

#控制生成的参数
    temperature: float = field(
        default=0.9,
        metadata={"help": "Temperature for sampling. The higher the temperature, the more random the completions."},#取样温度。温度越高，完井越随机。
    )
    top_p: float = field(
        default=1.0,
        metadata={
            "help": "Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. "#浮点数，控制要考虑的顶级令牌的累积概率。必须在（0，1]中。设置为1.0以考虑所有令牌。
            "Set to 1.0 to consider all tokens."
        },
    )
    top_k: Optional[int] = field(
        default=50,
        metadata={
            "help": "Number of highest probability vocabulary tokens to keep for top-k-filtering. If `None`, " #保留用于top-k过滤的最高概率词汇标记的数量。如果为“无”，则禁用top-k筛选。
            "top-k-filtering is disabled."
        },
    )
    min_p: Optional[float] = field(
        default=None,
        metadata={
            "help": "Minimum token probability, which will be scaled by the probability of the most likely token. It "
            "must be a value between 0.0 and 1.0. Typical values are in the 0.01-0.2 range."   #最小令牌概率，将根据最可能令牌的概率进行缩放。它必须是介于0.0和1.0之间的值。典型值在0.01-0.2范围内。
        },
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the prompt and the generated "
            "text so far. Values > 1.0 encourage the model to use new tokens, while values < 1.0 encourage the model "
            "to repeat tokens."  #浮点数，根据新标记是否出现在提示中以及到目前为止生成的文本来惩罚新标记。值>1.0鼓励模型使用新的令牌，而值<1.0鼓励模型重复令牌。
        },
    )
    cache_implementation: Optional[str] = field(
        default=None,
        metadata={"help": "Implementation of the cache method for faster generation when use_vllm is set to False."}, #当use_vllm设置为False时，实现缓存方法以加快生成速度。
    )

#控制vLLM驱动的发电加速的参数
    use_vllm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use vLLM for generating completions. If set to `True`, ensure that a vLLM server is "
            "running. To run the server, install vLLM (`pip install vllm`) and run `trl vllm-serve`."  #是否使用vLLM生成完井。如果设置为“True”，请确保vLLM服务器正在运行。要运行服务器，请安装vLLM（`pip install vLLM`）并运行`trl vLLM serve`。
        },
    )
    vllm_server_host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host of the vLLM server to connect to."},
    )
    vllm_server_port: int = field(
        default=8000,
        metadata={"help": "Port of the vLLM server to connect to."},
    )
    vllm_server_timeout: float = field(
        default=120.0,
        metadata={
            "help": "Total timeout duration in seconds to wait for the vLLM server to be up. If the server is not up "
            "after the timeout, a `ConnectionError` is raised."
        },
    )
    vllm_guided_decoding_regex: Optional[str] = field(
        default=None,
        metadata={"help": "Regex for vLLM guided decoding. If `None` (default), guided decoding is disabled."},
    )

#控制训练的参数
    learning_rate: float = field(
        default=1e-6,
        metadata={
            "help": "Initial learning rate for `AdamW` optimizer. The default value replaces that of "
            "`transformers.TrainingArguments`."  #“AdamW”优化器的初始学习率。默认值将替换`transformers.TrainingArguments`的默认值。
        },
    )
    beta: float = field(
        default=0.04,
        metadata={
            "help": "KL coefficient. If `0.0`, the reference model is not loaded, reducing memory usage and improving "
            "training speed, but may be numerically unstable for long training runs." #KL系数。如果为“0.0”，则不加载参考模型，从而减少内存使用并提高训练速度，但对于长时间的训练运行，其数值可能不稳定。
        },
    )
    num_iterations: int = field(
        default=1,
        metadata={"help": "Number of iterations per batch (denoted as μ in the algorithm)."},  #每批迭代次数（在算法中表示为μ）。
    )
    epsilon: float = field(
        default=0.2,
        metadata={"help": "Epsilon value for clipping."}, #剪切的Epsilon值。
    )
    epsilon_high: Optional[float] = field(
        default=None,
        metadata={
            "help": "Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the "
            "lower-bound specified in argument `epsilon`. Paper DAPO recommends `0.28`."  #剪切的上限epsilon值。如果未指定，则默认为与参数“epsilon”中指定的下限相同的值。DAPO论文建议为“0.28”。
        },
    )
    reward_weights: Optional[list[float]] = field(
        default=None,
        metadata={
            "help": "Weights for each reward function. Must match the number of reward functions. If `None`, all "
            "rewards are weighted equally with weight `1.0`."  #每个奖励函数的权重。必须与奖励函数的数量相匹配。如果为“无”，则所有奖励的权重均为“1.0”。
        },
    )
    scale_rewards: bool = field(
        default=True,
        metadata={
            "help": "Whether to scale the rewards by dividing them by their standard deviation. If `True` (default), "
            "the rewards are normalized by the standard deviation, ensuring they have unit variance. If `False`, no "
            "scaling is applied. The Dr. GRPO paper recommends not scaling the rewards, as scaling by the standard "
            "deviation introduces a question-level difficulty bias."  #是否通过将奖励除以标准差来衡量奖励。如果为“True”（默认值），则奖励将按标准偏差进行归一化，确保它们具有单位方差。如果为“False”，则不应用缩放。
                                                                    # GRPO博士的论文建议不要缩放奖励，因为按标准差缩放会引入问题级别的难度偏差。
        },
    )
    sync_ref_model: bool = field(
        default=False,
        metadata={
            "help": "Whether to synchronize the reference model with the active model every `ref_model_sync_steps` "
            "steps, using the `ref_model_mixup_alpha` parameter."   #是否使用“ref_model_mixup_alpha”参数，在每个“ref_model_sync_steps”步骤将参考模型与活动模型同步。
        },
    )
    ref_model_mixup_alpha: float = field(
        default=0.6,
        metadata={
            "help": "α parameter from the TR-DPO paper, which controls the mix between the current policy and the "  #来自TR-DPO论文的α参数，该参数控制更新期间当前策略和先前参考策略之间的混合。
             # 参考策略根据以下方程式进行更新：“π_ref=α*π_θ+（1-α）*π_ref_prev”。要使用此参数，必须设置`sync_ref_model=True`。
            "previous reference policy during updates. The reference policy is updated according to the equation: "
            "`π_ref = α * π_θ + (1 - α) * π_ref_prev`. To use this parameter, you must set `sync_ref_model=True`."
        },
    )
    ref_model_sync_steps: int = field(
        default=512,
        metadata={
            "help": "τ parameter from the TR-DPO paper, which determines how frequently the current policy is " #τ参数，用于确定当前策略与参考策略同步的频率。要使用此参数，必须设置`sync_ref_model=True`。
            "synchronized with the reference policy. To use this parameter, you must set `sync_ref_model=True`."
        },
    )
    use_liger_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use the Liger GRPO loss."}, #是否使用Liger GRPO损失。
    )

#控制日志记录的参数
    log_completions: bool = field(
        default=False,
        metadata={
            "help": "Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is "
            "installed, it prints the sample. If `wandb` logging is enabled, it logs it to `wandb`."  #是否记录每个“logging_step”步骤的（提示、完成）对样本。如果安装了“rich”，它将打印示例。如果启用了“wandb”日志记录，它会将其记录到“wandb'”。
        },
    )
    num_completions_to_print: Optional[int] = field(
        default=None,
        metadata={"help": "Number of completions to print with `rich`. If `None`, all completions are logged."}, #使用“丰富”打印的完成次数。如果为“无”，则记录所有完成情况。
    )
    wandb_log_unique_prompts: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to log unique prompts in wandb. If `True`, only unique prompts are logged. If `False`, " #是否在wandb中记录唯一提示。如果为“True”，则只记录唯一的提示。如果为“False”，则记录所有提示。
            "all prompts are logged."
        },
    )

#弃用的参数
    vllm_device: Optional[str] = field(
        default=None,
        metadata={
            "help": "This parameter is deprecated and will be removed in version 0.18.0. To use vLLM, start a vLLM "
            "server with the `trl vllm-serve` command."
        },
    )
    vllm_gpu_memory_utilization: Optional[float] = field(
        default=None,
        metadata={
            "help": "This parameter is deprecated and will be removed in version 0.18.0. To control the GPU memory "
            "utilization for vLLM, you should now use the `gpu_memory_utilization` parameter in the vLLM server "
            "configuration."
        },
    )
    vllm_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "This parameter is deprecated and will be removed in version 0.18.0. To control the data type for "
            "vLLM generation, you should now use the `dtype` parameter in the vLLM server configuration."
        },
    )
    vllm_max_model_len: Optional[int] = field(
        default=None,
        metadata={
            "help": "This parameter is deprecated and will be removed in version 0.18.0. To control the "
            "`max_model_len` for vLLM, you should now use the `max_model_len` parameter in the vLLM server "
            "configuration."
        },
    )
    vllm_enable_prefix_caching: Optional[bool] = field(
        default=None,
        metadata={
            "help": "This parameter is deprecated and will be removed in version 0.18.0. To control prefix caching in "
            "vLLM, you should now use the `enable_prefix_caching` parameter in the vLLM server configuration."
        },
    )

    def __post_init__(self):
        super().__post_init__()

        if self.vllm_device is not None:
            warnings.warn(
                "`vllm_device` is deprecated and will be removed in version 0.18.0. To use vLLM, start a vLLM server "
                "with the `trl vllm-serve` command.",
                DeprecationWarning,
            )

        if self.vllm_gpu_memory_utilization is not None:
            warnings.warn(
                "`vllm_gpu_memory_utilization` is deprecated and will be removed in v0.18. To control the GPU memory "
                "utilization for vLLM, you should now use the `gpu_memory_utilization` parameter in the vLLM server "
                "configuration.",
                DeprecationWarning,
            )

        if self.vllm_dtype is not None:
            warnings.warn(
                "`vllm_dtype` is deprecated and will be removed in version 0.18.0. To control the data type for vLLM "
                "generation, you should now use the `dtype` parameter in the vLLM server configuration.",
                DeprecationWarning,
            )

        if self.vllm_max_model_len is not None:
            warnings.warn(
                "`vllm_max_model_len` is deprecated and will be removed in version 0.18.0. To control the "
                "`max_model_len` for vLLM, you should now use the `max_model_len` parameter in the vLLM server "
                "configuration.",
                DeprecationWarning,
            )

        if self.vllm_enable_prefix_caching is not None:
            warnings.warn(
                "`vllm_enable_prefix_caching` is deprecated and will be removed in version 0.18.0. To control prefix "
                "caching in vLLM, you should now use the `enable_prefix_caching` parameter in the vLLM server "
                "configuration.",
                DeprecationWarning,
            )