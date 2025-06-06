## 仅需一个百来行代码的py文件即可实现R1的复现

### 进行环境配置
```bash
conda create -n ds python=3.10
conda activate ds
pip install -r requirements.txt
```

### 下载Qwen2.5-0.5B-Instruct模型

确保目录结构如下：
```
----deepseek_r1_train.py
----gsm8k_chinese
--------data
------------test-00000-of-00001.parquet
------------train-00000-of-00001.parquet
----Qwen2.5-0.5B-Instruct
--------config.json
--------configuration.json
--------generation_config.json
--------LICENSE
--------merges.txt
--------model.safetensors
--------README.md
--------tokenizer_config.json
--------tokenizer.json
--------vocab.json
```

### 启动训练
使用命令`python deepseek_r1_train.py`

### 得到输出
```
----output
--------added_tokens.json
--------special_tokens_map.json
--------tokenizer_config.json
--------tokenizer.json
--------training_args.bin
--------vocab.json
--------checkpoint-100
------------...
--------checkpoint-200
------------...
--------runs
```

#deepseek-r1解读

![deepseek系列相关模型架构](https://github.com/liuliAI/Deepseek-R1/blob/main/image/ds.png)

总体流程：在进行强化学习前引入了sft冷启动；随后进行推理强化学习训练；在RL过程接近收敛时，通过拒绝采样（rejection sampling）方法从RL检查点生成新的SFT数据，并结合写作、事实QA和自我认知等领域的监督数据重新训练模型；最后，使用新数据完成微调后的检查点进行额外的RL训练。包含两个RL阶段用于优化推理模式和人类偏好对齐，以及两个SFT阶段用于构建模型的推理和非推理基础能力。阶段1和2的训练过程，只是为了给真正的R1模型训练收集数据，真正微调的仍是DeepSeek-V3-Base，而非训练阶段1和2的中间模型。

## 常规的大模型训练方案
pretrain -> sft -> rl

## DeepSeek-R1-Zero
pretrain -> rl

缺陷：中英文混合、格式混乱

## DeepSeek-R1
pretrain -> sft一阶段 -> rl一阶段 -> sft二阶段 -> rl二阶段

### sft一阶段（冷启动）

目的：引入数千条高质量长推理链数据对基础模型微调，强制规范输出格式（如\<think>推理过程\</think>），提升可读性。\
数据来源：收集DeepSeek-R1-Zero的输出结果，以可读的格式呈现，最后通过人工标注者进行后处理以优化结果

### rl一阶段（推理导向的rl）

rl方法：GRPO\
奖励模型：基于规则的奖励（答案准确性和语言一致性），针对代码、数学、编程等有固定答案的任务设计奖励函数。

### sft二阶段

数据来源：推理数据和非推理数据合并

推理数据：rl一阶段checkpoint输出数据（60万）。rl一阶段，仅纳入了可以基于规则的奖励进行评估的数据。在sft二阶段，通过引入额外的数据来扩展数据集，其中一些数据通过将真实答案和模型预测输入DeepSeek-V3进行判断，使用生成式奖励模型。此外，由于模型输出有时会显得混乱且难以阅读，过滤掉了包含混合语言、长段落和代码块的推理链。对于每个提示，采样多个回答，仅保留正确的回答。收集了大约60万个与推理相关的训练样本。

非推理数据：如写作、事实问答、自我认知和翻译等，重用DeepSeek-V3监督微调数据集的部分内容。收集了大约20万个与推理无关的训练样本。

### rl二阶段(通用对齐的rl)

通用对齐RL（RLHF）：融入人类偏好奖励模型（Helpfulness & Harmlessness），确保模型在开放域任务中的安全性与实用性。