# Qwen3 8B 法律模型训练与验证报告

报告日期：2026-10-07  
项目：LexiLaw  
基座模型：`/workspace/Qwen3-8B`

## 结论摘要

本次 BF16 LoRA 微调有效提升了 Qwen3-8B 对现行法条、版本信息和目标回答格式的拟合能力。三轮训练中，第 2 轮 `checkpoint-270` 的验证集损失最低，应作为当前最佳模型。第 3 轮验证损失回升，表明继续训练已经出现轻微过拟合。

在相同的 30 条未参与训练的验证问题上，最佳 LoRA 的平均文本相似度从基座模型的 `0.1408` 提升到 `0.4557`，30 条全部优于基座；版本信息命中率从 `2/30` 提升到 `30/30`。但测试也发现“版本正确、法条正文错误”的串条案例，因此该模型不能脱离官方法条检索和结果校验直接用于真实法律业务。

推荐模型：

```text
/workspace/openel/outputs/qwen3-8b-legal-ascend-lora-eval/checkpoint-270
```

## 训练环境

| 项目 | 配置 |
|---|---|
| 加速设备 | 2 × Ascend 910 9382 |
| 单卡显存 | 64 GB |
| Python | 3.12.13 |
| PyTorch | 2.10.0+cpu，与 torch_npu 配合使用 |
| torch_npu | 2.10.0.post4 |
| Transformers | 5.5.4 |
| PEFT | 0.21.0 |
| TRL | 1.14.2 |
| DeepSpeed | 0.19.7 |
| 基座模型 | Qwen3-8B BF16 |
| 训练方式 | LoRA SFT，不是 INT4 QLoRA |

`torch=2.10.0+cpu` 是当前 torch_npu 软件栈的版本标识，不代表训练使用 CPU。训练期间两张 NPU 均被正常识别和使用。

## 数据划分

原始数据文件：`/workspace/openel/data/sft/current_law_sft.generated.json`

使用固定随机种子 `42` 按 8:2 划分：

| 数据集 | 数量 | 占比 | 文件 |
|---|---:|---:|---|
| 训练集 | 2,158 | 79.99% | `data/sft/current_law_sft.train.json` |
| 验证集 | 540 | 20.01% | `data/sft/current_law_sft.validation.json` |
| 合计 | 2,698 | 100% | - |

训练集和验证集 ID 重叠为 `0`，合并后覆盖全部 `2,698` 条记录。划分脚本为 `/workspace/openel/scripts/split_sft_dataset.py`。

## 训练参数

| 参数 | 值 |
|---|---:|
| Epochs | 3 |
| Learning rate | 2e-4 |
| 单卡训练 batch size | 4 |
| 梯度累积步数 | 2 |
| NPU 数量 | 2 |
| 有效全局 batch size | 16 |
| 单卡验证 batch size | 4 |
| 最大序列长度 | 512 |
| BF16 | 开启 |
| Gradient checkpointing | 关闭 |
| Packing | 关闭 |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| LoRA target modules | all-linear |
| DeepSpeed | ZeRO-3 |
| CPU parameter offload | 关闭 |
| CPU optimizer offload | 关闭 |
| 保存和验证策略 | 每个 epoch |

有效全局 batch size 为 `4 × 2 × 2 = 16`。关闭 packing 是为了规避当前 Ascend attention 实现下不同样本发生注意力串扰的风险。

## 训练运行结果

| 指标 | 结果 |
|---|---:|
| 总 optimizer steps | 405 |
| 每个 epoch steps | 135 |
| 训练总耗时 | 1,750 秒，约 29 分 10 秒 |
| 每秒训练样本数 | 3.699 |
| 每秒 optimizer steps | 0.231 |
| 最终训练 loss | 0.5034 |

训练输出目录：`/workspace/openel/outputs/qwen3-8b-legal-ascend-lora-eval`  
训练日志：`/workspace/openel/qwen_lora_eval.log`

## 验证集结果

每轮均在完整的 540 条验证集上计算 teacher-forcing 指标。

| Epoch | Checkpoint | Eval loss | Eval token accuracy | Eval entropy | 验证耗时 |
|---:|---|---:|---:|---:|---:|
| 1 | `checkpoint-135` | 0.583949 | 0.844197 | 0.573559 | 59.69 秒 |
| 2 | `checkpoint-270` | **0.579790** | **0.846086** | 0.525869 | 59.35 秒 |
| 3 | `checkpoint-405` | 0.610879 | 0.843957 | 0.464754 | 60.68 秒 |

第 2 轮验证损失最低。第 3 轮训练损失继续下降，但验证损失从 `0.579790` 上升到 `0.610879`，说明模型开始更加贴合训练集而没有继续改善验证集表现。

输出目录根部的 `adapter_model.safetensors` 是第 3 轮最终模型，不是验证结果最好的模型。当前应优先使用 `checkpoint-270`。

## 生成式对比测试

### 测试方法

从固定 seed 划分后的验证集中选取前 30 条未见样本，对原始 Qwen3-8B 和加载 `checkpoint-270` 的 LoRA 模型使用完全相同的问题进行确定性生成。

```text
do_sample=False
max_new_tokens=384
enable_thinking=False
```

测试对输出进行 Unicode NFKC 归一化，去除空白和大部分标点后，使用字符序列相似度比较生成结果与参考答案。该指标适合衡量同一批样本上的相对变化，但不能代替法律事实核验。

评估脚本：`/workspace/openel/scripts/evaluate_qwen_ascend.py`

### 汇总结果

| 指标 | 原始 Qwen3-8B | checkpoint-270 | 变化 |
|---|---:|---:|---:|
| 平均文本相似度 | 0.140797 | **0.455747** | +0.314950 |
| 中位数相似度 | 0.140978 | **0.404748** | +0.263770 |
| 相似度至少 0.2 | 2/30 | **28/30** | +26 |
| 相似度至少 0.4 | 0/30 | **16/30** | +16 |
| 相似度至少 0.6 | 0/30 | **6/30** | +6 |
| 相似度至少 0.8 | 0/30 | **1/30** | +1 |
| 版本信息命中 | 2/30 | **30/30** | +28 |
| 单题相似度胜出 | 0/30 | **30/30** | +30 |
| 严格归一化精确匹配 | 0/30 | 0/30 | 无变化 |
| 完整参考答案包含率 | 0/30 | 0/30 | 无变化 |

LoRA 平均相似度相对基座提高约 `223.69%`。严格精确匹配和完整包含率仍为零，原因包括引号和表述差异、回答格式变化，以及部分回答只覆盖法条的一部分；这也说明不能用训练 token accuracy 代替实际法条正确率。

详细结果：

```text
/workspace/openel/evaluation/base_30.json
/workspace/openel/evaluation/adapter270_30.json
```

## 典型案例

### 改善明显的案例

问题：`请给出中华人民共和国商业银行法第十一条的现行内容，并说明版本。`

参考答案要求给出 2015 年版本及商业银行设立审批、禁止未经批准吸收公众存款、名称中不得使用“银行”等内容。基座模型给出了较多解释性文字，正文不够贴近参考文本；LoRA 模型正确生成版本日期和主要法条正文，归一化相似度相对基座提高约 `0.7618`。

### 仍然存在的高风险案例

问题：`请给出金融资产管理公司条例第三十四条的现行内容，并说明版本。`

参考答案：

```text
金融资产管理公司条例第三十四条（2000年版本（2000-11-10公布））规定：本条例自公布之日起施行。
```

LoRA 模型正确生成了版本信息，却错误生成了另一条关于监管处罚的长文本。这是典型的“版本正确、正文串条”问题，说明模型学会了版本和格式模式，但没有稳定建立条号与正文之间的一一对应关系。

## 效果判断

已确认的提升：

- 模型明显学会了数据集中的法律版本表达方式。
- 回答格式更接近目标法律问答格式。
- 验证样本上的法条文本相似度显著提高。
- 最佳 LoRA 在测试的 30 条样本上全部优于原始基座。

尚未解决的问题：

- 存在法条串条和正文幻觉。
- 严格精确匹配率仍为零。
- 30 条生成测试只是验证集抽样，不能代表全部法律领域。
- 验证集按记录随机划分，同一部法律的不同条文可能同时存在于训练集和验证集，因此不能完全衡量跨法律泛化能力。
- 模型输出不应被视为正式法律意见。

## 使用建议

1. 当前部署或后续测试优先使用 `checkpoint-270`。
2. 推理时先根据法律名称、版本和条号从官方法条库检索正文，再让模型基于检索结果组织回答。
3. 对输出执行结构化校验，至少核对法律名称、条号、版本日期和正文哈希或文本相似度。
4. 增加“相邻条号混淆”“不存在条文拒答”“旧版与现行版区分”等专门测试集。
5. 后续训练可以减少到 2 epochs，或启用按 `eval_loss` 自动加载最佳 checkpoint。
6. 若需要评估跨法律泛化能力，应按法律或法规名称分组划分训练集和验证集，而不是按单条记录随机划分。

## 复现命令

使用最佳 checkpoint 做单条推理：

```bash
cd /workspace/openel
python inference_qwen_ascend.py \
  --model /workspace/Qwen3-8B \
  --adapter outputs/qwen3-8b-legal-ascend-lora-eval/checkpoint-270 \
  --device npu:0 \
  --question "《劳动合同法》第三十七条现行规定是什么？"
```

重新运行 30 条 LoRA 评估：

```bash
python scripts/evaluate_qwen_ascend.py \
  --model /workspace/Qwen3-8B \
  --adapter outputs/qwen3-8b-legal-ascend-lora-eval/checkpoint-270 \
  --data data/sft/current_law_sft.validation.json \
  --output evaluation/adapter270_30.json \
  --limit 30 \
  --max-new-tokens 384 \
  --device npu:0
```

