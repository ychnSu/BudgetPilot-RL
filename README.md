# BudgetPilot-RL

**面向预算约束仓库修复的 Coding Agent 后训练项目。**

BudgetPilot-RL 让模型在给定预算内修复代码仓库中的缺陷。模型读取 issue，检索和修改代码，根据测试反馈决定继续排查还是提交补丁。每轮输入都包含剩余 Token、工具调用次数和运行时间。

项目使用 Qwen3-8B 作为基础模型，LangGraph 管理交互，Docker 执行仓库工具。训练部分提供 SFT 数据导出、LoRA 配置，以及 GRPO 的奖励、组内优势和策略损失函数。

## 核心设计

- 每轮选择一个工具动作，包括查看文件、搜索、编辑、测试、检查 diff 和提交。
- 将测试判定与执行成本一起计入奖励，减少重复检索和无效重试。
- 每条轨迹使用独立容器；标准补丁和最终测试由外部评测器管理。
- 按仓库划分数据，导出 SWE-bench 预测，同时统计成功率和运行成本。

## 系统架构

```text
Issue + Repository Base Commit + Budget
                    |
           LangGraph Agent Loop
                    |
        Observe -> Decide -> Act
           ^                |
           |          Docker Repository Tools
           +------ Execution Feedback
                    |
                Final Patch
                    |
          Independent Test Verifier
                    |
        Outcome Reward + Budget Costs
                    |
          Group-Relative Advantages
                    |
           GRPO Objective Components
```

工具执行结果会进入下一轮上下文。轨迹文件保存模型请求、工具参数、返回结果、资源消耗和终止原因。

## Agentic RL 方法

只奖励最终修复成功，无法区分一次有效修改和多轮重复尝试。因此，奖励函数同时计算修复结果与执行成本：

```text
R = R_outcome
    - 0.15 × C_token
    - 0.10 × C_tool
    - 0.05 × C_time
    - 0.05 × C_repeat
```

修复成功计 +1，失败计 -1。Token、工具调用和耗时除以各自预算，上限取 1；重复动作按相同工具名称和参数的重复比例计算。权重可在 `configs/default.yaml` 中调整。

同一任务、同一预算下采样多条轨迹，计算组内标准化优势。`training.py` 实现带裁剪和 KL 项的 GRPO 损失，action mask 用于排除工具返回等非模型生成内容。

SFT 使用通过独立验证的训练轨迹，导出为工具对话后按 LoRA 配置训练。RL 使用多轮轨迹的奖励和组内优势；训练接口所需的 token ID、logprob 和 mask 见 [训练文档](docs/TRAINING.md)。

## 技术栈

| 层级 | 技术与组件 |
|---|---|
| 基础模型 | Qwen3-8B、OpenAI-compatible 模型服务 |
| 多轮交互 | LangGraph |
| 推理服务 | vLLM |
| 监督微调 | LLaMA-Factory、LoRA |
| 强化学习组件 | PyTorch、GRPO 奖励与损失、veRL 接口设计 |
| 执行环境 | Docker、Git、pytest |
| 数据处理 | Hugging Face Datasets、JSONL、Parquet |
| 评测 | SWE-bench harness、自定义成本统计 |

## 数据集

| 用途 | 数据源 | 链接 |
|---|---|---|
| 训练任务与轨迹采集 | SWE-Gym | [下载数据集](https://huggingface.co/datasets/SWE-Gym/SWE-Gym) |
| 仓库修复评测 | SWE-bench Lite | [下载数据集](https://huggingface.co/datasets/SWE-bench/SWE-bench_Lite) |
| 扩展评测 | SWE-bench Verified | [下载数据集](https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified) |
| 扩展执行环境 | R2E-Gym | [项目仓库](https://github.com/agentica-project/R2E-Gym) |

原始数据放在 `data/raw/`。转换后的 Agent 输入包含任务 ID、仓库、base commit 和问题描述。标准补丁与最终验证测试只供评测器使用。

划分脚本按仓库分组，并排除与评测集重叠的仓库、任务 ID 和 base commit。抽样评测时固定随机种子和任务清单，结果按子集报告。格式见 [数据说明](data/README.md)。

## 安装与配置

需要 Python 3.11+ 和 Docker，建议在 Linux 或 WSL2 下运行。模型推理与训练另需配置 CUDA 环境。

```bash
pip install -e '.[data]'

export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=EMPTY
export MODEL_NAME=Qwen/Qwen3-8B
```

vLLM 需配置对应版本的 Qwen3 工具调用 parser 和 chat template。服务必须返回 API usage，程序据此累计输入和输出 Token。

默认预算配置：

```yaml
budget:
  tokens: 16000
  tool_calls: 20
  wall_seconds: 600
  max_turns: 24
  max_completion_tokens: 2048
```

程序在每轮交互时检查预算。单次请求仍可能使累计 Token 超出阈值，轨迹中记录实际消耗。

## 数据准备

```bash
python scripts/local_dataset_to_jsonl.py \
  --input data/raw/swegym --split train \
  --output data/raw/swegym.jsonl

budgetpilot prepare \
  --input data/raw/swegym.jsonl \
  --output data/processed/swegym.jsonl

python scripts/local_dataset_to_jsonl.py \
  --input data/raw/swebench-lite --split test \
  --output data/raw/lite.jsonl

budgetpilot prepare \
  --input data/raw/lite.jsonl --limit 60 --seed 42 \
  --output data/processed/eval60.jsonl

python scripts/split_by_repo.py \
  --input data/processed/swegym.jsonl \
  --eval-tasks data/processed/eval60.jsonl \
  --dev-repos YOUR_DEV_REPO \
  --output-dir data/processed/split
```

将 `YOUR_DEV_REPO` 替换为开发集仓库名。以上脚本读取本地文件，运行前需下载数据集。

## 仓库修复与轨迹采集

每个任务需要一个 Docker 镜像，`/repo` 中放置指定 `base_commit` 的干净仓库，并安装任务依赖、Python、Git、pytest 和 sleep。镜像不包含标准补丁和最终测试补丁。

在 JSON 文件中指定每个任务的镜像：

```json
{
  "YOUR_INSTANCE_ID": "your-task-image:version"
}
```

采样并导出补丁：

```bash
budgetpilot rollout \
  --tasks data/processed/eval60.jsonl \
  --images data/processed/images.json \
  --config configs/default.yaml \
  --samples 4 \
  --output runs/rollouts.jsonl

budgetpilot export \
  --trajectories runs/rollouts.jsonl \
  --output-dir runs/predictions
```

每次采样创建新容器，结束后停止并保留供检查。输出文件包含补丁、动作序列、工具返回和资源消耗。

## 监督微调数据

将通过验证的训练轨迹导出为 ShareGPT 格式，再使用 LLaMA-Factory 的 LoRA 配置：

```bash
python scripts/export_sft.py \
  --scored runs/train-scored.jsonl \
  --train-tasks data/processed/split/train.jsonl \
  --output data/processed/sft.json

cp configs/dataset_info.json data/processed/dataset_info.json
llamafactory-cli train configs/llamafactory_sft.yaml
```

导出脚本会核对训练清单，拒绝清单外的任务。训练前需检查模板中的工具消息和 loss mask，工具反馈只作为上下文。

## 评测与统计

使用 SWE-bench 官方 harness 在独立容器中评测补丁。每个 sample 使用不同的 `run_id`：

```bash
pip install -e '.[eval]'

python -m swebench.harness.run_evaluation \
  --dataset_name SWE-bench/SWE-bench_Lite \
  --predictions_path runs/predictions/sample-0.jsonl \
  --run_id budgetpilot-sample-0
```

根据各任务的评测报告生成 verdicts 文件：

```json
{"instance_id":"YOUR_INSTANCE_ID","sample_id":0,"resolved":true}
```

计算奖励与组内优势，并汇总结果：

```bash
budgetpilot score \
  --trajectories runs/rollouts.jsonl \
  --verdicts runs/verdicts.jsonl \
  --output runs/scored.jsonl

python scripts/summarize.py --scored runs/scored.jsonl
```

| 指标 | 定义 |
|---|---|
| empirical pass@1 | 每题独立采样成功比例的任务平均值 |
| observed pass@k | 每题 k 次采样中至少成功一次的任务比例 |
| Token 消耗 | 每条轨迹累计 API prompt + completion tokens |
| 工具调用次数 | 每条轨迹实际执行的工具动作次数 |
| 执行耗时 | 每条轨迹的端到端运行时间 |

比较 SFT、结果奖励 GRPO 与预算感知 GRPO 时，保持任务、模型和预算一致。超时和环境错误单独记录，具体规则见 [评测文档](docs/EVALUATION.md)。

## 目录结构

```text
BudgetPilot-RL/
├── configs/                 # 预算、奖励、LoRA 与数据格式配置
├── data/                    # 数据来源说明与输入示例
├── docs/                    # 训练接口与评测协议
├── examples/                # 仓库任务环境示例
├── scripts/                 # 数据转换、划分、SFT 导出与统计
├── src/budgetpilot/
│   ├── agent.py             # LangGraph 多轮决策
│   ├── budget.py            # 预算与使用量追踪
│   ├── tools.py             # Docker 仓库工具
│   ├── rewards.py           # 结果奖励、成本与组内优势
│   ├── training.py          # GRPO 策略损失
│   └── cli.py               # 命令行入口
└── pyproject.toml
```

## 参考项目

- [ReTool](https://github.com/ReTool-RL/ReTool)：工具使用策略的强化学习方法。
- [veRL Agent Loop](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)：多轮 Agent rollout 接口。
- [SWE-Gym](https://github.com/SWE-Gym/SWE-Gym)：软件工程训练任务与执行环境。
- [SWE-bench](https://www.swebench.com/SWE-bench/guides/evaluation/)：仓库修复评测流程。
