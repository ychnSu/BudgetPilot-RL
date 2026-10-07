# BudgetPilot-RL

**面向预算约束仓库修复的 Coding Agent 与 Agentic RL 研究原型。**

基于 Qwen3-8B-compatible 推理服务与 LangGraph，将代码检索、编辑、测试和提交组织成多轮交互；在观测中暴露剩余预算，研究修复成功率与 Token、工具调用和时间成本之间的权衡。

> 状态：初版代码，未运行验证。无已训练模型、无实测 benchmark 分数。多轮 Agent 与奖励组件已编写，veRL 参数更新与 SFT 训练尚未接入。

## 设计

```text
Issue + base checkout + budget
              |
      LangGraph decision loop <---- tool observations
              |                           ^
      Qwen3 / vLLM API ------> Docker tools
              |
          final patch
              |
     independent SWE-bench verifier
              |
      outcome + cost reward
              |
       group advantages
              |
     veRL training adapter [TODO]
```

目前提供 7 个动作：list_files、read_file、search、replace_text、run_tests、diff、submit。模型每轮选择一个动作。Docker 容器禁用网络，不挂载宿主机目录；路径校验与测试文件写保护限制直接编辑测试，但并不构成抗恶意代码的完整安全证明。

奖励参考：`R = (1 if resolved else -1) - 0.15*C_token - 0.10*C_tool - 0.05*C_time - 0.05*C_repeat`。前三项成本按预算归一化并截断为 1；重复动作按完全相同工具参数计数。重复测试可能合理，该惩罚需要消融验证。

## 安装与 toy 演示

建议 Linux/WSL2 + Docker；正式训练需要支持 CUDA 的环境。当前未固定完整 GPU 依赖组合。

```bash
pip install -e '.[data]'
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=EMPTY
export MODEL_NAME=Qwen/Qwen3-8B
# 另行安装适配环境的 vLLM，再启动服务：
vllm serve Qwen/Qwen3-8B --enable-auto-tool-choice --tool-call-parser hermes
```

不同 vLLM/Qwen 版本的工具 parser、chat template 与 thinking 设置需实际验证，上述是启动参考。

```bash
docker build -t budgetpilot-toy:latest examples/toy
budgetpilot rollout --tasks data/examples/tasks.jsonl --images examples/toy/images.json --samples 4 --output runs/toy.jsonl
budgetpilot export --trajectories runs/toy.jsonl --output-dir runs/predictions
```

toy 是自编的算术缺陷，仅演示交互接口。内含测试是任务环境的一部分；本项目未执行这些命令，也未运行测试验证。

## 数据集准备

- 训练：[SWE-Gym](https://huggingface.co/datasets/SWE-Gym/SWE-Gym)
- 评测：[SWE-bench Lite](https://huggingface.co/datasets/SWE-bench/SWE-bench_Lite)、[SWE-bench Verified](https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified)
- 可选环境：[R2E-Gym](https://github.com/agentica-project/R2E-Gym)

数据由使用者下载，本仓库提供来源说明、转换脚本及原创示例。原始数据不提交 GitHub。详见 [data/README.md](data/README.md)。

```bash
python scripts/local_dataset_to_jsonl.py --input data/raw/swegym --split train --output data/raw/swegym.jsonl
budgetpilot prepare --input data/raw/swegym.jsonl --output data/processed/swegym.jsonl
python scripts/local_dataset_to_jsonl.py --input data/raw/swebench-lite --split test --output data/raw/lite.jsonl
budgetpilot prepare --input data/raw/lite.jsonl --limit 60 --seed 42 --output data/processed/eval60.jsonl
python scripts/split_by_repo.py --input data/processed/swegym.jsonl --eval-tasks data/processed/eval60.jsonl --dev-repos YOUR_DEV_REPO --output-dir data/processed/split
```

将 YOUR_DEV_REPO 换成实际训练侧仓库。划分脚本执行仓库、instance 和 base_commit 排除，尚未实现补丁相似性去重。所选 60 题仅是子集，不等同完整 Lite 结果。

真实任务需要预制镜像：`/repo` 下为准确 base_commit 的干净 Git checkout，安装仓库依赖和 pytest，支持 python/git/sleep，不包含 gold patch 或最终 test_patch。提供一个 JSON 文件映射 `instance_id -> image`，传给 rollout 的 `--images`。此版本没有自动构建真实 benchmark 镜像。

## 官方评测与奖励

```bash
pip install -e '.[eval]'
budgetpilot export --trajectories runs/rollouts.jsonl --output-dir runs/predictions
python -m swebench.harness.run_evaluation --dataset_name SWE-bench/SWE-bench_Lite --predictions_path runs/predictions/sample-0.jsonl --run_id budgetpilot-sample-0
# 每个 sample 重复运行，使用不同 run_id。
# 根据各 report.json 准备 verifier verdicts，之后：
budgetpilot score --trajectories runs/rollouts.jsonl --verdicts runs/verdicts.jsonl --output runs/scored.jsonl
python scripts/summarize.py --scored runs/scored.jsonl
```

verdicts 格式：`{"instance_id":"...","sample_id":0,"resolved":true}`。toy 任务不属于 SWE-bench，不能交给 SWE-bench harness；它需要单独的可信 toy verifier（尚未提供）。协议和统计定义见 [docs/EVALUATION.md](docs/EVALUATION.md)。

## 训练范围与下一步

已有 `training.grpo_loss` 参考实现与组内优势计算；没有可直接执行的 veRL trainer。API rollout 也不提供训练必需的完整 token/logprob。接入方案见 [docs/TRAINING.md](docs/TRAINING.md)。

- [x] 多轮预算观测与工具选择原型
- [x] 隔离 Docker 工具、轨迹记录、patch 导出
- [x] 数据转换、显式仓库划分、预算成本奖励
- [x] GRPO 优势与损失参考实现
- [ ] 跑通 toy、真实任务环境与官方 harness
- [ ] 严格预算预检、截断、超时子进程处理
- [ ] SFT 轨迹转换、LoRA 冷启动
- [ ] veRL on-policy AgentLoop、action mask 与参数更新
- [ ] 基线、消融、真实统计结果与 checkpoint

## GitHub 发布

上传源代码、configs、docs 和 data/examples。data/raw、runs、模型权重与本地凭证已经忽略。保存实际抽样清单和数据版本后，可另行公开无答案的任务 manifest。此仓库的实验结果暂为空，先前规划中的近似数字未写入。

## 参考

- [verl Agent Loop](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)
- [ReTool](https://github.com/ReTool-RL/ReTool)
- [SWE-Gym](https://github.com/SWE-Gym/SWE-Gym)
- [SWE-bench evaluation](https://www.swebench.com/SWE-bench/guides/evaluation/)

## 许可证

本项目原创代码采用 MIT；外部数据、模型及任务仓库分别遵循其原始许可证。
