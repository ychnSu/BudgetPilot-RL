# 数据集

仓库仅包含自编 toy 示例；不包含已下载的 SWE-Gym/SWE-bench 数据或任何真实实验结果。

| 用途 | 数据集 | 下载链接 |
|---|---|---|
| SFT 轨迹采集、RL 训练 | SWE-Gym | https://huggingface.co/datasets/SWE-Gym/SWE-Gym |
| 小规模与完整评测 | SWE-bench Lite（test 300 题） | https://huggingface.co/datasets/SWE-bench/SWE-bench_Lite |
| 扩展评测 | SWE-bench Verified（500 题） | https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified |
| 可选扩展训练环境 | R2E-Gym | https://github.com/agentica-project/R2E-Gym |

请自行下载 Parquet 或使用 datasets 保存到本地。下载数据不等于下载任务运行环境；仓库代码、对应 base_commit、依赖与容器需要另行准备。

原始数据放 `data/raw/`，不提交 Git。保留原始数据中的标准补丁和 test_patch 供独立评测器使用；`budgetpilot prepare` 只导出 instance_id、repo、base_commit、problem_statement 给 Agent。不要把原始数据挂载进 Agent 容器。

数据版本、源 split、抽样种子和任务 ID 列表必须保存。只跑 60 题时报告“固定 60 题子集”，不报告为完整 benchmark 成绩。发布下载数据前核对数据卡及底层项目许可证；代码许可证不覆盖外部数据。

`data/examples/tasks.jsonl` 是本项目原创演示任务，与公开 benchmark 无关。真实数据尚未下载、筛选、去重或验证。按仓库划分脚本提供 repo/instance/commit 排除；跨仓库补丁相似去重与预训练污染审计仍未实现。
