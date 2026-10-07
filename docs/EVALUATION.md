# 评测协议

使用相同模型与独立初始容器，对比 SFT、结果奖励 RL、预算感知 RL。当前仓库尚未提供任何已训练 checkpoint，只有预算感知推理与研究组件。

Agent 可运行修复前仓库已有测试。标准补丁与最终 test_patch 不进入 Agent 容器；独立官方 harness 对模型 patch 进行判定。Agent 工具返回的 pytest exit code 不作为最终 resolved。

1. 采集每题 4 次独立 rollout，每次创建全新容器。
2. 用 `budgetpilot export` 按 sample_id 导出官方 predictions JSONL，避免同一文件重复 instance_id。
3. 每个 sample 用独立 run_id 调用 SWE-bench harness。
4. 按任务逐项检查 report.json，生成 `instance_id, sample_id, resolved` 的 verifier verdicts JSONL；必须是 boolean。该报告转换当前需人工完成，不假设不同 harness 版本的报告结构一致。
5. 用 `budgetpilot score` 计算奖励，再用 summarize.py 汇总。

主指标：对每题独立采样的成功比例取均值作为 empirical pass@1；4 次至少成功一次作为 observed pass@4。每次运行的所有 API prompt+completion tokens 均计成本，包括重复输入历史。不要将 best-of-4 成绩命名为 pass@1。

预算当前是执行边界检查，并非严格 tokenizer 预检：单次请求可能突破累计 token 上限；超额后停止执行新动作，超额 Token 仍计成本。README 不应声称严格预算保证。需要在正式实验前增加服务端 tokenizer preflight、输入裁剪与 timeout 后容器子进程清理。

基础设施失败、超时、空补丁单独记录。固定任务集上的 Agent 失败计入分母；环境失败先修复重跑，无法恢复时披露数量与预定规则。样本数不一致时统计脚本拒绝汇总。

容器退出后保留用于检查，未自动删除。需要删除时先备份任务日志和 patch，并由操作者确认。
