# Agentic RL 接入边界

当前实现：在线 Agent rollout、外部 verifier 判定后的成本奖励、组内优势计算，以及 PyTorch GRPO 裁剪损失参考函数。**当前没有能直接启动并更新 Qwen3 参数的 veRL 训练器。**

在线 Agent 使用 OpenAI-compatible API，适合 vLLM 推理与数据采集。普通 API 轨迹没有完整的行为策略 token logprob 与 tokenizer token ID，不能直接当作严格 on-policy GRPO batch。

完成 veRL 接入需要：

1. 固定 veRL/vLLM/transformers 兼容版本及 Qwen3 chat template。
2. 用 veRL AgentLoop 实现多轮 rollout；每题在同一预算下采样 G 条轨迹，任务间可随机选择预算档位。
3. 获取实际生成 token IDs、old_logprobs 与工具观察 token；构建 response/action mask，工具返回、system、user token 不参与策略损失。
4. 每个 rollout 在独立任务镜像中执行，结束后将 patch 交给容器外 verifier；无法判定的基础设施失败不得冒充测试通过或静默丢弃。
5. 调用 `rewards.score` 获得终局奖励；按 instance_id、预算和策略版本分组，使用训练框架优势估计接口。
6. 配置 LoRA actor、reference model、KL 与 optimizer，再运行单步训练检查后启动正式任务。

官方参考：https://verl.readthedocs.io/en/latest/advance/agent_loop.html

`training.grpo_loss` 仅展示 clipped policy objective 与 KL 计算，不代替 veRL 的分布式训练、采样同步、梯度累积或 checkpoint 管理。`score` 导出的 advantage 适合审计与接入准备，不表示已完成参数更新。

SFT：从训练仓库采集成功且经过验证的轨迹，转换成 LLaMA-Factory 支持的工具对话格式；保留工具结果作为上下文，但配置工具观察 loss mask。转换和 LoRA 训练脚本尚待实现。禁止使用最终评测集轨迹进行 SFT 或选择 checkpoint。
