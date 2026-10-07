"""GRPO loss reference for a future veRL adapter, not a distributed trainer.

Token logprobs must be produced by the same tokenizer/chat template as rollout.
Observation, system and user tokens MUST have action_mask=0. The current OpenAI
rollout does not capture behavior-policy token logprobs; do not train from those
JSON traces as if they were complete on-policy samples.
"""


def grpo_loss(new_logprobs, old_logprobs, advantages, action_mask,
              reference_logprobs=None, clip=0.2, beta=0.01):
    import torch
    ratio = torch.exp(new_logprobs-old_logprobs.detach())
    advantage = advantages.detach().unsqueeze(-1)
    objective = torch.minimum(ratio*advantage,
                              ratio.clamp(1-clip,1+clip)*advantage)
    loss = -objective
    if reference_logprobs is not None:
        delta = reference_logprobs.detach()-new_logprobs
        loss = loss + beta*(torch.exp(delta)-delta-1)
    # Sequence-normalized reference implementation; match chosen veRL algorithm
    # and reduction semantics explicitly when integrating.
    per_sequence = (loss*action_mask).sum(-1)/action_mask.sum(-1).clamp_min(1)
    return per_sequence.mean()
