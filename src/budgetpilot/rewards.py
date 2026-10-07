"""Reward requires an external trusted verifier; tool test output is not truth."""
import json
import math


def score(trajectory, resolved, config):
    if not isinstance(resolved, bool):
        raise ValueError('resolved must be a trusted boolean verifier result')
    usage, budget = trajectory['usage'], trajectory['budget']
    signatures = [json.dumps([s['action'], s.get('arguments')],sort_keys=True)
                  for s in trajectory['steps'] if s['action'] != 'submit']
    repeats = len(signatures)-len(set(signatures))
    cost = (config['token_weight']*min(1,usage['tokens']/budget['tokens'])
            +config['tool_weight']*min(1,usage['tool_calls']/budget['tool_calls'])
            +config['time_weight']*min(1,usage['wall_seconds']/budget['wall_seconds'])
            +config['repeat_weight']*repeats/max(1,len(signatures)))
    # Failure stays below any success; cheap empty submissions cannot be rewarded.
    return {'reward': (1.0 if resolved else -1.0)-cost,
            'resolved':resolved,'cost':cost,'repeated_actions':repeats}


def group_advantages(rewards, epsilon=1e-6):
    if len(rewards) < 2: raise ValueError('GRPO needs at least two rollouts per group')
    mean = sum(rewards)/len(rewards)
    std = math.sqrt(sum((r-mean)**2 for r in rewards)/len(rewards))
    return [(r-mean)/(std+epsilon) for r in rewards]
