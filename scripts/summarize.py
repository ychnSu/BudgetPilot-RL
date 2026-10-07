"""Summarize complete externally verified rollout groups."""
import argparse
import json
from statistics import mean
from budgetpilot.cli import read_jsonl

p = argparse.ArgumentParser()
p.add_argument('--scored',required=True)
a = p.parse_args()
rows = read_jsonl(a.scored)
groups = {}
for r in rows: groups.setdefault(r['instance_id'],[]).append(r)
counts = {len(g) for g in groups.values()}
if len(counts)!=1: raise ValueError('Unequal samples per task; resolve incomplete runs first')
print(json.dumps({
    'tasks':len(groups),'samples_per_task':next(iter(counts)),
    'empirical_pass_at_1':mean(mean(r['resolved'] for r in g) for g in groups.values()),
    'observed_pass_at_k':mean(any(r['resolved'] for r in g) for g in groups.values()),
    'mean_tokens':mean(r['usage']['tokens'] for r in rows),
    'mean_tool_calls':mean(r['usage']['tool_calls'] for r in rows),
    'mean_wall_seconds':mean(r['usage']['wall_seconds'] for r in rows)
},indent=2))
