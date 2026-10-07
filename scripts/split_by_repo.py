"""Explicit repository splits with instance/commit overlap checks."""
import argparse
from budgetpilot.cli import read_jsonl, write_jsonl

p = argparse.ArgumentParser()
p.add_argument('--input',required=True)
p.add_argument('--dev-repos',nargs='+',required=True)
p.add_argument('--eval-tasks',required=True)
p.add_argument('--output-dir',required=True)
a = p.parse_args()
rows, evaluation = read_jsonl(a.input), read_jsonl(a.eval_tasks)
eval_repos = {r['repo'] for r in evaluation}
eval_ids = {r['instance_id'] for r in evaluation}
eval_commits = {(r['repo'],r['base_commit']) for r in evaluation}
seen, clean = set(), []
for row in rows:
    if row['instance_id'] in seen: continue
    seen.add(row['instance_id'])
    if (row['repo'] in eval_repos or row['instance_id'] in eval_ids
            or (row['repo'],row['base_commit']) in eval_commits): continue
    clean.append(row)
train = [r for r in clean if r['repo'] not in a.dev_repos]
dev = [r for r in clean if r['repo'] in a.dev_repos]
if not train or not dev: raise ValueError('Empty split; choose different development repositories')
write_jsonl(a.output_dir+'/train.jsonl',train)
write_jsonl(a.output_dir+'/dev.jsonl',dev)
print({'train':len(train),'dev':len(dev),'excluded_eval_repos':sorted(eval_repos)})
