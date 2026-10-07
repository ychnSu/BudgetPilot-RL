import argparse
import json
import random
from pathlib import Path
import yaml
from .agent import rollout
from .tools import DockerTools
from .rewards import score, group_advantages


def read_jsonl(path):
    with open(path,encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists(): raise FileExistsError(f'Refusing to overwrite {p}')
    with p.open('w',encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+'\n')


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--input',required=True,help='Local JSONL export; no automatic download')
    p.add_argument('--output',required=True)
    p.add_argument('--limit',type=int)
    p.add_argument('--seed',type=int,default=42)
    p = sub.add_parser('rollout')
    p.add_argument('--tasks',required=True)
    p.add_argument('--images',required=True,help='JSON mapping instance_id -> prepared Docker image')
    p.add_argument('--config',default='configs/default.yaml')
    p.add_argument('--output',required=True)
    p.add_argument('--samples',type=int,default=4)
    p = sub.add_parser('export')
    p.add_argument('--trajectories',required=True)
    p.add_argument('--output-dir',required=True)
    p = sub.add_parser('score')
    p.add_argument('--trajectories',required=True)
    p.add_argument('--verdicts',required=True,help='JSONL: instance_id, sample_id, resolved')
    p.add_argument('--config',default='configs/default.yaml')
    p.add_argument('--output',required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        rows = read_jsonl(args.input)
        random.Random(args.seed).shuffle(rows)
        if args.limit: rows = rows[:args.limit]
        # Intentionally discard patch, test_patch, hints and test lists from agent data.
        fields = ('instance_id','repo','base_commit','problem_statement')
        write_jsonl(args.output,[{k:r[k] for k in fields} for r in rows])
    elif args.command == 'rollout':
        config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
        images = json.loads(Path(args.images).read_text(encoding='utf-8'))
        output = Path(args.output)
        if output.exists(): raise FileExistsError(output)
        output.parent.mkdir(parents=True,exist_ok=True)
        # Persist each episode immediately so later failures don't lose earlier work.
        with output.open('x',encoding='utf-8') as f:
            for task in read_jsonl(args.tasks):
                for sample in range(args.samples):
                    with DockerTools(images[task['instance_id']],**config['tools']) as tools:
                        row = rollout(task,tools,config)
                        row['container_id'] = tools.container
                    row['sample_id'] = sample
                    f.write(json.dumps(row,ensure_ascii=False)+'\n')
                    f.flush()
    elif args.command == 'export':
        rows = read_jsonl(args.trajectories)
        for sample in sorted({r['sample_id'] for r in rows}):
            write_jsonl(Path(args.output_dir)/f'sample-{sample}.jsonl',[
                {k:r[k] for k in ('instance_id','model_name_or_path','model_patch')}
                for r in rows if r['sample_id']==sample])
    elif args.command == 'score':
        config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
        verdicts = {(r['instance_id'],r['sample_id']):r['resolved']
                    for r in read_jsonl(args.verdicts)}
        rows = read_jsonl(args.trajectories)
        groups = {}
        for row in rows:
            row.update(score(row,verdicts[(row['instance_id'],row['sample_id'])],config['reward']))
            key = (row['instance_id'],json.dumps(row['budget'],sort_keys=True))
            groups.setdefault(key,[]).append(row)
        for group in groups.values():
            for row, adv in zip(group,group_advantages([r['reward'] for r in group])):
                row['advantage'] = adv
        write_jsonl(args.output,rows)


if __name__ == '__main__': main()
