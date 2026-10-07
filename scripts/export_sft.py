"""Export externally verified TRAIN-only tool trajectories for LLaMA-Factory."""
import argparse
import json
from pathlib import Path
from budgetpilot.cli import read_jsonl
from budgetpilot.tools import SCHEMAS

p = argparse.ArgumentParser()
p.add_argument('--scored',required=True)
p.add_argument('--train-tasks',required=True)
p.add_argument('--output',required=True)
a = p.parse_args()
train_ids = {r['instance_id'] for r in read_jsonl(a.train_tasks)}
examples = []
for trajectory in read_jsonl(a.scored):
    if trajectory['instance_id'] not in train_ids:
        raise ValueError('SFT export received task outside training manifest')
    if not trajectory['resolved'] or not trajectory['steps']: continue
    steps = trajectory['steps']
    first = steps[0]['request_messages']
    system = next((m['content'] for m in first if m['role']=='system'),'')
    dialogue = []
    # Export each request's newly appended budget observation as well as tool turns.
    for index, step in enumerate(steps):
        if index == 0:
            dialogue.extend({'from':'human','value':m['content']} for m in first if m['role']=='user')
        else:
            message = step['request_messages'][-1]
            dialogue.append({'from':'human','value':message['content']})
        content = step['assistant'].get('content')
        if content: dialogue.append({'from':'gpt','value':content})
        dialogue.append({'from':'function_call','value':json.dumps(
            {'name':step['action'],'arguments':step['arguments']},ensure_ascii=False)})
        observation = step.get('observation')
        if observation is not None:
            dialogue.append({'from':'observation','value':json.dumps(observation,ensure_ascii=False)})
        elif step['action']=='submit':
            dialogue.append({'from':'observation','value':'Submission recorded.'})
            dialogue.append({'from':'gpt','value':'Repair submitted.'})
        else:
            # Do not teach incomplete/budget-aborted function calls.
            dialogue.pop()
    examples.append({'conversations':dialogue,'system':system,
                     'tools':json.dumps(SCHEMAS,ensure_ascii=False)})
dest = Path(a.output)
dest.parent.mkdir(parents=True,exist_ok=True)
with dest.open('x',encoding='utf-8') as f:
    json.dump(examples,f,ensure_ascii=False,indent=2)
print({'exported':len(examples),'format':'sharegpt; chat template validation required'})
