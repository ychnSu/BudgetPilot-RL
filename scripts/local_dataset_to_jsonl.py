"""Convert already-downloaded HF dataset directory or Parquet files; no network."""
import argparse
import json
from pathlib import Path
from datasets import load_dataset, load_from_disk

p = argparse.ArgumentParser()
p.add_argument('--input',required=True)
p.add_argument('--split',default='train')
p.add_argument('--output',required=True)
a = p.parse_args()
source = Path(a.input)
if source.is_file():
    ds = load_dataset('parquet',data_files=str(source),split='train')
elif (source/'state.json').exists() or (source/'dataset_dict.json').exists():
    ds = load_from_disk(str(source))
    if hasattr(ds,'keys'): ds = ds[a.split]
else:
    files = sorted(source.glob(f'**/{a.split}*.parquet'))
    if not files: raise ValueError('No matching local Parquet files')
    ds = load_dataset('parquet',data_files=[str(x) for x in files],split='train')
dest = Path(a.output)
dest.parent.mkdir(parents=True,exist_ok=True)
with dest.open('x',encoding='utf-8') as f:
    for row in ds: f.write(json.dumps(row,ensure_ascii=False)+'\n')
