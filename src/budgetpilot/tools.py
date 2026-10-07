"""All repository reads/edits/tests execute in a disposable, networkless container.

This is an engineering sandbox, not a formal security boundary against hostile code.
The container must contain the task's base checkout and installed dependencies.
"""
import json
import subprocess


SCHEMAS = [
    {"name": "list_files", "description": "List tracked repository files", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "read_file", "description": "Read a UTF-8 source file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"], "additionalProperties": False}},
    {"name": "search", "description": "Search literal text in tracked files", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"], "additionalProperties": False}},
    {"name": "replace_text", "description": "Replace one unique exact substring in a source file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}}, "required": ["path", "old", "new"], "additionalProperties": False}},
    {"name": "run_tests", "description": "Run existing pytest tests; supply test node IDs, no options", "parameters": {"type": "object", "properties": {"nodes": {"type": "array", "items": {"type": "string"}}}, "required": ["nodes"], "additionalProperties": False}},
    {"name": "diff", "description": "Inspect current patch", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "submit", "description": "Finish and submit the current patch", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
]

# Executed INSIDE the container. Never accepts a shell command from the model.
WORKER = r'''
import json, pathlib, subprocess, sys
name, args = json.loads(sys.argv[1])
root = pathlib.Path('/repo').resolve()
def path(value, write=False):
    p = (root / value).resolve()
    rel = p.relative_to(root)
    if any(x in {'.git', '.env', '.ssh'} for x in rel.parts):
        raise ValueError('protected path')
    if write and (any(x in {'tests', 'test', '.github'} for x in rel.parts)
                  or p.name.startswith('test_') or p.name.endswith('_test.py')
                  or p.name in {'conftest.py','pytest.ini','pyproject.toml','setup.cfg','tox.ini'}):
        raise ValueError('protected test/config file')
    return p
def run(argv):
    p = subprocess.run(argv, cwd=root, capture_output=True, text=True, timeout=55)
    return {'returncode':p.returncode,'output':(p.stdout+p.stderr)[-12000:]}
try:
    if name == 'list_files': result = run(['git','ls-files'])
    elif name == 'read_file': result = path(args['path']).read_text()[:12000]
    elif name == 'search': result = run(['git','grep','-n','-F','--',args['query']])
    elif name == 'diff': result = run(['git','diff','--no-ext-diff','--no-textconv','HEAD','--'])
    elif name == 'replace_text':
        p = path(args['path'], True)
        text = p.read_text()
        old = args['old']
        if not old or text.count(old) != 1: raise ValueError('old must occur exactly once')
        p.write_text(text.replace(old,args['new'],1))
        result = {'edited':args['path']}
    elif name == 'run_tests':
        nodes = args['nodes']
        for node in nodes:
            if node.startswith('-'): raise ValueError('test options are forbidden')
            path(node.split('::')[0])
        result = run(['python','-m','pytest','-q',*nodes])
    else: raise ValueError('unknown tool')
    print(json.dumps({'ok':True,'result':result}))
except Exception as e:
    print(json.dumps({'ok':False,'error':str(e)}))
'''


class DockerTools:
    def __init__(self, image, timeout_seconds=60, max_output_chars=12000):
        self.image = image
        self.timeout = timeout_seconds
        self.max_output = max_output_chars
        self.container = None

    def __enter__(self):
        self.container = subprocess.check_output([
            "docker", "run", "-d", "--network", "none", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--pids-limit", "256",
            "--memory", "4g", "--cpus", "2", "--entrypoint", "sleep",
            self.image, "infinity"], text=True).strip()
        return self

    def __exit__(self, *args):
        # Stop only; keep container for inspection. No automatic deletion.
        subprocess.run(["docker", "stop", self.container], capture_output=True)

    def call(self, name, args, timeout=None):
        if name not in {s['name'] for s in SCHEMAS} - {'submit'}:
            return {'ok': False, 'error': 'unknown tool'}
        try:
            result = subprocess.run([
                "docker", "exec", "-w", "/repo", self.container, "python", "-c",
                WORKER, json.dumps([name, args])], text=True, capture_output=True,
                timeout=min(self.timeout, timeout) if timeout else self.timeout)
            if result.returncode:
                return {'ok': False, 'error': result.stderr[-self.max_output:]}
            return json.loads(result.stdout)
        except Exception as e:
            return {'ok': False, 'error': str(e)}

    def patch(self):
        result = self.call('diff', {})
        if not result.get('ok'):
            raise RuntimeError(result)
        # Export full patch, not the observation's truncated diff.
        return subprocess.check_output([
            "docker", "exec", "-w", "/repo", self.container, "git", "diff",
            "--no-ext-diff", "--no-textconv", "HEAD", "--"], text=True, timeout=self.timeout)
