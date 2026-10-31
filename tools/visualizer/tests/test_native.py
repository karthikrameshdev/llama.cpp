import json
from pathlib import Path
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
OBSERVER=ROOT/'build/llama-visualizer-observer'
MODELS=REPO/'build/context-models'

def execute(model,observe=True,run_id='native-test'):
    commands=[{'kind':'load','path':str(model),'model_id':'fixture','inspection_id':'inspection:fixture:'+run_id,'context_size':256,'gpu_layers':0,'capture':'full','observe':observe},
              {'kind':'run','run_id':run_id,'token_ids':[1,2,3,4],'prompt':'synthetic token fixture','max_tokens':4,'chat':False,'capture':'full','observe':observe},
              {'kind':'shutdown'}]
    result=subprocess.run([str(OBSERVER)],input=''.join(json.dumps(c)+'\n' for c in commands),text=True,capture_output=True,timeout=60)
    assert result.returncode==0,result.stderr[-3000:]
    events=[json.loads(line)for line in result.stdout.splitlines()]
    errors=[e for e in events if e['kind'] in ('error','observation_error')]
    assert not errors,errors
    return events

@pytest.mark.parametrize('name',['llama-dense.gguf','llama-moe.gguf'])
def test_tracing_preserves_uninstrumented_output_and_token_ids(name):
    model=MODELS/name
    if not OBSERVER.exists() or not model.exists():pytest.skip('Build the observer and generate local synthetic models first')
    traced=execute(model,True,'traced')
    baseline=execute(model,False,'baseline')
    tokens=lambda events:[e['data']['id']for e in events if e['kind']=='token']
    assert tokens(traced)==tokens(baseline)
    assert any(e['kind']=='activation'for e in traced)
    channels=[e['data']['neurons'] for e in traced if e['kind']=='activation' and e['data'].get('neurons')]
    assert channels
    assert all(len(n['values'])==n['channels']*n['slots'] for n in channels)
    assert not any(e['kind']=='activation'for e in baseline)
    nodes=[e['data']['node']for e in traced if e['kind']=='node' and e['run_id']=='traced' and e['data']['step']==0]
    ids={n['id']for n in nodes}
    assert nodes and all(s in ids for n in nodes for s in n['sources'])
    if 'moe' in name:
        routes=[e['data']for e in traced if e['kind']=='routing'and e['run_id']=='traced']
        assert routes
        details=next(e['data']for e in traced if e['kind']=='model_loaded')
        n_experts=max(v for k,v in details['metadata'].items()if k.endswith('.expert_count'))
        assert all(0<=i<n_experts for r in routes for s in r['selections'] for i in s['experts'])


@pytest.mark.parametrize('name',['qwen3.5-0.8b-q4_0.gguf','granite-3.1-1b-moe-q4_k_m.gguf'])
def test_real_model_cpu_output_equivalence(name):
    model=ROOT/'models'/name
    if not OBSERVER.exists() or not model.exists():pytest.skip('Download the pinned verification models first')
    traced=execute(model,True,'real-traced')
    baseline=execute(model,False,'real-baseline')
    tokens=lambda events:[e['data']['id']for e in events if e['kind']=='token']
    assert tokens(traced)==tokens(baseline)
