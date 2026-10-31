"""Exercise a running application against real local dense/hybrid and MoE models."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import httpx

H={'X-Visualizer-Client':'1'}

def verify(base, output):
    c=httpx.Client(base_url=base,timeout=120)
    report={'base_url':base,'checks':[],'runs':[]}
    def check(name,condition,detail=None):
        assert condition,(name,detail)
        report['checks'].append({'name':name,'passed':True,'detail':detail})
        print('PASS',name,flush=True)
    def wait_state(status,timeout=120):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            state=c.get('/api/state').json()['engine']
            if state['status']==status:return state
            if state['status']=='error':raise AssertionError(state)
            time.sleep(.1)
        raise AssertionError(('engine timeout',state))
    def load(model,**options):
        wait_state('ready') if c.get('/api/state').json()['engine']['status']=='running' else None
        r=c.post('/api/models/load',headers=H,json=dict(model_id=model['id'],capture='full',context_size=2048,gpu_layers=99,**options))
        assert r.status_code==202,r.text
        wait_state('ready')
        model=c.get('/api/models/'+model['id']).json()
        graph=c.get('/api/runs/inspection:'+model['id']+'/graph').json()
        check('actual inspection graph: '+model['name'],len(graph['nodes'])>100)
        return model
    def prompt(text,max_tokens=3,**options):
        r=c.post('/api/runs',headers=H,json=dict(prompt=text,max_tokens=max_tokens,capture='full',**options))
        assert r.status_code==202,r.text
        run=r.json();end=time.monotonic()+120
        while time.monotonic()<end:
            run=c.get('/api/runs/'+run['id']).json()
            if run['status'] not in ('queued','running'):break
            time.sleep(.1)
        return run
    models=c.get('/api/models').json()
    qwen=next(m for m in models if 'qwen' in m['name'].lower())
    granite=next(m for m in models if 'granite' in m['name'].lower())
    qwen=load(qwen)
    qrun=prompt('In one sentence, explain why the sky appears blue.',4)
    check('real Qwen inference completes',qrun['status']=='completed',qrun.get('error'))
    qgraph=c.get('/api/runs/'+qrun['id']+'/graph').json()
    check('hybrid recurrent operations are discovered',any('ssm' in n['name'].lower() or 'GATED_DELTA' in n['op'] for n in qgraph['nodes']))
    check('Qwen tensor statistics are captured',any(a.get('stats') for a in qgraph['activations'].values()))
    check('Qwen hidden feature channels are captured',any(a.get('neurons') for a in qgraph['activations'].values()))
    first_graph_size=len(qgraph['nodes'])
    granite=load(granite)
    prompts=['What is a mixture of experts?','Write a Python function to add two numbers.','Translate good morning into French.','Explain gravity to a child.','What is the capital of Japan?','Give one benefit of reading books.','Name three primary colors.','How does a computer store numbers?','What makes a good software test?','Describe the ocean in a short sentence.']
    signatures=[];runs=[]
    for index,text in enumerate(prompts):
        run=prompt(text)
        check('Granite prompt '+str(index+1),run['status']=='completed',run.get('error'))
        graph=c.get('/api/runs/'+run['id']+'/graph',params={'step':0}).json()
        ids={n['id'] for n in graph['nodes']}
        check('complete dependency references '+str(index+1),all(source in ids for n in graph['nodes'] for source in n['sources']))
        check('real MoE expert IDs '+str(index+1),len(graph['routing'])>0)
        expert_counts=[v for k,v in granite['details']['metadata'].items() if k.endswith('.expert_count') and isinstance(v,int)]
        check('expert IDs within actual model range '+str(index+1),all(0<=e<max(expert_counts) for r in graph['routing'] for s in r['selections'] for e in s['experts']))
        channels=[a['neurons'] for a in graph['activations'].values() if a.get('neurons',{}).get('meaning')=='FFN hidden activation after nonlinearity']
        check('all routed FFN channels captured '+str(index+1),len(channels)==24 and all(n['channels']==512 and n['slots']==8 and len(n['values'])==4096 for n in channels))
        check('channel values retain expert identities '+str(index+1),all(len(n['expert_ids'])==8 and len(set(n['expert_ids']))==8 and all(0<=e<32 for e in n['expert_ids']) for n in channels))
        routes=c.get('/api/runs/'+run['id']+'/routes').json()
        signatures.append(hashlib.sha256(json.dumps(routes,sort_keys=True).encode()).hexdigest())
        runs.append(run['id']);report['runs'].append(dict(id=run['id'],prompt=text,nodes=len(graph['nodes']),routing_observations=len(graph['routing']),trace_bytes=run['trace_bytes']))
    check('ten independent prompt identities',len(set(runs))==10)
    check('prompt-specific expert routing changes',len(set(signatures))>1,{'distinct_routing_signatures':len(set(signatures))})
    check('model topology changes with model selection',first_graph_size!=report['runs'][0]['nodes'])
    exported=c.get('/api/runs/'+runs[0]+'/export').json()
    check('full graph and trace export',exported['run']['id']==runs[0] and any(e['kind']=='node' for e in exported['events']) and any(e['kind']=='routing' for e in exported['events']))
    lines=c.get('/api/runs/'+runs[0]+'/export?format=jsonl').text.splitlines()
    check('pure JSONL export',len(lines)==len(exported['events']) and all(json.loads(line) for line in lines))
    steps=c.get('/api/runs/'+runs[0]+'/steps').json()
    last=c.get('/api/runs/'+runs[0]+'/graph',params={'step':steps[-1]['step']}).json()
    check('prefill and decode replay remain separate',len(steps)>=2 and last['summary']['phase']=='decode' and last['summary']['complete'])
    delta=c.get('/api/compare',params={'left':runs[0],'right':runs[1]}).json()
    check('comparison contains actual routing differences',any(delta['routing_delta'].values()))
    playback=c.get('/api/runs/'+runs[0]+'/playback?step=0').json()
    check('operation replay retains original execution order',len(playback)>1000 and all(a['cursor']<b['cursor'] for a,b in zip(playback,playback[1:])) and all(e['data']['step']==0 for e in playback))
    weight=c.get('/api/models/'+granite['id']+'/weights',params={'tensor':'blk.0.ffn_up_exps.weight','index':511,'plane':3,'offset':29,'limit':40}).json()
    check('actual quantized expert weight row can be paged',weight['count']==40 and weight['total']==1024 and len(weight['values'])==40)
    column=c.get('/api/models/'+granite['id']+'/weights',params={'tensor':'blk.0.ffn_down_exps.weight','axis':'column','index':511,'plane':3,'offset':0,'limit':16}).json()
    check('actual neuron outgoing weights can be read',column['count']==16 and column['axis']=='column' and column['index']==511)
    r=c.post('/api/runs',headers=H,json={'prompt':'Count from one to one hundred and explain every number.','max_tokens':128,'capture':'full'})
    cancel_id=r.json()['id'];cancel=c.post('/api/runs/'+cancel_id+'/cancel',headers=H,json={})
    check('queued cancellation accepted',cancel.status_code==202)
    wait_state('ready')
    check('cancellation becomes terminal',c.get('/api/runs/'+cancel_id).json()['status']=='cancelled')
    overflow=prompt('hello '*5000,2)
    check('context overflow fails explicitly',overflow['status']=='failed' and 'context' in overflow.get('error',''))
    recovery=prompt('Say hello.',2)
    check('engine recovers after invalid prompt',recovery['status']=='completed')
    metadata=c.get('/api/models/'+granite['id']+'/metadata',params={'key':'tokenizer.ggml.tokens','offset':3,'limit':5})
    check('full GGUF metadata paging',metadata.status_code==200 and len(metadata.json()['value'])==5)
    mmproj=Path(qwen['path']).parent/'qwen3.5-0.8b-mmproj-q8.gguf'
    if mmproj.is_file():
        qwen=load(qwen,mmproj=str(mmproj))
        from PIL import Image
        buf=io.BytesIO();Image.new('RGB',(128,128),(240,60,30)).save(buf,format='PNG')
        upload=c.post('/api/media',headers=H,content=buf.getvalue()).json()
        media=prompt('What is the main color in this image?',2,media_id=upload['media_id'])
        check('real image encoder and decoder complete',media['status']=='completed',media.get('error'))
        media_graph=c.get('/api/runs/'+media['id']+'/graph',params={'step':0}).json()
        check('multimodal encoder nodes appear dynamically',any(n['component']=='multimodal_encoder' for n in media_graph['nodes']))
        ids={n['id'] for n in media_graph['nodes']}
        check('multimodal dependencies resolve across graph epochs',all(source in ids for n in media_graph['nodes'] for source in n['sources']))
        check('multimodal graph has no overwritten nodes',len(ids)==media_graph['summary']['node_count'])
        check('separate image and text graph epochs are retained',len({n['id'].split('n')[0] for n in media_graph['nodes']})>1)
        report['media_run']=media['id']
        load(qwen)
        saved=c.get('/api/runs/'+media['id']).json()
        check('prompt retains original projector configuration after reload',saved['model_snapshot']['details']['multimodal'] and saved['model_snapshot']['config']['mmproj']==str(mmproj))
        check('saved raw log returns latest ordered events',c.get('/api/events',params={'run_id':media['id'],'tail':True,'limit':3}).json()[-1]['kind']=='run_finished')
    check('all ten runs remain available',set(runs).issubset({r['id'] for r in c.get('/api/runs?limit=500').json()}))
    report['passed']=len(report['checks']);report['granite_id']=granite['id'];report['qwen_id']=qwen['id']
    Path(output).write_text(json.dumps(report,indent=2));print('Report:',output,flush=True)
    c.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',default='http://127.0.0.1:8787');p.add_argument('--output',default='data/live-verification.json');a=p.parse_args();verify(a.base,a.output)
