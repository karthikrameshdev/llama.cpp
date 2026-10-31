import json
import pytest
from fastapi.testclient import TestClient
from visualizer.app import create_app
from test_store import event

HEADERS={'X-Visualizer-Client':'1'}
@pytest.fixture
def client(tmp_path):
    models=tmp_path/'models';models.mkdir();(models/'test.gguf').write_bytes(b'GGUFtest')
    app=create_app(tmp_path/'data',[models],start_engine=False)
    with TestClient(app) as c:
        yield c,app,models

def test_loopback_state_and_directory_scan(client):
    c,app,models=client
    assert c.get('/api/state').status_code==200
    assert len(c.get('/api/models').json())==1
    assert c.get('/api/state',headers={'Host':'attacker.example'}).status_code==400

def test_cross_origin_and_missing_header_rejected(client):
    c,app,models=client
    assert c.post('/api/runs',json={'prompt':'test'}).status_code==403
    assert c.post('/api/runs',json={'prompt':'test'},headers=dict(HEADERS,Origin='https://attacker.example')).status_code==403

def test_model_symlink_escape_not_registered(client,tmp_path):
    c,app,models=client
    outside=tmp_path/'outside.gguf';outside.write_bytes(b'GGUF')
    (models/'escape.gguf').symlink_to(outside)
    assert len(c.get('/api/models').json())==1

def test_invalid_commands_and_unknown_resources(client):
    c,app,_=client
    assert c.post('/api/runs',headers=HEADERS,json={'prompt':'','max_tokens':2}).status_code==422
    assert c.post('/api/runs',headers=HEADERS,json={'prompt':'test','max_tokens':10000}).status_code==422
    assert c.post('/api/runs',headers=HEADERS,json={'prompt':'test','capture':'imaginary'}).status_code==422
    assert c.get('/api/runs/missing').status_code==404
    assert c.get('/api/models/missing').status_code==404
    assert c.post('/api/models/load',headers=HEADERS,json={'model_id':'missing'}).status_code==404
    assert c.get('/api/events?after=-1').status_code==422

def test_engine_readiness_and_overlapping_run_rejection(client):
    c,app,_=client;m=c.get('/api/models').json()[0]
    assert c.post('/api/runs',headers=HEADERS,json={'prompt':'test'}).status_code==409
    e=app.state.engine;e.state.update(status='ready',connected=True,model_id=m['id'])
    first=c.post('/api/runs',headers=HEADERS,json={'prompt':'first'})
    assert first.status_code==202
    assert c.post('/api/runs',headers=HEADERS,json={'prompt':'second'}).status_code==409
    assert c.post('/api/models/load',headers=HEADERS,json={'model_id':m['id']}).status_code==409
    assert c.post('/api/runs/'+first.json()['id']+'/cancel',headers=HEADERS,json={}).status_code==202

def test_exports_preserve_prompt_events_and_schema(client):
    c,app,_=client;m=c.get('/api/models').json()[0];s=app.state.store;r=s.create_run(m['id'],'<script>test</script>',{})
    s.ingest(event(1,r['id'],'run_started',{},m['id']))
    s.ingest(event(2,r['id'],'run_finished',{'status':'completed','output':'✓'},m['id']))
    d=c.get('/api/runs/'+r['id']+'/export').json()
    assert d['schema_version']==1 and d['run']['prompt']==r['prompt']
    assert d['model']==r['model_snapshot']
    assert len(d['events'])==2
    rows=c.get('/api/runs/'+r['id']+'/export?format=jsonl').text.splitlines()
    assert len(rows)==2 and json.loads(rows[1])['data']['output']=='✓'
    assert c.get('/api/runs/'+r['id']+'/export?format=exe').status_code==422

def test_upload_is_bounded_and_unknown_media_rejected(client):
    c,app,_=client
    assert c.post('/api/media',headers=HEADERS,content=b'').status_code==422
    assert c.post('/api/media',headers=HEADERS,content=b'not-yet-decoded').status_code==201
    assert c.post('/api/runs',headers=HEADERS,json={'prompt':'describe','media_id':'../secret'}).status_code==404

def test_compare_is_separate_from_trace_graphs(client):
    c,app,_=client;m=c.get('/api/models').json()[0];s=app.state.store
    a=s.create_run(m['id'],'one',{});b=s.create_run(m['id'],'two',{})
    s.ingest(event(1,a['id'],'routing',{'step':0,'component':'decoder','layer':0,'selections':[{'token_index':0,'experts':[1]}]},m['id']))
    s.ingest(event(2,b['id'],'routing',{'step':0,'component':'decoder','layer':0,'selections':[{'token_index':0,'experts':[2]}]},m['id']))
    d=c.get('/api/compare',params={'left':a['id'],'right':b['id']}).json()
    assert d['routing_delta']=={'decoder:0:1':-1,'decoder:0:2':1}

def test_weight_api_validation_and_unknown_replay_resources(client):
    c,app,_=client;m=c.get('/api/models').json()[0]
    assert c.get('/api/models/missing/weights?tensor=x').status_code==404
    assert c.get('/api/models/'+m['id']+'/weights?tensor=x&index=-1').status_code==422
    assert c.get('/api/models/'+m['id']+'/weights?tensor=x&limit=65536').status_code==422
    assert c.get('/api/models/'+m['id']+'/weights?tensor=x&component=projector').status_code==404
    assert c.get('/api/runs/missing/playback').status_code==404
