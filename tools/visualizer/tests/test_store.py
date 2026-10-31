import json
import pytest
from visualizer.store import Store, TraceBudgetExceeded

@pytest.fixture
def store(tmp_path):
    s=Store(tmp_path/'traces.sqlite3')
    p=tmp_path/'model.gguf'; p.write_bytes(b'GGUFtest')
    m=s.register(p)
    yield s,m
    s.close()

def event(seq,run,kind,data,model_id='m'):
    return dict(session='test',sequence=seq,run_id=run,model_id=model_id,kind=kind,data=data,schema_version=1)

def test_model_registration_and_replacement(store,tmp_path):
    s,m=store
    assert s.register(m['path'])['id']==m['id']
    p=tmp_path/'model.gguf';p.write_bytes(b'GGUFchanged')
    newer=s.register(p)
    assert newer['id']!=m['id']
    assert s.model(m['id']) is not None

def test_durable_completed_run_and_separate_steps(store):
    s,m=store;r=s.create_run(m['id'],'hello',{})
    s.ingest(event(1,r['id'],'run_started',{}))
    s.ingest(event(2,r['id'],'node',{'step':0,'node':{'id':'n0','name':'prefill','sources':[]}}))
    s.ingest(event(3,r['id'],'node',{'step':1,'node':{'id':'n0','name':'decode','sources':[]}}))
    s.ingest(event(4,r['id'],'run_finished',{'status':'completed','output':'world'}))
    other=Store(s.path)
    assert other.run(r['id'])['output']=='world'
    assert other.graph(r['id'],0)['nodes'][0]['name']=='prefill'
    assert other.graph(r['id'],1)['nodes'][0]['name']=='decode'
    other.close()

def test_crash_recovery_marks_incomplete_runs(store):
    s,m=store;r=s.create_run(m['id'],'a',{})
    other=Store(s.path)
    assert other.run(r['id'])['status']=='interrupted'
    other.close()

def test_duplicate_envelopes_are_idempotent(store):
    s,m=store;e=event(1,'x','hello',{})
    assert s.ingest(e)==s.ingest(e)
    assert len(s.events())==1

def test_event_cursor_pages_have_no_gaps(store):
    s,m=store
    for i in range(17):s.ingest(event(i,'x','hello',{}))
    after=0;seen=[]
    while page:=s.events(after,limit=3):
        seen.extend(e['sequence'] for e in page);after=page[-1]['cursor']
    assert seen==list(range(17))
    assert [e['sequence'] for e in s.events(limit=3,tail=True)]==[14,15,16]

def test_routing_counts_deduplicate_dispatch_ops(store):
    s,m=store;r=s.create_run(m['id'],'prompt',{})
    data={'step':0,'layer':2,'component':'decoder','selections':[{'token_index':0,'experts':[1,4]},{'token_index':1,'experts':[4,7]}]}
    for i in range(3):s.ingest(event(i,r['id'],'routing',data))
    assert s.route_summary(r['id'])=={'decoder:2:1':1,'decoder:2:4':2,'decoder:2:7':1}

def test_trace_budget_fails_explicitly(store):
    s,m=store;s.budget=200;r=s.create_run(m['id'],'hello',{})
    with pytest.raises(TraceBudgetExceeded):s.ingest(event(1,r['id'],'node',{'huge':'x'*500}))
    assert not s.events(run_id=r['id'])

def test_failed_trace_cannot_be_overwritten_by_completion(store):
    s,m=store;r=s.create_run(m['id'],'hello',{})
    s.fail_run(r['id'],'trace incomplete')
    s.ingest(event(1,r['id'],'run_finished',{'status':'completed','output':'hello'}))
    assert s.run(r['id'])['status']=='failed'

def test_reload_inspection_graphs_are_isolated(store):
    s,m=store
    for seq,run,name in [(1,'inspection:'+m['id']+':first','first'),(4,'inspection:'+m['id']+':second','second')]:
        s.ingest(event(seq,run,'load_started',{},m['id']))
        s.ingest(event(seq+1,run,'step_begin',{'step':0},m['id']))
        s.ingest(event(seq+2,run,'node',{'step':0,'node':{'id':'n0','name':name,'sources':[]}},m['id']))
    assert s.graph('inspection:'+m['id'])['nodes'][0]['name']=='second'
    assert s.graph('inspection:'+m['id']+':first')['nodes'][0]['name']=='first'

def test_saved_model_configuration_survives_reload(store):
    s,m=store
    s.set_config(m['id'],{'context_size':512,'mmproj':'first-projector.gguf'})
    s.ingest(event(1,'inspection:'+m['id']+':first','model_loaded',{'context_size':512,'projector':{'tensors':[{'name':'encoder.weight'}]}},m['id']))
    r=s.create_run(m['id'],'first prompt',{})
    s.set_config(m['id'],{'context_size':4096,'mmproj':''})
    s.ingest(event(2,'inspection:'+m['id']+':second','model_loaded',{'context_size':4096},m['id']))
    snapshot=s.run(r['id'])['model_snapshot']
    assert snapshot['config']['context_size']==512
    assert snapshot['details']['projector']['tensors'][0]['name']=='encoder.weight'
    assert s.model(m['id'])['details']['context_size']==4096

def test_operation_playback_keeps_original_order_and_step_boundaries(store):
    s,m=store;r=s.create_run(m['id'],'replay',{})
    for sequence,kind,step in [(1,'node',0),(2,'activation',0),(3,'routing',0),(4,'activation',1),(5,'token',0)]:
        s.ingest(event(sequence,r['id'],kind,{'step':step},m['id']))
    assert [e['kind'] for e in s.playback(r['id'],0)]==['activation','routing','token']
    assert [e['sequence'] for e in s.playback(r['id'],1)]==[4]
