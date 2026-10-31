import grpc
import pytest
from visualizer.engine import Engine
from visualizer.protocol import pb, rpc
from test_store import event


def test_grpc_rejects_unauthenticated_local_publishers(store):
    s,_=store
    engine=Engine(s,'/not-used')
    engine.server.start()
    channel=grpc.insecure_channel('127.0.0.1:'+str(engine.port))
    try:
        stream=rpc.TelemetryStub(channel).Connect(iter([pb.EventBatch()]),metadata=[('authorization','Bearer invalid')],timeout=3)
        with pytest.raises(grpc.RpcError) as error:
            next(stream)
        assert error.value.code()==grpc.StatusCode.UNAUTHENTICATED
    finally:
        channel.close();engine.stop()


def test_duplex_grpc_transfers_commands_and_real_event_batches(store):
    s,m=store
    engine=Engine(s,'/not-used')
    engine.server.start()
    channel=grpc.insecure_channel('127.0.0.1:'+str(engine.port))
    import json, threading
    hold=threading.Event()
    def incoming():
        e=event(1,'','hello',{},m['id'])
        yield pb.EventBatch(events=[pb.Event(sequence=1,kind='hello',payload_json=json.dumps(e))])
        hold.wait(5)
    stream=rpc.TelemetryStub(channel).Connect(incoming(),metadata=[('authorization','Bearer '+engine.token)],timeout=5)
    try:
        engine.send('load',{'model_id':m['id']})
        command=next(stream)
        assert command.kind=='load'
        assert json.loads(command.payload_json)['model_id']==m['id']
        assert engine.ready.wait(3)
        assert s.events()[0]['kind']=='hello'
    finally:
        hold.set();stream.cancel();channel.close();engine.stop()


def test_trace_budget_requests_cancellation_and_marks_failed(store):
    s,m=store;s.budget=400
    r=s.create_run(m['id'],'prompt',{})
    engine=Engine(s,'/not-used')
    engine.state.update(status='running',active_run=r['id'],model_id=m['id'])
    engine._event(event(1,r['id'],'node',{'step':0,'huge':'x'*1000},m['id']))
    assert engine.commands.get_nowait().kind=='cancel'
    assert s.run(r['id'])['status']=='failed'
    assert s.events()[0]['kind']=='trace_error'
    engine.stop()


def test_engine_failure_marks_active_run_interrupted(store):
    s,m=store;r=s.create_run(m['id'],'prompt',{})
    engine=Engine(s,'/not-used');engine.state.update(active_run=r['id'],status='running',connected=True)
    engine._event(event(1,r['id'],'bridge_error',{'message':'native exited'},m['id']))
    assert s.run(r['id'])['status']=='interrupted'
    assert engine.snapshot()['connected'] is False
    engine.stop()
