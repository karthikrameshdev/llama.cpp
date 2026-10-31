"""Private native process boundary; real telemetry transport uses duplex gRPC."""
import argparse
import json
import os
import queue
import signal
import subprocess
import threading
import time
import uuid
import grpc
from .protocol import pb, rpc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', required=True)
    parser.add_argument('--observer', required=True)
    args = parser.parse_args()
    process = subprocess.Popen([args.observer], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace', bufsize=1)
    events = queue.Queue(maxsize=4096)
    stop = threading.Event()
    lock = threading.Lock()
    session = uuid.uuid4().hex
    sequence = 0
    current = {'run_id': '', 'model_id': ''}

    def publish(event):
        nonlocal sequence
        with lock:
            sequence += 1
            event['native_sequence'] = event.pop('sequence', None)
            event.update(sequence=sequence, session=session)
            item = pb.Event(sequence=sequence, run_id=event.get('run_id', ''), model_id=event.get('model_id', ''), kind=event['kind'], payload_json=json.dumps(event, ensure_ascii=False))
            while not stop.is_set():
                try:
                    events.put(item, timeout=.2)
                    break
                except queue.Full:
                    continue

    def stdout():
        try:
            for line in process.stdout:
                publish(json.loads(line))
        except Exception as exc:
            publish({'kind': 'bridge_error', 'data': {'message': str(exc)}, **current})
        code = process.wait()
        if not stop.is_set():
            publish({'kind': 'bridge_error', 'data': {'message': f'Native observer exited with code {code}'}, **current})

    def stderr():
        for line in process.stderr:
            if line.strip():
                publish({'kind': 'engine_log', 'time_us': time.time_ns() // 1000, 'data': {'text': line.rstrip()[:16384]}, **current})

    threading.Thread(target=stdout, daemon=True).start()
    threading.Thread(target=stderr, daemon=True).start()

    def batches():
        while not stop.is_set():
            try:
                first = events.get(timeout=.2)
            except queue.Empty:
                continue
            batch = [first]
            size = len(first.payload_json)
            while len(batch) < 128 and size < 1024 * 1024:
                try:
                    event = events.get_nowait()
                    batch.append(event); size += len(event.payload_json)
                except queue.Empty:
                    break
            yield pb.EventBatch(events=batch)

    channel = grpc.insecure_channel(args.target, options=[('grpc.max_send_message_length', 64 * 1024 * 1024), ('grpc.max_receive_message_length', 4 * 1024 * 1024)])
    try:
        grpc.channel_ready_future(channel).result(timeout=20)
        commands = rpc.TelemetryStub(channel).Connect(batches(), metadata=[('authorization', 'Bearer ' + os.environ['VISUALIZER_GRPC_TOKEN'])])
        for command in commands:
            body = json.loads(command.payload_json)
            if command.kind == 'cancel':
                if hasattr(signal, 'SIGUSR1') and process.poll() is None:
                    process.send_signal(signal.SIGUSR1)
                else:
                    process.terminate()
                continue
            if command.kind == 'shutdown':
                break
            current.update(run_id=body.get('run_id', body.get('inspection_id', 'inspection:' + body.get('model_id', ''))), model_id=body.get('model_id', current['model_id']))
            process.stdin.write(json.dumps(dict(body, kind=command.kind)) + '\n')
            process.stdin.flush()
    except (grpc.RpcError, grpc.FutureTimeoutError, BrokenPipeError) as exc:
        print(f'Engine bridge stopped: {exc}', flush=True)
    finally:
        stop.set()
        channel.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()

if __name__ == '__main__':
    main()
