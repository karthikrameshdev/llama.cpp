"""Build protobuf bindings locally from the checked-in wire contract."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / 'visualizer' / 'generated'

def generate():
    GENERATED.mkdir(exist_ok=True)
    (GENERATED / '__init__.py').touch()
    if not (GENERATED / 'telemetry_pb2.py').exists() or (ROOT / 'proto/telemetry.proto').stat().st_mtime > (GENERATED / 'telemetry_pb2.py').stat().st_mtime:
        subprocess.run([sys.executable, '-m', 'grpc_tools.protoc', '-I', str(ROOT / 'proto'),
                        '--python_out=' + str(GENERATED), '--grpc_python_out=' + str(GENERATED), str(ROOT / 'proto/telemetry.proto')], check=True)
        stub = GENERATED / 'telemetry_pb2_grpc.py'
        stub.write_text(stub.read_text().replace('import telemetry_pb2 as', 'from . import telemetry_pb2 as'))

generate()
from .generated import telemetry_pb2 as pb, telemetry_pb2_grpc as rpc  # noqa: E402
