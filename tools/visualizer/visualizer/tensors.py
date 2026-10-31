"""Bounded GGUF weight windows; quantized blocks use the repository's decoder."""
from functools import lru_cache
from pathlib import Path
import math
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'gguf-py'))
from gguf import GGUFReader, GGML_QUANT_SIZES
from gguf.quants import dequantize


@lru_cache(maxsize=3)
def _reader(path, size, mtime_ns):
    return GGUFReader(path)


def weight_window(path, tensor_name, axis='row', index=0, plane=0, offset=0, limit=32):
    path = Path(path)
    stat = path.stat()
    reader = _reader(str(path), stat.st_size, stat.st_mtime_ns)
    tensor = next((t for t in reader.tensors if t.name == tensor_name), None)
    if tensor is None:
        raise KeyError('Unknown weight tensor')
    shape = [int(n) for n in tensor.shape]
    dims = shape + [1] * (4 - len(shape))
    columns, rows = dims[:2]
    planes = math.prod(dims[2:])
    total = columns if axis == 'row' else rows
    index_total = rows if axis == 'row' else columns
    if axis not in ('row', 'column') or not (0 <= index < index_total and 0 <= plane < planes and 0 <= offset < total and 1 <= limit <= 1024):
        raise ValueError('Weight index, plane or window is outside the tensor')
    count = min(limit, total - offset)
    qtype = tensor.tensor_type
    # Typed scalar tensors preserve the reader's byte order directly.
    if tensor.data.dtype.kind in 'fi' and tensor.data.dtype.itemsize > 1:
        data = tensor.data.reshape(planes, rows, columns)
        values = data[plane, index, offset:offset + count] if axis == 'row' else data[plane, offset:offset + count, index]
        values = values.astype(np.float32)
    else:
        if reader.byte_order != 'I':
            raise NotImplementedError('Byte-swapped quantized weight windows are not supported')
        block, size = GGML_QUANT_SIZES[qtype]
        if columns % block:
            raise ValueError('Tensor row does not contain complete quantization blocks')
        raw = tensor.data.view(np.uint8).reshape(-1)
        row_bytes = columns // block * size
        if axis == 'row':
            first = offset // block
            last = (offset + count + block - 1) // block
            begin = (plane * rows + index) * row_bytes + first * size
            decoded = dequantize(raw[begin:begin + (last - first) * size].copy(), qtype).reshape(-1)
            values = decoded[offset % block:offset % block + count]
        else:
            starts = (plane * rows + np.arange(offset, offset + count)) * row_bytes + index // block * size
            blocks = raw[starts[:, None] + np.arange(size)[None, :]].copy()
            values = dequantize(blocks, qtype).reshape(count, block)[:, index % block]
    return dict(tensor=tensor_name, shape=shape, quantization=qtype.name, axis=axis, index=index, plane=plane,
                offset=offset, total=total, count=count,
                values=[float(v) if np.isfinite(v) else None for v in values],
                meaning='Effective stored GGUF weights, dequantized for display; this window is paged, not a summary statistic')
