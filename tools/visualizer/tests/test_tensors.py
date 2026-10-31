import numpy as np
import pytest
from visualizer.tensors import weight_window
from gguf import GGUFWriter, GGMLQuantizationType
from gguf.quants import quantize, dequantize


@pytest.mark.parametrize('qtype', [GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.Q4_0, GGMLQuantizationType.Q8_0])
def test_weight_windows_match_exact_rows_columns_and_expert_planes(tmp_path, qtype):
    original = np.linspace(-2, 3, 2 * 3 * 64, dtype=np.float32).reshape(2, 3, 64)
    encoded = quantize(original, qtype)
    expected = dequantize(encoded, qtype)
    path = tmp_path / (qtype.name + '.gguf')
    writer = GGUFWriter(path, 'llama')
    writer.add_tensor('blk.0.ffn_up_exps.weight', encoded, raw_dtype=qtype)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    row = weight_window(path, 'blk.0.ffn_up_exps.weight', 'row', index=2, plane=1, offset=29, limit=12)
    assert row['total'] == 64
    np.testing.assert_array_equal(row['values'], expected[1, 2, 29:41])
    column = weight_window(path, 'blk.0.ffn_up_exps.weight', 'column', index=33, plane=1, offset=1, limit=2)
    np.testing.assert_array_equal(column['values'], expected[1, 1:3, 33])
    with pytest.raises(ValueError): weight_window(path, 'blk.0.ffn_up_exps.weight', plane=2)
    with pytest.raises(ValueError): weight_window(path, 'blk.0.ffn_up_exps.weight', offset=64)
    with pytest.raises(ValueError): weight_window(path, 'blk.0.ffn_up_exps.weight', limit=4096)
    with pytest.raises(KeyError): weight_window(path, 'missing')
