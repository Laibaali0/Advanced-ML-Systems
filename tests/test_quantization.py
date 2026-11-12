import numpy as np
import pytest

from mlsystems.quantization import compare, dequantize, quantize, quantized_linear


@pytest.mark.parametrize("shape, group", [((2, 5), 3), ((9, 17), 8), ((3, 64), 32), ((1, 1), 1)])
def test_odd_group_sizes_and_shapes_roundtrip(shape, group):
    weights = np.random.default_rng(42).normal(size=shape)
    packed = quantize(weights, group_size=group)
    decoded = dequantize(packed)
    assert decoded.shape == shape
    assert np.isfinite(decoded).all()


def test_all_zeros_are_exact():
    weights = np.zeros((10, 41), dtype=np.float32)
    decoded = dequantize(quantize(weights, group_size=7))
    np.testing.assert_array_equal(decoded, weights)


def test_small_weights_have_reasonable_error():
    rng = np.random.default_rng(7)
    weights = rng.normal(size=(20, 63)).astype(np.float32)
    decoded = dequantize(quantize(weights, group_size=16))
    assert np.mean(abs(weights - decoded)) < 0.2


def test_bias_and_matmul_shape():
    rng = np.random.default_rng(7)
    weights = rng.normal(size=(15, 21)).astype(np.float32)
    x = rng.normal(size=(4, 21)).astype(np.float32)
    bias = np.arange(15, dtype=np.float32)
    packed = quantize(weights, group_size=12)
    observed = quantized_linear(x, packed, bias)
    expected = x @ dequantize(packed).T + bias
    np.testing.assert_allclose(observed, expected)


def test_packed_matrix_reduces_storage_for_larger_weights():
    rng = np.random.default_rng(42)
    packed = quantize(rng.normal(size=(256, 256)), group_size=32)
    assert packed.stored_bytes < packed.fp32_weight_bytes


def test_nonfinite_weights_rejected():
    with pytest.raises(ValueError):
        quantize(np.array([[np.inf, 0]]))


def test_invalid_bias_dimension_rejected():
    packed = quantize(np.ones((5, 7)))
    with pytest.raises(ValueError):
        quantized_linear(np.ones((3, 7)), packed, bias=np.ones(6))


def test_comparison_contains_only_cpu_measurements():
    result = compare(rows=16, columns=23, batch=3, repeats=2, seed=9)
    assert result["storage_ratio"] > 1
    assert result["relative_output_l2_error"] >= 0
    assert "CPU" in result["note"]
