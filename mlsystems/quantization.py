"""Groupwise signed 4-bit weight packing and CPU reconstruction experiments.

This is a pedagogical NumPy reference, not optimized CUDA inference. A scale
is stored for each row/group. Two quantized signed values fit in one byte.
"""
from dataclasses import dataclass
from time import perf_counter
import numpy as np


@dataclass(frozen=True)
class PackedInt4:
    packed: np.ndarray
    scales: np.ndarray
    rows: int
    columns: int
    group_size: int
    padded_columns: int

    @property
    def stored_bytes(self):
        return int(self.packed.nbytes + self.scales.nbytes)

    @property
    def fp32_weight_bytes(self):
        return self.rows * self.columns * 4


def quantize(weights, group_size=32):
    weights = np.asarray(weights, dtype=np.float32)
    if weights.ndim != 2 or not weights.size:
        raise ValueError("Expected a non-empty 2D weight matrix")
    if not np.all(np.isfinite(weights)):
        raise ValueError("Weights must be finite")
    if not isinstance(group_size, int) or group_size <= 0:
        raise ValueError("group_size must be a positive integer")
    rows, columns = weights.shape
    groups = (columns + group_size - 1) // group_size
    padded_columns = groups * group_size
    padded = np.pad(weights, ((0, 0), (0, padded_columns - columns)))
    group_view = padded.reshape(rows, groups, group_size)
    max_abs = np.max(np.abs(group_view), axis=2)
    scales = np.where(max_abs > 0, max_abs / 7, 1).astype(np.float32)
    signed = np.clip(np.rint(group_view / scales[:, :, None]), -7, 7).astype(np.int8)
    flat = signed.reshape(rows, padded_columns)
    if padded_columns % 2:
        flat = np.pad(flat, ((0, 0), (0, 1)))
    as_nibbles = (flat.astype(np.int16) + 8).astype(np.uint8)
    packed = as_nibbles[:, ::2] | (as_nibbles[:, 1::2] << 4)
    return PackedInt4(packed, scales, rows, columns, group_size, padded_columns)


def dequantize(matrix: PackedInt4):
    if matrix.padded_columns % matrix.group_size:
        raise ValueError("Corrupt packed matrix group structure")
    if matrix.packed.shape != (matrix.rows, (matrix.padded_columns + 1) // 2):
        raise ValueError("Corrupt packed matrix shape")
    if matrix.scales.shape != (matrix.rows, matrix.padded_columns // matrix.group_size):
        raise ValueError("Corrupt scale shape")
    unpacked = np.empty((matrix.rows, matrix.packed.shape[1] * 2), dtype=np.float32)
    unpacked[:, ::2] = (matrix.packed & 15).astype(np.int16) - 8
    unpacked[:, 1::2] = ((matrix.packed >> 4) & 15).astype(np.int16) - 8
    grouped = unpacked[:, :matrix.padded_columns].reshape(matrix.rows, -1, matrix.group_size)
    return (grouped * matrix.scales[:, :, None]).reshape(matrix.rows, matrix.padded_columns)[:, :matrix.columns]


def quantized_linear(inputs, matrix: PackedInt4, bias=None):
    inputs = np.asarray(inputs, dtype=np.float32)
    if inputs.shape[-1] != matrix.columns:
        raise ValueError("Last input dimension must match weight columns")
    result = inputs @ dequantize(matrix).T
    if bias is not None:
        bias = np.asarray(bias, dtype=np.float32)
        if bias.shape != (matrix.rows,):
            raise ValueError("Bias shape does not match output features")
        result += bias
    return result


def compare(seed=42, rows=256, columns=257, batch=32, group_size=32, repeats=8):
    """Measure numerical error and *illustrative CPU* timings (not throughput)."""
    if rows <= 0 or columns <= 0 or batch <= 0 or repeats <= 0:
        raise ValueError("Sizes and repeats must be positive")
    rng = np.random.default_rng(seed)
    weights = rng.normal(size=(rows, columns)).astype(np.float32)
    inputs = rng.normal(size=(batch, columns)).astype(np.float32)
    packed = quantize(weights, group_size=group_size)
    restored = dequantize(packed)
    reference = inputs @ weights.T
    approximation = quantized_linear(inputs, packed)
    relative_error = float(np.linalg.norm(approximation - reference) / max(np.linalg.norm(reference), 1e-12))

    def median_runtime(callback):
        samples = []
        for _ in range(repeats):
            start = perf_counter()
            callback()
            samples.append((perf_counter() - start) * 1000)
        return float(np.median(samples))

    return {
        "seed": seed, "shape": [rows, columns], "batch": batch,
        "fp32_weight_bytes": packed.fp32_weight_bytes,
        "packed_int4_plus_scales_bytes": packed.stored_bytes,
        "storage_ratio": packed.fp32_weight_bytes / packed.stored_bytes,
        "mean_absolute_weight_error": float(np.mean(np.abs(weights - restored))),
        "relative_output_l2_error": relative_error,
        "fp32_cpu_median_ms": median_runtime(lambda: inputs @ weights.T),
        "int4_decode_and_matmul_cpu_median_ms": median_runtime(lambda: quantized_linear(inputs, packed)),
        "synthetic_data": True,
        "note": "CPU NumPy timings include dequantization; no GPU speed or LLM quality claim",
    }
