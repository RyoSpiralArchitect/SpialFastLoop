from collections.abc import Mapping
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from hypothesis import HealthCheck, given, settings, strategies as st

from spiralfastloop.engine import (
    _concatenate_batches,
    _ensure_loss_vector,
    _infer_batch_size,
)


MAX_DEPTH = 3


def _tensor_strategy(batch_dim: int):
    """Generate rank-1 to rank-3 tensors with a fixed batch dimension."""

    rest = st.lists(st.integers(min_value=1, max_value=4), min_size=0, max_size=2)
    return rest.map(lambda dims: torch.randn((batch_dim, *dims), dtype=torch.float32))


@st.composite
def batch_structure(draw, depth: int = 0, batch_dim: int | None = None):
    if batch_dim is None:
        batch_dim = draw(st.integers(min_value=1, max_value=4))
    if depth >= MAX_DEPTH:
        return draw(_tensor_strategy(batch_dim))

    choice = draw(st.sampled_from(["tensor", "list", "tuple", "dict"]))
    if choice == "tensor":
        return draw(_tensor_strategy(batch_dim))
    if choice == "list":
        size = draw(st.integers(min_value=1, max_value=3))
        return [draw(batch_structure(depth + 1, batch_dim=batch_dim)) for _ in range(size)]
    if choice == "tuple":
        size = draw(st.integers(min_value=1, max_value=3))
        values = [draw(batch_structure(depth + 1, batch_dim=batch_dim)) for _ in range(size)]
        return tuple(values)
    # mapping
    keys = draw(
        st.lists(
            st.text(min_size=1, max_size=5),
            min_size=1,
            max_size=3,
            unique=True,
        )
    )
    base = {k: draw(batch_structure(depth + 1, batch_dim=batch_dim)) for k in keys}
    return base


def _extra_like(base):
    if torch.is_tensor(base):
        return base + 1.0
    if isinstance(base, list):
        return type(base)(_extra_like(x) for x in base)
    if isinstance(base, tuple):
        converted = tuple(_extra_like(x) for x in base)
        return type(base)(converted)
    if isinstance(base, Mapping):
        converted = {k: _extra_like(v) for k, v in base.items()}
        if hasattr(base, "default_factory"):
            new_mapping = type(base)(getattr(base, "default_factory"))
            new_mapping.update(converted)
            return new_mapping
    # Fall back to constructing the same mapping type when possible.
        try:
            return type(base)(converted)
        except TypeError:
            new_mapping = type(base)()
            new_mapping.update(converted)
            return new_mapping
    raise TypeError(f"Unsupported batch leaf: {type(base)!r}")


@st.composite
def batch_pairs(draw):
    base = draw(batch_structure())
    extra = _extra_like(base)
    return base, extra


def _assert_structure(result, base, extra):
    if torch.is_tensor(base):
        assert torch.is_tensor(result)
        expected_batch = base.shape[0] + extra.shape[0]
        assert result.shape[0] == expected_batch
        assert torch.allclose(result[: base.shape[0]], base)
        assert torch.allclose(result[base.shape[0] :], extra)
        return
    if isinstance(base, list):
        assert isinstance(result, list)
        assert len(result) == len(base)
        for r, b, e in zip(result, base, extra):
            _assert_structure(r, b, e)
        return
    if isinstance(base, tuple):
        assert isinstance(result, tuple)
        assert len(result) == len(base)
        for r, b, e in zip(result, base, extra):
            _assert_structure(r, b, e)
        return
    if isinstance(base, Mapping):
        assert isinstance(result, Mapping)
        assert set(result.keys()) == set(base.keys())
        for key in base.keys():
            _assert_structure(result[key], base[key], extra[key])
        return
    raise AssertionError(f"Unexpected structure {type(base)!r}")


def _structures_close(first, second):
    if torch.is_tensor(first) and torch.is_tensor(second):
        assert torch.allclose(first, second)
        return
    if isinstance(first, list):
        assert isinstance(second, list)
        assert len(first) == len(second)
        for f, s in zip(first, second):
            _structures_close(f, s)
        return
    if isinstance(first, tuple):
        assert isinstance(second, tuple)
        assert len(first) == len(second)
        for f, s in zip(first, second):
            _structures_close(f, s)
        return
    if isinstance(first, Mapping):
        assert isinstance(second, Mapping)
        assert set(first.keys()) == set(second.keys())
        for key in first.keys():
            _structures_close(first[key], second[key])
        return
    raise AssertionError(f"Unexpected structure {type(first)!r}")


def _expected_batch_size(batch):
    if torch.is_tensor(batch):
        return int(batch.shape[0])
    if isinstance(batch, Mapping):
        sizes = {_expected_batch_size(v) for v in batch.values()}
        assert len(sizes) == 1
        return sizes.pop()
    if isinstance(batch, (list, tuple)):
        sizes = [_expected_batch_size(v) for v in batch]
        if sizes:
            unique = set(sizes)
            assert len(unique) == 1
            return sizes[0]
        return len(batch)
    raise AssertionError(f"Unsupported batch element {type(batch)!r}")


@settings(max_examples=120, suppress_health_check=[HealthCheck.too_slow])
@given(batch_pairs())
def test_concatenate_batches_preserves_structure(pair):
    base, extra = pair
    combined = _concatenate_batches(base, extra)
    _assert_structure(combined, base, extra)
    base_size = _expected_batch_size(base)
    extra_size = _expected_batch_size(extra)
    combined_size = _expected_batch_size(combined)
    assert combined_size == base_size + extra_size


@settings(max_examples=120, suppress_health_check=[HealthCheck.too_slow])
@given(batch_structure())
def test_concatenate_handles_none_base(extra):
    combined = _concatenate_batches(None, extra)
    _structures_close(combined, extra)


@settings(max_examples=120, suppress_health_check=[HealthCheck.too_slow])
@given(batch_structure())
def test_infer_batch_size_matches_manual(batch):
    expected = _expected_batch_size(batch)
    inferred = _infer_batch_size(batch)
    assert inferred == expected


@st.composite
def loss_tensors(draw):
    variant = draw(st.sampled_from(["scalar", "vector", "matrix", "tensor3"]))
    if variant == "scalar":
        value = draw(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False))
        return torch.tensor(value, dtype=torch.float32)
    if variant == "vector":
        size = draw(st.integers(min_value=1, max_value=6))
        data = [
            draw(
                st.floats(
                    min_value=-10,
                    max_value=10,
                    allow_nan=False,
                    allow_infinity=False,
                )
            )
            for _ in range(size)
        ]
        return torch.tensor(data, dtype=torch.float32)
    if variant == "matrix":
        rows = draw(st.integers(min_value=1, max_value=5))
        cols = draw(st.integers(min_value=1, max_value=5))
        return torch.randn(rows, cols, dtype=torch.float32)
    # 3D tensor variant
    dim0 = draw(st.integers(min_value=1, max_value=4))
    dim1 = draw(st.integers(min_value=1, max_value=4))
    dim2 = draw(st.integers(min_value=1, max_value=4))
    return torch.randn(dim0, dim1, dim2, dtype=torch.float32)


@settings(max_examples=120, suppress_health_check=[HealthCheck.too_slow])
@given(loss_tensors())
def test_ensure_loss_vector_behaviour(loss):
    vector = _ensure_loss_vector(loss)
    assert vector.ndim == 1
    if loss.ndim == 0:
        assert vector.shape == (1,)
        assert torch.allclose(vector, loss.unsqueeze(0))
    elif loss.ndim == 1:
        assert torch.allclose(vector, loss)
    else:
        expected = loss.reshape(loss.shape[0], -1).mean(dim=1)
        assert torch.allclose(vector, expected)
    # The returned vector should never contain NaNs for finite inputs.
    assert torch.isfinite(vector).all()
