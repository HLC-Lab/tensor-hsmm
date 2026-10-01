from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from tensor_viterbi.hsmm import HSMM

try:
    from . import _native
except ImportError:
    _native = None


def _ensure_loaded() -> None:
    if _native is None:
        raise RuntimeError(
            "[native] Native extension not available.\n"
            "Build it with:\n"
            "  cmake -B build -DBUILD_GPU=OFF\n"
            "  cmake --build build\n"
        )


def decode_tensor_viterbi_cpp(
        n_states,
        trans_mat,
        emission_probs,
        duration_probs_linear,
        start_probs,
        duration_probs,
        obs_seq,
) -> np.ndarray:
    _ensure_loaded()
    return _native.decode_tensor_viterbi_cpp(
        n_states, trans_mat, emission_probs,
        duration_probs_linear, start_probs, duration_probs, obs_seq,
    )


def decode_tensor_viterbi_cuda(
        n_states,
        trans_mat,
        emission_probs,
        duration_probs_linear,
        start_probs,
        duration_probs,
        obs_seq,
) -> np.ndarray:
    _ensure_loaded()
    fn = getattr(_native, "decode_tensor_viterbi_cuda", None)
    if fn is None:
        raise RuntimeError("CUDA backend not available in this build.")
    return fn(n_states, trans_mat, emission_probs,
              duration_probs_linear, start_probs, duration_probs, obs_seq)


def decode_tensor_viterbi_omp(
        n_states,
        trans_mat,
        emission_probs,
        duration_probs_linear,
        start_probs,
        duration_probs,
        obs_seq,
) -> np.ndarray:
    _ensure_loaded()
    fn = getattr(_native, "decode_tensor_viterbi_omp", None)
    if fn is None:
        raise RuntimeError("OMP backend not available in this build.")
    return fn(n_states, trans_mat, emission_probs,
              duration_probs_linear, start_probs, duration_probs, obs_seq)
