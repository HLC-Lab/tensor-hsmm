# tensor-viterbi

Python library for Hidden Semi-Markov Model (HSMM) Viterbi decoding with CPU and GPU backends, plus a genomic gene-structure prediction application built on top of it.

---

## Table of Contents

- [Implemented Backends](#implemented-backends)
- [Repository Structure](#repository-structure)
- [The `tensor_viterbi` Library](#the-tensor_viterbi-library)
  - [HSMM Class](#hsmm-class)
  - [FastaReader Class](#fastareader-class)
  - [Viterbi Decoders](#viterbi-decoders)
- [Gene Prediction Application](#gene-prediction-application)
- [Requirements](#requirements)
- [Installation](#installation)

---

## Implemented Backends

| Function | Status | Description |
|---|---|---|
| `decode_log_tensor_viterbi_cached` | ✅ Active | Vectorized log-space tensor implementation with emission caching (Python) |
| `decode_tensor_viterbi_cpp` | ✅ Active | C++ tensor implementation (via pybind11) |
| `decode_tensor_viterbi_omp` | ✅ Active | OpenMP-parallelized C++ tensor implementation |
| `decode_tensor_viterbi_omp_opt` | ✅ Active | Optimized OpenMP C++ tensor implementation |
| `decode_tensor_viterbi_cuda` | ✅ Active | GPU tensor implementation (CUDA / ROCm via pybind11) |
| `decode_vanilla_viterbi` | ✅ Active | Reference O(T·N²·D) triple-loop implementation |
| `decode_log_tensor_viterbi_no_cache` | ⚠️ Deprecated | Log-space tensor without emission caching |
| `decode_tensor_viterbi` | ⚠️ Deprecated | Linear-space tensor, underflows after ~370 timesteps |

---

## Repository Structure

```
tensor-viterbi/
├── tensor_viterbi/              # Python library package
│   ├── __init__.py
│   ├── hsmm.py                  # HSMM model class
│   ├── fasta.py                 # FASTA file reader
│   ├── metrics.py               # Hardware metrics collectors (Cray PM)
│   └── viterbi/
│       ├── tensor.py            # Vectorized Python Viterbi (primary)
│       ├── vanilla.py           # Reference triple-loop implementation
│       ├── native.py            # Lazy wrappers for C++/CUDA/OMP extensions
│       ├── _native.pyi          # Type stubs
│       └── <system>/<toolchain>/
│           └── _native.so       # Per-system compiled extension
├── src/                         # C++ / CUDA / ROCm sources
│   ├── bindings.cpp             # pybind11 module definition
│   ├── hsmm.cu / hsmm.hpp       # CPU and GPU Viterbi implementations
│   └── kernels.cu / kernels.cuh # CUDA/HIP GPU kernels
├── viterbi_app.py               # Gene structure prediction application
├── gff3_downloader.py           # Genomic annotation downloader
├── requirements.txt
└── CMakeLists.txt
```

---

## The `tensor_viterbi` Library

### HSMM Class

`tensor_viterbi.HSMM` is a builder-style class that holds all model parameters and exposes decoding and parameter re-estimation.

```python
from tensor_viterbi import HSMM
import numpy as np

hsmm = (
    HSMM(states=["Sleep", "Wake"])
    .set_emissions(["low", "high"], emission_probs)   # shape (O, N)
    .set_transitions(trans_mat)                        # shape (N, N)
    .set_duration_probs(duration_probs)                # shape (D, N)
    .set_start_probs(start_probs)                      # shape (N,)
    .set_observations(obs_seq)                         # shape (T,) integer indices
)
```

#### Builder Methods

| Method | Shape | Description |
|---|---|---|
| `set_emissions(symbols, probs)` | probs: (O, N) | Emission symbol list and probability matrix |
| `set_transitions(trans_mat)` | (N, N) | State-to-state transition probabilities, rows sum to 1 |
| `set_duration_probs(probs)` | (D, N) | Duration distribution per state (linear space), columns sum to 1 |
| `set_start_probs(probs)` | (N,) | Initial state distribution, sums to 1 |
| `set_observations(obs_seq)` | (T,) | Integer observation sequence (indices into the symbols list) |

#### Key Methods

**`decode() → np.ndarray`**

Runs Viterbi decoding using the vectorized Python backend. Converts to log space internally and returns the most-likely state sequence of shape (T,).

```python
path = hsmm.decode()
```

---

**`to_log_space()`**

Converts `trans_mat`, `emission_probs`, `start_probs`, and `duration_probs` to log space in-place (adds a small smoothing term of 1e-30 before taking the log). `duration_probs_linear` is left unchanged — the native backends need it in linear space.

Call this before passing the model to any native decoder:

```python
hsmm.to_log_space()
result = decode_tensor_viterbi_omp(
    N, hsmm.trans_mat, hsmm.emission_probs,
    hsmm.duration_probs_linear, hsmm.start_probs,
    hsmm.duration_probs, hsmm.obs_seq,
)
```

---

**`reestimate(result: np.ndarray) → HSMM`**

Re-estimates all model parameters from a Viterbi-decoded state path using hard-assignment EM (Viterbi training). Returns a **new** `HSMM` object in linear probability space, ready for the next iteration.

Parameter updates:

- **Emissions** — counts how many times each symbol is emitted by each state, then normalizes per state.
- **Transitions** — counts consecutive state pairs at segment boundaries, then normalizes per source state.
- **Durations** — builds a histogram of observed segment lengths per state, then applies a uniform ±3-bin neighbourhood smoothing kernel before normalizing. This prevents zero probability mass on durations close to — but not exactly equal to — observed lengths.
- **Start probabilities** — set deterministically to place all mass on the first decoded state.

```python
for _ in range(n_iterations):
    hsmm.to_log_space()
    result = decode_tensor_viterbi_omp(...)
    hsmm = hsmm.reestimate(result)   # returns new HSMM with updated linear-space parameters
```

---

**`print_model()`**

Prints a diagnostic summary of the model dimensions, states, start probabilities, transition matrix, emission matrix, and duration distributions. Call this **before** `to_log_space()` to display meaningful linear-space values.

---

**`load_model(json_path) → HSMM`** *(static)*

Loads a model from a JSON configuration file. See [Data Format](#data-format) for the expected schema.

---

### FastaReader Class

`tensor_viterbi.FastaReader` reads a FASTA file and converts the sequence to an integer index array.

```python
from tensor_viterbi import FastaReader

reader = FastaReader("sequence.fa", symbols=["A", "T", "C", "G"])
obs = reader.read()          # np.ndarray, dtype int64, shape (T,)
```

Symbol matching is case-insensitive; characters not in `symbols` are silently skipped.

---

### Viterbi Decoders

All decoders return a (T,) integer array of the decoded state sequence.

#### Python Backends

```python
from tensor_viterbi import decode_log_tensor_viterbi_cached, decode_vanilla_viterbi

# Both require the HSMM to be in log space already
path = decode_log_tensor_viterbi_cached(hsmm)   # vectorized, sliding-window emission cache
path = decode_vanilla_viterbi(hsmm)             # O(T·N²·D) reference, naive nested loops
```

#### Native Backends (C++ / OMP / CUDA)

```python
from tensor_viterbi import (
    decode_tensor_viterbi_cpp,
    decode_tensor_viterbi_omp,
    decode_tensor_viterbi_cuda,
)

# Shared signature for all three:
path = decode_tensor_viterbi_omp(
    N,                          # int   — number of states
    trans_mat,                  # (N, N) log-space
    emission_probs,             # (O, N) log-space
    duration_probs_linear,      # (D, N) linear space
    start_probs,                # (N,)   log-space
    duration_probs,             # (D, N) log-space
    obs_seq,                    # (T,)   integer indices
)
```

Native backends raise `RuntimeError` if the extension has not been compiled (see [Installation](#installation)).

---

## Gene Prediction Application

`viterbi_app.py` demonstrates the library on a real genomics task: annotating a nucleotide sequence with a three-state gene-structure model (Intergenic / Exon / Intron) via iterative Viterbi training.

### Model

Three states with biologically motivated emission probabilities:

| State | Character | A | T | C | G |
|---|---|---|---|---|---|
| Intergenic | very AT-rich | 0.35 | 0.35 | 0.15 | 0.15 |
| Exon | GC-rich | 0.20 | 0.20 | 0.30 | 0.30 |
| Intron | AT-rich | 0.30 | 0.30 | 0.20 | 0.20 |

Initial transition and duration distributions are uniform (max duration D = 1000 bp). Parameters are refined over 5 Viterbi-training iterations.

### Data

The application targets the euchromatic MSY region of chrY in the T2T-CHM13 v2.0 assembly (coordinates 2 458 320 – 26 673 214), fetched automatically from the UCSC REST API on first run.

### Usage

```bash
python viterbi_app.py [--outdir <directory>]
```

| Argument | Default | Description |
|---|---|---|
| `--outdir` | `.` | Directory for downloaded FASTA and output annotation |

### Workflow

1. **Download** — fetches `chrY:2458320-26673214` from the UCSC Genome Browser API and writes `T2T-CHM13v2.0_chrY_euchromatic_MSY.fa` (skipped if the file already exists).
2. **Load** — reads up to 10 000 nucleotides and initialises the HSMM.
3. **Iterate** — runs 5 rounds of Viterbi decode → `reestimate()`, printing the fraction of bases assigned to each state per round.
4. **Write** — saves the final decoded annotation as `<input>.gene`.

### Output `.gene` Format

Plain text, 30 state-index characters per line:

```
# Gene structure predictions | source: T2T-CHM13v2.0_chrY_euchromatic_MSY.fa | generated by tensor-viterbi
000000000001111111111222222222200
000000001111111111112222222222222
...
```

`0` = Intergenic, `1` = Exon, `2` = Intron.

---

## Requirements

- Python >= 3.10
- CMake >= 3.20 *(only for native C++/OMP/CUDA backends)*
- A C++17 compiler with OpenMP support: GCC, Clang, Intel ICX, Cray CC, or Fujitsu FCC *(native backends only, OpenMP is required even for the GPU build)*
- CUDA >= 12.0 (or ROCm, auto-detected from the environment) *(GPU backend only)*

Python packages:

```
numpy >= 2.4
pybind11 >= 3.0
requests >= 2.30
```

```bash
pip install -r requirements.txt
```

---

## Installation

```bash
git clone https://github.com/lor3ny/tensor-viterbi.git
cd tensor-viterbi
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The pure-Python backends (`decode_log_tensor_viterbi_cached`, `decode_vanilla_viterbi`) work immediately, with no further steps. `decode_tensor_viterbi_cpp`, `decode_tensor_viterbi_omp`, and `decode_tensor_viterbi_cuda` raise `RuntimeError` until the native extension below is compiled.

### Compiling the native extension (C++ / OMP / CUDA)

The native backends are a single pybind11 module (`tensor_viterbi/viterbi/_native.so`) built from `src/` via the root `CMakeLists.txt`. Configure once, then build:

```bash
# CPU / OpenMP only (no GPU toolchain required)
cmake -B build -DBUILD_GPU=OFF
cmake --build build -j$(nproc)

# GPU backend (CUDA or ROCm — auto-detected, or set -DGPU_PLATFORM=CUDA|ROCM)
cmake -B build -DBUILD_GPU=ON
cmake --build build -j$(nproc)
```

Both commands must be run from the repository root — the build writes `_native.so` directly into `tensor_viterbi/viterbi/`, which is where `native.py` looks for it. `build/` is a disposable, gitignored CMake working directory; delete it and re-run `cmake -B build ...` to reconfigure from scratch (e.g. when switching between `-DBUILD_GPU=ON`/`OFF`).

Once built, `decode_tensor_viterbi_omp` (and, if compiled with `-DBUILD_GPU=ON`, `decode_tensor_viterbi_cuda`) are available immediately — no reinstall step needed, since `tensor_viterbi` imports the compiled `.so` directly:

```python
from tensor_viterbi.viterbi import decode_tensor_viterbi_omp
```

This is exactly what `viterbi_app.py` uses.
