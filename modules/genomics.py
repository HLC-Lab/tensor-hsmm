
from collections import deque
from pathlib import Path
import sys

import numpy as np
import textwrap
import requests


API_URL = "https://api.genome.ucsc.edu/getData/sequence" # UCSC REST API for fetching genomic sequences
OBS_LIMIT = None                                         # max observations read from FASTA (None = whole file)
DNA_BASES = ["A", "T", "C", "G"]
RNA_BASES = ["A", "U", "C", "G"]

ASSEMBLY = "hs1"  # T2T-CHM13 v2.0


# Approximate euchromatic MSY boundaries.
# - PAR1 end  -> start of euchromatic MSY
# - Yq12 het. -> end of euchromatic MSY
# NOTE: the centromere lies inside this interval; in hg19 it is a gap (Ns),
REGIONS = {
    "hs1": {            # T2T-CHM13 v2.0
        "chrom": "chrY",
        "start": 2_458_320,   # end of PAR1 (Rhie et al. 2023)
        "end":   26_673_214,  # start of Yq12 heterochromatin
        "label": "T2T-CHM13v2.0_chrY_euchromatic_MSY",
    },
}

#! -------------------------------------------------------------
#!
#! UTILITIES, maybe I'll need to create a library fire for these
#!
#! -------------------------------------------------------------



def fetch_sequence(genome: str, chrom: str, start: int, end: int) -> str:

    params = {"genome": genome, "chrom": chrom, "start": start, "end": end}
    print(f"[info] Requesting {chrom}:{start}-{end} from {genome} ...",
          file=sys.stderr)
    r = requests.get(API_URL, params=params, timeout=300)
    r.raise_for_status()
    data = r.json()
    if "dna" not in data:
        raise RuntimeError(f"Unexpected response: {data}")
    return data["dna"]


def write_fasta(path: Path, header: str, sequence: str, width: int = 60) -> None:
    """Write a single-record FASTA, wrapping the sequence at `width` columns."""
    with path.open("w") as fh:
        fh.write(f">{header}\n")
        for line in textwrap.wrap(sequence, width=width):
            fh.write(line + "\n")


def read_fasta(path: str | Path) -> np.ndarray:
    """Read a FASTA file and return its sequence as an array of the genome's bases.

    Header lines (starting with '>') are skipped; a multi-record file is
    concatenated into a single flat array. Characters are upper-cased.
    """
    def _iter_chars():
        with open(path) as f:
            for line in f:
                if line.startswith(">"):
                    continue
                yield from line.strip().upper()

    return np.fromiter(_iter_chars(), dtype="<U1")


class UCSCRegion:
    """A genomic region identified by assembly/chromosome/coordinates, downloadable from UCSC.

    Parameters
    ----------
    assembly:
        UCSC genome assembly name (e.g. "hs1" for T2T-CHM13 v2.0).
    chrom:
        Chromosome name (e.g. "chrY").
    start, end:
        0-based half-open region coordinates.
    label:
        Name used for the downloaded FASTA file (`<label>.fa`) and its header.
    """

    def __init__(self, assembly: str, chrom: str, start: int, end: int, label: str):
        self.assembly = assembly
        self.chrom = chrom
        self.start = start
        self.end = end
        self.label = label

    def download(self, outdir: str | Path = ".") -> Path:
        """Fetch this region from the UCSC REST API and write it to `<outdir>/<label>.fa`.

        Skips the download if the file already exists. Returns the FASTA path.
        """
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        out_path = outdir / f"{self.label}.fa"

        if out_path.exists():
            print(f"[skip] {out_path} already exists, skipping download.", file=sys.stderr)
            return out_path

        seq = fetch_sequence(self.assembly, self.chrom, self.start, self.end)
        header = f"{self.label} {self.chrom}:{self.start}-{self.end} len={len(seq)}"
        write_fasta(out_path, header, seq)

        n_count = seq.upper().count("N")
        print(f"[done] Wrote {out_path}", file=sys.stderr)
        print(f"[stats] length = {len(seq):,} bp", file=sys.stderr)
        print(f"[stats] Ns     = {n_count:,} ({100 * n_count / len(seq):.2f}%)", file=sys.stderr)
        return out_path


class FastaReader:
    """Read a FASTA file and map nucleotides to observation indices, optionally in k-mers.

    Symbols must be defined before reading so the mapping is fixed.
    Characters not in the symbol set are skipped silently.

    Parameters
    ----------
    path:
        Path to the .fa / .fasta file.
    symbols:
        Ordered list of valid characters (e.g. ["A","C","G","T","N"]).
        Position in the list is the observation index passed to the model.
        Matching is case-insensitive.
    k:
        k-mer size (default 1, i.e. one observation per nucleotide). With
        k > 1, each observation is a sliding window of k consecutive symbols
        (step 1), encoded as a single integer in [0, len(symbols)**k) via
        positional/mixed-radix encoding — so it's still a plain 1D array of
        indices, ready to feed as an HSMM observation sequence with an
        alphabet of size len(symbols)**k.

        E.g. for symbols=["A","C","G","T"] and sequence "ACGT":
        k=1 -> indices of [A, C, G, T]  (4 observations)
        k=3 -> indices of [ACG, CGT]    (2 observations — the first k-1
               positions can't start a full window on their own, so the
               first output only appears once k symbols have been seen)
    """

    def __init__(self, path: str | Path, symbols: list[str], k: int = 1):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self._path = Path(path)
        self.symbols = symbols
        self.k = k
        self._index: dict[str, int] = {s.upper(): i for i, s in enumerate(symbols)}
        self._base = len(symbols)

    # ------------------------------------------------------------------
    # Core generators — skip headers and unknown characters
    # ------------------------------------------------------------------

    def _iter_indices(self):
        with open(self._path) as f:
            for line in f:
                if line.startswith(">"):
                    continue
                for ch in line.rstrip():
                    idx = self._index.get(ch.upper())
                    if idx is not None:
                        yield idx

    def _iter_kmers(self):
        """Yield each k-mer window, encoded as a single base-`len(symbols)` integer.

        Reduces to `_iter_indices()` unchanged when k == 1.
        """
        base = self._base
        k = self.k
        modulus = base ** k
        value = 0
        seen = 0
        for idx in self._iter_indices():
            value = (value * base + idx) % modulus
            seen += 1
            if seen >= k:
                yield value

    # ------------------------------------------------------------------
    # Public reading API
    # ------------------------------------------------------------------

    def read(self) -> np.ndarray:
        """Load the entire sequence as a numpy array of (k-mer) observation indices."""
        return np.fromiter(self._iter_kmers(), dtype=np.int64)

    def iter_chars(self):
        """Yield one (k-mer) observation index at a time (memory-efficient)."""
        yield from self._iter_kmers()

    def iter_windows(self, k: int):
        """Yield raw sliding windows of length k (step=1) as int64 arrays of per-symbol indices.

        Unlike `read()`/`iter_chars()`, this ignores `self.k` and returns the
        window's individual symbol indices rather than a single encoded value.
        """
        if k < 1:
            raise ValueError(f"Window size must be >= 1, got {k}")
        buf = deque()
        for idx in self._iter_indices():
            buf.append(idx)
            if len(buf) == k:
                yield np.array(buf, dtype=np.int64)
                buf.popleft()

