#!/usr/bin/env python3
"""CpG island detection over chrY's euchromatic MSY region (T2T-CHM13 v2.0),
via a single Viterbi decoding pass on the native OpenMP backend.
"""

import time
from pathlib import Path

import numpy as np

from tensor_viterbi import HSMM
from tensor_viterbi.viterbi import decode_tensor_viterbi_omp
from modules.genomics import UCSCRegion, FastaReader, DNA_BASES, fetch_cpg_ground_truth

# k-mer size the model reads the FASTA with (k=2 -> dinucleotides).
K = 2


def build_cpg_hsmm(obs_seq: np.ndarray) -> HSMM:

    # HSMM model definition for CpG island detection: two states (Background,
    # CpG-island), read as dinucleotides (k=2) since the defining feature of a
    # CpG island isn't just GC content but the CG dinucleotide itself — genome-wide
    # it's strongly suppressed (methylated C deaminates to T over evolutionary
    # time), while inside CpG islands that suppression is absent (unmethylated).
    STATES = ["Background", "CpG-island"]

    # Dinucleotide emission probabilities per state, each summing to 1.
    # Background: bulk genomic composition (30% A/T, 20% C/G) with CG suppressed
    # to ~1/4 of its naive frequency. CpG-island: C/G-enriched (35% C/G) with CG
    # left unsuppressed and then some.
    BACKGROUND_EMIT = {
        "AA": 0.092784, "AT": 0.092784, "AC": 0.061856, "AG": 0.061856,
        "TA": 0.092784, "TT": 0.092784, "TC": 0.061856, "TG": 0.061856,
        "CA": 0.061856, "CT": 0.061856, "CC": 0.041237, "CG": 0.010309,
        "GA": 0.061856, "GT": 0.061856, "GC": 0.041237, "GG": 0.041237,
    }
    ISLAND_EMIT = {
        "AA": 0.021201, "AT": 0.021201, "AC": 0.049470, "AG": 0.049470,
        "TA": 0.021201, "TT": 0.021201, "TC": 0.049470, "TG": 0.049470,
        "CA": 0.049470, "CT": 0.049470, "CC": 0.115430, "CG": 0.173145,
        "GA": 0.049470, "GT": 0.049470, "GC": 0.115430, "GG": 0.115430,
    }

    MAX_DURATION = 1000  # uniform 1/D per state


    N = len(STATES)
    emissions = [a + b for a in DNA_BASES for b in DNA_BASES]
    emission_probs = np.array([
        [BACKGROUND_EMIT[dinuc], ISLAND_EMIT[dinuc]] for dinuc in emissions
    ])

    duration_probs = np.full((MAX_DURATION, N), 1.0 / MAX_DURATION)
    trans_mat = np.full((N, N), 1.0 / N)
    start_probs = np.full(N, 1.0 / N)

    return (
        HSMM(STATES)
        .set_emissions(emissions, emission_probs)
        .set_transitions(trans_mat)
        .set_duration_probs(duration_probs)
        .set_start_probs(start_probs)
        .set_observations(obs_seq)
    )



#! NON MI CONVINCE, SEMBRA MOLTO DIVERSO
def score_against_ground_truth(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """Fraction of positions where `predicted` (decoded 0/1 states) agrees with
    `ground_truth` (UCSC's own CpG-island calls, same 0/1 notation). The two
    are truncated to the shorter length first, since `predicted` is indexed by
    dinucleotide start (k=2) while `ground_truth` is indexed per base."""
    n = min(len(predicted), len(ground_truth))
    return float(np.mean(predicted[:n] == ground_truth[:n]))


def _extract_islands(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of 1s in a 0/1 array, as inclusive (start, end) index pairs."""
    if len(mask) == 0:
        return []
    mask = mask.astype(np.int8)
    diff = np.diff(mask)
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]
    if mask[0] == 1:
        starts = np.insert(starts, 0, 0)
    if mask[-1] == 1:
        ends = np.append(ends, len(mask) - 1)
    return list(zip(starts.tolist(), ends.tolist()))


def score_against_ground_truth_detailed(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    max_shift: int = 50,
    found_threshold: float = 0.5,
) -> dict:
    """Per-CpG-island comparison of `predicted` against `ground_truth`, on top
    of the same whole-sequence `overall_score` as `score_against_ground_truth`.
    The two arrays are truncated to the shorter length first.

    Every maximal run of 1s in `predicted` is treated as one predicted island.
    An island is first checked for any overlap at all with `ground_truth`
    within [-max_shift, +max_shift] characters of its own position; with none,
    it's hopeless (no shift could ever match it) and is skipped without
    further analysis — just tallied into `not_found_in_ground_truth`. Islands
    that do have some overlap get analyzed:
      - `score`: fraction of the island's own positions that are also 1 in
        `ground_truth`, i.e. a one-by-one match at the island's own coordinates.
      - a search over `shift` in [-max_shift, +max_shift] characters looks for
        a better-matching offset — `shift` > 0 means the matching ground-truth
        island sits `shift` characters forward (higher coordinate); `shift` < 0
        means it sits `abs(shift)` characters backward. `shifted_score` is the
        match fraction at that best offset (equal to `score` when shift is 0).
    If the best `shifted_score` still falls short of `found_threshold`, the
    island is likewise dropped and only counted in `not_found_in_ground_truth`
    rather than reported in detail — it's a false positive with no real
    counterpart.

    The same check runs in reverse over `ground_truth`'s own islands against
    `predicted`, to catch real islands the model missed entirely — only the
    count is kept (`not_found_in_predicted`), not per-island detail.

    Returns a dict with:
      - "overall_score": float, whole-sequence agreement fraction (same
        definition as `score_against_ground_truth`).
      - "islands": list of dicts, one per predicted CpG island *found* in
        `ground_truth`, each with "start", "end" (inclusive), "length",
        "score", "shift", "shifted_score".
      - "not_found_in_ground_truth": number of predicted islands with no
        counterpart in `ground_truth` (skipped or below `found_threshold`).
      - "not_found_in_predicted": number of ground-truth islands with no
        counterpart in `predicted` (false negatives the model missed).
      - "min_length_found" / "max_length_found": shortest/longest length among
        the *found* islands (None if none were found).
    """
    n = min(len(predicted), len(ground_truth))
    predicted, ground_truth = predicted[:n], ground_truth[:n]
    overall_score = float(np.mean(predicted == ground_truth))

    # Prefix sums let any window's 1-count be fetched in O(1).
    prefix_gt = np.concatenate(([0], np.cumsum(ground_truth.astype(np.int64))))
    prefix_pred = np.concatenate(([0], np.cumsum(predicted.astype(np.int64))))

    def make_window_sum(prefix: np.ndarray):
        def window_sum(a: int, b: int) -> int:
            """Count of 1s in the referenced array's [a, b]; out-of-bounds positions count as 0."""
            lo, hi = max(a, 0), min(b, n - 1)
            if lo > hi:
                return 0
            return int(prefix[hi + 1] - prefix[lo])
        return window_sum

    window_sum_gt = make_window_sum(prefix_gt)
    window_sum_pred = make_window_sum(prefix_pred)

    def best_shifted_score(start: int, end: int, length: int, window_sum) -> tuple[int, float] | None:
        """Best (shift, score) in [-max_shift, +max_shift], or None if there's
        no overlap anywhere in that range (hopeless, not worth searching)."""
        if window_sum(start - max_shift, end + max_shift) == 0:
            return None
        best_shift, best_score = 0, window_sum(start, end) / length
        for shift in range(-max_shift, max_shift + 1):
            if shift == 0:
                continue
            s = window_sum(start + shift, end + shift) / length
            if s > best_score:
                best_shift, best_score = shift, s
        return best_shift, best_score

    islands = []
    not_found_in_ground_truth = 0
    for start, end in _extract_islands(predicted):
        length = end - start + 1
        result = best_shifted_score(start, end, length, window_sum_gt)
        if result is None or result[1] < found_threshold:
            not_found_in_ground_truth += 1
            continue
        best_shift, best_score = result
        islands.append({
            "start": start,
            "end": end,
            "length": length,
            "score": window_sum_gt(start, end) / length,
            "shift": best_shift,
            "shifted_score": best_score,
        })

    not_found_in_predicted = 0
    for start, end in _extract_islands(ground_truth):
        length = end - start + 1
        result = best_shifted_score(start, end, length, window_sum_pred)
        if result is None or result[1] < found_threshold:
            not_found_in_predicted += 1

    lengths = [island["length"] for island in islands]
    return {
        "overall_score": overall_score,
        "islands": islands,
        "not_found_in_ground_truth": not_found_in_ground_truth,
        "not_found_in_predicted": not_found_in_predicted,
        "min_length_found": min(lengths) if lengths else None,
        "max_length_found": max(lengths) if lengths else None,
    }


def write_islands(states: np.ndarray, fasta_path: Path, out_path: Path) -> None:
    with open(out_path, "w") as f:
        f.write(f"# CpG island predictions | source: {fasta_path.name} | generated by tensor-viterbi\n")
        flat = states.astype(int)
        for i in range(0, len(flat), 30):
            f.write("".join(str(s) for s in flat[i:i + 30]) + "\n")


def main() -> None:

    #! ----- FASTA DOWNLOAD ----- 
    dna_region = UCSCRegion(
        assembly="hs1",
        chrom="chrY",
        start=2_458_320,   # end of PAR1 (Rhie et al. 2023)
        end=26_673_214,     # start of Yq12 heterochromatin
        label="T2T-CHM13v2.0_chrY_euchromatic_MSY",
    )
    OBS_LIMIT = None  # max observations read from FASTA (None = whole file)
    fasta_path = dna_region.download()

    print(f"Loading {fasta_path} ...")

    obs_seq = FastaReader(fasta_path, symbols=DNA_BASES, k=K).read()
    if OBS_LIMIT is not None:
        obs_seq = obs_seq[:OBS_LIMIT]
    #! ----- FASTA DOWNLOAD -----


    #! ----- BUILD HSMM -----
    cpg_hsmm = build_cpg_hsmm(obs_seq)
    cpg_hsmm.print_model()
    #! ----- BUILD HSMM -----


    #! ----- VITERBI DECODING -----
    t0 = time.perf_counter()
    result = decode_tensor_viterbi_omp(
        cpg_hsmm.N, cpg_hsmm.trans_mat, cpg_hsmm.emission_probs,
        cpg_hsmm.duration_probs_linear, cpg_hsmm.start_probs,
        cpg_hsmm.duration_probs, cpg_hsmm.obs_seq,
    )
    elapsed = time.perf_counter() - t0

    out_path = fasta_path.with_suffix(".cpg")

    #* SPECIFIC OF CPG ISLANDS, I SHOULD FIND A MORE GENERAL WAY TO WRITE THE RESULTS
    n_background = int(np.sum(result == 0))
    n_island     = int(np.sum(result == 1))
    print(f"-> decode time={elapsed:.4f} s  Background={n_background} ({100*n_background/cpg_hsmm.T:.2f}%)  "
            f"CpG-island={n_island} ({100*n_island/cpg_hsmm.T:.2f}%)")
    #! ----- VITERBI DECODING -----


    #! ----- UCSC GROUND TRUTH COMPARISON -----
    ground_truth = fetch_cpg_ground_truth(dna_region)
    score = score_against_ground_truth_detailed(result, ground_truth)
    print(f"-> agreement with UCSC '{dna_region.chrom}' cpgIslandExt = {100*score['overall_score']:.2f}%")

    print(f"-> predicted CpG islands found in ground truth ({len(score['islands'])}):")
    for island in score["islands"]:
        direction = "forward" if island["shift"] > 0 else "backward" if island["shift"] < 0 else "none"
        print(f"     {island['start']}-{island['end']} (len={island['length']}): "
              f"{100*island['score']:.2f}%  "
              f"best_shift={island['shift']} ({direction}) -> {100*island['shifted_score']:.2f}%")

    print(f"-> predicted islands not found in ground truth: {score['not_found_in_ground_truth']}")
    print(f"-> ground truth islands not found in predicted: {score['not_found_in_predicted']}")
    print(f"-> length range among found islands: "
          f"{score['min_length_found']}-{score['max_length_found']}")
    #! ----- UCSC GROUND TRUTH COMPARISON -----

    write_islands(result, fasta_path, out_path)
    print(f"\nResult written -> {out_path}")


if __name__ == "__main__":
    main()
