import random
import numpy as np
import json

_LOG_SMOOTHING = 1e-30


def _log(arr: np.ndarray) -> np.ndarray:
    return np.log(arr + _LOG_SMOOTHING)


class HSMM:
    """HSMM parameters are always stored in log space (`trans_mat`,
    `emission_probs`, `start_probs`, `duration_probs`), converted
    automatically as soon as they are set — via the constructor or the
    builder setters below. The linear-space values remain available as
    `*_linear` attributes, e.g. for native backends or display.
    """

    def __init__(self, states, emissions=None, trans_mat=None, emission_prob=None,
                 duration_probs=None, start_probs=None):
        self.states = states
        self.emissions = emissions
        self.obs_seq = None

        self.trans_mat_linear = None
        self.emission_probs_linear = None
        self.duration_probs_linear = None
        self.start_probs_linear = None

        self.trans_mat = None
        self.emission_probs = None
        self.duration_probs = None
        self.start_probs = None

        if trans_mat is not None:
            self.set_transitions(trans_mat)
        if emission_prob is not None:
            self.set_emissions(emissions, emission_prob)
        if duration_probs is not None:
            self.set_duration_probs(duration_probs)
        if start_probs is not None:
            self.set_start_probs(start_probs)

    # ------------------------------------------------------------------
    # Builder setters — each one converts to log space immediately.
    # ------------------------------------------------------------------

    def set_transitions(self, trans_mat: np.ndarray) -> "HSMM":
        """Transition matrix, shape (N, N), given in linear space. Rows must sum to 1."""
        self.trans_mat_linear = np.asarray(trans_mat, dtype=float)
        self.trans_mat = _log(self.trans_mat_linear)
        return self

    def set_emissions(self, emissions, emission_probs: np.ndarray) -> "HSMM":
        """Emission symbols and probability matrix, shape (O, N), given in linear space."""
        self.emissions = emissions
        self.emission_probs_linear = np.asarray(emission_probs, dtype=float)
        self.emission_probs = _log(self.emission_probs_linear)
        return self

    def set_duration_probs(self, duration_probs: np.ndarray) -> "HSMM":
        """Duration probabilities, shape (D, N), given in linear space.
        Stored as both duration_probs (log space) and duration_probs_linear
        (kept in linear space for native backends)."""
        self.duration_probs_linear = np.asarray(duration_probs, dtype=float)
        self.duration_probs = _log(self.duration_probs_linear)
        return self

    def set_start_probs(self, start_probs: np.ndarray) -> "HSMM":
        """Initial state distribution, shape (N,), given in linear space."""
        self.start_probs_linear = np.asarray(start_probs, dtype=float)
        self.start_probs = _log(self.start_probs_linear)
        return self

    def set_observations(self, obs_seq: np.ndarray) -> "HSMM":
        """Observation sequence, shape (T,)."""
        self.obs_seq = np.asarray(obs_seq, dtype=float)
        return self

    # ------------------------------------------------------------------
    # Completeness check
    # ------------------------------------------------------------------

    def _missing(self) -> list[str]:
        required = {
            "transitions (set_transitions)":     self.trans_mat,
            "emissions (set_emissions)":          self.emission_probs,
            "duration probs (set_duration_probs)": self.duration_probs,
            "start probs (set_start_probs)":      self.start_probs,
            "observations (set_observations)":    self.obs_seq,
        }
        return [name for name, val in required.items() if val is None]

    def is_complete(self) -> bool:
        return len(self._missing()) == 0

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(self) -> np.ndarray:
        """Run Viterbi decoding. Raises if any required field is not set."""
        missing = self._missing()
        if missing:
            raise RuntimeError(
                "HSMM model is incomplete. Missing fields:\n"
                + "\n".join(f"  - {m}" for m in missing)
            )
        from tensor_viterbi.viterbi.tensor import decode_log_tensor_viterbi_cached
        return decode_log_tensor_viterbi_cached(self)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def set_obs_sequence(self, obs_seq):
        self.obs_seq = obs_seq

    def reestimate(self, result: np.ndarray) -> "HSMM":
        """Re-estimate parameters from a Viterbi-decoded state sequence.

        Computes new emission, transition, and duration probabilities by counting
        statistics directly from the decoded path, then normalizing. Duration
        probabilities are smoothed with a uniform ±3 neighbourhood kernel.
        Returns a new HSMM, converted to log space, ready for the next iteration.
        """
        N       = len(self.states)
        O       = len(self.emissions)
        D       = self.duration_probs_linear.shape[0]
        obs_seq = self.obs_seq.astype(int)
        T       = len(obs_seq)

        # Parse decoded path into (state, start_t, length) segments
        segments: list[tuple[int, int, int]] = []
        t = 0
        while t < T:
            s = int(result[t])
            length = 1
            while t + length < T and int(result[t + length]) == s:
                length += 1
            segments.append((s, t, length))
            t += length

        # Emission counts
        emit_counts = np.zeros((O, N), dtype=float)
        for t in range(T):
            emit_counts[int(obs_seq[t]), int(result[t])] += 1
        col_sums = emit_counts.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        new_emission_probs = emit_counts / col_sums

        # Transition counts
        trans_counts = np.zeros((N, N), dtype=float)
        for idx in range(len(segments) - 1):
            trans_counts[segments[idx][0], segments[idx + 1][0]] += 1
        row_sums = trans_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        new_trans_mat = trans_counts / row_sums

        # Duration counts with uniform ±3 neighbourhood smoothing
        dur_counts = np.zeros((D, N), dtype=float)
        for s, _, length in segments:
            dur_counts[min(length, D) - 1, s] += 1
        kernel = np.ones(7)
        for n in range(N):
            dur_counts[:, n] = np.convolve(dur_counts[:, n], kernel, mode='same')
        col_sums = dur_counts.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        new_duration_probs = dur_counts / col_sums

        # Start probs: deterministic from the first decoded segment
        new_start_probs = np.zeros(N, dtype=float)
        new_start_probs[segments[0][0]] = 1.0

        new_hsmm = HSMM(self.states)
        new_hsmm.set_emissions(self.emissions, new_emission_probs)
        new_hsmm.set_transitions(new_trans_mat)
        new_hsmm.set_duration_probs(new_duration_probs)
        new_hsmm.set_start_probs(new_start_probs)
        new_hsmm.set_obs_sequence(self.obs_seq)
        return new_hsmm

    def print_model(self):
        """Prints the model's linear-space probabilities (internally the
        model always holds the log-space versions used for decoding)."""
        N = len(self.states)
        O = len(self.emissions) if self.emissions is not None else "?"
        D = self.duration_probs_linear.shape[0] if self.duration_probs_linear is not None else "?"
        T = len(self.obs_seq) if self.obs_seq is not None else "?"

        print("===== HSMM MODEL =====")
        print(f"\nDimensions:")
        print(f"  N (states)    = {N}")
        print(f"  O (emissions) = {O}")
        print(f"  D (max dur)   = {D}")
        print(f"  T (obs len)   = {T}")

        print(f"\nStates ({N}):")
        for i, s in enumerate(self.states):
            print(f"  [{i}] {s}")

        if self.start_probs_linear is not None:
            print("\nStart probabilities (pi):")
            for i, s in enumerate(self.states):
                print(f"  {s}: {self.start_probs_linear[i]:.6f}")

        if self.trans_mat_linear is not None:
            print("\nTransition matrix (N x N):")
            for i in range(N):
                row = "  ".join(f"{self.trans_mat_linear[i, j]:8.6f}" for j in range(N))
                print(f"  {row}")

        if self.emission_probs_linear is not None:
            print(f"\nEmission probabilities (O x N):")
            for o in range(len(self.emissions)):
                row = "  ".join(f"{self.emission_probs_linear[o, s]:8.6f}" for s in range(N))
                print(f"  Obs {o}: {row}")

        if self.duration_probs_linear is not None:
            D = self.duration_probs_linear.shape[0]
            print("\nDuration probabilities:")
            for s in range(N):
                row = "  ".join(f"{self.duration_probs_linear[d, s]:.6f}" for d in range(D))
                print(f"  State {self.states[s]}: [ {row} ]")

        print("\n======================")

    @staticmethod
    def load_model(json_path: str = "hsmm_config.json") -> "HSMM":
        with open(json_path, "r") as f:
            cfg = json.load(f)

        n_bins       = int(cfg["n_bins"])
        seed         = int(cfg["seed"])

        np.random.seed(seed)
        random.seed(seed)

        sleep_states = [s["name"] for s in cfg["states"]]
        sleep_emissions = np.arange(n_bins)
        sleep_obs_seq = np.array(cfg["obs_seq"], dtype=float) - 1

        sleep_trans_mat = np.array(cfg["trans_mat"], dtype=float)

        emission_by_state = np.array(
            [s["emission_probs"] for s in cfg["states"]], dtype=float
        )
        sleep_emission_probs = emission_by_state.T

        sleep_start_probs = np.array(cfg["pi"], dtype=float)

        sleep_duration_probs = np.array(
            [s["duration_probs"] for s in cfg["states"]], dtype=float
        )

        hsmm_sleep = HSMM(
            sleep_states,
            sleep_emissions,
            sleep_trans_mat.T,
            sleep_emission_probs,
            sleep_duration_probs.T,
            sleep_start_probs,
        )
        hsmm_sleep.set_obs_sequence(sleep_obs_seq)

        return hsmm_sleep
