"""Shared information-aware time-window contract for training and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


MAD_TO_STD = 0.6744897501960817


@dataclass(frozen=True)
class WindowContract:
    """A fixed-shape contract shared by sampling and overlap inference."""

    length: int = 2560
    nominal_stride: int = 1280
    priority_floor: float = 0.20
    priority_power: float = 1.50
    occupancy_snr_threshold: float = 1.0
    blend_floor: float = 0.05
    score_quantiles: tuple[float, float] = (0.10, 0.90)
    version: str = "consistent-information-window-v1"

    def validate(self, trace_length: int) -> None:
        if self.length < 1 or self.length > trace_length:
            raise ValueError(
                f"Window length {self.length} is invalid for {trace_length}"
            )
        if self.nominal_stride < 1 or self.nominal_stride > self.length:
            raise ValueError("Stride must be in [1, window length]")
        if not 0.0 < self.priority_floor <= 1.0:
            raise ValueError("priority_floor must be in (0, 1]")
        if self.priority_power <= 0.0:
            raise ValueError("priority_power must be positive")
        if not 0.0 < self.blend_floor <= 1.0:
            raise ValueError("blend_floor must be in (0, 1]")
        low, high = self.score_quantiles
        if not 0.0 <= low < high <= 1.0:
            raise ValueError("score_quantiles must satisfy 0 <= low < high <= 1")

    def to_dict(self) -> dict:
        result = asdict(self)
        result["score_quantiles"] = list(self.score_quantiles)
        return result


@dataclass(frozen=True)
class WindowPlan:
    """Deterministic candidates and priorities derived from measured channels."""

    trace_length: int
    starts: np.ndarray
    information_scores: np.ndarray
    normalized_information: np.ndarray
    fusion_priorities: np.ndarray
    sampling_probabilities: np.ndarray
    normalized_rms: np.ndarray
    median_signal_snr: np.ndarray
    occupancies: np.ndarray
    channel_noise_rms: np.ndarray
    contract: WindowContract

    @property
    def stops(self) -> np.ndarray:
        return self.starts + self.contract.length

    def sample_start(self, rng: np.random.Generator) -> tuple[int, int]:
        index = int(
            rng.choice(len(self.starts), p=self.sampling_probabilities)
        )
        return int(self.starts[index]), index

    def to_dict(self, include_noise: bool = False) -> dict:
        result = {
            "trace_length": self.trace_length,
            "starts": self.starts.tolist(),
            "stops": self.stops.tolist(),
            "information_scores": self.information_scores.tolist(),
            "normalized_information": self.normalized_information.tolist(),
            "fusion_priorities": self.fusion_priorities.tolist(),
            "sampling_probabilities": self.sampling_probabilities.tolist(),
            "normalized_rms": self.normalized_rms.tolist(),
            "median_signal_snr": self.median_signal_snr.tolist(),
            "occupancies": self.occupancies.tolist(),
            "contract": self.contract.to_dict(),
        }
        if include_noise:
            result["channel_noise_rms"] = self.channel_noise_rms.tolist()
        return result


def coverage_starts(
    trace_length: int,
    contract: WindowContract,
    minimum_start: int = 0,
    maximum_stop: int | None = None,
) -> np.ndarray:
    """Return deterministic, approximately uniform starts with full coverage."""

    contract.validate(trace_length)
    maximum_stop = trace_length if maximum_stop is None else maximum_stop
    minimum_start = int(minimum_start)
    maximum_stop = int(maximum_stop)
    if minimum_start < 0 or maximum_stop > trace_length:
        raise ValueError("Coverage bounds exceed the trace")
    span = maximum_stop - minimum_start
    if span < contract.length:
        raise ValueError("Coverage interval is shorter than one window")
    maximum_start = maximum_stop - contract.length
    start_span = maximum_start - minimum_start
    count = int(np.ceil(start_span / contract.nominal_stride)) + 1
    starts = np.rint(
        np.linspace(minimum_start, maximum_start, count)
    ).astype(np.int64)
    starts[0] = minimum_start
    starts[-1] = maximum_start
    return np.unique(starts)


def robust_channel_noise_rms(measured: np.ndarray) -> np.ndarray:
    """Estimate per-channel noise without requiring a known background interval."""

    centered = measured.astype(np.float64, copy=False)
    centered = centered - np.median(centered, axis=1, keepdims=True)
    noise = np.median(np.abs(centered), axis=1) / MAD_TO_STD
    positive = noise[noise > 0.0]
    fallback = float(np.median(positive)) if len(positive) else 1.0
    return np.maximum(noise, fallback * 1e-3 + 1e-12)


def _normalize_scores(
    scores: np.ndarray,
    contract: WindowContract,
) -> np.ndarray:
    low = float(np.quantile(scores, contract.score_quantiles[0]))
    high = float(np.quantile(scores, contract.score_quantiles[1]))
    if high <= low + 1e-12:
        return np.zeros_like(scores, dtype=np.float64)
    normalized = np.clip((scores - low) / (high - low), 0.0, 1.0)
    return normalized.astype(np.float64, copy=False)


def build_window_plan(
    measured_channels: np.ndarray,
    contract: WindowContract = WindowContract(),
    minimum_start: int = 0,
    maximum_stop: int | None = None,
    starts: np.ndarray | None = None,
) -> WindowPlan:
    """Score windows using only channels available to the deployed model."""

    measured = np.asarray(measured_channels)
    if measured.ndim != 2:
        raise ValueError("measured_channels must have shape (channels, time)")
    if measured.shape[0] < 1:
        raise ValueError("At least one measured channel is required")
    if not np.all(np.isfinite(measured)):
        raise ValueError("Measured channels contain non-finite values")
    trace_length = int(measured.shape[1])
    maximum_stop = trace_length if maximum_stop is None else int(maximum_stop)
    if starts is None:
        starts = coverage_starts(
            trace_length,
            contract,
            minimum_start=minimum_start,
            maximum_stop=maximum_stop,
        )
    else:
        starts = np.asarray(starts, dtype=np.int64)
        if starts.ndim != 1 or len(starts) < 1:
            raise ValueError("Explicit window starts must be a nonempty vector")
        if np.any(np.diff(starts) <= 0):
            raise ValueError("Explicit window starts must be strictly increasing")
        if int(starts[0]) != int(minimum_start):
            raise ValueError("Explicit windows do not start at the coverage bound")
        if int(starts[-1] + contract.length) != maximum_stop:
            raise ValueError("Explicit windows do not end at the coverage bound")
        if np.any(np.diff(starts) > contract.length):
            raise ValueError("Explicit window starts leave uncovered samples")
        if starts[0] < 0 or starts[-1] + contract.length > trace_length:
            raise ValueError("Explicit window starts exceed the trace")
    noise_rms = robust_channel_noise_rms(measured)
    reference_noise = float(
        np.sqrt(np.mean(noise_rms.astype(np.float64) ** 2)) + 1e-12
    )

    scores = []
    normalized_rms = []
    median_snrs = []
    occupancies = []
    for start in starts:
        window = measured[:, start : start + contract.length].astype(
            np.float64,
            copy=False,
        )
        channel_rms = np.sqrt(np.mean(window**2, axis=1) + 1e-12)
        signal_rms = np.sqrt(
            np.maximum(channel_rms**2 - noise_rms**2, 0.0)
        )
        signal_snr = signal_rms / (noise_rms + 1e-12)
        occupancy = float(
            np.mean(signal_snr >= contract.occupancy_snr_threshold)
        )
        median_snr = float(np.median(signal_snr))
        window_rms = float(np.sqrt(np.mean(window**2) + 1e-12))
        energy_snr = float(
            np.sqrt(np.mean(signal_rms**2)) / reference_noise
        )
        score = float(
            np.log1p(max(energy_snr, 0.0))
            * np.log1p(max(median_snr, 0.0))
            * np.sqrt(0.05 + occupancy)
        )
        scores.append(score)
        normalized_rms.append(window_rms / reference_noise)
        median_snrs.append(median_snr)
        occupancies.append(occupancy)

    score_array = np.asarray(scores, dtype=np.float64)
    normalized = _normalize_scores(score_array, contract)
    priorities = contract.priority_floor + (
        1.0 - contract.priority_floor
    ) * normalized**contract.priority_power
    probabilities = priorities / priorities.sum()
    return WindowPlan(
        trace_length=trace_length,
        starts=starts,
        information_scores=score_array,
        normalized_information=normalized,
        fusion_priorities=priorities,
        sampling_probabilities=probabilities,
        normalized_rms=np.asarray(normalized_rms, dtype=np.float64),
        median_signal_snr=np.asarray(median_snrs, dtype=np.float64),
        occupancies=np.asarray(occupancies, dtype=np.float64),
        channel_noise_rms=noise_rms.astype(np.float64, copy=False),
        contract=contract,
    )


def blending_window(contract: WindowContract) -> np.ndarray:
    """A nonzero taper that is valid even at trace boundaries."""

    if contract.length == 1:
        return np.ones(1, dtype=np.float32)
    hann = np.hanning(contract.length).astype(np.float64)
    blend = contract.blend_floor + (1.0 - contract.blend_floor) * hann
    return blend.astype(np.float32)


class OverlapAccumulator:
    """Streaming overlap fusion for fixed-shape neural predictions."""

    def __init__(
        self,
        output_shape: tuple[int, ...],
        plan: WindowPlan,
        dtype: np.dtype = np.float32,
    ) -> None:
        if output_shape[-1] != plan.trace_length:
            raise ValueError("Output time length does not match the plan")
        self.plan = plan
        self.value = np.zeros(output_shape, dtype=np.float64)
        self.weight = np.zeros(plan.trace_length, dtype=np.float64)
        self.output_dtype = np.dtype(dtype)
        self.taper = blending_window(plan.contract).astype(np.float64)

    def add(self, candidate_index: int, prediction: np.ndarray) -> None:
        start = int(self.plan.starts[candidate_index])
        stop = start + self.plan.contract.length
        expected = self.value.shape[:-1] + (self.plan.contract.length,)
        if prediction.shape != expected:
            raise ValueError(
                f"Prediction shape {prediction.shape} does not match {expected}"
            )
        weight = (
            self.taper
            * float(self.plan.fusion_priorities[candidate_index])
        )
        self.value[..., start:stop] += prediction.astype(
            np.float64, copy=False
        ) * weight
        self.weight[start:stop] += weight

    def finish(self) -> np.ndarray:
        if np.any(self.weight <= 0.0):
            missing = np.flatnonzero(self.weight <= 0.0)
            raise RuntimeError(
                f"Window plan left {len(missing)} time samples uncovered"
            )
        result = self.value / self.weight
        return result.astype(self.output_dtype)


def plan_audit(plan: WindowPlan) -> dict:
    """Summarize the train-sampling and inference-fusion distributions."""

    probabilities = plan.sampling_probabilities
    priorities = plan.fusion_priorities
    normalized = plan.normalized_information
    effective_sampling_information = float(
        np.sum(probabilities * normalized)
    )
    fusion_distribution = priorities / priorities.sum()
    effective_fusion_information = float(
        np.sum(fusion_distribution * normalized)
    )
    return {
        "format_version": "consistent-time-window-audit-v1",
        "window_count": int(len(plan.starts)),
        "trace_length": plan.trace_length,
        "coverage_fraction": 1.0,
        "fixed_input_length": plan.contract.length,
        "nominal_overlap_fraction": float(
            1.0
            - plan.contract.nominal_stride / plan.contract.length
        ),
        "sampling_probability_sum": float(probabilities.sum()),
        "effective_training_information": effective_sampling_information,
        "effective_fusion_information": effective_fusion_information,
        "train_inference_priority_l1": float(
            np.sum(np.abs(probabilities - fusion_distribution))
        ),
        "minimum_sampling_probability": float(probabilities.min()),
        "maximum_sampling_probability": float(probabilities.max()),
        "minimum_fusion_priority": float(priorities.min()),
        "maximum_fusion_priority": float(priorities.max()),
        "plan": plan.to_dict(include_noise=False),
    }
