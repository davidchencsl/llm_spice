import csv
from dataclasses import dataclass
from itertools import islice
from typing import Iterator, Optional, Sequence, Callable
import matplotlib.pyplot as plt
import numpy as np

from llm_spice.utils.registry import register_trace, TRACE_REGISTRY
from llm_spice.utils.common import PROJECT_ROOT


@dataclass
class Request:
    id: int
    input_tokens: int
    output_tokens: int
    enqueue_time: float = 0.0
    prefill_start_time: float = 0.0
    dequeue_time: float = 0.0
    prefill_finish_time: float = 0.0

    def __hash__(self):
        return self.id

    @property
    def ttft(self):
        return self.prefill_finish_time - self.enqueue_time

    @property
    def ttc(self):
        return self.dequeue_time - self.enqueue_time

    @property
    def tpot(self):
        return (
            (self.ttc - self.ttft) / self.output_tokens
            if self.output_tokens > 0
            else 0.0
        )


class Trace:
    def __init__(self, name: str):
        self.name = name
        self._id = 0

    def __iter__(self) -> Iterator[Request]:
        self._id += 1
        return self  # type: ignore

    def __next__(self) -> Request:
        raise NotImplementedError

    def __getitem__(self, slice: slice):
        return islice(self, slice.start, slice.stop, slice.step)

    def __len__(self):
        raise NotImplementedError

    def average_input_output_tokens(self) -> tuple[int, int]:
        raise NotImplementedError

    def reset(self):
        self._id = 0

    def close(self):
        pass

    @staticmethod
    def create(name: str) -> "Trace":
        cls, kwargs = TRACE_REGISTRY[name]
        return cls(**kwargs)

    @staticmethod
    def get_all_traces():
        return list(TRACE_REGISTRY.keys())


@register_trace(
    "AzureLLMCode2023",
    trace_file=f"{PROJECT_ROOT}/traces/azure/AzureLLMInferenceTrace_code.csv",
)
@register_trace(
    "AzureLLMChat2023",
    trace_file=f"{PROJECT_ROOT}/traces/azure/AzureLLMInferenceTrace_conv.csv",
)
@register_trace(
    "AzureLLMCode2024",
    trace_file=f"{PROJECT_ROOT}/traces/azure/AzureLLMInferenceTrace_code_1week.csv",
)
@register_trace(
    "AzureLLMChat2024",
    trace_file=f"{PROJECT_ROOT}/traces/azure/AzureLLMInferenceTrace_conv_1week.csv",
)
class AzureTrace(Trace):
    def __init__(self, trace_file: str, name: str = "AzureTrace"):
        super().__init__(name)
        self.trace_file = open(trace_file, "r")
        self.csv_reader = csv.reader(self.trace_file)
        self.trace_length = sum(1 for _ in self.trace_file) - 1  # -1 for the header
        self.reset()

    def __iter__(self):
        for i, row in enumerate(self.csv_reader):
            if i == 0:
                continue
            req = Request(
                id=self._id, input_tokens=int(row[1]), output_tokens=int(row[2])
            )
            super().__iter__()
            yield req

    def __len__(self):
        return self.trace_length

    def reset(self):
        super().reset()
        self.trace_file.seek(0)

    def close(self):
        self.trace_file.close()

    def average_input_output_tokens(self) -> tuple[int, int]:
        self.reset()
        input_tokens = [req.input_tokens for req in self]
        self.reset()
        output_tokens = [req.output_tokens for req in self]
        self.reset()
        result = (
            round(sum(input_tokens) / len(input_tokens) if input_tokens else 0.0),
            round(sum(output_tokens) / len(output_tokens) if output_tokens else 0.0),
        )
        return result


def create_pdf_sampler(pdf_points: Sequence[Sequence[float]]) -> Callable[[int], float]:
    """
    Return a sampler for a piecewise-linear PDF defined by (x, pdf) points.
    The sampler draws *continuous* samples (floats), no step behavior at knots.

    Args:
        pdf_points: iterable of [x, pdf(x)] pairs. x must be strictly increasing.

    Returns:
        sampler(n) -> float (ignores its arg; kept for your call site)
    """
    pts = sorted(pdf_points, key=lambda p: p[0])
    x = np.asarray([p[0] for p in pts], dtype=float)
    y = np.asarray([p[1] for p in pts], dtype=float)

    if np.any(np.diff(x) <= 0):
        raise ValueError("x values must be strictly increasing")

    # Clip tiny negatives from numerical noise (optional)
    y = np.clip(y, 0.0, None)

    dx = np.diff(x)
    # Trapezoid area per segment
    seg_area = 0.5 * (y[:-1] + y[1:]) * dx
    total_area = seg_area.sum()
    if not np.isfinite(total_area) or total_area <= 0:
        raise ValueError("Total area must be positive; check your pdf points")

    # CDF at segment edges (length = len(x))
    cdf_edges = np.concatenate([[0.0], np.cumsum(seg_area)]) / total_area

    def sample(_: int = 0) -> float:
        u = np.random.random()  # in [0,1)
        # find segment k such that cdf_edges[k] <= u < cdf_edges[k+1]
        k = np.searchsorted(cdf_edges, u, side="right") - 1
        k = int(np.clip(k, 0, len(dx) - 1))

        x0, x1 = x[k], x[k + 1]
        y0, y1 = y[k], y[k + 1]
        width = x1 - x0
        m = (y1 - y0) / width  # slope of pdf on this segment

        # area we need to accumulate inside this segment (in absolute area units)
        A_target = (u - cdf_edges[k]) * total_area

        # Invert A(t) = y0*t + 0.5*m*t^2 for t in [0, width]
        if abs(m) < 1e-14:
            # pdf ~ constant across the segment
            if y0 <= 0:
                # degenerate flat-zero segment: just return the left edge
                t = 0.0
            else:
                t = A_target / y0
        else:
            disc = y0 * y0 + 2.0 * m * A_target
            # numerical guard
            if disc < 0:
                disc = 0.0
            t = (-y0 + np.sqrt(disc)) / m

        # clamp to segment bounds (guards rare floating error)
        if t < 0.0:
            t = 0.0
        elif t > width:
            t = width

        return x0 + t

    return sample


@register_trace(
    "Common",
    input_tokens=15671,
    output_tokens=885,
    input_dist=create_pdf_sampler(
        [
            [1, 0.01],
            [1290, 0.04],
            [4282, 0.20],
            [7809, 0.25],
            [14439, 0.25],
            [27238, 0.20],
            [75074, 0.05],
            [117301, 0],
        ]
    ),
    output_dist=create_pdf_sampler(
        [
            [1, 0.01],
            [7, 0.04],
            [30, 0.20],
            [283, 0.25],
            [753, 0.25],
            [1568, 0.25],
            [4557, 0.05],
            [29418, 0],
        ]
    ),
)
@register_trace(
    "Normal4096:128",
    input_tokens=4096,
    output_tokens=128,
    input_dist=lambda x: np.random.normal(x, x / 2),
    output_dist=lambda x: np.random.normal(x, x / 2),
)
@register_trace(
    "Normal4096:512",
    input_tokens=4096,
    output_tokens=128,
    input_dist=lambda x: np.random.normal(x, x / 2),
    output_dist=lambda x: np.random.normal(x, x / 2),
)
@register_trace(
    "Normal512:512",
    input_tokens=512,
    output_tokens=512,
    input_dist=lambda x: np.random.normal(x, x / 2),
    output_dist=lambda x: np.random.normal(x, x / 2),
)
@register_trace(
    "Normal1024:1024",
    input_tokens=512,
    output_tokens=512,
    input_dist=lambda x: np.random.normal(x, x / 2),
    output_dist=lambda x: np.random.normal(x, x / 2),
)
@register_trace(
    "Normal1024:4096",
    input_tokens=1024,
    output_tokens=4096,
    input_dist=lambda x: np.random.normal(x, x / 2),
    output_dist=lambda x: np.random.normal(x, x / 2),
)
class SynthethicTrace(Trace):
    def __init__(
        self,
        input_tokens: int,
        output_tokens: int,
        input_dist: Callable[[int], int],
        output_dist: Callable[[int], int],
        name: str,
    ):
        super().__init__(name)
        self.trace_length = 100000
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.input_dist = input_dist
        self.output_dist = output_dist
        self.reset()

    def __iter__(self):
        for _ in range(self.trace_length):
            req = Request(
                id=self._id,
                input_tokens=round(max(32, self.input_dist(self.input_tokens))),
                output_tokens=round(max(32, self.output_dist(self.output_tokens))),
            )
            super().__iter__()
            yield req

    def __len__(self):
        return self.trace_length

    def average_input_output_tokens(self) -> tuple[int, int]:
        return int(self.input_tokens), int(self.output_tokens)


def plot_trace_token_distribution(trace: Trace, length=10000, show_title=True):
    """
    Plot the distribution of input and output tokens from a trace.

    Args:
        trace: Trace object to sample from
        length: Number of requests to sample (default: 10000)

    Returns:
        Figure with two subplots showing input and output token distributions
    """
    trace.reset()

    # Sample requests from the trace
    requests = list(islice(trace, length))

    if not requests:
        raise ValueError("No requests to plot.")

    input_tokens = [req.input_tokens for req in requests]
    output_tokens = [req.output_tokens for req in requests]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot input token distribution
    ax1.hist(input_tokens, bins=50, color="tab:blue", alpha=0.7, edgecolor="black")
    ax1.set_xlabel("Input Tokens")
    ax1.set_ylabel("Frequency")
    if show_title:
        ax1.set_title(f"Input Token Distribution\n(n={len(input_tokens)})")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # Add statistics text for input tokens
    input_stats = (
        f"Mean: {np.mean(input_tokens):.1f}\n"
        f"Median: {np.median(input_tokens):.1f}\n"
        f"Std: {np.std(input_tokens):.1f}"
    )
    ax1.text(
        0.98,
        0.97,
        input_stats,
        transform=ax1.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Plot output token distribution
    ax2.hist(output_tokens, bins=50, color="tab:orange", alpha=0.7, edgecolor="black")
    ax2.set_xlabel("Output Tokens")
    ax2.set_ylabel("Frequency")
    if show_title:
        ax2.set_title(f"Output Token Distribution\n(n={len(output_tokens)})")
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    # Add statistics text for output tokens
    output_stats = (
        f"Mean: {np.mean(output_tokens):.1f}\n"
        f"Median: {np.median(output_tokens):.1f}\n"
        f"Std: {np.std(output_tokens):.1f}"
    )
    ax2.text(
        0.98,
        0.97,
        output_stats,
        transform=ax2.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    if show_title:
        plt.suptitle(f"Token Distribution - {trace.name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    trace.reset()
    return fig


def plot_request_timeline(
    requests: Sequence,  # sequence of Request objects
    *,
    sort_by: str = "id",  # or "id", "ttft", "ttc"
    annotate: bool = False,  # add small text labels on bars
    figsize: tuple = (12, 0.5),  # width, height-per-request (will scale)
    ax: Optional[plt.Axes] = None,  # type: ignore
    title: Optional[str] = "Request Timeline",
    ylabels: bool = True,  # show request IDs on y-axis
    grid: bool = True,  # show vertical gridlines
):
    """
    Plot a per-request timeline with three segments:
      - QUEUE:  enqueue_time -> prefill_start_time (red)
      - PREFILL: prefill_start_time -> prefill_finish_time (blue)
      - DECODE: prefill_finish_time -> dequeue_time (orange; skipped if output_tokens == 0)
    """
    if not requests:
        raise ValueError("No requests to plot.")

    reqs = list(requests)

    # Sort
    key_map = {
        "enqueue_time": lambda r: r.enqueue_time,
        "id": lambda r: r.id,
        "ttft": lambda r: getattr(r, "ttft"),
        "ttc": lambda r: getattr(r, "ttc"),
    }
    if sort_by not in key_map:
        raise ValueError(f"sort_by must be one of {list(key_map)}, got {sort_by!r}")
    reqs.sort(key=key_map[sort_by])

    # Check consistency
    for r in reqs:
        if r.prefill_start_time < r.enqueue_time:
            raise ValueError(f"Request {r.id}: prefill_start_time < enqueue_time.")
        if r.prefill_finish_time < r.prefill_start_time:
            raise ValueError(
                f"Request {r.id}: prefill_finish_time < prefill_start_time."
            )
        if r.dequeue_time < r.prefill_finish_time:
            raise ValueError(f"Request {r.id}: dequeue_time < prefill_finish_time.")

    # Build axes
    height = max(2.0, len(reqs) * figsize[1])
    if ax is None:
        fig, ax = plt.subplots(figsize=(figsize[0], height))

    y_positions = list(range(len(reqs)))
    bar_height = 0.8

    t_min = min(r.enqueue_time for r in reqs)
    t_max = max(r.dequeue_time for r in reqs)
    pad = 0.02 * (t_max - t_min if t_max > t_min else 1.0)
    ax.set_xlim(t_min - pad, t_max + pad)

    queue_handles, prefill_handles, decode_handles = [], [], []

    for row, r in enumerate(reqs):
        y0 = y_positions[row] - bar_height / 2

        # Queue: red
        queue_dur = r.prefill_start_time - r.enqueue_time
        if queue_dur > 0:
            coll_queue = ax.broken_barh(
                [(r.enqueue_time, queue_dur)], (y0, bar_height), facecolors="tab:gray"
            )
            queue_handles.append(coll_queue)
            if annotate:
                ax.text(
                    r.enqueue_time + queue_dur / 2,
                    y_positions[row],
                    f"{queue_dur:.2f}s",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

        # Prefill: blue
        prefill_dur = r.prefill_finish_time - r.prefill_start_time
        if prefill_dur > 0:
            coll_prefill = ax.broken_barh(
                [(r.prefill_start_time, prefill_dur)],
                (y0, bar_height),
                facecolors="tab:blue",
            )
            prefill_handles.append(coll_prefill)
            if annotate:
                ax.text(
                    r.prefill_start_time + prefill_dur / 2,
                    y_positions[row],
                    f"{prefill_dur:.2f}s",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

        # Add token count annotation (input:output) on the left side of each bar
        ax.text(
            r.enqueue_time - pad * 0.5,
            y_positions[row],
            f"{r.input_tokens}:{r.output_tokens}",
            ha="right",
            va="center",
            fontsize=7,
            color="black",
        )

        # Decode: orange
        decode_dur = r.dequeue_time - r.prefill_finish_time
        if r.output_tokens > 0 and decode_dur > 0:
            coll_dec = ax.broken_barh(
                [(r.prefill_finish_time, decode_dur)],
                (y0, bar_height),
                facecolors="tab:orange",
            )
            decode_handles.append(coll_dec)
            if annotate:
                txt = f"{decode_dur:.2f}s, {r.output_tokens} tok"
                ax.text(
                    r.prefill_finish_time + decode_dur / 2,
                    y_positions[row],
                    txt,
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    # Y ticks / labels
    ax.set_ylim(-1, len(reqs))
    if ylabels:
        ax.set_yticks(y_positions)
        ax.set_yticklabels([str(r.id) for r in reqs])
        ax.set_ylabel("Request ID")
    else:
        ax.set_yticks([])

    ax.set_xlabel("Time (s)")
    if title:
        ax.set_title(title)

    if grid:
        ax.grid(axis="x", linestyle="--", linewidth=0.5)

    # Legend
    handles, labels = [], []
    if queue_handles:
        handles.append(queue_handles[0])
        labels.append("Queue")
    if prefill_handles:
        handles.append(prefill_handles[0])
        labels.append("Prefill")
    if decode_handles:
        handles.append(decode_handles[0])
        labels.append("Decode")
    if handles:
        ax.legend(handles=handles, labels=labels, loc="upper right")

    return ax
