import statistics
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Any, Callable, Dict


class TimingTracker:
    """A utility class to track timing metrics with automatic averaging."""

    __slots__ = ("timings", "active_timers", "_maxlen", "_deque_factory")

    def __init__(self, maxlen: int = 10):
        """Initialize the timing tracker.

        Args:
            maxlen: Maximum number of timing samples to keep for averaging
        """
        self._maxlen = maxlen
        self._deque_factory: Callable[[], deque] = lambda: deque(maxlen=maxlen)
        self.timings: Dict[str, deque] = defaultdict(self._deque_factory)
        self.active_timers: Dict[str, float] = {}

    @contextmanager
    def time(self, name: str) -> Any:
        """Context manager for timing code blocks.

        Args:
            name: Name of the timing metric

        Usage:
            with timer.time("inference"):
                # code to time
                pass
        """
        start_time = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name].append(time.perf_counter() - start_time)

    def start_timer(self, name: str) -> None:
        """Start a named timer.

        Args:
            name: Name of the timing metric
        """
        self.active_timers[name] = time.perf_counter()

    def stop_timer(self, name: str) -> float:
        """Stop a named timer and record the elapsed time.

        Args:
            name: Name of the timing metric

        Returns:
            Elapsed time in seconds
        """
        if name not in self.active_timers:
            raise ValueError(f"Timer '{name}' was not started")

        elapsed = time.perf_counter() - self.active_timers[name]
        self.timings[name].append(elapsed)
        del self.active_timers[name]
        return elapsed

    def add_timing(self, name: str, elapsed: float) -> None:
        """Manually add a timing measurement.

        Args:
            name: Name of the timing metric
            elapsed: Elapsed time in seconds
        """
        self.timings[name].append(elapsed)

    def get_mean(self, name: str) -> float:
        """Get the mean timing for a metric.

        Args:
            name: Name of the timing metric

        Returns:
            Mean timing in seconds
        """
        timing_data = self.timings.get(name)
        if timing_data is None or len(timing_data) == 0:
            return 0.0
        return float(statistics.mean(timing_data))

    def get_latest(self, name: str) -> float:
        """Get the latest timing for a metric.

        Args:
            name: Name of the timing metric

        Returns:
            Latest timing in seconds
        """
        timing_data = self.timings.get(name)
        if timing_data is None or len(timing_data) == 0:
            return 0.0
        return float(timing_data[-1])

    def get_all_means(self) -> Dict[str, float]:
        """Get mean timings for all tracked metrics.

        Returns:
            Dictionary of metric names to mean timings
        """
        result = {}
        for name, timing_data in self.timings.items():
            if len(timing_data) > 0:
                result[name] = float(statistics.mean(timing_data))
            else:
                result[name] = 0.0
        return result

    def clear(self) -> None:
        """Clear all timing data."""
        self.timings.clear()
        self.active_timers.clear()

    def reset_metric(self, name: str) -> None:
        """Reset timing data for a specific metric.

        Args:
            name: Name of the timing metric to reset
        """
        if name in self.timings:
            self.timings[name].clear()
        if name in self.active_timers:
            del self.active_timers[name]
