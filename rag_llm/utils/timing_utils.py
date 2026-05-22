import time
from contextlib import contextmanager
from typing import Dict, Optional


class TimingRecorder:
    """
    用來記錄程式各階段的執行時間。

    Example:
        timer = TimingRecorder()

        with timer.measure("load_data_seconds"):
            load_data()

        print(timer.get_timings())
    """

    def __init__(self):
        self.timings: Dict[str, float] = {}

    @contextmanager
    def measure(self, key: str):
        start = time.perf_counter()

        try:
            yield

        finally:
            elapsed = time.perf_counter() - start
            self.timings[key] = elapsed

    def add(self, key: str, value: float):
        self.timings[key] = value

    def get(self, key: str, default: Optional[float] = None):
        return self.timings.get(key, default)

    def get_timings(self, round_digits: int = 6) -> Dict[str, float]:
        return {
            key: round(value, round_digits)
            for key, value in self.timings.items()
        }

    def reset(self):
        self.timings.clear()
