
import json
import time
from pathlib import Path
from threading import Lock

METRICS_FILE = Path("benchmark_results.jsonl")
_file_lock = Lock()


def save_benchmark(data):
    """Append one benchmark record to a JSONL file."""
    with _file_lock:
        with METRICS_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(data, default=str) + "\n")


def elapsed_ms(start, end=None):
    end = end if end is not None else time.perf_counter()
    return round((end - start) * 1000, 2)
