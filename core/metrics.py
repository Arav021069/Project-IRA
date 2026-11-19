import time
import json
from pathlib import Path


class RequestMetrics:
    def __init__(self):
        self.started_at = time.perf_counter()
        self.first_token_at = None
        self.finished_at = None

        self.model_calls = []
        self.tool_calls = []

    def record_first_token(self):
        if self.first_token_at is None:
            self.first_token_at = time.perf_counter()

    def record_model_call(self, data):
        self.model_calls.append(data)

    def record_tool_call(self, name, duration):
        self.tool_calls.append({
            "name": name,
            "duration_ms": round(duration * 1000, 2)
        })

    def finish(self):
        self.finished_at = time.perf_counter()

    def report(self):
        end = self.finished_at or time.perf_counter()

        return {
            "total_latency_ms": round(
                (end - self.started_at) * 1000, 2
            ),
            "end_to_end_ttft_ms": (
                round(
                    (self.first_token_at - self.started_at) * 1000,
                    2
                )
                if self.first_token_at else None
            ),
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls
        }


def save_metrics(metrics, path="benchmark_results.jsonl"):
    path = Path(path)

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(metrics) + "\n")
