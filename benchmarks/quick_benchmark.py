
import csv
import time
import statistics
from pathlib import Path
from datetime import datetime

import httpx

# Configuration
URL = "http://127.0.0.1:8000/api/chat"
MODEL = "gemma4:12b"

WARMUP = 2
REPETITIONS = 5
TIMEOUT = 300

PROMPTS = {
    "simple": "Hi! Give me a short greeting.",
    "general": "Explain how Python dictionaries work.",
    "long": "Explain transformer architecture in detail, including attention mechanisms."
}


def percentile(values, p):
    """Nearest-rank percentile."""
    values = sorted(values)
    index = max(0, int(len(values) * p + 0.999999) - 1)
    return values[index]


def run_request(client, scenario, prompt):
    payload = {
        "message": prompt,
        "model": MODEL
    }

    start = time.perf_counter()
    first_chunk = None
    response_parts = []

    with client.stream(
        "POST",
        URL,
        json=payload,
        timeout=TIMEOUT
    ) as response:

        response.raise_for_status()
        session_id = response.headers.get("X-Session-ID")

        for chunk in response.iter_bytes():
            if chunk:
                if first_chunk is None:
                    first_chunk = time.perf_counter()

                response_parts.append(chunk)

    end = time.perf_counter()

    return {
        "scenario": scenario,
        "model": MODEL,
        "session_id": session_id,
        "first_chunk_ms": (
            round((first_chunk - start) * 1000, 2)
            if first_chunk else None
        ),
        "total_latency_ms": round((end - start) * 1000, 2),
        "response_length": sum(len(x) for x in response_parts),
        "timestamp": datetime.now().isoformat(timespec="seconds")
    }


def summarize(name, results):
    first_chunks = [
        r["first_chunk_ms"] for r in results
        if r["first_chunk_ms"] is not None
    ]

    latencies = [r["total_latency_ms"] for r in results]

    print(f"\n{name.upper()} ({len(results)} requests)")
    print("-" * 42)

    if first_chunks:
        print(f"First chunk median : {statistics.median(first_chunks):.2f} ms")
        print(f"First chunk p95    : {percentile(first_chunks, 0.95):.2f} ms")

    print(f"Total median       : {statistics.median(latencies):.2f} ms")
    print(f"Total mean         : {statistics.mean(latencies):.2f} ms")
    print(f"Total p95          : {percentile(latencies, 0.95):.2f} ms")
    print(f"Total stdev        : {statistics.stdev(latencies):.2f} ms")


def main():
    all_results = []

    print(f"Model: {MODEL}")
    print(f"Warm-ups: {WARMUP} | Measured requests: {REPETITIONS}")
    print("=" * 50)

    with httpx.Client() as client:

        for scenario, prompt in PROMPTS.items():
            print(f"\nTesting: {scenario}")

            # Warm-up requests are excluded from measurements.
            for _ in range(WARMUP):
                run_request(client, scenario, prompt)

            scenario_results = []

            for i in range(REPETITIONS):
                result = run_request(client, scenario, prompt)

                scenario_results.append(result)
                all_results.append(result)

                print(
                    f"{i + 1}/{REPETITIONS} | "
                    f"First chunk: {result['first_chunk_ms']} ms | "
                    f"Total: {result['total_latency_ms']} ms"
                )

            summarize(scenario, scenario_results)

    output = Path("benchmarks/quick_benchmark_results.csv")

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=all_results[0].keys()
        )
        writer.writeheader()
        writer.writerows(all_results)

    print("\n" + "=" * 50)
    print("OVERALL RESULTS")

    summarize("all scenarios", all_results)

    print(f"\nRaw results saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
