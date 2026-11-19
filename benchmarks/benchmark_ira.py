
import csv
import time
import statistics
from pathlib import Path

import httpx

URL = "http://127.0.0.1:8000/api/chat"

# Change this to a model shown by `ollama list`.
MODEL = "gemma4:12b"

PROMPTS = {
    "simple": "Hi! Give me a short greeting.",
    "general": "Explain how Python dictionaries work.",
    "long": "Explain how transformer architecture works in detail.",
}

WARMUP = 2
REPETITIONS = 30


def run_request(client, prompt, session_id=None):
    payload = {
        "message": prompt,
        "model": MODEL,
        "session_id": session_id
    }

    start = time.perf_counter()
    first_byte = None
    response_chunks = []
    session_id = None

    try:
        with client.stream(
            "POST",
            URL,
            json=payload,
            timeout=300
        ) as response:

            response.raise_for_status()

            # The API creates a fresh session automatically.
            session_id = response.headers.get("X-Session-ID")

            for chunk in response.iter_text():
                if chunk:
                    if first_byte is None:
                        first_byte = time.perf_counter()

                    response_chunks.append(chunk)

    except Exception as e:
        print(f"\nBenchmark request failed: {e}")
        print(f"Received response so far: {''.join(response_chunks)}")
        raise

    end = time.perf_counter()

    return {
        "ttft_ms": (
            round((first_byte - start) * 1000, 2)
            if first_byte else None
        ),
        "total_latency_ms": round(
            (end - start) * 1000, 2
        ),
        "response_length": len("".join(response_chunks)),
        "session_id": session_id
    }



def main():
    results = []

    with httpx.Client() as client:

        for scenario, prompt in PROMPTS.items():
            print(f"\nTesting: {scenario}")

            # Warm-up runs, excluded from results.
            for i in range(WARMUP):
                run_request(client, prompt)

            for i in range(REPETITIONS):
                result = run_request(
                    client,
                    prompt,
                    f"benchmark-{scenario}-{i}"
                )

                result["scenario"] = scenario
                result["iteration"] = i + 1

                results.append(result)

                print(
                    f"{i + 1:02d}/{REPETITIONS} "
                    f"TTFT: {result['ttft_ms']} ms | "
                    f"Total: {result['total_latency_ms']} ms"
                )

    output = Path("benchmark_results.csv")

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "iteration",
                "ttft_ms",
                "total_latency_ms",
                "response_length"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n--- SUMMARY ---")

    for scenario in PROMPTS:
        subset = [
            r for r in results
            if r["scenario"] == scenario
        ]

        ttfts = [r["ttft_ms"] for r in subset if r["ttft_ms"]]
        latencies = [
            r["total_latency_ms"] for r in subset
        ]

        print(f"\n{scenario}")
        print(f"TTFT median: {statistics.median(ttfts):.2f} ms")
        print(
            f"Total median: "
            f"{statistics.median(latencies):.2f} ms"
        )
        print(
            f"Total mean: "
            f"{statistics.mean(latencies):.2f} ms"
        )
        print(
            f"Total stdev: "
            f"{statistics.stdev(latencies):.2f} ms"
        )

    print(f"\nSaved results to {output}")


if __name__ == "__main__":
    main()
