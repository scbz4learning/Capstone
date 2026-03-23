import numpy as np


def compute_throughput(tokens: float, latency: float) -> float:
    if latency <= 0:
        return 0.0
    return tokens / latency


def compute_itl(total_time: float, tokens: float) -> float:
    if tokens <= 0:
        return float('inf')
    return total_time / tokens


def summarize_latencies(latencies):
    if len(latencies) == 0:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "mean": 0.0}
    return {
        "p50": float(np.percentile(latencies, 50)),
        "p90": float(np.percentile(latencies, 90)),
        "p99": float(np.percentile(latencies, 99)),
        "mean": float(np.mean(latencies)),
    }
