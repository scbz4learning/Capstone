import time
from .metrics import compute_throughput, compute_itl


class UnifiedProfiler:
    def __init__(self, model_adapter, backend, device='cpu'):
        self.model_adapter = model_adapter
        self.backend = backend
        self.device = device

    def _measure(self, fn, runs=3):
        latencies = []
        for _ in range(runs):
            start = time.time()
            fn()
            end = time.time()
            latencies.append(end - start)
        return {
            'mean_latency_s': sum(latencies) / len(latencies),
            'latencies_s': latencies,
        }

    def run(self, raw_input, max_new_tokens=32):
        inputs = self.model_adapter.preprocess(raw_input)

        # prefill (forward pass)
        prefill = self._measure(lambda: self.backend.forward(self.model_adapter, inputs))

        # ttft (partial generation)
        ttft = self._measure(lambda: self.backend.generate(self.model_adapter, inputs, max_new_tokens=1), runs=1)

        # full generation for ITL and throughput
        token_count = max_new_tokens
        decode = self._measure(lambda: self.backend.generate(self.model_adapter, inputs, max_new_tokens=max_new_tokens), runs=1)

        itl = compute_itl(decode['mean_latency_s'], token_count)
        throughput = compute_throughput(token_count, decode['mean_latency_s'])

        return {
            'device': self.device,
            'model': self.model_adapter.name,
            'backend': self.backend.name,
            'prefill_mean_latency_s': prefill['mean_latency_s'],
            'ttft_latency_s': ttft['mean_latency_s'],
            'decode_latency_s': decode['mean_latency_s'],
            'itl_s_per_token': itl,
            'throughput_tokens_s': throughput,
            'prefill_latencies_s': prefill['latencies_s'],
            'ttft_latencies_s': ttft['latencies_s'],
        }
