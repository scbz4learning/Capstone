import torch
from .base import BackendBase


class TorchBackend(BackendBase):
    name = 'torch'

    def __init__(self, device='cpu'):
        self.device = device

    def forward(self, model_adapter, inputs):
        with torch.inference_mode():
            return model_adapter.model(**inputs)

    def generate(self, model_adapter, inputs, **kwargs):
        with torch.inference_mode():
            return model_adapter.model.generate(**inputs, **kwargs)

    def profile(self, fn, output_dir):
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            fn()
        trace_path = f"{output_dir}/torch_trace.json"
        prof.export_chrome_trace(trace_path)
        return trace_path
