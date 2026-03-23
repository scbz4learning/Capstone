import onnxruntime as ort
from .base import BackendBase


class ONNXBackend(BackendBase):
    name = 'onnxruntime'

    def __init__(self, model_path, provider='CPUExecutionProvider'):
        self.session = ort.InferenceSession(model_path, providers=[provider])

    def forward(self, model_adapter, inputs):
        ort_inputs = {k: v.cpu().numpy() if hasattr(v, 'cpu') else v for k, v in inputs.items()}
        return self.session.run(None, ort_inputs)

    def generate(self, model_adapter, inputs, **kwargs):
        raise NotImplementedError('ONNX generate not supported directly')

    def profile(self, fn, output_dir):
        # ONNX profiling can use built-in profile option in session, here placeholder
        fn()
        return None
