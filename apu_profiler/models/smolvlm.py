from transformers import AutoProcessor, AutoModelForImageTextToText
from .base import ModelAdapterBase


class SmolVLMAdapter(ModelAdapterBase):
    name = 'SmolVLM'

    def __init__(self, model_id, device='cpu', dtype=None):
        self.processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForImageTextToText.from_pretrained(model_id)
        if dtype is not None:
            model = model.to(dtype=dtype)
        self.model = model.to(device)

    def preprocess(self, raw_input):
        # raw_input expects dict with text + images
        return self.processor(**raw_input, return_tensors='pt').to(self.model.device)

    def generate(self, inputs, **kwargs):
        return self.model.generate(**inputs, **kwargs)
