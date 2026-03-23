from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from .base import ModelAdapterBase


class VGGTAdapter(ModelAdapterBase):
    name = 'VGGT'

    def __init__(self, model_id, device='cpu', dtype=None):
        self.model = VGGT.from_pretrained(model_id).to(device)
        if dtype is not None:
            self.model = self.model.to(dtype)

    def preprocess(self, raw_input):
        # raw_input expects list of image paths
        images = load_and_preprocess_images(raw_input)
        return images.to(self.model.device)

    def forward(self, inputs):
        return self.model(inputs)

    def generate(self, inputs, **kwargs):
        # VGGT has no generate; use forward as proxy
        return self.forward(inputs)
