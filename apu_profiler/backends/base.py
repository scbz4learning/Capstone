class BackendBase:
    name = 'base'

    def forward(self, model_adapter, inputs):
        raise NotImplementedError()

    def generate(self, model_adapter, inputs, **kwargs):
        raise NotImplementedError()

    def profile(self, fn, output_dir):
        raise NotImplementedError()
