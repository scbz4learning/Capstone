class ModelAdapterBase:
    name = 'base'

    def preprocess(self, raw_input):
        raise NotImplementedError()

    def forward(self, inputs):
        raise NotImplementedError()

    def generate(self, inputs, **kwargs):
        raise NotImplementedError()
