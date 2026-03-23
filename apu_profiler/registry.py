MODEL_REGISTRY = {}
BACKEND_REGISTRY = {}

def register_model(name):
    def wrapper(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return wrapper


def register_backend(name):
    def wrapper(cls):
        BACKEND_REGISTRY[name] = cls
        return cls
    return wrapper
