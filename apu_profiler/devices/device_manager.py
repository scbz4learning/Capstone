class DeviceManager:
    @staticmethod
    def get_device(name: str):
        name = name.lower()
        if name in ("cpu", "cpu"):
            return "cpu"
        if name in ("igpu", "cuda", "gpu"):
            return "cuda"
        if name in ("npu", "vitisai", "tensorflowlite"):
            return "npu"
        raise ValueError(f"Unsupported device: {name}")
