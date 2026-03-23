from setuptools import setup, find_packages

setup(
    name='apu_profiler',
    version='0.1.0',
    description='Unified APU profiling toolkit',
    author='Auto Generated',
    packages=find_packages(exclude=['tests', 'benchmark', 'docs']),
    install_requires=[
        # ROCm-specific torch from ROCm-nightlies; no plain torch wheel on pip for ROCm.
        'torch==2.12.0a0+rocm7.13.0a20260317',
        'numpy==1.26.4',
        'pandas==3.0.1',
        'matplotlib==3.10.8',
        'transformers==5.3.0',
        # ONNX Runtime ROCm variant
        'onnxruntime-rocm==1.22.2.post1',
        'pyyaml==6.0.3',
    ],
    python_requires='>=3.8',
)
