import argparse
import os
import yaml

from apu_profiler.core.profiler import UnifiedProfiler
from apu_profiler.backends.torch_backend import TorchBackend
from apu_profiler.backends.onnx_backend import ONNXBackend
from apu_profiler.models.smolvlm import SmolVLMAdapter
from apu_profiler.models.vggt import VGGTAdapter
from apu_profiler.devices.device_manager import DeviceManager
from apu_profiler.utils.io import write_csv, write_json
from apu_profiler.utils.plotting import plot_latency, plot_throughput


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_adapter(model_name, model_id, device, dtype):
    if model_name.lower() == 'smolvlm':
        return SmolVLMAdapter(model_id=model_id, device=device, dtype=dtype)
    elif model_name.lower() == 'vggt':
        return VGGTAdapter(model_id=model_id, device=device, dtype=dtype)
    raise ValueError(f'Unsupported model {model_name}')


def main():
    parser = argparse.ArgumentParser(description='APU Profiler benchmark runner')
    parser.add_argument('--model', choices=['smolvlm', 'vggt'], default='smolvlm')
    parser.add_argument('--backend', choices=['torch', 'onnx'], default='torch')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'igpu'], default='cpu')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--output', type=str, default='outputs')
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--num-runs', type=int, default=3)
    args = parser.parse_args()

    config_path = args.config or f'configs/{args.model}.yaml'
    config = load_config(config_path)

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, 'csv'), exist_ok=True)
    os.makedirs(os.path.join(args.output, 'traces'), exist_ok=True)
    os.makedirs(os.path.join(args.output, 'plots'), exist_ok=True)

    device_name = DeviceManager.get_device(args.device)
    dtype = None
    if 'precision' in config:
        if config['precision'] == 'bfloat16':
            dtype = None  # torch dtype mapping optional; keep model defaults

    model_id = config.get('model_id', '')
    adapter = create_adapter(config['model'], model_id, device_name, dtype)

    backend = None
    if args.backend == 'torch':
        backend = TorchBackend(device=device_name)
    elif args.backend == 'onnx':
        onnx_path = config.get('onnx_path')
        if not onnx_path or not os.path.exists(onnx_path):
            raise FileNotFoundError('ONNX model path is required for onnx backend')
        backend = ONNXBackend(onnx_path)

    prof = UnifiedProfiler(adapter, backend, device=device_name)

    raw_input = {}
    if args.model == 'smolvlm':
        from transformers.image_utils import load_image

        image1 = load_image('https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg')
        image2 = load_image('https://huggingface.co/spaces/merve/chameleon-7b/resolve/main/bee.jpg')

        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image'},
                    {'type': 'image'},
                    {'type': 'text', 'text': 'Describe the images briefly.'},
                ],
            }
        ]

        prompt = adapter.processor.apply_chat_template(messages, add_generation_prompt=True)
        raw_input = {
            'text': prompt,
            'images': [image1, image2],
        }
    else:
        raw_input = ['vggt/examples/kitchen/images/00.png', 'vggt/examples/kitchen/images/01.png']


    result = prof.run(raw_input=raw_input, max_new_tokens=args.max_new_tokens)
    result['model_id'] = model_id
    result['backend'] = args.backend
    result['config'] = config_path

    csv_path = os.path.join(args.output, 'csv', f'{args.model}_{args.backend}_{args.device}.csv')
    json_path = os.path.join(args.output, 'traces', f'{args.model}_{args.backend}_{args.device}.json')

    write_csv(csv_path, [result], fieldnames=list(result.keys()))
    write_json(json_path, result)
    plot_latency([result], os.path.join(args.output, 'plots', f'{args.model}_{args.device}_latency.png'))
    plot_throughput([result], os.path.join(args.output, 'plots', f'{args.model}_{args.device}_throughput.png'))

    print('Benchmark done:', result)


if __name__ == '__main__':
    main()
