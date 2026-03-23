import argparse
import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText


def export_smolvlm(model_id, output_path, device='cpu', max_new_tokens=32):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id).to(device)
    model.eval()

    dummy_image = torch.randn(1, 3, 224, 224).to(device)
    # 这里只是示意，真实会拿到模型的输入格式
    dummy_input = processor(text=['hello'], images=[dummy_image], return_tensors='pt').to(device)

    inputs = {
        'pixel_values': dummy_input['pixel_values'].cpu(),
        'input_ids': dummy_input['input_ids'].cpu(),
        'attention_mask': dummy_input['attention_mask'].cpu()
    }

    torch.onnx.export(
        model,
        (inputs['pixel_values'], inputs['input_ids'], inputs['attention_mask']),
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['pixel_values', 'input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'pixel_values': {0: 'batch', 2: 'height', 3: 'width'},
            'input_ids': {0: 'batch', 1: 'seq'},
            'attention_mask': {0: 'batch', 1: 'seq'},
            'logits': {0: 'batch', 1: 'seq'}
        }
    )


def main():
    parser = argparse.ArgumentParser(description='Export model to ONNX')
    parser.add_argument('--model', choices=['smolvlm', 'vggt'], default='smolvlm')
    parser.add_argument('--model-id', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.model == 'smolvlm':
        export_smolvlm(args.model_id, args.output, device=args.device)
    else:
        raise NotImplementedError('VGGT ONNX export is not implemented yet')

    print('Exported ONNX model at', args.output)


if __name__ == '__main__':
    main()
