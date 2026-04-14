import onnx

def fix_to_static(input_file, output_file):
    model = onnx.load(input_file)
    graph = model.graph
    
    # 修改输入形状
    for i in graph.input:
        for dim in i.type.tensor_type.shape.dim:
            if dim.dim_param in ['batch_size', 'num_images', 'sequence_length', 'total_sequence_length', 'past_sequence_length']:
                dim.dim_value = 1
                dim.ClearField('dim_param')
            elif dim.dim_value <= 0: # 处理 -1
                dim.dim_value = 1

    # 修改输出形状（部分模型需要）
    for o in graph.output:
        for dim in o.type.tensor_type.shape.dim:
            if dim.dim_param in ['batch_size', 'num_images', 'sequence_length', 'total_sequence_length', 'past_sequence_length']:
                dim.dim_value = 1
                dim.ClearField('dim_param')
            elif dim.dim_value <= 0:
                dim.dim_value = 1

    onnx.save(model, output_file)
    print(f"Static model saved to {output_file}")

if __name__ == "__main__":
    base_path = r"C:\Users\AMD_Capstone_Bokai\Documents\Capstone\models\SmolVLM-Instruct\onnx"
    # 仅针对 Vision Encoder 尝试固定，因为它在 NPU 上的兼容性相对最高
    fix_to_static(f"{base_path}\\vision_encoder_int8.onnx", f"{base_path}\\vision_encoder_int8_static.onnx")
