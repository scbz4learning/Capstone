import onnxruntime
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoProcessor, AutoTokenizer
from transformers.image_utils import load_image

# --- 1. 初始化配置与会话 ---
model_id = "HuggingFaceTB/SmolVLM-Instruct"
model_base_path = r"C:\Users\AMD_Capstone_Bokai\Documents\Capstone\models\smolvlm"

print("正在初始化配置与处理器...")
config = AutoConfig.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 强制开启基础优化，避开 LayerNormFusion Bug
sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
providers = ['CPUExecutionProvider'] # ['VitisAIExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']

print("正在加载 ONNX 会话...")
v_ses = onnxruntime.InferenceSession(f"{model_base_path}\\vision_encoder_fp16.onnx", sess_options=sess_options, providers=providers)
e_ses = onnxruntime.InferenceSession(f"{model_base_path}\\embed_tokens_fp16.onnx", sess_options=sess_options, providers=providers)
d_ses = onnxruntime.InferenceSession(f"{model_base_path}\\decoder_model_merged_fp16.onnx", sess_options=sess_options, providers=providers)

# 获取 Decoder 期待的输入类型字典（关键：处理混合精度）
decoder_input_types = {i.name: i.type for i in d_ses.get_inputs()}

# --- 2. 准备输入数据 ---
NUM_PATCHES = 832 
image_tags = "<image>" * NUM_PATCHES
messages = [{"role": "user", "content": [{"type": "text", "text": f"{image_tags}\nCan you describe this image?"}]}]

image = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = tokenizer(text=prompt, return_tensors="pt")
pixel_inputs = processor(images=[image], return_tensors="pt")

def resize_static_inputs(pv, pam):
    b, n, c, h, w = pv.shape
    pv_res = F.interpolate(pv.view(b*n, c, h, w).float(), size=(512, 512), mode='bilinear')
    pam_res = F.interpolate(pam.view(b*n, 1, h, w).float(), size=(512, 512), mode='nearest')
    return pv_res.view(b, n, c, 512, 512).numpy().astype(np.float32), \
           pam_res.view(b, n, 512, 512).numpy().astype(np.bool_)

print("执行维度对齐 (512x512)...")
pixel_values, pixel_attention_mask = resize_static_inputs(pixel_inputs['pixel_values'], pixel_inputs['pixel_attention_mask'])

# 初始化 KV Cache
past_key_values = {
    f'past_key_values.{i}.{kv}': np.zeros([1, config.text_config.num_key_value_heads, 0, config.text_config.head_dim], dtype=np.float32)
    for i in range(config.text_config.num_hidden_layers) for kv in ('key', 'value')
}

image_injected = False
input_ids = inputs['input_ids'].numpy().astype(np.int64)
attention_mask = inputs['attention_mask'].numpy().astype(np.int64)
position_ids = (np.cumsum(attention_mask, axis=-1) - 1).astype(np.int64)

# --- 3. 类型适配辅助函数 ---
def match_types(feeds, expected_types):
    matched_feeds = {}
    for name, value in feeds.items():
        if name not in expected_types:
            matched_feeds[name] = value
            continue
        
        target_type = expected_types[name]
        if "float16" in target_type:
            matched_feeds[name] = value.astype(np.float16)
        elif "float" in target_type: # 对应 inputs_embeds 的 tensor(float)
            matched_feeds[name] = value.astype(np.float32)
        else:
            matched_feeds[name] = value
    return matched_feeds

# --- 4. 生成循环 ---
print("\n--- 模型输出 ---")
for i in range(128):
    # 1. 词向量嵌入
    inputs_embeds = e_ses.run(None, {'input_ids': input_ids})[0]

    # 2. 第一帧：注入视觉特征
    if not image_injected:
        v_out = v_ses.run(None, {'pixel_values': pixel_values, 'pixel_attention_mask': pixel_attention_mask})
        raw_features = v_out[0].reshape(-1, 576) # (832, 576)
        
        # 修复乱码尝试：使用平铺 (Tile) 替代补零 (Padding)
        target_dim = 2048
        repeat_factor = (target_dim // 576) + 1
        full_features = np.tile(raw_features, (1, repeat_factor))[:, :target_dim]
        
        # 注入位置
        mask = (inputs['input_ids'].numpy() == config.image_token_id)
        inputs_embeds = inputs_embeds.astype(np.float32) # 确保 embeds 基础类型
        inputs_embeds[mask] = full_features.astype(np.float32)
        image_injected = True

    # 3. 构造并匹配输入类型
    current_feeds = {
        'inputs_embeds': inputs_embeds,
        'attention_mask': attention_mask,
        'position_ids': position_ids,
        **past_key_values
    }
    final_feeds = match_types(current_feeds, decoder_input_types)

    # 4. Decoder 推理
    outputs = d_ses.run(None, final_feeds)
    logits = outputs[0]
    
    # 5. 更新 KV Cache (直接保存 output 以维持 fp16)
    new_pkv_list = outputs[1:]
    for j, key in enumerate(past_key_values):
        past_key_values[key] = new_pkv_list[j]

    # 6. 生成下一个字符
    next_id = logits[:, -1].argmax(-1, keepdims=True)
    word = tokenizer.decode(next_id[0])
    print(word, end='', flush=True)

    # 7. 准备下一轮输入
    input_ids = next_id.astype(np.int64)
    attention_mask = np.concatenate([attention_mask, [[1]]], axis=1)
    position_ids = np.array([[attention_mask.shape[1] - 1]], dtype=np.int64)

    if next_id[0] == config.text_config.eos_token_id:
        break

print("\n\n--- 运行完成 ---")