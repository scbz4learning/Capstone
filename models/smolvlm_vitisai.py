"""
SmolVLM-Instruct ONNX inference on Vitis AI (NPU).
Note: Vitis AI EP often requires specific layout (NHWC) and may have limitations
on which layers can be accelerated. This script uses a hybrid approach.
"""

import time
import numpy as np
import onnxruntime
from transformers import AutoConfig, AutoProcessor
from transformers.image_utils import load_image


def main():
    model_dir = r"C:\Users\AMD_Capstone_Bokai\Documents\Capstone\models\SmolVLM-Instruct"
    onnx_dir = f"{model_dir}/onnx"

    # ── 1. Load config and processor ─────────────────────────────────────
    print("Loading config and processor...")
    config = AutoConfig.from_pretrained(model_dir)
    processor = AutoProcessor.from_pretrained(model_dir)

    # ── 2. Load ONNX sessions (Vitis AI NPU - Forced Static) ──────────
    print("Attempting to force NPU (Vitis AI) for Vision Encoder (Static Shape)...")
    
    # Session options
    sess_opts = onnxruntime.SessionOptions()
    sess_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
    
    # Enable Operator Assignment Report to verify NPU offloading
    import os
    os.environ["XLNX_ONNX_EP_REPORT_FILE"] = "vitisai_static_report.json"
    
    # NPU Provider with fixed shape hint if needed (via provider_options)
    vitis_ai_provider = ("VitisAIExecutionProvider", {
        "target": "X2",
        "cacheDir": r"C:\Users\AMD_Capstone_Bokai\Documents\Capstone\models\cache_static",
        "cacheKey": "smolvlm_static",
        "enable_cache_file_io_in_mem": "0"
    })
    
    cpu_provider = "CPUExecutionProvider"

    print("  Initializing Vision Encoder (INT8-STATIC) on NPU...")
    try:
        vision_session = onnxruntime.InferenceSession(
            f"{onnx_dir}/vision_encoder_int8_static.onnx", sess_opts, providers=[vitis_ai_provider, cpu_provider]
        )
    except Exception as e:
        print(f"  NPU Session failed to init: {e}. Falling back to DML.")
        vision_session = onnxruntime.InferenceSession(
            f"{onnx_dir}/vision_encoder_int8_static.onnx", sess_opts, providers=["DmlExecutionProvider", cpu_provider]
        )
    
    print("  Initializing Embed Tokens (INT8) on CPU...")
    embed_session = onnxruntime.InferenceSession(
        f"{onnx_dir}/embed_tokens_int8.onnx", sess_opts, providers=[cpu_provider]
    )

    print("  Initializing Decoder (INT8) on CPU...")
    decoder_session = onnxruntime.InferenceSession(
        f"{onnx_dir}/decoder_model_merged_int8.onnx", sess_opts, providers=[cpu_provider]
    )

    # ── 3. Extract config values ─────────────────────────────────────────
    num_key_value_heads = config.text_config.num_key_value_heads
    head_dim = config.text_config.head_dim
    num_hidden_layers = config.text_config.num_hidden_layers
    eos_token_id = config.text_config.eos_token_id
    image_token_id = config.image_token_id

    # ── 4. Prepare inputs ────────────────────────────────────────────────
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Can you describe this image?"},
            ],
        },
    ]

    print("Downloading test image...")
    image = load_image(
        "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
    )

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="np")

    # ── 5. Prepare decoder state ─────────────────────────────────────────
    batch_size = inputs["input_ids"].shape[0]

    past_key_values = {
        f"past_key_values.{layer}.{kv}": np.zeros(
            [batch_size, num_key_value_heads, 0, head_dim], dtype=np.float32
        )
        for layer in range(num_hidden_layers)
        for kv in ("key", "value")
    }

    image_features = None
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # ── 6. Generation loop ───────────────────────────────────────────────
    max_new_tokens = 50
    generated_tokens = np.array([[]], dtype=np.int64)

    print("\n--- Generating (Vitis AI / Hybrid) ---\n")
    gen_start = time.time()

    for i in range(max_new_tokens):
        # Embed input tokens
        inputs_embeds = embed_session.run(None, {"input_ids": input_ids.astype(np.int64)})[0]

        if image_features is None:
            # Compute vision features
            try:
                image_features = vision_session.run(
                    ["image_features"],
                    {
                        "pixel_values": inputs["pixel_values"].astype(np.float32),
                        "pixel_attention_mask": inputs["pixel_attention_mask"].astype(np.bool_),
                    },
                )[0]
            except Exception as e:
                print(f"\nVision Encoder error: {e}")
                print("Falling back to CPU for Vision...")
                # Fallback to CPU session manually if NPU fails
                vision_session = onnxruntime.InferenceSession(
                    f"{onnx_dir}/vision_encoder.onnx", sess_opts, providers=[cpu_provider]
                )
                image_features = vision_session.run(
                    ["image_features"],
                    {
                        "pixel_values": inputs["pixel_values"].astype(np.float32),
                        "pixel_attention_mask": inputs["pixel_attention_mask"].astype(np.bool_),
                    },
                )[0]

            # Merge text and vision embeddings
            inputs_embeds[inputs["input_ids"] == image_token_id] = (
                image_features.reshape(-1, image_features.shape[-1])
            )

        # Run decoder
        logits, *present_key_values_list = decoder_session.run(
            None,
            dict(
                inputs_embeds=inputs_embeds.astype(np.float32),
                attention_mask=attention_mask.astype(np.int64),
                **past_key_values,
            ),
        )

        input_ids = logits[:, -1].argmax(-1, keepdims=True)
        attention_mask = np.concatenate(
            [attention_mask, np.ones([batch_size, 1], dtype=attention_mask.dtype)],
            axis=-1,
        )

        for j, key in enumerate(past_key_values):
            past_key_values[key] = present_key_values_list[j]

        generated_tokens = np.concatenate([generated_tokens, input_ids], axis=-1)
        if (input_ids == eos_token_id).all():
            break

        print(processor.decode(input_ids[0]), end="", flush=True)

    elapsed = time.time() - gen_start
    print(f"\n\n--- Done in {elapsed:.1f}s ---\n")


if __name__ == "__main__":
    main()
