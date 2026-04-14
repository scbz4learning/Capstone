"""
SmolVLM-Instruct ONNX inference on DirectML (GPU).
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

    # ── 2. Load ONNX sessions (DirectML for Vision, CPU for Decoder) ──────
    print("Loading ONNX models (Vision on DML, Decoder on CPU)...")
    sess_opts = onnxruntime.SessionOptions()
    sess_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Provider lists
    dml_provider = [("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    cpu_provider = ["CPUExecutionProvider"]

    t0 = time.time()
    vision_session = onnxruntime.InferenceSession(
        f"{onnx_dir}/vision_encoder.onnx", sess_opts, providers=dml_provider
    )
    print(f"  vision_encoder loaded in {time.time()-t0:.1f}s")

    t0 = time.time()
    embed_session = onnxruntime.InferenceSession(
        f"{onnx_dir}/embed_tokens.onnx", sess_opts, providers=cpu_provider
    )
    print(f"  embed_tokens loaded in {time.time()-t0:.1f}s")

    t0 = time.time()
    decoder_session = onnxruntime.InferenceSession(
        f"{onnx_dir}/decoder_model_merged.onnx", sess_opts, providers=cpu_provider
    )
    print(f"  decoder_model_merged loaded in {time.time()-t0:.1f}s")

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
    max_new_tokens = 100
    generated_tokens = np.array([[]], dtype=np.int64)

    print("\n--- Generating (DirectML) ---\n")
    gen_start = time.time()

    for i in range(max_new_tokens):
        # Embed input tokens
        inputs_embeds = embed_session.run(None, {"input_ids": input_ids.astype(np.int64)})[0]

        if image_features is None:
            # Compute vision features
            image_features = vision_session.run(
                ["image_features"],
                {
                    "pixel_values": inputs["pixel_values"],
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
