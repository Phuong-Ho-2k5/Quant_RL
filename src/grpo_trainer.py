import torch
import sys
import os
import io
from PIL import Image
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoProcessor, LlavaForConditionalGeneration
from datasets import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.lora_setup import apply_lora_for_llava, load_existing_lora_for_quantized_model
from src.rewards import format_reward_func, accuracy_reward_func
from src.utils import prepare_scienceqa_for_grpo 

def logging_reward_func(prompts, completions, **kwargs):
    ground_truths = kwargs.get('answer', kwargs.get('ground_truth', ['N/A'] * len(prompts)))
    print("\n" + "="*60)
    print("🔍 [DEBUG TRACE] BATCH HIỆN TẠI (LLAVA-7B)")
    print("="*60)
    if len(prompts) > 0:
        output = completions[0][0]["content"] if isinstance(completions[0], list) else completions[0]
        print(f"🔹 [INPUT PROMPT]:\n{prompts[0][:200]}...\n")
        print(f"🔸 [MODEL OUTPUT]:\n{output}\n")
        print(f"✅ [GROUND TRUTH]:\n{ground_truths[0]}\n")
    print("="*60 + "\n")
    return [0.0] * len(prompts)

def train_llava_grpo(model_dir: str, train_data, output_dir: str, sft_lora_dir: str = None):
    # 1. Load Processor chuẩn Llava
    processor = AutoProcessor.from_pretrained(model_dir)
    processor.tokenizer.padding_side = "left"

    # 2. Load Model với PEFT
    if sft_lora_dir and os.path.exists(sft_lora_dir):
        print(f"🔄 Đang tải Adapter từ SFT: {sft_lora_dir}")
        peft_model = load_existing_lora_for_quantized_model(model_dir, sft_lora_dir)
    else:
        print("🆕 Khởi tạo Adapter mới cho Llava-7B.")
        peft_model = apply_lora_for_llava(model_dir)

    # 3. Quản lý Token đặc biệt
    if peft_model.generation_config.pad_token_id is None:
        peft_model.generation_config.pad_token_id = processor.tokenizer.pad_token_id

    # 4. Chuẩn bị Dataset (Sử dụng format USER: <image>\n{Q} ASSISTANT:)
    grpo_dataset = prepare_scienceqa_for_grpo(train_data)

    def decode_and_sanitize_data(batch):
        """Hiệu chỉnh format prompt đặc thù cho Llava Next/1.5."""
        new_batch = {"prompt": [], "images": []}
        
        # Xử lý Text Prompt
        for prompts in batch.get('prompt', []):
            new_batch["prompt"].append(prompts)

        if "images" in batch:
            for img_list in batch["images"]:
                if img_list and len(img_list) > 0:
                    img = img_list[0]
                    if isinstance(img, dict) and 'bytes' in img:
                        img = Image.open(io.BytesIO(img['bytes'])).convert("RGB")
                    new_batch["images"].append([img])
                
        # Giữ lại ground_truth cho Reward Func
        if "ground_truth" in batch:
            new_batch["ground_truth"] = batch["ground_truth"]
            
        return new_batch

    grpo_dataset.set_transform(decode_and_sanitize_data)

    # 5. Cấu hình Training (Tối ưu cho 7B Model)
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=5e-6,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        max_steps=500,
        save_steps=50,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        
        num_generations=4,
        temperature=0.9,
        max_completion_length=512,
        beta=0.04,
        fp16=True,
        bf16=False,
        tf32=False,
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = GRPOTrainer(
        model=peft_model,
        processing_class=processor,
        reward_funcs=[format_reward_func, accuracy_reward_func, logging_reward_func],
        args=training_args,
        train_dataset=grpo_dataset,
    )

    trainer.train()
    
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)