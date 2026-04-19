import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["DISABLE_WANDB"] = "1"

import torch
import sys
import io
from PIL import Image
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoProcessor, LlavaForConditionalGeneration
from model.lora_setup import apply_lora_for_llava, load_existing_lora_for_quantized_model
from rewards import format_reward_func, accuracy_reward_func
from utils import prepare_scienceqa_for_grpo 

def train_llava_grpo(model_dir: str, train_data, output_dir: str, sft_lora_dir: str = None):
    """Train Llava với GRPO"""
    
    # Kiểm tra GPU
    if not torch.cuda.is_available():
        print("⚠️ Warning: GPU không khả dụng, training sẽ chậm!")
        device = "cpu"
    else:
        device = "cuda"
        torch.set_default_dtype(torch.float16)
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_dir)
    if hasattr(processor, 'tokenizer') and processor.tokenizer:
        processor.tokenizer.padding_side = "left"
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # Load model
    if sft_lora_dir and os.path.exists(sft_lora_dir):
        peft_model = load_existing_lora_for_quantized_model(model_dir, sft_lora_dir)
    else:
        peft_model = apply_lora_for_llava(model_dir, use_4bit=True)
    
    peft_model.config.use_cache = False
    
    # Prepare dataset
    grpo_dataset = prepare_scienceqa_for_grpo(train_data)
    
    # Training config
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=5e-6,
        optim="adamw_torch",
        max_steps=1,  # Tăng lên từ 1
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        num_generations=4,
        max_completion_length=256,
        fp16=torch.cuda.is_available(),
        report_to="none",
        use_vllm=False,
        logging_steps=10,
        save_steps=50,
    )
    
    trainer = GRPOTrainer(
        model=peft_model,
        processing_class=processor,
        reward_funcs=[format_reward_func, accuracy_reward_func],
        args=training_args,
        train_dataset=grpo_dataset,
    )
    
    print("--- Bắt đầu huấn luyện GRPO cho Llava-7B ---")
    trainer.train()
    trainer.save_model(output_dir)
    print(f"✅ Đã lưu model tại {output_dir}")