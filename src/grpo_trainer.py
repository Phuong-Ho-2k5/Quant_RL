import os
# Chặn đứng wandb ngay từ đầu để tránh lỗi protobuf telemetry
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["DISABLE_WANDB"] = "1"

import torch
import sys
import io
from PIL import Image
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoProcessor
from model.lora_setup import apply_lora_for_llava, load_existing_lora_for_quantized_model
from src.rewards import format_reward_func, accuracy_reward_func
from src.utils import prepare_scienceqa_for_grpo 
from datasets import Dataset

def train_llava_grpo(model_dir: str, train_data, output_dir: str, sft_lora_dir: str = None):
    """Train Llava với GRPO"""
    
    # 1. Cấu hình thiết bị và kiểu dữ liệu
    if not torch.cuda.is_available():
        print("⚠️ Warning: GPU không khả dụng, training sẽ chậm!")
    else:
        torch.set_default_dtype(torch.float16)
    
    # 2. Load processor chuẩn Llava
    processor = AutoProcessor.from_pretrained(model_dir)
    if hasattr(processor, 'tokenizer') and processor.tokenizer:
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # 3. Load model
    if sft_lora_dir and os.path.exists(sft_lora_dir):
        print(f"🔄 Đang nạp Adapter từ SFT: {sft_lora_dir}")
        peft_model = load_existing_lora_for_quantized_model(model_dir, sft_lora_dir)
    else:
        print("🆕 Khởi tạo Adapter LoRA mới cho GRPO.")
        peft_model = apply_lora_for_llava(model_dir)

    peft_model.config.use_cache = False
    
    print("🔄 Đang chuẩn bị dataset cho GRPO...")
    
    grpo_data = []
    answer_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
    
    for idx, example in enumerate(train_data):
        try:
            # Lấy câu hỏi
            question = example.get('question', '')
            if not question:
                question = example.get('prompt', '')
            
            # Lấy choices
            choices = example.get('choices', [])
            if not choices:
                choices = example.get('options', ['A', 'B', 'C', 'D'])
            
            # Format choices
            choices_str = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
            
            # Tạo prompt - YÊU CẦU THẺ THINK
            prompt = (
                f"USER: </td>\nQuestion: {question}\nChoices:\n{choices_str}\n\n"
                f"Please reason step by step. Put your reasoning inside <think> tags "
                f"and the final answer letter inside <answer> tags.\n"
                f"ASSISTANT: <think>"
            )
            
            # Lấy ground truth
            answer = example.get('answer', 0)
            if isinstance(answer, int):
                answer_letter = answer_map.get(answer, "A")
            else:
                answer_letter = str(answer).upper()
                if len(answer_letter) > 1:
                    answer_letter = answer_letter[0]
            
            ground_truth = f"<answer>{answer_letter}</answer>"
            
            grpo_data.append({
                'prompt': prompt,
                'ground_truth': ground_truth
            })
            
            # Debug: in sample đầu tiên
            if idx == 0:
                print(f"   Sample 0 prompt: {prompt[:200]}...")
                print(f"   Sample 0 ground_truth: {ground_truth}")
                
        except Exception as e:
            print(f"⚠️ Lỗi ở sample {idx}: {e}")
            continue
    
    # Tạo HuggingFace Dataset
    grpo_dataset = Dataset.from_list(grpo_data)
    
    print(f"✅ Dataset ready: {len(grpo_dataset)} samples")
    print(f"   Columns: {grpo_dataset.column_names}")
    
    # 5. Cấu hình GRPO tối ưu cho 2xT4 Kaggle
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=5e-6,
        optim="adamw_8bit",
        max_steps=200,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        
        num_generations=2,
        max_completion_length=256,
        beta=0.04,
        dataloader_pin_memory=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
        use_vllm=False,
        logging_steps=1,
        save_steps=50,
    )
    
    # 6. Khởi tạo Trainer
    trainer = GRPOTrainer(
        model=peft_model,
        processing_class=processor,
        reward_funcs=[format_reward_func, accuracy_reward_func],
        args=training_args,
        train_dataset=grpo_dataset,
    )
    
    print("--- 🚀 Bắt đầu huấn luyện GRPO cho Llava-7B ---")
    trainer.train()
    
    # 7. Lưu kết quả
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print(f"✅ Đã lưu model thành công tại {output_dir}")