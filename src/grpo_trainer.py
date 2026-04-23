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
    
    # 3. Load model (FIX LỖI: Bỏ tham số use_4bit)
    if sft_lora_dir and os.path.exists(sft_lora_dir):
        print(f"🔄 Đang nạp Adapter từ SFT: {sft_lora_dir}")
        peft_model = load_existing_lora_for_quantized_model(model_dir, sft_lora_dir)
    else:
        print("🆕 Khởi tạo Adapter LoRA mới cho GRPO.")
        peft_model = apply_lora_for_llava(model_dir)

    peft_model.config.use_cache = False
    
    # 4. Hàm chuyển đổi dataset sang conversational format
    def convert_to_conversation(example):
        """Chuyển mỗi example sang định dạng hội thoại"""
        
        # Format choices
        choices_str = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(example['choices'])])
        
        # User prompt với ảnh
        user_content = [
            {"type": "image"},
            {"type": "text", "text": f"Question: {example['question']}\nChoices:\n{choices_str}\nThink step by step and provide the answer in <answer> tag."}
        ]
        
        # Assistant response (lấy đáp án đúng)
        answer_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
        answer = answer_map.get(example['answer'], "A")
        assistant_content = f"<answer>{answer}</answer>"
        
        # Tạo conversation
        conversation = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
        
        return {
            "conversations": conversation,
            "image": example['image']  # Giữ nguyên ảnh
        }
    
    # 5. Chuẩn bị dataset (format USER: <tr>\n{Q} ASSISTANT:)
    grpo_dataset = prepare_scienceqa_for_grpo(train_data)
    
    # 6. Chuyển dataset sang conversational format (FIX LỖI GRPO)
    print("🔄 Đang chuyển dataset sang conversational format...")
    
    # Kiểm tra xem dataset đã có cột 'conversations' chưa
    if "conversations" not in grpo_dataset.column_names:
        # Áp dụng chuyển đổi
        grpo_dataset = grpo_dataset.map(
            convert_to_conversation, 
            remove_columns=grpo_dataset.column_names
        )
        print("✅ Đã chuyển đổi dataset thành công!")
    else:
        print("✅ Dataset đã ở định dạng conversational, bỏ qua bước chuyển đổi.")
    
    # 7. Cấu hình GRPO tối ưu cho 2xT4 Kaggle
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=5e-6,
        optim="adamw_8bit",
        max_steps=10,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        
        num_generations=2,
        max_completion_length=1024,
        beta=0.04,
        dataloader_pin_memory=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
        use_vllm=False,
        logging_steps=1,
        save_steps=50,
    )
    
    # 8. Khởi tạo Trainer
    trainer = GRPOTrainer(
        model=peft_model,
        processing_class=processor,
        reward_funcs=[format_reward_func, accuracy_reward_func],
        args=training_args,
        train_dataset=grpo_dataset,
    )
    
    print("--- 🚀 Bắt đầu huấn luyện GRPO cho Llava-7B ---")
    trainer.train()
    
    # 9. Lưu kết quả
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print(f"✅ Đã lưu model thành công tại {output_dir}")