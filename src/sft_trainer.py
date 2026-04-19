import torch
import os
from trl import SFTConfig, SFTTrainer
from transformers import AutoProcessor
from model.lora_setup import apply_lora_for_llava 
from src.utils import prepare_minicap_for_sft

def train_llava_sft(model_dir: str, train_data, output_dir: str):
    # Luôn dùng float16 trên T4
    torch.set_default_dtype(torch.float16)
    
    processor = AutoProcessor.from_pretrained(model_dir)
    # Rất quan trọng: Llava cần token image để map features
    if "<image>" not in processor.tokenizer.additional_special_tokens:
        processor.tokenizer.add_tokens(["<image>"], special_tokens=True)
    
    peft_model = apply_lora_for_llava(model_dir)
    peft_model.resize_token_embeddings(len(processor.tokenizer))
    peft_model.config.use_cache = False

    sft_dataset = prepare_minicap_for_sft(train_data) 

    # Định nghĩa Data Collator để gộp ảnh và text đúng cách
    def collate_fn(examples):
        texts = [example["text"] for example in examples]
        images = [example["images"][0] for example in examples]
        
        # Processor sẽ tự động tạo input_ids và pixel_values
        batch = processor(text=texts, images=images, padding=True, return_tensors="pt")
        
        # Tạo labels để tính loss (bản sao của input_ids)
        batch["labels"] = batch["input_ids"].clone()
        return batch

    training_args = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        learning_rate=2e-5,          
        max_steps=1, 
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=8,
        gradient_checkpointing=True, 
        fp16=True,
        remove_unused_columns=False, # Quan trọng: Giữ lại cột 'images'
        report_to="none",
    )
    
    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=sft_dataset,
        data_collator=collate_fn, # Sử dụng collator tùy chỉnh
        processing_class=processor,
    )
    
    print("--- 🚀 Bắt đầu huấn luyện SFT (Fix Image Token) ---")
    trainer.train()
    trainer.save_model(output_dir)