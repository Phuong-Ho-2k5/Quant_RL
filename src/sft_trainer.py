import torch
import os
from trl import SFTConfig, SFTTrainer
from transformers import AutoProcessor
from model.lora_setup import apply_lora_for_llava 
from src.utils import prepare_minicap_for_sft

def train_llava_sft(model_dir: str, train_data, output_dir: str):
    torch.set_default_dtype(torch.float16)
    
    processor = AutoProcessor.from_pretrained(model_dir)
    
    # SỬA LỖI: Sử dụng phương pháp an toàn để thêm token
    image_token = "<image>"
    if image_token not in processor.tokenizer.get_vocab():
        processor.tokenizer.add_tokens([image_token], special_tokens=True)
    
    peft_model = apply_lora_for_llava(model_dir)
    peft_model.resize_token_embeddings(len(processor.tokenizer))
    peft_model.config.use_cache = False

    sft_dataset = prepare_minicap_for_sft(train_data) 

    # Data collator để xử lý đồng thời text và ảnh
    def collate_fn(examples):
        texts = [example["text"] for example in examples]
        images = [example["images"][0] for example in examples]
        
        batch = processor(text=texts, images=images, padding=True, return_tensors="pt")
        batch["labels"] = batch["input_ids"].clone()
        return batch

    training_args = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        learning_rate=2e-5,          
        max_steps=500, 
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=8,
        gradient_checkpointing=True, 
        fp16=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to="none",
    )
    
    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=sft_dataset,
        data_collator=collate_fn,
        processing_class=processor,
    )
    
    print("--- 🚀 Bắt đầu huấn luyện SFT  ---")
    trainer.train()
    trainer.save_model(output_dir)