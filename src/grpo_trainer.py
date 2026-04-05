import torch
import sys
import os
import io
from PIL import Image
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoProcessor
from datasets import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.lora_setup import apply_lora_to_quantized_model, load_existing_lora_for_quantized_model
from src.rewards import format_reward_func, accuracy_reward_func
from src.utils import prepare_scienceqa_for_grpo 

def logging_reward_func(prompts, completions, **kwargs):
    ground_truths = kwargs.get('answer', kwargs.get('ground_truth', ['N/A'] * len(prompts)))
    print("\n" + "="*60)
    print("🔍 [DEBUG TRACE] BATCH HIỆN TẠI")
    print("="*60)
    if len(prompts) > 0:
        print(f"🔹 [INPUT PROMPT]:\n{prompts[0]}\n")
        print(f"🔸 [MODEL OUTPUT (Generation 1/{len(completions)})]:\n{completions[0]}\n")
        print(f"✅ [GROUND TRUTH]:\n{ground_truths[0]}\n")
    print("="*60 + "\n")
    return [0.0] * len(prompts)

def train_r3_quant_grpo(model_dir: str, train_data, output_dir: str, sft_lora_dir: str = None):

    processor = AutoProcessor.from_pretrained(model_dir)
    
    if sft_lora_dir and os.path.exists(sft_lora_dir):
        print(f"Đang tải Adapter từ SFT: {sft_lora_dir}")
        peft_model = load_existing_lora_for_quantized_model(model_dir, sft_lora_dir)
    else:
        print("Không tìm thấy Adapter SFT, khởi tạo Adapter mới.")
        peft_model = apply_lora_to_quantized_model(model_dir)
        
    grpo_dataset = prepare_scienceqa_for_grpo(train_data)

    def decode_and_sanitize_data(batch):
        new_batch = {}
        for k, v in batch.items():
            if k not in ["prompt", "images"]: 
                new_batch[k] = v
        
        new_batch["prompt"] = []
        for prompts in batch.get('prompt', []):
            new_prompts = []
            for message in prompts:
                new_msg = {"role": message["role"]}
                if isinstance(message.get("content"), list):
                    new_content = []
                    for content in message["content"]:
                        if content.get('type') == 'text':
                            cleaned_text = content['text']
                            for tag in ['<image>', '<|image_pad|>', '<|vision_start|>', '<|vision_end|>']:
                                cleaned_text = cleaned_text.replace(tag, '')
                            new_content.append({"type": "text", "text": cleaned_text.strip()})
                        elif content.get('type') == 'image':
                            new_content.append({"type": "image"})
                        else:
                            new_content.append(content)
                    new_msg["content"] = new_content
                else:
                    new_msg["content"] = message.get("content")
                new_prompts.append(new_msg)
            new_batch["prompt"].append(new_prompts)
            
        if "images" in batch:
            new_batch["images"] = []
            for img_list in batch["images"]:
                new_img_list = []
                for img_data in img_list:
                    if isinstance(img_data, dict):
                        if img_data.get('bytes'):
                            new_img_list.append(Image.open(io.BytesIO(img_data['bytes'])))
                        elif img_data.get('path'):
                            new_img_list.append(Image.open(img_data['path']))
                    elif img_data is not None:
                        new_img_list.append(img_data)
                new_batch["images"].append(new_img_list)
                
        return new_batch

    grpo_dataset.set_transform(decode_and_sanitize_data)

    training_args = GRPOConfig(
        output_dir=output_dir,
        
        learning_rate=1e-5,                  
        optim="adamw_8bit",                   
        lr_scheduler_type="cosine",
        
        logging_steps=1,           
        max_steps=500,
        
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,       
        gradient_checkpointing=True, 
        
        num_generations=4,                    
        temperature=1.0,                      
        max_completion_length=4096,          
        
        epsilon=0.2,                        
        epsilon_high=0.28,                   
        beta=0.04,                            
        
        bf16=True,                   
        remove_unused_columns=False, 
        report_to="none"             
    )

    reward_funcs = [
        format_reward_func,           
        accuracy_reward_func,
        logging_reward_func           
    ]

    trainer = GRPOTrainer(
        model=peft_model,
        processing_class=processor,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=grpo_dataset,
    )

    trainer.train()
    
    print(f"\nĐang lưu mô hình LoRA tại: {output_dir}")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)