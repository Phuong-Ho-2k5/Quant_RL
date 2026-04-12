import torch
import sys
import os
import random
from trl import SFTConfig, SFTTrainer
from transformers import AutoProcessor, TrainerCallback, LlavaForConditionalGeneration
from datasets import load_dataset
from peft import PeftModel
from accelerate import Accelerator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Đảm bảo dùng hàm apply_lora dành cho Llava (target_modules: q_proj, v_proj,...)
from model.lora_setup import apply_lora_for_llava 
from src.utils import prepare_minicap_for_sft

class SFTVisualizerCallback(TrainerCallback):
    """
    Callback in ra kết quả thực tế để theo dõi khả năng học format <think><answer>
    """
    def __init__(self, processor, dataset, sample_every=20):
        self.processor = processor
        self.dataset = dataset
        self.sample_every = sample_every

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % self.sample_every == 0 and state.global_step > 0:
            model = kwargs['model']
            model.eval()
            
            idx = random.randint(0, len(self.dataset) - 1)
            item = self.dataset[idx]
            
            # Llava format: USER: <image>\n{prompt} ASSISTANT:
            # Lấy tin nhắn từ dataset đã qua xử lý ở utils.py
            messages = item["messages"]
            ground_truth = messages[2]["content"][0]["text"]
            
            # Chuẩn bị input cho inference
            # prompt_text thường có dạng "USER: <image>\n{Question}\nASSISTANT:"
            prompt_text = f"USER: <image>\n{messages[1]['content'][1]['text']}\nASSISTANT:"
            image = item["images"][0]
            
            inputs = self.processor(text=prompt_text, images=image, return_tensors="pt").to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=256)
                # Cắt phần prompt khỏi output
                generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True
                )[0]

            print("\n" + "🚀" * 15)
            print(f"📊 [SFT DEBUG] STEP: {state.global_step}")
            print(f"✅ [GT]: {ground_truth[:150]}...")
            print(f"🤖 [MODEL]: {output_text[:150]}...")
            print("🚀" * 15 + "\n")
            
            model.train()

def train_llava_sft(model_dir: str, train_data, output_dir: str):
    # 1. Processor & Config
    processor = AutoProcessor.from_pretrained(model_dir)
    # Llava 1.5 thường dùng size 336x336, processor sẽ tự resize
    
    # 2. Apply LoRA cho quantized model (Llava-7B)
    peft_model = apply_lora_for_llava(model_dir)

    # 3. Chuẩn bị dataset (phải map sang format USER/ASSISTANT của Llava)
    sft_dataset = prepare_minicap_for_sft(train_data) 

    training_args = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        # max_seq_length=1024,
        learning_rate=2e-5,          
        lr_scheduler_type="cosine",
        logging_steps=5,           
        max_steps=500, 
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=8,
        gradient_checkpointing=True, 
        fp16=True,   
        bf16=False,
        tf32=False, 
        dataloader_pin_memory=False,             
        remove_unused_columns=False, 
        report_to="none",
    )

    training_args.fp16_backend = "automatic" 
    visualizer = SFTVisualizerCallback(processor, sft_dataset, sample_every=25)
    accelerator = Accelerator(
        fp16 = True,
        mixed_precision = "fp16",
        gradient_accumulation_steps = training_args.gradient_accumulation_steps
    )
    trainer = SFTTrainer(
        model=peft_model,
        processing_class=processor,
        args=training_args,
        # max_seq_length=1024,
        train_dataset=sft_dataset,
        callbacks=[visualizer],
        accelerator=accelerator
    )

    print("--- Bắt đầu huấn luyện SFT cho Llava-7B ---")
    trainer.train()
    
    # Lưu Adapter LoRA
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)