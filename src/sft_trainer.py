import sys
import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
import torch
import random
from trl import SFTConfig, SFTTrainer
from transformers import AutoProcessor, TrainerCallback
from accelerate import Accelerator
from model.lora_setup import apply_lora_for_llava 
from src.utils import prepare_minicap_for_sft

class SFTVisualizerCallback(TrainerCallback):
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
            
            full_text = item.get("text", "")
            images = item.get("images", [])
            
            if not images:
                return
                
            image = images[0]

            if "ASSISTANT:" in full_text:
                prompt_text = full_text.split("ASSISTANT:")[0] + "ASSISTANT:"
                ground_truth = full_text.split("ASSISTANT:")[1].strip()
            else:
                prompt_text = full_text[:100]
                ground_truth = "N/A"
                
            inputs = self.processor(text=prompt_text, images=image, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=256)
                generated_ids_trimmed = generated_ids[:, inputs['input_ids'].shape[1]:]
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
    """Train Llava với SFT"""
    
    # Check GPU
    if torch.cuda.is_available():
        torch.set_default_dtype(torch.float16)
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_dir)
    if hasattr(processor, 'tokenizer') and processor.tokenizer:
        processor.tokenizer.padding_side = "right"
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # Load model
    peft_model = apply_lora_for_llava(model_dir, use_4bit=True)
    peft_model.config.use_cache = False
    
    # Prepare dataset
    sft_dataset = prepare_minicap_for_sft(train_data, max_samples=1000)
    
    # Training config
    training_args = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        max_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        bf16=False,
        remove_unused_columns=False,
        report_to="none",
        dataloader_pin_memory=False,
        logging_steps=10,
        save_steps=100,
    )
    
    # Callback
    visualizer = SFTVisualizerCallback(processor, sft_dataset, sample_every=50)

    trainer = SFTTrainer(
        model=peft_model,
        processing_class=processor,
        args=training_args,
        train_dataset=sft_dataset,
        callbacks=[visualizer],
    )

    print("--- Bắt đầu huấn luyện SFT cho Llava-7B ---")
    trainer.train()
    
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print(f"✅ Đã lưu model tại {output_dir}")