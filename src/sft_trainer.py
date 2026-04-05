import torch
import sys
import os
import random
from trl import SFTConfig, SFTTrainer
from transformers import AutoProcessor, TrainerCallback
from datasets import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.lora_setup import apply_lora_to_quantized_model
from src.utils import prepare_minicap_for_sft

class SFTVisualizerCallback(TrainerCallback):
    """
    Callback để in ra Ground Truth và Output của mô hình trong quá trình SFT.
    """
    def __init__(self, processor, dataset, sample_every=10):
        self.processor = processor
        self.dataset = dataset
        self.sample_every = sample_every

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % self.sample_every == 0 and state.global_step > 0:
            model = kwargs['model']
            model.eval()
            
            idx = random.randint(0, len(self.dataset) - 1)
            item = self.dataset[idx]
            
            messages = item["messages"]
            ground_truth = messages[2]["content"][0]["text"]
            
            input_messages = messages[:2] 
            text = self.processor.apply_chat_template(input_messages, tokenize=False, add_generation_prompt=True)
            
            image_inputs = item.get("images", [None])[0]
            
            inputs = self.processor(
                text=[text],
                images=[image_inputs] if image_inputs else None,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=128)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

            print("\n" + "-" * 30)
            print(f"📊 [SFT DEBUG] STEP: {state.global_step}")
            print(f"✅ [GROUND TRUTH]:\n{ground_truth}")
            print(f"🤖 [MODEL OUTPUT]:\n{output_text}")
            print("✨" * 30 + "\n")
            
            model.train()

def train_sft_baseline(model_dir: str, train_data, output_dir: str):
    processor = AutoProcessor.from_pretrained(model_dir)
    
    processor.image_processor.min_pixels = 256 * 28 * 28
    processor.image_processor.max_pixels = 512 * 28 * 28 

    peft_model = apply_lora_to_quantized_model(model_dir)
    sft_dataset = prepare_minicap_for_sft(train_data) 

    training_args = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="messages", 
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=2048,
        learning_rate=2e-5,          
        lr_scheduler_type="cosine",
        logging_steps=1,           
        max_steps=500, 
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=4,
        gradient_checkpointing=True, 
        bf16=True,                   
        remove_unused_columns=False, 
        report_to="none",
    )

    visualizer = SFTVisualizerCallback(processor, sft_dataset, sample_every=10)

    trainer = SFTTrainer(
        model=peft_model,
        processing_class=processor,
        args=training_args,
        train_dataset=sft_dataset,
        callbacks=[visualizer], 
    )

    trainer.train()
    
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)