import torch
import pandas as pd
import io
import re
import os
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

class LlavaDeepEvaluator:
    def __init__(self, model_path, data_path, num_samples=20, mode="fp16"):
        self.model_path = model_path
        self.data_path = data_path
        self.num_samples = num_samples
        self.choices_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
        
        print(f"\n🛠️ Đang khởi tạo model: {model_path} ({mode})")
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        # Cấu hình load dựa trên mode
        load_args = {
            "device_map": "auto",
            "low_cpu_mem_usage": True
        }
        
        if mode == "4bit":
            # Dành cho lúc bạn vừa nén xong bằng BitsAndBytes/GPTQ
            load_args["torch_dtype"] = torch.float16
        else:
            # Dành cho model gốc (FP16/BF16)
            load_args["torch_dtype"] = torch.bfloat16

        self.model = LlavaForConditionalGeneration.from_pretrained(model_path, **load_args)
        self.model.eval()

    def load_test_data(self):
        df = pd.read_parquet(self.data_path)
        return df[df['image'].notnull()].head(self.num_samples)

    @staticmethod
    def extract_answer(text):
        # Ưu tiên XML tag cho GRPO
        match = re.search(r'<answer>\s*([A-E])\s*</answer>', text, re.IGNORECASE)
        if match: return match.group(1).upper()
        
        # Fallback cho model gốc (Tìm chữ cái in hoa đứng sau dấu hai chấm hoặc "is")
        fallback = re.findall(r'(?:answer is|is|:)\s*([A-E])', text, re.IGNORECASE)
        return fallback[-1].upper() if fallback else ""

    def evaluate(self):
        df = self.load_test_data()
        correct = 0
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating"):
            img = Image.open(io.BytesIO(row['image']['bytes'])).convert("RGB")
            prompt = (
                f"USER: <image>\nQuestion: {row['question']}\nChoices: {row['choices']}\n"
                "Think step by step and provide the final answer letter inside <answer> tags.\nASSISTANT:"
            )
            
            inputs = self.processor(text=prompt, images=img, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
            
            gen_text = self.processor.batch_decode(output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            if self.extract_answer(gen_text) == self.choices_map[int(row['answer'])]:
                correct += 1
                
        return (correct / len(df)) * 100

# --- SCRIPT CHẠY SO SÁNH 3 GIAI ĐOẠN ---
def run_comparison():
    DATA_PATH = "./data/science_qa/test-00000-of-00001-f0e719df791966ff.parquet"
    checkpoints = [
        {"name": "1. GỐC (Chưa nén)", "path": "./weights/llava-1.5-7b-hf", "mode": "fp16"},
        {"name": "2. NÉN (Baseline)", "path": "./weights/llava-1.5-7b-hf-GPTQ-Int4", "mode": "4bit"},
        
        {"name": "3. SAU GRPO (Final)", "path": "./r3_quant_checkpoints", "mode": "4bit"}
    ]
    
    final_results = {}

    for ckpt in checkpoints:
        if not os.path.exists(ckpt["path"]) and "llava-hf" not in ckpt["path"]:
            print(f"⚠️ Bỏ qua {ckpt['name']} vì không tìm thấy đường dẫn.")
            continue
            
        evaluator = LlavaDeepEvaluator(ckpt["path"], DATA_PATH, num_samples=20, mode=ckpt["mode"])
        acc = evaluator.evaluate()
        final_results[ckpt["name"]] = acc
        
        # Dọn dẹp RAM cực kỳ quan trọng trên Kaggle
        del evaluator.model
        del evaluator.processor
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    print("\n" + "="*45)
    print(f"{'GIAI ĐOẠN':<25} | {'ACCURACY':<10}")
    print("-" * 45)
    for stage, score in final_results.items():
        print(f"{stage:<25} | {score:>8.2f}%")
    print("="*45)

if __name__ == "__main__":
    run_comparison()