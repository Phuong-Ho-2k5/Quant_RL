import torch
import pandas as pd
import io
import re
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

class LlavaEvaluator:
    def __init__(self, model_path, data_path, num_samples=50):
        self.model_path = model_path
        self.data_path = data_path
        self.num_samples = num_samples
        self.choices_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

        print(f"📦 Đang tải Llava-7B (Nén) từ: {model_path}...")
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        # Nạp model (đã qua GPTQ)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        self.model.eval()

    def load_test_data(self):
        df = pd.read_parquet(self.data_path)
        mask = df['image'].notnull()
        df = df[mask].head(self.num_samples)
        return df

    @staticmethod
    def extract_answer(text):
        """Trích xuất chữ cái đáp án từ định dạng <answer>X</answer> hoặc text tự do."""
        # Ưu tiên tìm trong thẻ XML (kết quả sau GRPO)
        match = re.search(r'<answer>\s*([A-E])\s*</answer>', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # Fallback cho model chưa học format (SFT baseline hoặc base model)
        text = text.strip().upper()
        if len(text) > 0 and text[0] in "ABCDE":
            return text[0]
        return ""

    def evaluate(self):
        df = self.load_test_data()
        correct_count = 0
        total = len(df)

        print(f"\n🚀 Bắt đầu đánh giá Llava-7B trên {total} mẫu ScienceQA...")
        
        for idx, row in tqdm(df.iterrows(), total=total):
            ans_idx = int(row['answer'])
            target_letter = self.choices_map[ans_idx]
            
            # Xử lý ảnh
            image_bytes = row['image']['bytes']
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Format Prompt chuẩn Llava-1.5
            # Sau khi qua SFT/GRPO, model sẽ nhạy với yêu cầu về thẻ XML
            prompt_text = (
                f"USER: <image>\nQuestion: {row['question']}\nChoices: {row['choices']}\n"
                "Think step by step and provide the final answer letter inside <answer> tags.\nASSISTANT:"
            )

            # Khác với Qwen, Llava dùng processor trực tiếp cho cả text và image
            inputs = self.processor(text=prompt_text, images=img, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=256)
                
            # Cắt bỏ phần prompt đầu vào khỏi output
            generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
            pred_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]

            pred_letter = self.extract_answer(pred_text)
            score = 1.0 if pred_letter == target_letter else 0.0
            correct_count += score

            if idx == df.index[0]:
                print(f"\n[Ví dụ Output Thực Tế]")
                print(f"Target: {target_letter}")
                print(f"Model Gen: {pred_text}")
                print(f"Extracted: {pred_letter}")
                print(f"Chấm điểm: {'ĐÚNG' if score else 'SAI'}\n")

        accuracy = (correct_count / total) * 100
        print("\n" + "="*50)
        print(f"🏆 ĐỘ CHÍNH XÁC LLAVA-7B: {accuracy:.2f}%")
        print("="*50)

if __name__ == "__main__":
    # Đường dẫn tới checkpoint sau khi bạn đã Quantize/SFT/GRPO
    MODEL_PATH = r"./weights/llava-1.5-7b-gptq-int4" 
    DATA_PATH = r"./data/science_qa/test-00000-of-00001-f0e719df791966ff.parquet"
    
    evaluator = LlavaEvaluator(MODEL_PATH, DATA_PATH, num_samples=50)
    evaluator.evaluate()