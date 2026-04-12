import torch
import pandas as pd
import io
import re
import os
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset, Dataset
import json
from datetime import datetime

class LlavaDeepEvaluator:
    def __init__(self, model_path, data_path, num_samples=500, mode="fp16", lora_path=None):
        self.model_path = model_path
        self.data_path = data_path
        self.num_samples = num_samples
        self.choices_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
        self.lora_path = lora_path
        
        print(f"\n🛠️ Đang khởi tạo model: {model_path} ({mode})")
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        # Cấu hình load dựa trên mode
        if mode == "4bit":
            # Load với quantization config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        else:
            # Load model gốc (FP16)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        
        # Load LoRA adapter nếu có
        if lora_path and os.path.exists(lora_path):
            print(f"🔧 Đang load LoRA adapter từ: {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.model = self.model.merge_and_unload()  # Merge LoRA vào base model
        
        self.model.eval()
        
        # Disable cache cho evaluation
        self.model.config.use_cache = True

    def load_test_data(self):
        df = pd.read_parquet(self.data_path)
        # Lọc các sample có image
        df_with_image = df[df['image'].notnull()]
        if len(df_with_image) < self.num_samples:
            print(f"⚠️ Chỉ có {len(df_with_image)} samples có image, lấy toàn bộ")
            return df_with_image
        return df_with_image.head(self.num_samples)

    @staticmethod
    def extract_answer(text):
        # Ưu tiên XML tag cho GRPO/SFT
        match = re.search(r'<answer>\s*([A-E])\s*</answer>', text, re.IGNORECASE)
        if match: 
            return match.group(1).upper()
        
        # Tìm trong <think> tags
        match = re.search(r'<think>.*?<answer>\s*([A-E])\s*</answer>', text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # Fallback: Tìm chữ cái A-E đứng riêng
        match = re.search(r'\b([A-E])\b', text)
        if match:
            return match.group(1)
        
        # Fallback cuối: Tìm sau dấu hai chấm hoặc "is"
        fallback = re.findall(r'(?:answer is|is|:)\s*([A-E])', text, re.IGNORECASE)
        return fallback[-1].upper() if fallback else ""

    def evaluate_basic(self):
        """Đánh giá cơ bản chỉ lấy accuracy"""
        df = self.load_test_data()
        correct = 0
        total = len(df)
        
        print(f"\n📊 Bắt đầu đánh giá trên {total} samples...")
        
        for _, row in tqdm(df.iterrows(), total=total, desc="Evaluating"):
            try:
                # Xử lý image
                if isinstance(row['image'], dict) and 'bytes' in row['image']:
                    img = Image.open(io.BytesIO(row['image']['bytes'])).convert("RGB")
                else:
                    continue
                
                # Format choices
                choices = row['choices']
                if isinstance(choices, list):
                    choices_str = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
                else:
                    choices_str = str(choices)
                
                prompt = (
                    f"USER: <tr>\n"
                    f"Question: {row['question']}\n"
                    f"Choices:\n{choices_str}\n"
                    "Think step by step and provide the final answer letter inside <answer> tags.\n"
                    "ASSISTANT:"
                )
                
                inputs = self.processor(
                    text=prompt, 
                    images=img, 
                    return_tensors="pt"
                ).to(self.model.device)
                
                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs, 
                        max_new_tokens=512, 
                        do_sample=False,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id
                    )
                
                gen_text = self.processor.batch_decode(
                    output_ids[:, inputs.input_ids.shape[1]:], 
                    skip_special_tokens=True
                )[0]
                
                pred_answer = self.extract_answer(gen_text)
                true_answer = self.choices_map[int(row['answer'])]
                
                if pred_answer == true_answer:
                    correct += 1
                    
            except Exception as e:
                continue
        
        accuracy = (correct / total) * 100
        print(f"\n✅ Accuracy: {accuracy:.2f}% ({correct}/{total})")
        return accuracy

    def evaluate_with_logging(self, log_file="model_reasoning.txt", print_samples=5, save_json=True):
        """
        Đánh giá và log chi tiết những gì model nghĩ ra
        
        Args:
            log_file: File để lưu output dạng text
            print_samples: Số sample in ra màn hình
            save_json: Có lưu file JSON chi tiết không
        """
        df = self.load_test_data()
        correct = 0
        total = len(df)
        wrong_samples = []
        all_results = []
        
        # Tạo tên file với timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if save_json:
            json_file = log_file.replace('.txt', f'_{timestamp}.json')
        
        print(f"\n📊 Bắt đầu đánh giá trên {total} samples...")
        print(f"📝 Sẽ lưu log vào: {log_file}")
        if save_json:
            print(f"📊 Sẽ lưu JSON vào: {json_file}")
        
        # Mở file log
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write("SCIENCE QA EVALUATION - MODEL REASONING LOG\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"LoRA: {self.lora_path if self.lora_path else 'None'}\n")
            f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*100 + "\n\n")
            
            for idx, row in tqdm(df.iterrows(), total=total, desc="Evaluating"):
                try:
                    # Xử lý image
                    if isinstance(row['image'], dict) and 'bytes' in row['image']:
                        img = Image.open(io.BytesIO(row['image']['bytes'])).convert("RGB")
                    else:
                        continue
                    
                    # Format choices
                    choices = row['choices']
                    if isinstance(choices, list):
                        choices_str = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
                    else:
                        choices_str = str(choices)
                    
                    prompt = (
                        f"USER: <tr>\n"
                        f"Question: {row['question']}\n"
                        f"Choices:\n{choices_str}\n"
                        "Think step by step and provide the final answer letter inside <answer> tags.\n"
                        "ASSISTANT:"
                    )
                    
                    inputs = self.processor(
                        text=prompt, 
                        images=img, 
                        return_tensors="pt"
                    ).to(self.model.device)
                    
                    with torch.no_grad():
                        output_ids = self.model.generate(
                            **inputs, 
                            max_new_tokens=512, 
                            do_sample=False,
                            pad_token_id=self.processor.tokenizer.pad_token_id,
                            eos_token_id=self.processor.tokenizer.eos_token_id
                        )
                    
                    gen_text = self.processor.batch_decode(
                        output_ids[:, inputs.input_ids.shape[1]:], 
                        skip_special_tokens=True
                    )[0]
                    
                    pred_answer = self.extract_answer(gen_text)
                    true_answer = self.choices_map[int(row['answer'])]
                    is_correct = pred_answer == true_answer
                    
                    if is_correct:
                        correct += 1
                    else:
                        wrong_samples.append({
                            'id': idx,
                            'question': row['question'],
                            'predicted': pred_answer,
                            'true': true_answer,
                            'model_output': gen_text
                        })
                    
                    # Lưu kết quả
                    result = {
                        'id': idx,
                        'question': row['question'],
                        'choices': choices_str,
                        'model_output': gen_text,
                        'predicted_answer': pred_answer,
                        'true_answer': true_answer,
                        'is_correct': is_correct
                    }
                    all_results.append(result)
                    
                    # Ghi vào file text
                    f.write(f"\n{'='*100}\n")
                    f.write(f"SAMPLE #{idx + 1}\n")
                    f.write(f"{'='*100}\n")
                    f.write(f"QUESTION: {row['question']}\n\n")
                    f.write(f"CHOICES:\n{choices_str}\n\n")
                    f.write(f"MODEL REASONING:\n{'-'*50}\n{gen_text}\n{'-'*50}\n\n")
                    f.write(f"PREDICTED: {pred_answer} | GROUND TRUTH: {true_answer}\n")
                    f.write(f"STATUS: {'✅ CORRECT' if is_correct else '❌ WRONG'}\n")
                    
                    # In ra màn hình cho sample đầu
                    if idx < print_samples:
                        print("\n" + "="*80)
                        print(f"SAMPLE #{idx + 1} - {'✅ CORRECT' if is_correct else '❌ WRONG'}")
                        print("="*80)
                        print(f"QUESTION: {row['question'][:150]}...")
                        print(f"\n🤖 WHAT THE MODEL THINKS:")
                        print("-"*50)
                        print(gen_text)
                        print("-"*50)
                        print(f"🎯 PREDICTED: {pred_answer} | TRUE: {true_answer}")
                        
                except Exception as e:
                    print(f"\n⚠️ Lỗi ở sample {idx}: {e}")
                    f.write(f"\n⚠️ ERROR at sample {idx}: {e}\n")
                    continue
            
            # Summary
            accuracy = (correct / total) * 100
            f.write(f"\n{'='*100}\n")
            f.write(f"SUMMARY\n")
            f.write(f"{'='*100}\n")
            f.write(f"Total samples: {total}\n")
            f.write(f"Correct: {correct}\n")
            f.write(f"Wrong: {total - correct}\n")
            f.write(f"Accuracy: {accuracy:.2f}%\n")
            
            # Ghi các câu sai
            if wrong_samples:
                f.write(f"\n{'='*100}\n")
                f.write(f"WRONG SAMPLES ({len(wrong_samples)} samples)\n")
                f.write(f"{'='*100}\n")
                for ws in wrong_samples[:20]:  # Chỉ lấy 20 sample đầu
                    f.write(f"\nID {ws['id']}: Pred={ws['predicted']} | True={ws['true']}\n")
                    f.write(f"Q: {ws['question'][:100]}...\n")
        
        # Lưu file JSON
        if save_json:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'model_path': self.model_path,
                    'lora_path': self.lora_path,
                    'total_samples': total,
                    'correct': correct,
                    'accuracy': accuracy,
                    'results': all_results,
                    'wrong_samples': wrong_samples
                }, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*80}")
        print(f"✅ FINAL ACCURACY: {accuracy:.2f}% ({correct}/{total})")
        print(f"📝 Detailed reasoning saved to: {log_file}")
        if save_json:
            print(f"📊 JSON results saved to: {json_file}")
        print(f"{'='*80}\n")
        
        return accuracy, all_results, wrong_samples

    def evaluate_interactive(self, num_samples=5):
        """Đánh giá tương tác - cho phép xem từng sample"""
        df = self.load_test_data().head(num_samples)
        correct = 0
        
        print("\n" + "🎯"*40)
        print("INTERACTIVE EVALUATION MODE")
        print("🎯"*40)
        
        for idx, row in df.iterrows():
            print("\n" + "🔍"*40)
            print(f"Sample {idx + 1}/{len(df)}")
            print("🔍"*40)
            
            print(f"\n📖 QUESTION: {row['question']}")
            print(f"\n📋 CHOICES:")
            if isinstance(row['choices'], list):
                for i, choice in enumerate(row['choices']):
                    print(f"   {chr(65+i)}. {choice}")
            
            input("\n⏸️  Press Enter để xem câu trả lời của model...")
            
            # Inference
            img = Image.open(io.BytesIO(row['image']['bytes'])).convert("RGB")
            choices_str = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(row['choices'])])
            
            prompt = (
                f"USER: <tr>\n"
                f"Question: {row['question']}\n"
                f"Choices:\n{choices_str}\n"
                "Think step by step and provide the final answer letter inside <answer> tags.\n"
                "ASSISTANT:"
            )
            
            inputs = self.processor(text=prompt, images=img, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
            
            gen_text = self.processor.batch_decode(output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            
            print("\n🤖 MODEL'S ANSWER:")
            print("-"*40)
            print(gen_text)
            print("-"*40)
            
            pred_answer = self.extract_answer(gen_text)
            true_answer = self.choices_map[int(row['answer'])]
            
            print(f"\n📊 PREDICTED: {pred_answer} | CORRECT: {true_answer}")
            
            if pred_answer == true_answer:
                print("✅ CORRECT!")
                correct += 1
            else:
                print("❌ WRONG!")
            
            if idx < len(df) - 1:
                input("\n⏩ Press Enter để tiếp tục...")
        
        print(f"\n🎯 ACCURACY: {(correct/len(df))*100:.2f}% ({correct}/{len(df)})")

# --- SCRIPT CHẠY SO SÁNH CÁC GIAI ĐOẠN ---
def run_comparison():
    DATA_PATH = "./data/science_qa/test-00000-of-00001-f0e719df791966ff.parquet"
    
    # Download test dataset nếu chưa có
    if not os.path.exists(DATA_PATH):
        print(f"📥 Đang tải dataset ScienceQA test... (khoảng 100MB)")
        os.makedirs("./data/science_qa", exist_ok=True)
        dataset = load_dataset("derek-thomas/ScienceQA", split="test", cache_dir="./data/cache")
        dataset.to_parquet(DATA_PATH)
        print(f"✅ Đã lưu dataset tại: {DATA_PATH}")
    else:
        print(f"✅ Đã tìm thấy dataset tại: {DATA_PATH}")
    
    # Định nghĩa các checkpoints cần đánh giá
    checkpoints = [
        {
            "name": "1. GỐC (Chưa nén)", 
            "path": "./weights/llava-1.5-7b-hf", 
            "mode": "fp16",
            "lora": None
        },
        {
            "name": "2. NÉN (Baseline 4-bit)", 
            "path": "./weights/llava-1.5-7b-hf", 
            "mode": "4bit",
            "lora": None
        },
        {
            "name": "3. SAU SFT", 
            "path": "./weights/llava-1.5-7b-hf", 
            "mode": "4bit",
            "lora": "./sft_baseline_checkpoints"
        },
        {
            "name": "4. SAU GRPO (Final)", 
            "path": "./weights/llava-1.5-7b-hf", 
            "mode": "4bit",
            "lora": "./r3_quant_checkpoints"
        }
    ]
    
    final_results = {}
    all_details = {}

    for ckpt in checkpoints:
        # Kiểm tra model path
        if not os.path.exists(ckpt["path"]):
            print(f"⚠️ Bỏ qua {ckpt['name']} vì không tìm thấy {ckpt['path']}")
            continue
        
        # Kiểm tra LoRA path nếu có
        if ckpt["lora"] and not os.path.exists(ckpt["lora"]):
            print(f"⚠️ Bỏ qua {ckpt['name']} vì không tìm thấy LoRA: {ckpt['lora']}")
            continue
        
        print(f"\n{'='*60}")
        print(f"🎯 Đánh giá: {ckpt['name']}")
        print(f"{'='*60}")
        
        try:
            evaluator = LlavaDeepEvaluator(
                ckpt["path"], 
                DATA_PATH, 
                num_samples=500, 
                mode=ckpt["mode"],
                lora_path=ckpt["lora"]
            )
            
            # Chọn mode đánh giá
            # Mode 1: Chỉ lấy accuracy
            # acc = evaluator.evaluate_basic()
            
            # Mode 2: Đánh giá với logging chi tiết
            log_filename = f"results_{ckpt['name'].replace(' ', '_').replace('.', '')}.txt"
            acc, results, wrong = evaluator.evaluate_with_logging(
                log_file=log_filename, 
                print_samples=500,  # In 500 sample đầu
                save_json=True
            )
            
            final_results[ckpt["name"]] = acc
            all_details[ckpt["name"]] = {
                'accuracy': acc,
                'total_correct': len(results) - len(wrong),
                'total_samples': len(results),
                'wrong_samples': wrong[:10]  # Lưu 10 sample sai đầu
            }
            
        except Exception as e:
            print(f"❌ Lỗi khi đánh giá {ckpt['name']}: {e}")
            import traceback
            traceback.print_exc()
            final_results[ckpt["name"]] = 0.0
        
        # Dọn dẹp RAM
        if 'evaluator' in locals():
            del evaluator
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    # In kết quả
    print("\n" + "="*60)
    print(f"{'🏆 KẾT QUẢ ĐÁNH GIÁ':^60}")
    print("="*60)
    print(f"{'GIAI ĐOẠN':<35} | {'ACCURACY':<10} | {'CORRECT/TOTAL':<15}")
    print("-" * 60)
    for stage, score in final_results.items():
        if stage in all_details:
            correct = all_details[stage]['total_correct']
            total = all_details[stage]['total_samples']
            print(f"{stage:<35} | {score:>8.2f}% | {correct:>3}/{total:<3}")
        else:
            print(f"{stage:<35} | {score:>8.2f}% | {'N/A':<15}")
    print("="*60)
    
    # Tìm model tốt nhất
    if final_results:
        best_model = max(final_results, key=final_results.get)
        best_acc = final_results[best_model]
        print(f"\n🏆 Model tốt nhất: {best_model} với {best_acc:.2f}%")
    
    # Lưu kết quả tổng hợp
    summary_file = "evaluation_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': final_results,
            'details': all_details
        }, f, indent=2, ensure_ascii=False)
    print(f"\n📊 Đã lưu kết quả tổng hợp vào: {summary_file}")
    
    return final_results, all_details

if __name__ == "__main__":
    results, details = run_comparison()
    
    # Hoặc đánh giá một model cụ thể với interactive mode
    # evaluator = LlavaDeepEvaluator(
    #     model_path="./weights/llava-1.5-7b-hf",
    #     data_path="./data/science_qa/test-00000-of-00001-f0e719df791966ff.parquet",
    #     num_samples=10,
    #     mode="4bit",
    #     lora_path="./r3_quant_checkpoints"
    # )
    # evaluator.evaluate_interactive(num_samples=5)  # Interactive mode
    # evaluator.evaluate_with_logging(log_file="my_model_outputs.txt", print_samples=5)  # Logging mode