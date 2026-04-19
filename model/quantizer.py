import sys
import os
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from optimum.gptq import GPTQQuantizer
from data.dataset_loader import ScienceQALocalLoader 

class LlavaGPTQQuantizer:
    def __init__(self, base_model_path, save_path, data_path):
        self.base_model_path = base_model_path
        self.save_path = save_path
        self.data_path = data_path

    def quantize_and_save(self, bits=3):
        print(f"--- 🛠️ Đang cấu hình nén {bits}-bit (GPTQ) cho Llava-7B... ---")
        
        try:
            # Chuẩn bị calibration dataset
            loader = ScienceQALocalLoader(self.data_path, subset_size=128)
            df_samples = loader.preprocess_for_r3_quant()
            
            # Tạo calibration texts
            calibration_dataset = [
                f"Question: {row['question']} Answer: {row['reasoning']}" 
                for _, row in df_samples.iterrows()
            ]
            
            # Khởi tạo GPTQ quantizer
            quantizer = GPTQQuantizer(
                bits=bits,
                dataset=calibration_dataset,
                model_seqlen=2048,
                block_size=128,
            )
            
            print(f"--- ⏳ Đang nạp và nén model ({bits}-bit)... ---")
            
            # Load model
            model = LlavaForConditionalGeneration.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                low_cpu_mem_usage=True
            )
            
            if not hasattr(model.config, 'use_cache'):
                print("🔧 Adding use_cache attribute to config...")
                model.config.use_cache = False
                
            # Quantize model
            quantized_model = quantizer.quantize_model(model)
            
            # Save quantized model
            os.makedirs(self.save_path, exist_ok=True)
            quantized_model.save_pretrained(self.save_path)
            
            # Save processor
            processor = AutoProcessor.from_pretrained(self.base_model_path)
            processor.save_pretrained(self.save_path)
            
            print(f"--- ✅ [SUCCESS] Đã lưu model GPTQ {bits}-bit tại: {self.save_path} ---")
            
        except Exception as e:
            print(f"--- ❌ Lỗi nghiêm trọng: {e} ---")
            import traceback
            traceback.print_exc()
            sys.exit(1)