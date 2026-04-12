import sys
import os
import torch
from transformers import (
    AutoProcessor, 
    LlavaForConditionalGeneration,
    BitsAndBytesConfig, 
    AutoConfig
)

class LlavaGPTQQuantizer:
    def __init__(self, base_model_path, save_path, data_path):
        self.base_model_path = base_model_path
        self.save_path = save_path
        self.data_path = data_path

    def quantize_and_save(self, bits=4):
        # Sử dụng BitsAndBytes thay vì GPTQ để tránh lỗi biên dịch thư viện
        print(f"--- Đang cấu hình nén 4-bit (NF4) cho Llava-7B... ---")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",          # Kỹ thuật nén 4-bit chất lượng cao
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,     # Tiết kiệm RAM hơn nữa
        )

        config = AutoConfig.from_pretrained(self.base_model_path)
        config.use_cache = False 

        try:
            print(f"--- Đang nạp và nén model... (Quá trình này tốn khoảng 5-10 phút) ---")
            # Nạp model với cấu hình nén
            model = LlavaForConditionalGeneration.from_pretrained(
                self.base_model_path,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )

            # Lưu cấu hình và trọng số đã nén
            os.makedirs(self.save_path, exist_ok=True)
            model.save_pretrained(self.save_path)
            
            # Lưu kèm Processor (cần thiết cho Llava)
            processor = AutoProcessor.from_pretrained(self.base_model_path)
            processor.save_pretrained(self.save_path)
            
            print(f"--- [SUCCESS] Đã lưu model nén tại: {self.save_path} ---")
            
        except Exception as e:
            print(f"--- Lỗi nghiêm trọng: {e} ---")
            import traceback
            traceback.print_exc()
            sys.exit(1)