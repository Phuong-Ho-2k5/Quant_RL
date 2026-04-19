import sys
import os
import torch
from transformers import (
    AutoProcessor, 
    LlavaForConditionalGeneration,
    AutoConfig,
    GPTQConfig
)
from data.dataset_loader import ScienceQALocalLoader 

class LlavaGPTQQuantizer:
    def __init__(self, base_model_path, save_path, data_path):
        self.base_model_path = base_model_path
        self.save_path = save_path
        self.data_path = data_path

    def quantize_and_save(self, bits=3):
        print(f"--- 🛠️ Đang cấu hình nén {bits}-bit (GPTQ) cho Llava-7B... ---")
        
        loader = ScienceQALocalLoader(self.data_path, subset_size=128)
        df_samples = loader.preprocess_for_r3_quant()

        calibration_dataset = [
            f"Question: {row['question']} Answer: {row['reasoning']}" 
            for _, row in df_samples.iterrows()
        ]

        quantization_config = GPTQConfig(
            bits=bits,
            dataset=calibration_dataset,
            tokenizer=self.base_model_path,
            desc_act=False,
            sym=True,
        )

        try:
            print(f"--- ⏳ Đang nạp và nén model... ---")
            model = LlavaForConditionalGeneration.from_pretrained(
                self.base_model_path,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )

            os.makedirs(self.save_path, exist_ok=True)
            model.save_pretrained(self.save_path)
            
            processor = AutoProcessor.from_pretrained(self.base_model_path)
            processor.save_pretrained(self.save_path)
            
            print(f"--- ✅ [SUCCESS] Đã lưu model GPTQ {bits}-bit thành công! ---")
            
        except Exception as e:
            print(f"--- ❌ Lỗi nghiêm trọng: {e} ---")
            sys.exit(1)