import sys
import os
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, GPTQConfig, AutoConfig
from data.dataset_loader import ScienceQALocalLoader 

class LlavaGPTQQuantizer:
    def __init__(self, base_model_path, save_path, data_path):
        self.base_model_path = base_model_path
        self.save_path = save_path
        self.data_path = data_path

    def quantize_and_save(self, bits=3):
        print(f"--- 🛠️ Đang cấu hình nén {bits}-bit (GPTQ) cho Llava-7B... ---")
        
        processor = AutoProcessor.from_pretrained(self.base_model_path)
        pure_tokenizer = processor.tokenizer 

        loader = ScienceQALocalLoader(self.data_path, subset_size=128)
        df_samples = loader.preprocess_for_r3_quant()
        calibration_dataset = [
            f"Question: {row['question']} Answer: {row['reasoning']}" 
            for _, row in df_samples.iterrows()
        ]

        quantization_config = GPTQConfig(
            bits=bits,
            dataset=calibration_dataset,
            tokenizer=pure_tokenizer,
            desc_act=False,
            sym=True,
        )

        try:
            # Load cấu hình và ép thuộc tính use_cache để bypass lỗi của optimum
            config = AutoConfig.from_pretrained(self.base_model_path)
            config.use_cache = False #

            print(f"--- ⏳ Đang nạp và nén model (3-bit)... ---")
            model = LlavaForConditionalGeneration.from_pretrained(
                self.base_model_path,
                config=config, # Truyền config đã sửa [cite: 1]
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )

            os.makedirs(self.save_path, exist_ok=True)
            model.save_pretrained(self.save_path)
            processor.save_pretrained(self.save_path)
            print(f"--- ✅ [SUCCESS] Đã lưu model tại: {self.save_path} ---")
            
        except Exception as e:
            print(f"--- ❌ Lỗi nghiêm trọng: {e} ---")
            import traceback
            traceback.print_exc()
            sys.exit(1)