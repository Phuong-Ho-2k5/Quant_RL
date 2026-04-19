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
        print(f"--- 🛠️ Đang cấu hình nén {bits}-bit (GPTQ) cho riêng LLM... ---")
        
        # 1. Chuẩn bị dữ liệu Calibration (128 mẫu là tối ưu cho T4)
        loader = ScienceQALocalLoader(self.data_path, subset_size=128)
        df_samples = loader.preprocess_for_r3_quant()
        calibration_dataset = [
            f"Question: {row['question']} Answer: {row['reasoning']}"[:512] 
            for _, row in df_samples.iterrows()
        ]

        # 2. Cấu hình GPTQ chỉ nhắm vào LLM
        quantization_config = GPTQConfig(
            bits=bits,
            dataset=calibration_dataset,
            tokenizer=self.base_model_path,
            # QUAN TRỌNG: Không nén phần nhìn và projector
            modules_to_not_convert=["vision_tower", "multi_modal_projector"], 
            desc_act=False,
            sym=True,
            model_seqlen=512 # Tiết kiệm bộ nhớ trong lúc nén
        )

        try:
            # Bypass lỗi AttributeError của optimum trên Llava
            config = AutoConfig.from_pretrained(self.base_model_path)
            config.use_cache = False 

            print(f"--- ⏳ Đang nạp model lên CPU để nén (Tránh OOM T4)... ---")
            model = LlavaForConditionalGeneration.from_pretrained(
                self.base_model_path,
                config=config,
                quantization_config=quantization_config,
                device_map="cpu", # Nạp RAM trước khi đẩy từng layer lên GPU nén
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )

            os.makedirs(self.save_path, exist_ok=True)
            print(f"--- 💾 Đang lưu model tại: {self.save_path} ---")
            model.save_pretrained(self.save_path)
            
            processor = AutoProcessor.from_pretrained(self.base_model_path)
            processor.save_pretrained(self.save_path)
            
            print(f"--- ✅ [SUCCESS] Đã lưu model GPTQ 3-bit thành công! ---")
            
        except Exception as e:
            print(f"--- ❌ Lỗi nghiêm trọng: {e} ---")
            import traceback
            traceback.print_exc()
            sys.exit(1)