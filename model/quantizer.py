import sys
import os
import torch
import gc
from transformers import AutoProcessor, LlavaForConditionalGeneration, GPTQConfig, AutoConfig
from data.dataset_loader import ScienceQALocalLoader 

class LlavaGPTQQuantizer:
    def __init__(self, base_model_path, save_path, data_path):
        self.base_model_path = base_model_path
        self.save_path = save_path
        self.data_path = data_path

    def quantize_and_save(self, bits=3):
        # Dọn dẹp VRAM trước khi bắt đầu
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"--- 🛠️ Nén 3-bit: Chế độ Siêu tiết kiệm VRAM cho 2xT4 ---")
        
        processor = AutoProcessor.from_pretrained(self.base_model_path)
        loader = ScienceQALocalLoader(self.data_path, subset_size=16)
        calibration_dataset = [
            f"Q: {row['question']} A: {row['reasoning']}"[:256] 
            for _, row in loader.preprocess_for_r3_quant().iterrows()
        ]

        quantization_config = GPTQConfig(
            bits=bits,
            dataset=calibration_dataset,
            tokenizer=self.base_model_path,
            modules_to_not_convert=["vision_tower", "multi_modal_projector"],
            model_seqlen=256, # GIẢM XUỐNG 256: Đây là chìa khóa để thoát OOM
            desc_act=False,
            sym=True,
        )

        try:
            config = AutoConfig.from_pretrained(self.base_model_path)
            config.use_cache = False 

            print(f"--- ⏳ Đang nạp model lên CPU (Offloading mode)... ---")
            model = LlavaForConditionalGeneration.from_pretrained(
                self.base_model_path,
                config=config,
                quantization_config=quantization_config,
                # BUỘC DÙNG cpu để dành toàn bộ 15GB VRAM cho việc tính ma trận Hessian
                device_map="cpu", 
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )

            os.makedirs(self.save_path, exist_ok=True)
            model.save_pretrained(self.save_path)
            processor.save_pretrained(self.save_path)
            print(f"--- ✅ [SUCCESS] Đã vượt qua block 32/32! ---")
            
        except Exception as e:
            print(f"--- ❌ Lỗi: {e} ---")
            sys.exit(1)