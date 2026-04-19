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
        # Dọn dẹp bộ nhớ trước khi bắt đầu
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"--- 🛠️ Đang cấu hình nén {bits}-bit (GPTQ) cho Llava-7B... ---")
        
        processor = AutoProcessor.from_pretrained(self.base_model_path)
        pure_tokenizer = processor.tokenizer 

        loader = ScienceQALocalLoader(self.data_path, subset_size=128)
        df_samples = loader.preprocess_for_r3_quant()
        
        # Chỉ lấy text ngắn để calibration
        calibration_dataset = [
            f"Q: {row['question']} A: {row['reasoning']}"[:512] 
            for _, row in df_samples.iterrows()
        ]

        quantization_config = GPTQConfig(
            bits=bits,
            dataset=calibration_dataset,
            tokenizer=pure_tokenizer,
            model_seqlen=512, # Giới hạn độ dài để cứu VRAM
            desc_act=False,
            sym=True,
        )

        try:
            config = AutoConfig.from_pretrained(self.base_model_path)
            config.use_cache = False 

            print(f"--- ⏳ Đang nạp model lên CPU và bắt đầu nén từng layer... ---")
            model = LlavaForConditionalGeneration.from_pretrained(
                self.base_model_path,
                config=config,
                quantization_config=quantization_config,
                # Không dùng device_map="auto" ở đây vì nó sẽ nạp hết vào GPU ngay lập tức
                device_map="cpu", 
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )

            os.makedirs(self.save_path, exist_ok=True)
            print(f"--- 💾 Đang lưu kết quả... ---")
            model.save_pretrained(self.save_path)
            processor.save_pretrained(self.save_path)
            
            print(f"--- ✅ [SUCCESS] Đã lưu model 3-bit GPTQ thành công! ---")
            
        except Exception as e:
            print(f"--- ❌ Lỗi nghiêm trọng: {e} ---")
            sys.exit(1)