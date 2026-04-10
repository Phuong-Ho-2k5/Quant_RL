import os
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch

class ModelDownloader:
    def __init__(self, model_id="llava-hf/llava-1.5-7b-hf", local_dir="./weights/llava-1.5-7b-hf"):
        self.model_id = model_id
        self.local_dir = local_dir

    def download(self):
        print(f"🚀 Bắt đầu tải Llava-7B: {self.model_id} -> {self.local_dir}")
        os.makedirs(self.local_dir, exist_ok=True)
        snapshot_download(
            repo_id=self.model_id,
            local_dir=self.local_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.pt", "*.msgpack", "*.bin"]
        )
        print("✅ Tải thành công!")

    def test_load_local(self):
        print(f"🔄 Đang kiểm tra nạp mô hình từ: {self.local_dir}...")

        processor = AutoProcessor.from_pretrained(self.local_dir)
        
   
        model = LlavaForConditionalGeneration.from_pretrained(
            self.local_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        print(f"💎 Model loaded on: {model.device} | Dtype: {model.dtype}")
        return model, processor

if __name__ == "__main__":
    # Khởi tạo downloader cho Llava 7B
    downloader = ModelDownloader(
        model_id="llava-hf/llava-1.5-7b-hf", 
        local_dir="./weights/llava-1.5-7b-hf"
    )

    downloader.download()
    
    try:
        model, processor = downloader.test_load_local()
        print("🚀 Sẵn sàng cho bước Quantization tiếp theo!")
    except torch.cuda.OutOfMemoryError:
        print("❌ Lỗi: Không đủ VRAM để load bản 16-bit. Đừng lo, bước Quantize sẽ giải quyết vấn đề này.")