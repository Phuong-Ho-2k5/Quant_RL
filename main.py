import os
import sys
from huggingface_hub import snapshot_download
from datasets import load_dataset, Dataset

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from model.quantizer import LlavaGPTQQuantizer 
from src.grpo_trainer import train_llava_grpo 
from src.sft_trainer import train_llava_sft

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# 1. Cấu hình Model Llava-7B
BASE_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
QUANT_BITS = 4

def setup_environment():
    print("--- 1. Khởi tạo cấu trúc thư mục ---")
    directories = ["data/science_qa", "weights", "r3_quant_checkpoints", "sft_baseline_checkpoints"]
    for folder in directories:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Đã tạo thư mục: {folder}")

def download_data():
    print("\n--- 2. Đang tải/đọc Dataset ScienceQA ---")
    target_path = "./data/science_qa/validation-00000-of-00001-6c7328ff6c84284c.parquet"
    if not os.path.exists(target_path):
        print("Đang tải dataset từ Hugging Face...")
        dataset = load_dataset("derek-thomas/ScienceQA", split="validation", cache_dir="./data/cache")
        dataset.to_parquet(target_path)
        print(f"Đã lưu dataset tại: {target_path}")
    else:
        print(f"Dataset đã tồn tại tại {target_path}, đang load...")
        dataset = load_dataset("parquet", data_files=target_path, split="train")
    return dataset

def download_sft_data():
    print("\n--- 2.5 Đang tải Dataset Mini CoT 8k (Pha mồi SFT) ---")
    target_path = "./data/mini_cot_8k_verified/train-00000-of-00001.parquet"
    if not os.path.exists("./data/mini_cot_8k_verified/train-00000-of-00001.parquet"):
        print("Đang tải dataset Mini CoT 8k từ Hugging Face...")
        sft_dataset = load_dataset("luodian/mini_cot_8k_verified", split="train")
        sft_dataset.save_to_disk("./data/mini_cot_8k_verified")
        print("Đã lưu dataset Mini CoT 8k tại: ./data/mini_cot_8k_verified")
    else:
        print("Dataset Mini CoT 8k đã tồn tại, đang load...")
        sft_dataset = load_dataset("parquet", data_files=target_path, split="train")
    return sft_dataset

def download_model(model_id):
    print(f"\n--- 3. Đang tải Model {model_id} ---")
    model_name = model_id.split("/")[-1]
    local_dir = f"./weights/{model_name}"
    
    if not os.path.exists(os.path.join(local_dir, "config.json")):
        print(f"Đang tải model {model_id} (~26GB, hãy đảm bảo ổ cứng đủ chỗ)...")
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.bin", "*.pt", "*.msgpack"]
        )
        print(f"Model đã được tải về: {local_dir}")
    else:
        print(f"Model đã tồn tại ở: {local_dir}")
    return local_dir

def run_quantization(base_model_dir, dataset_path, bits):
    model_name = os.path.basename(base_model_dir)
    save_dir = f"./weights/{model_name}-GPTQ-Int{bits}"
    
    print(f"\n--- 4. Bắt đầu lượng tử hoá Llava-7B (GPTQ Int{bits}) ---")
    if not os.path.exists(os.path.join(save_dir, "config.json")):
        quantizer = LlavaGPTQQuantizer(base_model_dir, save_dir, dataset_path)
        quantizer.quantize_and_save(bits=bits)
        print("[SUCCESS] Lượng tử hóa Llava-7B hoàn tất!")
    else:
        print(f"Model lượng tử hóa đã tồn tại ở: {save_dir}")
    return save_dir

def run_rl_pipeline(quant_model_dir, sft_dataset, grpo_dataset):
    sft_output_dir = "./sft_baseline_checkpoints"
    grpo_output_dir = "./r3_quant_checkpoints"
    
    # Bước SFT: Dạy format XML cho Llava
    checkpoint_exists = os.path.exists(sft_output_dir) and \
                        os.path.exists(os.path.join(sft_output_dir, "adapter_config.json"))

    if checkpoint_exists:
        print(f"\n--- [SKIP] Đã tìm thấy SFT Checkpoint. ---")
    else:
        print("\n--- 5. Bắt đầu SFT (Supervised Fine-Tuning) ---")
        train_llava_sft(quant_model_dir, sft_dataset, sft_output_dir)
        print(f"\n[SUCCESS] Hoàn tất SFT!")

    # Bước GRPO: Tối ưu khả năng suy luận
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    
    import time
    time.sleep(2)
    print("\n--- 6. Bắt đầu huấn luyện RL (GRPO) ---")
    train_llava_grpo(quant_model_dir, grpo_dataset, grpo_output_dir, sft_lora_dir=None)
    print(f"\n[SUCCESS] Hoàn tất GRPO!")

def main():
    print("==========================================")
    print(" 🚀 LLAVA-7B END-TO-END QUANT & RL PIPELINE")
    print(f" TARGET MODEL: {BASE_MODEL_ID}")
    print("==========================================\n")
    
    setup_environment()
    
    # Chuẩn bị dữ liệu
    grpo_dataset = download_data()
    sft_dataset = download_sft_data()
    dataset_path = "./data/science_qa/validation-00000-of-00001-6c7328ff6c84284c.parquet"
    
    # Tải và Nén
    base_model_dir = download_model(BASE_MODEL_ID)
    quant_model_dir = run_quantization(base_model_dir, dataset_path, QUANT_BITS)
    
    # Huấn luyện SFT -> GRPO
    run_rl_pipeline(quant_model_dir, sft_dataset, grpo_dataset)

if __name__ == "__main__":
    main()