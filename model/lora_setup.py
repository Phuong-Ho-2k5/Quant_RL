import torch
from transformers import LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def get_t4_bnb_config():
    """Cấu hình BitsAndBytes tối ưu cho GPU T4 (Dành cho bản nén 4-bit)"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16, # Sử dụng float16 để tiết kiệm VRAM
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

def apply_lora_for_llava(model_path):
    """
    Nạp model Llava (đã nén hoặc gốc) và áp dụng cấu hình LoRA.
    Chỉ nhắm mục tiêu vào phần Language Model (LLM).
    """
    # Nạp model với float16 để đồng nhất dữ liệu trên T4
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Chuẩn bị model cho k-bit training (quan trọng cho model đã quantized)
    model = prepare_model_for_kbit_training(model)
    
    # Các module mục tiêu trong kiến trúc Llama-2 của Llava-7B
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ]
    
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=target_modules,
        # LOẠI TRỪ Vision Tower và Projector để giữ nguyên mắt của mô hình
        exclude_modules=["vision_tower", "multi_modal_projector"],
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM",
    )
    
    print(f"✅ Đã áp dụng LoRA cho mô hình tại {model_path}")
    return get_peft_model(model, lora_config)

def load_existing_lora_for_quantized_model(base_model_dir, lora_dir):
    """
    Nạp các trọng số Adapter LoRA đã có sẵn (ví dụ từ bước SFT) 
    lên trên một mô hình nền đã được lượng tử hóa.
    """
    # Nạp model nền (Base model)
    base_model = LlavaForConditionalGeneration.from_pretrained(
        base_model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ]
    
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=target_modules,
        exclude_modules=["vision_tower", "multi_modal_projector"],
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(base_model, lora_config)
    
    # Nạp trọng số từ thư mục checkpoint LoRA
    print(f"🔄 Đang nạp LoRA adapter từ: {lora_dir}")
    peft_model.load_adapter(lora_dir, adapter_name="default")
    
    return peft_model