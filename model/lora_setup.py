import torch
from transformers import LlavaForConditionalGeneration, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

def apply_lora_for_llava(model_path):
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    model  = prepare_model_for_kbit_training(model)

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        exclude_modules=["vision_tower", "multi_modal_projector"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    return get_peft_model(model, lora_config)

def load_existing_lora_for_quantized_model(base_model_dir, lora_dir):
    base_model = LlavaForConditionalGeneration.from_pretrained(
        base_model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

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

    peft_model.load_adapter(lora_dir)

    return peft_model