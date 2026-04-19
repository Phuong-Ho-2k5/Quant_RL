import torch
from transformers import LlavaForConditionalGeneration, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import warnings

def get_t4_bnb_config():
    """BitsAndBytes config cho T4 GPU"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

def apply_lora_for_llava(model_path, use_4bit=True):
    """Apply LoRA cho Llava model"""
    try:
        if use_4bit:
            bnb_config = get_t4_bnb_config()
            model = LlavaForConditionalGeneration.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else "cpu",
            )
            model = prepare_model_for_kbit_training(model)
        else:
            model = LlavaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else "cpu",
            )
        
        # Target modules cho LoRA
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
        
        peft_model = get_peft_model(model, lora_config)
        peft_model.print_trainable_parameters()
        return peft_model
        
    except Exception as e:
        print(f"❌ Lỗi khi apply LoRA: {e}")
        raise

def load_existing_lora_for_quantized_model(base_model_dir, lora_dir):
    """Load LoRA adapter cho quantized model"""
    try:
        bnb_config = get_t4_bnb_config()
        base_model = LlavaForConditionalGeneration.from_pretrained(
            base_model_dir,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else "cpu",
        )
        
        base_model = prepare_model_for_kbit_training(base_model)
        
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
        
        # Load adapter weights
        peft_model.load_adapter(lora_dir, adapter_name="default")
        peft_model.set_adapter("default")
        
        return peft_model
        
    except Exception as e:
        print(f"❌ Lỗi khi load LoRA adapter: {e}")
        raise