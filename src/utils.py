import torch
from datasets import Dataset

def build_scienceqa_prompt(question: str, choices: list) -> str:
    """Xây dựng nội dung câu hỏi và các lựa chọn."""
    prompt = f"{question}\n\nChoices:\n"
    labels = ["A", "B", "C", "D", "E"]
    
    if not choices:
        return prompt

    for i, choice in enumerate(choices):
        prompt += f"{labels[i]}. {choice}\n"
        
    return prompt

def prepare_minicap_for_sft(raw_dataset, max_samples=None):
    """
    Chuẩn bị dữ liệu cho SFT. Đối với Llava-7B, chúng ta cần gộp 
    thành một cột 'text' hoàn chỉnh bao gồm cả Prompt và Label.
    """
    SYSTEM_PROMPT = (
        "A conversation between a user and an AI assistant. The assistant "
        "thinks step-by-step and encloses the reasoning in <think> tags "
        "and the final answer in <answer> tags."
    )

    def format_sft_row(item):
        # Format chuẩn Llava: USER: <image>\n{Question} ASSISTANT: <think>{Reasoning}</think><answer>{Answer}</answer>
        question_text = build_scienceqa_prompt(item.get("question", ""), item.get("choices", []))
        
        # Giả sử dataset của bạn có cột 'solution' hoặc 'reasoning' cho phần CoT
        solution = item.get("solution", item.get("reasoning", ""))
        answer_letter = ["A", "B", "C", "D", "E"][item.get("answer", 0)]
        
        # Tạo chuỗi text hoàn chỉnh cho SFT Trainer
        full_text = (
            f"USER: <image>\n{question_text}\n{SYSTEM_PROMPT}\n"
            f"ASSISTANT: <think>{solution}</think><answer>{answer_letter}</answer>"
        )
        
        return {
            "text": full_text,
            "images": [item["image"]] 
        }

    dataset = raw_dataset.filter(lambda x: x.get("image") is not None)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    dataset = dataset.map(format_sft_row, num_proc=4)
    return dataset

def prepare_scienceqa_for_grpo(raw_dataset, max_samples=None):
    """
    Chuẩn bị dữ liệu cho GRPO. GRPO cần cột 'prompt' (đầu vào) 
    và 'ground_truth' (để hàm reward so sánh).
    """
    labels = ["A", "B", "C", "D", "E"]

    def format_row(item):
        question_text = build_scienceqa_prompt(item['question'], item['choices'])
        
        # Với Llava trong GRPOTrainer, ta dùng format string để tránh lỗi template
        prompt_text = f"USER: <image>\n{question_text}\nThink step by step. Output reasoning in <think> and final letter in <answer>.\nASSISTANT:"
        
        return {
            "prompt": prompt_text,
            "images": [item['image']],  
            "ground_truth": labels[item['answer']]
        }

    dataset = raw_dataset.filter(lambda x: x.get('image') is not None)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        
    dataset = dataset.map(format_row, num_proc=4) 
    return dataset