import torch
from datasets import Dataset
from PIL import Image
import io

def build_scienceqa_prompt(question: str, choices: list) -> str:
    """Xây dựng nội dung câu hỏi và các lựa chọn."""
    prompt = f"{question}\n\nChoices:\n"
    labels = ["A", "B", "C", "D", "E"]
    
    if not choices:
        return prompt

    for i, choice in enumerate(choices):
        if i < len(labels):
            prompt += f"{labels[i]}. {choice}\n"
        
    return prompt

def prepare_minicap_for_sft(raw_dataset, max_samples=None):
    """Chuẩn bị dữ liệu cho SFT"""
    SYSTEM_PROMPT = (
        "A conversation between a user and an AI assistant. The assistant "
        "thinks step-by-step and encloses the reasoning in <think> tags "
        "and the final answer in <answer> tags."
    )
    
    def format_sft_row(item):
        question_text = build_scienceqa_prompt(item.get("question", ""), item.get("choices", []))
        solution = item.get("solution", item.get("reasoning", ""))
        answer_idx = item.get("answer", 0)
        answer_letter = ["A", "B", "C", "D", "E"][answer_idx] if isinstance(answer_idx, int) and 0 <= answer_idx < 5 else "A"
        
        full_text = (
            f"USER: <image>\n{question_text}\n\n{SYSTEM_PROMPT}\n"
            f"ASSISTANT: <think>{solution}</think><answer>{answer_letter}</answer>"
        )

        # Xử lý image
        image = item.get("image")
        if image is None:
            return None
            
        return {
            "text": full_text,
            "images": [image]
        }

    # Filter valid rows
    dataset = raw_dataset.filter(lambda x: x.get("image") is not None)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    # Apply formatting
    dataset = dataset.map(format_sft_row, num_proc=1, remove_columns=raw_dataset.column_names)
    
    # Remove None values
    dataset = dataset.filter(lambda x: x is not None)
    
    return dataset

def prepare_scienceqa_for_grpo(raw_dataset, max_samples=None):
    labels = ["A", "B", "C", "D", "E"]
    MAX_PROMPT_TEXT_LENGTH = 512

    def format_row(item):
        question_text = build_scienceqa_prompt(item.get('question', ''), item.get('choices', []))
        if len(question_text) > MAX_PROMPT_TEXT_LENGTH:
            question_text = question_text[:MAX_PROMPT_TEXT_LENGTH] + "..."
        
        # Format chuẩn đa phương thức cho TRL GRPOTrainer
        prompt_conversational = [
            {
                "role": "user",
                "content": [
                    {"type": "image"}, # TRL sẽ tự động thay thế bằng <image> token
                    {"type": "text", "text": f"{question_text}\nThink step by step. Output reasoning in <think> and final letter in <answer>."}
                ],
            }
        ]
        
        answer_idx = item.get('answer', 0)
        ground_truth = labels[answer_idx] if isinstance(answer_idx, int) and 0 <= answer_idx < 5 else "A"
        
        # Đảm bảo ảnh luôn ở định dạng RGB
        image = item.get('image')
        if isinstance(image, dict) and 'bytes' in image:
            image = Image.open(io.BytesIO(image['bytes'])).convert("RGB")
        
        return {
            "prompt": prompt_conversational,
            "images": [image] if image is not None else [],
            "ground_truth": ground_truth
        }

    # Filter rows with images
    dataset = raw_dataset.filter(lambda x: x.get('image') is not None)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        
    dataset = dataset.map(format_row, num_proc=1)
    
    return dataset