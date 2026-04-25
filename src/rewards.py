# rewards.py - Với hỗ trợ think tag
import re
from typing import List

def format_reward_func(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """
    Format Reward: Kiểm tra có cả <think> và <answer> tag không
    """
    rewards = []
    for completion in completions:
        text = completion if isinstance(completion, str) else str(completion)
        
        has_think = '<think>' in text and '</think>' in text
        has_answer = '<answer>' in text and '</answer>' in text
        
        # Thưởng tối đa nếu có cả hai
        if has_think and has_answer:
            rewards.append(1.0)
        elif has_answer:  # Chỉ có answer, không có think
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    
    return rewards

def accuracy_reward_func(prompts: List[str], completions: List[str], ground_truth: List[str], **kwargs) -> List[float]:
    """
    Accuracy Reward: So sánh đáp án trong <answer> với ground_truth
    """
    rewards = []
    
    for completion, truth in zip(completions, ground_truth):
        text = completion if isinstance(completion, str) else str(completion)
        
        # Tìm đáp án trong <answer> tag
        match = re.search(r'<answer>\s*([A-E])\s*</answer>', text, re.IGNORECASE)
        if match:
            pred = match.group(1).upper()
        else:
            # Nếu không có tag, tìm chữ cái A-E bất kỳ
            match = re.search(r'\b([A-E])\b', text.upper())
            pred = match.group(1) if match else ''
        
        # Xử lý ground truth
        if isinstance(truth, int):
            true_answer = ['A', 'B', 'C', 'D', 'E'][truth] if truth < 5 else ''
        elif isinstance(truth, str):
            true_match = re.search(r'<answer>\s*([A-E])\s*</answer>', truth, re.IGNORECASE)
            if true_match:
                true_answer = true_match.group(1).upper()
            else:
                true_answer = truth.upper().strip()
                if len(true_answer) > 1:
                    true_answer = true_answer[0] if true_answer[0] in 'ABCDE' else ''
        else:
            true_answer = str(truth).upper()
        
        rewards.append(1.0 if pred == true_answer else 0.0)
    
    return rewards