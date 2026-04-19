import re

def extract_xml_answer(text: str) -> str:
    """Trích xuất nội dung nằm trong thẻ <answer>...</answer>"""
    if not text:
        return ""
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""

def format_reward_func(completions, **kwargs) -> list[float]:
    """
    Format Reward: Kiểm tra cấu trúc <think>...</think><answer>...</answer>
    """
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    rewards = []
    
    for comp in completions:
        text = comp[0]["content"] if isinstance(comp, list) and comp else str(comp)
        
        text_clean = text.strip()
        
        if re.match(pattern, text_clean, re.DOTALL | re.IGNORECASE):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
            
    return rewards

def accuracy_reward_func(completions, ground_truth, **kwargs) -> list[float]:
    """
    Accuracy Reward: So sánh đáp án trong thẻ <answer> với ground_truth
    """
    index_to_letter = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'} 
    rewards = []
    
    for comp, truth in zip(completions, ground_truth):
        text = comp[0]["content"] if isinstance(comp, list) and comp else str(comp)
        pred_answer = extract_xml_answer(text)

        # Convert ground truth
        if isinstance(truth, int):
            truth_clean = index_to_letter.get(truth, str(truth)).lower().strip()
        else:
            truth_clean = str(truth).lower().strip()
            
        pred_clean = pred_answer.lower().strip()
        
        # Extract single letter if longer
        if len(pred_clean) > 1:
            match_letter = re.search(r'[a-e]', pred_clean)
            if match_letter:
                pred_clean = match_letter.group(0)
        
        # Compare
        if pred_clean == truth_clean:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
            
    return rewards