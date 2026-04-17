import re
from typing import Set, Dict, Any
import ast
def extract_answer(text: str) -> dict | None:
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)

    if not match:
        return None
    
    content_str = match.group(1)

    cleaned_str = content_str.strip()

    try:
        py_dict = ast.literal_eval(cleaned_str)
        if not isinstance(py_dict, dict):
            return None
        return py_dict
    except:
        return None

def extract_think(text):
    pattern = r"<think>(.*?)</think>"
    match = re.search(pattern, text, re.DOTALL)

    if not match:
        return None
    else:
        return match.group(1).strip()

def format_reward(predict_str: str) -> float:

    pattern = re.compile(r"<think>.*?</think>.*?<answer>.*?</answer>", re.DOTALL)
    match = re.fullmatch(pattern, predict_str.strip())
    if match==None:
        return 0.0
    ans_dict = extract_answer(predict_str)
    if ans_dict==None:
        return 0.0
    think_process = extract_think(predict_str)
    if think_process==None:
        return 0.0
    pattern = re.compile(r"\\boxed{Question Analysis}.*?\\boxed{Evidence Localization}.*?\\boxed{Reasoning Process}.*?", re.DOTALL)
    match = re.fullmatch(pattern, think_process.strip())
    if match==None:
        return 0.0
    return 1.0


def page_reward(predict_str: str, ground_truth_pages: Set[int], beta: float = 2.0) -> float:

    ans_dict = extract_answer(predict_str)
    if ans_dict==None:
        return 0.0
    try:
        evidence_pages_raw = ans_dict.get("evidence_pages")
        if evidence_pages_raw is None:
            predicted_pages = set()
        else:
            flat_pages = []
            for item in evidence_pages_raw:
                if isinstance(item, list):
                    flat_pages.extend(item)
                else:
                    flat_pages.append(item)
            predicted_pages = set(flat_pages)
    except:
        predicted_pages = set()
    if -1 in ground_truth_pages and len(ground_truth_pages)==1:
        return 1.0

    if not predicted_pages:
        return 0.0

    true_positives = len(predicted_pages.intersection(ground_truth_pages))
    if true_positives == 0:
        return 0.0

    precision = true_positives / len(predicted_pages)
    recall = true_positives / len(ground_truth_pages)
    
    if (beta**2 * precision) + recall == 0:
        return 0.0
        
    f_beta_score = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    return f_beta_score


def answer_reward(predict_str: str, ground_truth_answer: str) -> float:
    try:
        ans_dict = extract_answer(predict_str)
        predicted_answer = str(ans_dict["answer"])
    except:
        predicted_answer = None

    if not predicted_answer:
        return 0.0
        
    def levenshtein_distance(s1, s2):
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]

    def anls_compute(prediction, groundtruth, threshold=0.0):
        dist = levenshtein_distance(groundtruth.lower(), prediction.lower())
        length = max(len(groundtruth), len(prediction))
        if length == 0:
            return 1.0
        value = float(dist) / float(length)
        anls = 1.0 - value
        return anls if anls >= threshold else 0.0

    return anls_compute(predicted_answer, ground_truth_answer)


def compute_score(predict_str: str, ground_truth: str, extra_info: dict) -> float:

    weights = {
        "format": 0.1,
        "page": 0.3,
        "answer": 0.6 
    }
    format_r = format_reward(predict_str)
    
    page_r = page_reward(predict_str, set(extra_info.get("gt_page", [])))
    answer_r = answer_reward(predict_str, ground_truth)
    if extra_info.get("gt_page", [])==[-1]:
        page_r = (weights["format"] * format_r + weights["answer"] * answer_r) / (weights["format"]+weights["answer"])
    final_score = (
        weights["format"] * format_r +
        weights["page"] * page_r +
        weights["answer"] * answer_r
    )
    
    return final_score
