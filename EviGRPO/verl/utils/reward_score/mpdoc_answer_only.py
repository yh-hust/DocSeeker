import re
from typing import Set, Dict, Any
import ast
def extract_answer(text: str) -> dict | None:
    """
    从文本中提取<answer>标签内的内容，清理并将其转换为Python字典。

    :param text: 包含<answer>标签的原始字符串。
    :return: 转换后的字典，如果找不到或转换失败则返回 None。
    """
    # 步骤 1: 使用正则表达式提取内容
    # r"<answer>(.*?)</answer>" 是正则表达式模式：
    # - <answer> 和 </answer> 是要匹配的字面标签。
    # - (.*?) 是一个非贪婪匹配的捕获组，它会捕获两个标签之间的所有字符。
    # - re.DOTALL 标志让 '.' 也能匹配换行符 \n，这对于多行内容至关重要。
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)

    if not match:
        #print("错误：在文本中未找到 <answer>...</answer> 标签。")
        return None

    # match.group(0) 是整个匹配的文本（包括标签）
    # match.group(1) 是第一个捕获组的内容（标签之间的内容）
    content_str = match.group(1)

    # 步骤 2: 清理字符串，删除头尾的 \n 和其他空白字符
    cleaned_str = content_str.strip()

    # 步骤 3: 使用 ast.literal_eval() 安全地将字符串转换为字典
    try:
        py_dict = ast.literal_eval(cleaned_str)
        if not isinstance(py_dict, dict):
            #print(f"错误：转换后的对象不是字典，而是 {type(py_dict)}。")
            return None
        return py_dict
    except:
        #print(f"错误：无法将内容转换为字典。内容：'{cleaned_str}'，错误：{e}")
        return None

def extract_think(text):
    pattern = r"<think>(.*?)</think>"
    match = re.search(pattern, text, re.DOTALL)

    if not match:
        #print("错误：在文本中未找到 <think>...</think> 标签。")
        return None
    else:
        return match.group(1).strip()

# --- Reward 组件 (这部分代码与您提供的一致，无需修改) ---

def format_reward(predict_str: str) -> float:
    """
    检查格式是否完整包含 <think>, <page>, 和 \\boxed{}。
    """
    pattern = re.compile(r"<think>.*?</think>.*?<answer>.*?</answer>", re.DOTALL)
    match = re.fullmatch(pattern, predict_str.strip())
    if match==None:
        return 0.0
    ans_dict = extract_answer(predict_str)
    if ans_dict==None:
        return 0.0
    # think_process = extract_think(predict_str)
    # if think_process==None:
    #     return 0.0
    # pattern = re.compile(r"\\boxed{Question Analysis}.*?\\boxed{Evidence Localization}.*?\\boxed{Reasoning Process}.*?", re.DOTALL)
    # match = re.fullmatch(pattern, think_process.strip())
    # if match==None:
    #     return 0.0
    return 1.0

def answer_reward(predict_str: str, ground_truth_answer: str) -> float:
    """
    计算答案的正确性得分 (ANLS)。
    """
    try:
        ans_dict = extract_answer(predict_str)
        predicted_answer = str(ans_dict["answer"])
    except:
        predicted_answer = None
    #predicted_answer = answer_match.group(1).strip() if answer_match else ""

    if not predicted_answer:
        return 0.0
        
    # def levenshtein_distance(s1, s2):
    #     if len(s1) > len(s2):
    #         s1, s2 = s2, s1
    #     distances = range(len(s1) + 1)
    #     for i2, c2 in enumerate(s2):
    #         distances_ = [i2 + 1]
    #         for i1, c1 in enumerate(s1):
    #             if c1 == c2:
    #                 distances_.append(distances[i1])
    #             else:
    #                 distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
    #         distances = distances_
    #     return distances[-1]

    # def anls_compute(prediction, groundtruth, threshold=0.5):
    #     dist = levenshtein_distance(groundtruth.lower(), prediction.lower()) # 忽略大小写
    #     length = max(len(groundtruth), len(prediction))
    #     if length == 0:
    #         return 1.0 # 两个空字符串是完全匹配的
    #     value = float(dist) / float(length)
    #     anls = 1.0 - value
    #     # 如果相似度低于阈值，直接判为0分
    #     return anls if anls >= threshold else 0.0
    def anls_compute(predicted_answer, ground_truth_answer):
        if predicted_answer.lower()==ground_truth_answer.lower():
            return 1.0
        else:
            return 0.0
    return anls_compute(predicted_answer, ground_truth_answer)


# --- 总分计算 (核心修改处) ---

def compute_score(predict_str: str, ground_truth: str, extra_info: dict) -> float:
    """
    计算最终的加权总分。
    即使格式不完整，只要能提取出内容，也会给出相应的分数。
    """
    # 1. 定义各部分权重，总和为1
    weights = {
        "format": 0.1,  # 完整格式的奖励
        "answer": 0.9   # 答案正确性的奖励
    }

    # 2. 分别计算每个部分的 reward
    # format_reward 现在只作为加分项，而不是否决项
    format_r = format_reward(predict_str)
    
    # 只要<page>和\\boxed{}存在，就能计算内容分数
    #page_r = page_reward(predict_str, set(extra_info.get("gt_page", [])))
    answer_r = answer_reward(predict_str, ground_truth)
    # if extra_info.get("gt_page", [])==[-1]:
    #     page_r = (weights["format"] * format_r + weights["answer"] * answer_r) / (weights["format"]+weights["answer"])
    # 3. 直接计算加权总和
    final_score = (
        weights["format"] * format_r +
        weights["answer"] * answer_r
    )
    
    return final_score


# --- 使用示例 (已更新以反映新逻辑) ---

if __name__ == '__main__':
    # 示例1: 全部正确，格式完整
    predict_1 = "<think>...</think><page>7, 9</page>\\boxed{Correct Answer}"
    gt_1 = {"answer": "Correct Answer", "pages": [7, 9]}
    score_1 = compute_score(predict_1, gt_1)
    print(f"示例1 (全部正确) 分数: {score_1:.4f}") # 应该等于 0.1*1 + 0.4*1 + 0.5*1 = 1.0

    # 示例2: 页面召回不全，答案正确，格式完整
    predict_2 = "<think>...</think><page>7</page>\\boxed{Correct Answer}"
    gt_2 = {"answer": "Correct Answer", "pages": [7, 9]}
    score_2 = compute_score(predict_2, gt_2)
    # 页面 F_4(recall=0.5, prec=1.0) 分数约为 0.58. 
    # 预期总分: 0.1*1 + 0.4*0.58 + 0.5*1 = 0.1 + 0.232 + 0.5 = 0.832
    print(f"示例2 (页面召回不全) 分数: {score_2:.4f}") 

    # 示例3: 答案错误，页面正确，格式完整
    predict_3 = "<think>...</think><page>7, 9</page>\\boxed{2}"
    gt_3 = {"answer": "3", "pages": [7, 9]}
    score_3 = compute_score(predict_3, gt_3)
    # 预期总分: 0.1*1 + 0.4*1 + 0.5*0 = 0.1 + 0.4 = 0.5
    print(f"示例3 (答案错误) 分数: {score_3:.4f}")

    # 示例4: 格式错误 (缺少<think>), 但核心内容正确
    predict_4 = "<page>7, 9</page>\\boxed{Correct Answer}" # 缺少 <think>
    gt_4 = {"answer": "Correct Answer", "pages": [7, 9]}
    score_4 = compute_score(predict_4, gt_4)
    # format_r=0, page_r=1, answer_r=1
    # 预期总分: 0.1*0 + 0.4*1 + 0.5*1 = 0.9
    print(f"示例4 (格式不完整) 分数: {score_4:.4f}") 
    
    # 示例5: 只有答案正确
    predict_5 = "\\boxed{Correct Answer}" # 缺少 <think> 和 <page>
    gt_5 = {"answer": "Correct Answer", "pages": [7, 9]}
    score_5 = compute_score(predict_5, gt_5)
    # format_r=0, page_r=0, answer_r=1
    # 预期总分: 0.1*0 + 0.4*0 + 0.5*1 = 0.5
    print(f"示例5 (只有答案) 分数: {score_5:.4f}")