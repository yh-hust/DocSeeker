import logging
import numpy as np
from munkres import Munkres, make_cost_matrix


def levenshtein_distance(s1: str, s2: str) -> int:
    """计算两个字符串之间的莱文斯坦距离（编辑距离）"""
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

def parse_answers(pred_answers: list) -> str:
    """解析预测答案列表，确保返回的是单个字符串"""
    if not pred_answers:
        # 如果预测列表为空，视为无法回答
        return ""

    if isinstance(pred_answers, list):
        if len(pred_answers) > 1:
            logging.warning(f"警告：预测答案是包含多个元素的列表 {pred_answers}，默认取第一个。")
        # 即使列表只有一个元素，也将其取出
        return pred_answers[0]
    return str(pred_answers) # 以防万一传入的不是列表

def get_NLS(gt_answers: list, pred_answers: list, threshold: float) -> float:
    """计算单个问题的归一化莱文斯坦相似度 (NLS)"""
    values = []
    
    # 解析预测答案，确保是单个字符串
    pred_answer_str = parse_answers(pred_answers)

    if not gt_answers: # 如果标准答案为空（例如，对于无法回答的问题）
        gt_answers = [""]

    for answer in gt_answers:
        # 预处理：去除首尾空格，转为小写，合并多余空格
        gt_answer = " ".join(answer.strip().lower().split())
        det_answer = " ".join(pred_answer_str.strip().lower().split())

        dist = levenshtein_distance(gt_answer, det_answer)
        length = max(len(gt_answer), len(det_answer))
        
        # 归一化距离
        normalized_dist = 0.0 if length == 0 else float(dist) / float(length)
        values.append(normalized_dist)

    # 相似度 = 1 - 最小的归一化距离（即与最接近的标准答案的相似度）
    question_result = 1 - min(values)

    # 如果相似度低于阈值，则直接判为0分
    if question_result < threshold:
        question_result = 0

    return question_result

def get_best_matches_hungarian_munkers(anchor_list: list, matching_list: list) -> tuple:
    """为匈牙利算法准备成本矩阵"""
    cost_matrix = []
    for anchor_item in anchor_list:
        row = []
        for matching_item in matching_list:
            # 注意：匈牙利算法求的是最小成本，而NLS是相似度（越大越好）
            # 所以成本应该是 1 - NLS
            nls_score = get_NLS([anchor_item], [matching_item], threshold=0.5)
            cost = 1.0 - nls_score
            row.append(cost)
        cost_matrix.append(row)
    return cost_matrix

def get_NLSL(gt_list: list, pred_list: list) -> float:
    """使用匈牙利算法计算列表答案的相似度 (NLSL)"""
    if not gt_list and not pred_list:
        return 1.0
    if not gt_list or not pred_list:
        return 0.0

    # 确保 anchor_list 是较短的列表
    if len(gt_list) < len(pred_list):
        anchor_list, matching_list = gt_list, pred_list
    else:
        anchor_list, matching_list = pred_list, gt_list

    cost_matrix = get_best_matches_hungarian_munkers(anchor_list, matching_list)
    
    # 分母应该是两个列表中较长的那个的长度
    num_answers = max(len(set(gt_list)), len(pred_list))

    m = Munkres()
    indexes = m.compute(cost_matrix)
    
    total_similarity = 0
    for row, column in indexes:
        # 从成本矩阵中取出的值是 cost = 1 - similarity
        # 所以 similarity = 1 - cost
        similarity = 1.0 - cost_matrix[row][column]
        total_similarity += similarity

    nlsl_score = total_similarity / num_answers
    return nlsl_score



def calculate_score(predict: list, answers: list, answer_type: str, anls_threshold: float = 0.5) -> float:
    """
    根据答案类型计算单个问题的得分。

    Args:
        predict (list): 模型预测的答案列表，例如 ["my answer"]。
        answers (list): 标准答案列表，例如 ["the right answer", "a correct answer"]。
        answer_type (str): 问题的答案类型，例如 "abstractive" 或 "list/extractive"。
        anls_threshold (float): ANLS 阈值，低于此值的单句答案得分将归零。默认为 0.5。

    Returns:
        float: 计算出的得分 (0.0 到 1.0 之间)。
    """
    if answer_type=='not-answerable':
        answers = [""]
    if predict==['Not answerable']:
        predict = [""]
    if "list" in answer_type:
        if ' ' in predict[0]:
            predict = predict[0].split(' ')
        if ',' in predict[0]:
            predict = predict[0].split(',')
    

    if "list" in answer_type:
        # 如果答案类型包含 "list"，使用 NLSL 算法
        #import pdb;pdb.set_trace()
        score = get_NLSL(answers, predict)
    else:
        # 否则，使用 NLS 算法
        score = get_NLS(answers, predict, threshold=anls_threshold)
    
    return score