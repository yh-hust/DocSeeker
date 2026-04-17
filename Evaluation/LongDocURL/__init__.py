CALL_LLM=True
#import pdb;pdb.set_trace()
from .longdocurl_llm_call import llm_process
from .utils_score_v3 import eval_score,show_results
import json
import os
import numpy as np
def LLM_CALL(data_list,model_name='gemini-2.5-flash'):
    return llm_process(data_list,model_name)

def append_to_json(file_path, new_data):
    # 读取已有数据（如果存在），并追加新数据
    # 将合并后的数据保存回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

def show_results(score_dict, avg_acc, show_path=None):
    """
    将总体平均准确率和细粒度评估的 score_dict 格式化并保存到 txt 文件中
    
    参数:
        score_dict: 包含各个维度细粒度评测结果的字典
        avg_acc: 整体的平均准确率
        show_path: TXT 文件的保存路径
    """
    if not show_path:
        print("[Warning] 未提供 show_path，无法保存评测报告。")
        return

    # 确保保存路径的文件夹存在
    os.makedirs(os.path.dirname(show_path) or '.', exist_ok=True)
    
    # 强制将后缀改为 .txt，防止传错了路径名
    if not show_path.endswith('.txt'):
        show_path = os.path.splitext(show_path)[0] + '.txt'

    # 1. 定义递归展平函数，将多层嵌套字典转为一层
    def flatten_results(d, parent_key=''):
        items = []
        for k, v in d.items():
            # 拼接层级名，使用 " - " 分隔
            new_key = f"{parent_key} - {k}" if parent_key else k
            if isinstance(v, dict):
                # 如果还是字典，继续向下递归
                items.extend(flatten_results(v, new_key).items())
            else:
                # 如果已经是具体分值，则加入列表
                items.append((new_key, v))
        return dict(items)

    # 获取展平后的字典
    flat_scores = flatten_results(score_dict)

    # 2. 开始写入 TXT 文件
    try:
        with open(show_path, "w", encoding="utf-8") as f:
            f.write(f"Overall Average Accuracy: {avg_acc:.4f}\n")
            f.write("--------------------------------------------------\n\n")
            f.write("--- Fine-Grained Breakdown ---\n")
            # 遍历展平后的字典，按统一格式写入
            for key, val in flat_scores.items():
                # 处理精度，如果数值是浮点数，保留四位小数
                if isinstance(val, (float, np.float64, np.float32)):
                    formatted_val = f"{val:.4f}"
                else:
                    formatted_val = str(val)
                    
                # 使用对齐格式写入，保证冒号对齐，看起来更整齐
                f.write(f"{key:<45} | Accuracy: {formatted_val}\n")
                
            f.write("\n==================================================\n")
            f.write("Report Generated Successfully.\n")
            
        print(f"✅ 评估结果已成功保存至: {show_path}")
        
    except Exception as e:
        print(f"[Error] 保存评估报告时发生错误: {e}")

def calculate_accuracy_fine_grained(samples, score_dict):
    for sample in samples:
        pred_ans, annotation, answer_format = sample["extracted_res"], sample["answer"], sample["answer_format"]
        if pred_ans == "Fail to extract":
            score_v3 = 0.0
        elif pred_ans == []:
            score_v3 = 0.0
        else:
            score_v3 = eval_score(annotation, pred_ans, answer_format)
            
        sample["score_v3"] = score_v3
        
    # Main_Task
    for sample in samples:
        score_dict["Main_Task"][sample["task_tag"]] += sample["score_v3"]
    
    # Element_Type
    for sample in samples:
        for evidence_source in sample["evidence_sources"]:
            if evidence_source in ["Text", "Layout", "Figure", "Table"]:
                score_dict["Element_Type"][evidence_source] += sample["score_v3"]

    # Evidence_Pages
    for sample in samples:
        if len(sample["evidence_pages"]) > 1:
            score_dict["Evidence_Pages"]["Multi_Page"] += sample["score_v3"]
        elif len(sample["evidence_pages"]) == 1:
            score_dict["Evidence_Pages"]["Single_Page"] += sample["score_v3"]

    # Num_of_Element_Types
    for sample in samples:
        if len(sample["evidence_sources"]) > 1:
            score_dict["Num_of_Element_Types"]["Cross_Element"] += sample["score_v3"]

    # Fine_Grained
    for sample in samples:
        sub_score_dict = score_dict["Fine_Grained"][sample["task_tag"]]
        if sample["task_tag"] in ["Understanding", "Reasoning"]:
            if len(sample["evidence_pages"]) > 1:
                sub_sub_score_dict = sub_score_dict["Multi_Page"]
            elif len(sample["evidence_pages"]) == 1:
                sub_sub_score_dict = sub_score_dict["Single_Page"]

            for evidence_source in sample["evidence_sources"]:
                if evidence_source in ["Text", "Layout", "Figure", "Table"]:
                    sub_sub_score_dict[evidence_source] += sample["score_v3"]

            if len(sample["evidence_pages"]) > 1:
                sub_score_dict["Multi_Page"] = sub_sub_score_dict
            elif len(sample["evidence_pages"]) == 1:
                sub_score_dict["Single_Page"] = sub_sub_score_dict

        elif sample["task_tag"] in ["Locating"]:
            sub_sub_score_dict = sub_score_dict["Cross_Element"]
            if sample["question_type"] == "topic2title":
                sub_sub_score_dict["Cross_Title"] += sample["score_v3"]
            elif sample["question_type"] == "summary2title":
                sub_sub_score_dict["Para_Title"] += sample["score_v3"]
            elif sample["question_type"] == "summary2tab":
                sub_sub_score_dict["Cross_Table"] += sample["score_v3"]
            elif sample["question_type"] == "extract_fig2tab":
                sub_sub_score_dict["Figure_Table"] += sample["score_v3"]
            
            sub_score_dict["Cross_Element"] = sub_sub_score_dict
        
        score_dict["Fine_Grained"][sample["task_tag"]] = sub_score_dict


    return score_dict


def save_dict_to_txt(score_dict, avg_acc, save_path):
    txt_path = save_path.replace("json", "txt")
    
    # 定义递归展平函数
    def flatten_results(d, parent_key=''):
        items = []
        for k, v in d.items():
            # 拼接层级名
            new_key = f"{parent_key} - {k}" if parent_key else k
            if isinstance(v, dict):
                # 如果还是字典，继续向下递归
                items.extend(flatten_results(v, new_key).items())
            else:
                # 已经是具体分值
                items.append((new_key, v))
        return dict(items)

    flat_scores = flatten_results(score_dict)

def score_cal_and_save(data_list, save_path):
    gt_answers = [item['answer'] for item in data_list]
    preds = [item['extracted_res'] for item in data_list]
    answer_formats = [item['answer_format'] for item in data_list]
    acc_list = []
    
    with open('./LongDocURL/scores_sample_fine_grained.json', "r", encoding="utf-8") as rf:
        _ = json.load(rf)
        score_dict, sample_cnt_dict = _["scores"], _["sample_cnt"]
    score_dict = calculate_accuracy_fine_grained(data_list, score_dict)

    def generalize_score_dict(score_dict, sample_cnt_dict):
        for key, value in score_dict.items():
            if isinstance(value, dict):
                generalize_score_dict(value, sample_cnt_dict[key])
                score_dict[key] = value
            else:
                score_dict[key] /= sample_cnt_dict[key]

    generalize_score_dict(score_dict, sample_cnt_dict)
    
    for item in data_list:
        acc_list.append(item['score_v3'])
    avg_acc = sum(acc_list) / len(acc_list) if acc_list else 0
    print(f'avg_acc:{avg_acc}')
    append_to_json(save_path, data_list)

    txt_path = save_path.replace("json", "txt")
    show_results(score_dict,avg_acc,show_path=txt_path)


    #import pdb;pdb.set_trace()
    # for gt,pred,ans,item in zip(gt_answers,preds,answer_formats,data_list):
    #     import pdb;pdb.set_trace()
    #     if pred == []:
    #         #import pdb;pdb.set_trace()
    #         pred =['Not answerable']
    #     acc = eval_score(gt,pred,ans)
    #     item['score_v3'] = acc
    #     acc_list.append(acc)
    #     # if not isinstance(item['evidence_pages'],str):
    #     #     item['evidence_pages'] = str(item['evidence_pages'])
    #     # if not isinstance(item["evidence_sources"],str):
    #     #     item["evidence_sources"] = str(item["evidence_sources"])
    # avg_acc = sum(acc_list) / len(acc_list) if acc_list else 0
    # print(f'avg_acc:{avg_acc}')
    # append_to_json(save_path, data_list)
    # txt_path = save_path.replace("json", "txt")
    # with open(txt_path, "w", encoding="utf-8") as f:
    #     f.write(f"Average acc: {avg_acc:.4f}\n")
    #import pdb;pdb.set_trace()
    #show_results(data_list,txt_path)
        
