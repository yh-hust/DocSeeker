import os
import json
CALL_LLM=False
from .evaluate import eval_qa
def append_to_json(file_path, new_data):
    # 读取已有数据（如果存在），并追加新数据
    # 将合并后的数据保存回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

def score_cal_and_save(data_list, save_path):
    gt_answers = [item['answer'] for item in data_list]
    preds = [item['extracted_res'] for item in data_list]
    questions = [item['question'] for item in data_list]
    f1_list = []
    em_list = []
    for gt,pred,question,item in zip(gt_answers,preds,questions,data_list):
        f1,em = eval_qa(gt,pred,question)
        item['f1'] = f1
        item['em'] = em
        f1_list.append(f1)
        em_list.append(em)
    avg_f1 = sum(f1_list) / len(f1_list) if f1_list else 0
    avg_em = sum(em_list) / len(em_list) if em_list else 0
    print(f'avg_f1:{avg_f1}')
    print(f'avg_em:{avg_em}')
    append_to_json(save_path, data_list)
    txt_path = save_path.replace("json", "txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Average f1: {avg_f1:.4f}\n")
        f.write(f"Average em: {avg_em:.4f}\n")
        