CALL_LLM=True
from .mmlongbench_llm_cal import llm_process
import json
import re
from .eval.eval_score import eval_score, eval_acc_and_f1, show_results
# /home/ma-user/work/dataset/dataset_yh/yh/DocSeeker/Evaluation/mmlongbench_doc/mmlongbench_llm_cal.py
def LLM_CALL(data_list,model_name='gemini-2.5-flash'):
    return llm_process(data_list,model_name)

def append_to_json(file_path, new_data):
    # 读取已有数据（如果存在），并追加新数据
    # 将合并后的数据保存回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

def score_cal_and_save(samples, save_path):
    #import pdb;pdb.set_trace()
    for sample in samples:
        # if sample['question']=="Which category has the most increase from 2005 to 2010 for time spent on weedends?":
        #     import pdb;pdb.set_trace()
        try:
            pred_ans = sample['extracted_res'].split("Answer format:")[0].split("Extracted answer:")[1].strip()
            #import pdb;pdb.set_trace()
            score = eval_score(sample["answer"], pred_ans, sample["answer_format"])
        except:
            pred_ans = "Failed to extract"
            score = 0.0
        sample['response'] = sample['pred']
        sample["pred"] = pred_ans
        sample["score"] = score
        acc, f1 = eval_acc_and_f1(samples)
        print("--------------------------------------")
        print("Question: {}".format(sample["question"]))
        print("Response: {}".format(sample["response"]))
        print("Gt: {}\tPred: {}\tScore: {}".format(sample["answer"], sample["pred"], sample["score"]))
        print("Avg acc: {}".format(acc))
        print("Avg f1: {}".format(f1))
    
    append_to_json(save_path,samples)
    show_results(samples,re.sub(r"\.json$", ".txt", save_path))