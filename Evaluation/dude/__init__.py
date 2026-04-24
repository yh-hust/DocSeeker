CALL_LLM=False
from .eval import calculate_score
import json
from tqdm import tqdm
def append_to_json(file_path, new_data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

def score_cal_and_save(data_list, save_path):

    gt_answers = [item['answer'] for item in data_list]
    preds = [item['extracted_res'] for item in data_list]
    answer_types = [item["evidence_sources"] for item in data_list]
    scores = []
    
    if not all(x == '' for x in gt_answers):
        assert len(preds) == 6318
        for gt,pred,answer_type in tqdm(zip(gt_answers,preds,answer_types)):
            scores.append(calculate_score([pred],gt,answer_type))

        for item,score in zip(data_list,scores):
            item['anls_score'] = score
        
        avg_anls = sum(scores) / len(scores) if scores else 0

        append_to_json(save_path, data_list)

        txt_path = save_path.replace("json", "txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Average Anls: {avg_anls:.4f}\n")
    else:
        assert len(preds) == 11402
        ori_data_list = json.load(open('dude/sample_test.json'))
        submit_data_list = []
        save_path = save_path.replace(".json","_submit.json")
        for pred,ori in zip(preds,ori_data_list):
            sample = {
                "questionId":ori["questionId"],
                "answer":pred if pred.lower() != "not answerable" else "",
                "answer_confidence":1.0,
                "answer_abstain":False
            }
            submit_data_list.append(sample)
        append_to_json(save_path,submit_data_list)
