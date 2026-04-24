CALL_LLM=False
from .eval import Evaluator
import json
evaluator = Evaluator()


def append_to_json(file_path, new_data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

def score_cal_and_save(data_list, save_path):
    gt_answers = [item['answer'] for item in data_list]
    preds = [item['extracted_res'] for item in data_list]
    if not all(x == '' for x in gt_answers):
        assert 5187 == len(preds)
        metrics = evaluator.get_metrics(gt_answers, preds)
        for item,acc,anls in zip(data_list,metrics['accuracy'],metrics['anls']):
            item['acc_score'] = acc
            item['anls_score'] = anls
        avg_acc = sum(metrics['accuracy']) / len(metrics['accuracy']) if metrics['accuracy'] else 0
        avg_anls = sum(metrics['anls']) / len(metrics['anls']) if metrics['anls'] else 0
        append_to_json(save_path, data_list)
        txt_path = save_path.replace("json", "txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Average Accuracy: {avg_acc:.4f}\n")
            f.write(f"Average ANLS: {avg_anls:.4f}\n")
    else:
        assert 5019 == len(preds)
        ori_data_list = json.load(open('mpdocvqa/sample_test.json'))
        save_path = save_path.replace(".json","_submit.json")
        submit_data_list = []
        for pred,ori in zip(data_list,ori_data_list):
            
            sample = {
                "questionId":ori["questionId"],
                "answer":pred['extracted_res'],
                "answer_page":pred['extracted_page'][0]
            }
            
            submit_data_list.append(sample)
        append_to_json(save_path,submit_data_list)
        #pass
