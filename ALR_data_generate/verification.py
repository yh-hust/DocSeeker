import json
import re
import os
import ast
from tqdm import tqdm
import traceback
import multiprocessing
from openai import OpenAI
import argparse

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")

client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )

def get_clean_string(s):

    s = str(s).lower().strip()

    for suffix in ["miles", "mile", "million"]:
        if s.endswith(suffix):
            s = s[:-len(suffix)].strip()
            break

    s = re.sub(r'\s*\([^)]*\)', '', s).strip()
    s = re.sub(r"^['\"]|['\"]$", "", s).strip()
    s = s.lstrip('$').rstrip('%').strip()
    
    return s

def append_to_json(file_path, new_data):

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

def read_jsonl(file_path):

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                data.append(item)
    return data

def check_answer(response,gt):
    #import pdb;pdb.set_trace()
    cleaned_response = get_clean_string(response)
    if isinstance(gt,list):
        cleaned_gt = [get_clean_string(gt_item) for gt_item in gt]
        is_match = cleaned_response in cleaned_gt

    elif isinstance(gt,str):
        cleaned_response = get_clean_string(response)
        cleaned_gt = get_clean_string(gt)
        is_match = cleaned_response == cleaned_gt

    return is_match

def replacer(ep_ans):
    # 提取 evidence_pages（数字列表）
    gt_page_match = re.search(r'"evidence_pages"\s*:\s*(\[.*?\])\s*,', ep_ans, re.DOTALL)
    # 提取 answer 原始字符串（可能含非法引号）
    ans_match = re.search(r'"answer"\s*:\s*"(.*)"\s*}', ep_ans, re.DOTALL)

    if not gt_page_match or not ans_match:
        print("匹配失败，原始字符串格式可能有误")
        return None

    evidence_pages_str = gt_page_match.group(1)
    raw_answer = ans_match.group(1)

    # 修复未转义的双引号（忽略已经转义的）
    fixed_answer = re.sub(r'(?<!\\)"', r'\\"', raw_answer)

    # 拼接为标准 JSON 字符串
    json_str = f'{{"evidence_pages": {evidence_pages_str}, "answer": "{fixed_answer}"}}'
    return json_str

def extract_answer(question, prompt, model_name="gpt-4o"):
  
    try:
        response = client.chat.completions.create(
            model=model_name,
            max_tokens=768,
            temperature=0.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            messages=[
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
        )
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"API Error: {e}")
        traceback.print_exc()
        return "Failed"

def map_function(args):
    model_name, item, prompt = args
    formatted_answer = item.pop('formatted_answer')
    response = formatted_answer['answer']
    formatted_question = (
                f"question: {item['conversations'][0]['value']}\n"
                f"response: {response}\n"
                f"answer: {item['conversations'][1]['value'][0]}"
            )
    #print(formatted_question)
    while True:
        try:
            #text_input = "\n\nQuestion:{}\nAnalysis:{}\n".format(question, output)
            #print(output)
            out = extract_answer(formatted_question,prompt,model_name)
            assert out != "Failed"
            item["final_answer"]=out
            print(f"item_id:{item['id']}\nLLM output:{out}")
            return item
        except Exception as e:
            print(e)
            traceback.print_exc()

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default="data/stage1_synthesis/mpdoc_expand_data_0_10/data.json")
    parser.add_argument("--model_name", type=str, default="gemini-2.5-flash")
    parser.add_argument("--max_workers",type=int,default=5)
    args = parser.parse_args()

    file_path = args.file_path
    folder_path = os.path.dirname(file_path)
    
    exact_match_path = os.path.join(folder_path, 'exact_match.json') 
    semantic_match_path = os.path.join(folder_path, 'semantic_match.json')

    data_list = json.load(open(file_path, 'r', encoding='utf-8'))
    
    exact_match_count = 0
    exact_match_list = []
    pending_eval_list = []

    for item in tqdm(data_list):

        sd_ans = item['conversations'][1]['value']
        ep_ans = item['expand_answer']
        question = item['conversations'][0]['value']
        
        ep_thought_match = re.search(r"<think>(.*?)</think>", item['expand_answer'], re.DOTALL)
        if ep_thought_match:
            ep_thought = ep_thought_match.group(1).strip().strip('\n')
        else:
            ep_thought = None
            
        ep_ans_match = re.search(r"<answer>(.*?)</answer>", item['expand_answer'], re.DOTALL)
        if ep_ans_match:
            ep_ans = ep_ans_match.group(1).strip().replace('\\n', '').replace('\n', '').replace('\\', '')
        else:
            ep_ans = None
            
        try:
            assert "evidence_pages" in ep_ans and "answer" in ep_ans
            ep_ans = replacer(ep_ans)
            ep_ans = ast.literal_eval(ep_ans)
        except Exception:
            continue
            
        if ep_thought is None or ep_ans is None:
            assert False
        
        if check_answer(ep_ans['answer'], sd_ans):
            item['final_answer'] = sd_ans
            exact_match_count += 1
            exact_match_list.append(item)
        else:
            item['formatted_answer'] = ep_ans
            pending_eval_list.append(item)
            continue

    append_to_json(exact_match_path,exact_match_list)
    print(f"Exact match count: {exact_match_count}")

    data_list = pending_eval_list
    with open("./sys_prompt/filter.md", "r", encoding="utf-8") as f:
        sys_prompt = f.read()

    dist_args = [(args.model_name, item, sys_prompt) for i, item in enumerate(data_list)]   
    inferred_results = [] 
    #map_function(dist_args[0])

    with multiprocessing.Pool(processes=args.max_workers) as pool:
        new_results = pool.map(map_function, dist_args)
        all_results = inferred_results + new_results
        valid_results = [d for d in all_results if d is not None]
        
    append_to_json(semantic_match_path, valid_results)