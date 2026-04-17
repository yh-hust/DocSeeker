# 输入是dataset_name, 判断是否需要调用大模型, 调用score_cal

import json
import os
import argparse
import ast
import re
from typing import Set, Dict, Any
def extract_ans(text: str) -> dict | None:
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
    #import pdb;pdb.set_trace()
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
            return cleaned_str
        return py_dict
    except:
        #print(f"错误：无法将内容转换为字典。内容：'{cleaned_str}'，错误：{e}")
        return cleaned_str


def append_to_json(file_path, new_data):
    # 读取已有数据（如果存在），并追加新数据
    # 将合并后的数据保存回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

def clean_func(text: str) -> str:
    # 去掉前后空白符
    assert text is not None
    cleaned = text.strip()
    
    # 删除各种形式的 output 及后续符号 (:、：、- 等)，忽略大小写
    cleaned = re.sub(r'output', '', text, flags=re.IGNORECASE)
    cleaned = re.sub(r'^[\s:：-]+|[\s:：-]+$', '', cleaned)

    # 删除换行符、制表符，再去掉首尾空格
    cleaned = cleaned.replace("\n", "").replace("\t", "").strip()
    
    return cleaned

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="dude/output/sft_expand_ck_6996/val")
    parser.add_argument("--model_name", type=str, default="gemini-2.5-flash")
    args = parser.parse_args()
    dataset_name = args.output_path.split('/')[0]
    folder_path = ''
    llm_extract_function=None
    #import pdb;pdb.set_trace()
    if dataset_name.lower()=='dude':
        folder_path = 'dude'
        from dude import CALL_LLM
    elif dataset_name.lower()=='mpdocvqa':
        folder_path = 'mpdocvqa'
        from mpdocvqa import CALL_LLM
    elif dataset_name.lower()=='longdocurl':
        folder_path = 'LongDocURL'
        from LongDocURL import CALL_LLM,LLM_CALL
        llm_extract_function = LLM_CALL
    elif dataset_name.lower() == 'mmlongbench_doc':
        folder_path = 'mmlongbench_doc'
        from mmlongbench_doc import CALL_LLM,LLM_CALL
        llm_extract_function = LLM_CALL
    elif dataset_name.lower() == 'slidevqa':
        from SlideVQA import CALL_LLM
    else:
        raise NotImplementedError
        
    #save_dir = os.path.join(folder_path,args.save_dir,args.subset)    
    save_name = f'{dataset_name}'
    save_path = os.path.join(args.output_path,f'{save_name}.json')
    data_list = json.load(open(save_path))
    metric_save_path = save_path.replace('.json',f'_{args.model_name}_metric.json')
    #import pdb;pdb.set_trace()
    ## 判断是否调用大模型，如果不调用预处理：从response中提取json，删除output 得到final_answer
    #import pdb;pdb.set_trace()
    if not CALL_LLM:
        for item in data_list:
            #import pdb;pdb.set_trace()
            #item['pred'] = [item['pred']]
            #import pdb;pdb.set_trace()
            try:
                if isinstance(extract_ans(item['pred'][0]),dict):
                    extract_answer = clean_func(extract_ans(item['pred'][0])['answer'])
                    item["extracted_page"] = extract_ans(item['pred'][0])['evidence_pages']
                elif isinstance(extract_ans(item['pred'][0]),str):
                    extract_answer = clean_func(extract_ans(item['pred']))
                    item["extracted_page"]  = extract_ans(item['pred'])['evidence_pages']
                else:
                    extract_answer = item['pred'][0]
                    item["extracted_page"] = [0]
            except:
                extract_answer = item['pred'][0]
                item["extracted_page"] = [0]

            item["extracted_res"] = extract_answer

            
    ## 调用大模型抽取答案
    else:
        assert llm_extract_function is not None
        #import pdb;pdb.set_trace()
        data_list = llm_extract_function(data_list,args.model_name)
        
    ## 保存调用结果
    #import pdb;pdb.set_trace()
    for item in data_list:
        if isinstance(item['extracted_res'],complex):
            item['extracted_res'] = str(item['extracted_res'])
    append_to_json(metric_save_path, data_list)

if __name__=='__main__':
    main()