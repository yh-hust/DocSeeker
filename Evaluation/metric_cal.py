import json
import os
import argparse
import re
from typing import Set, Dict, Any

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="dude/output/sft_expand_ck_6996/val")
    parser.add_argument("--model_name", type=str, default="gemini-2.5-flash")
    args = parser.parse_args()
    dataset_name = args.output_path.split('/')[0]
    folder_path = ''
    llm_extract_function=None
    if dataset_name.lower() =='dude':
        folder_path = 'dude'
        from dude import score_cal_and_save
        # 不支持 test数据集指标计算
        #assert args.output_path.split('/')[-1] != 'test'
    elif dataset_name.lower() =='mpdocvqa':
        folder_path = 'mpdocvqa'
        from mpdocvqa import score_cal_and_save
    elif dataset_name.lower() == 'longdocurl':
        folder_path = 'LongDocURL'
        from LongDocURL import score_cal_and_save
    elif dataset_name.lower() == 'mmlongbench_doc':
        from mmlongbench_doc import score_cal_and_save
    elif dataset_name.lower() == 'slidevqa':
        from SlideVQA import score_cal_and_save
    else:
        raise NotImplementedError
    
    save_name = f'{dataset_name}'
    save_path = os.path.join(args.output_path,f'{save_name}_{args.model_name}_metric.json')
    data_list = json.load(open(save_path))
    score_cal_and_save(data_list,os.path.join(args.output_path,f'{save_name}_{args.model_name}_acc.json'))
   

if __name__=='__main__':
    main()