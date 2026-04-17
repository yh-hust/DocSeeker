import os
import json
import re
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge ALR data from exact and semantic matches.")
    parser.add_argument(
        "--folder_path", 
        type=str, 
        required=True, 
        help="The directory path containing exact_match.json and semantic_match.json"
    )
    args = parser.parse_args()

    folder = Path(args.folder_path)
    exact_path = folder / "exact_match.json"
    semantic_path = folder / "semantic_match.json"
    output_path = folder / "ALR_data.json"

    if not exact_path.exists() or not semantic_path.exists():
        print(f"Error: Source files not found. Please ensure the following paths exist:\n  - {exact_path}\n  - {semantic_path}")
        exit(1)

    print(f"Reading target directory: {folder}")
    with open(exact_path, 'r', encoding='utf-8') as f:
        exact_data = json.load(f)
        
    with open(semantic_path, 'r', encoding='utf-8') as f:
        semantic_data = json.load(f)

    final_alr_data = []

    for item in exact_data:
        item['conversations'][1]['value'] = item.get('expand_answer', '')
        final_alr_data.append(item)

    valid_semantic_count = 0
    error_count = 0
    
    for item in semantic_data:
        final_ans = item.get('final_answer', '')

        if 'error' in final_ans.lower():
            error_count += 1
            continue

        gt_ans = item['conversations'][1]['value']
        expand_ans = item.get('expand_answer', '')
        
        ans_match = re.search(r"<answer>(.*?)</answer>", expand_ans, re.DOTALL)
        if ans_match:
            try:
                ans_str = ans_match.group(1).strip()
                ans_dict = json.loads(ans_str)

                ans_dict['answer'] = gt_ans
                new_ans_str = json.dumps(ans_dict, ensure_ascii=False)
                
                new_expand_ans = expand_ans.replace(ans_match.group(1), f"\n{new_ans_str}\n")
                
                item['conversations'][1]['value'] = new_expand_ans

                item.pop('expand_answer', None)
                item.pop('final_answer', None)

                final_alr_data.append(item)
                valid_semantic_count += 1
                
            except json.JSONDecodeError:
                print(f"Warning: <answer> block for ID {item.get('id')} is not a valid JSON format. Skipped.")
                continue
        else:
            print(f"Warning: No <answer> tag matched for ID {item.get('id')}.")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_alr_data, f, ensure_ascii=False, indent=4)

    print("==================================================")
    print("ALR Data Processing Completed!")
    print(f"Exact matches processed   : {len(exact_data)}")
    print(f"Semantic errors removed   : {error_count}")
    print(f"Semantic matches aligned  : {valid_semantic_count}")
    print(f"Total ALR data generated  : {len(final_alr_data)}")
    print(f"Saved to                  : {output_path}")
    print("==================================================")