# -*- coding: utf-8 -*-

"""
@Time    : 2023/4/2 11:28 下午
@Author  : hcai
@Email   : hua.cai@unidt.com
"""
import os
import argparse
import json
import random

from tqdm import tqdm

# file_root = os.path.dirname(__file__)
file_root = "/data/renma/unigpt/"

def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        # context += f"Input: {example['input']}\n"
        input_prompt = example['input'].strip("输入：")
        context += f"{input_prompt}\n"
    context += "Answer: "
    target = example["output"]
    return {"context": context, "target": target}


def get_random_candidates(choices, target):
    cand = random.sample(choices, random.randint(4,8))
    cand.append(target)
    random.shuffle(cand)
    return list(set(cand))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, 
                        default=os.path.join(file_root, "GPT-4-LLM/data/alpaca_gpt4_data_zh.json"))
    parser.add_argument("--save_path", type=str, 
                        default=os.path.join(file_root, "GPT-4-LLM/data/alpaca_gpt4_data_zh.jsonl"))

    args = parser.parse_args()
    with open(args.data_path, 'r', encoding='utf-8') as f:
        if 'belle' in args.data_path:
            examples = [json.loads(i) for i in f.readlines()]
        elif 'pclue' in args.data_path:
            examples = []
            lines = f.readlines()
            for l in lines:
                d = json.loads(l)
                clean = {'instruction': d['input'], 'output': d['target']}
                if d.get('answer_choices', None) is not None and len(d['answer_choices']) > 8:
                    text1 = '，'.join(d['answer_choices'])
                    text2 = ','.join(d['answer_choices'])
                    cand = get_random_candidates(d['answer_choices'], d['target'])
                    if random.random() < 0.5:
                        cand_text = '，'.join(cand)
                    else:
                        cand_text = ','.join(cand)
                    try:
                        clean_input = d['input'].replace(text1, cand_text)
                    except:
                        clean_input = d['input'].replace(text2, cand_text)
                    clean['instruction'] = clean_input
                examples.append(clean)
        elif 'firefly' in args.data_path:
            examples = []
            lines = f.readlines()
            for l in lines:
                d = json.loads(l)
                clean = {'instruction': d['input'], 'output': d['target']}
                examples.append(clean)
        else:
            examples = json.load(f)

    with open(args.save_path, 'w') as f:
        for example in tqdm(examples, desc="formatting.."):
            f.write(json.dumps(format_example(example), ensure_ascii=False) + '\n')


if __name__ == "__main__":
    main()
