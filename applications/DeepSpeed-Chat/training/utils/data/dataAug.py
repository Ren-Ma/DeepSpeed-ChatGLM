import os
import re
import sys
import json
import argparse
import numpy as np
import pandas as pd
import requests
import time
from rich import print
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
CHAT_URL = "your url"

def _generate_sequence(self, prompts, mask):

    max_min_length = 256 + prompts.shape[1]
    min_length = 0
    with torch.no_grad():
        # ? 这一步很慢而且GPU利用率很低，是因为autoregressive生成？
        seq = self.generate(
            prompts,
            attention_mask=mask,
            max_length=max_min_length,
            min_length=min_length) 
        # ! 好像这一步会生成256长度的回复，且后面的不是padding的。。。。是硬生成的
    batch_size = seq.shape[0]
    prompt_length = prompts.shape[1]
    ans = seq[:, prompt_length:]
    self.prompt_length = prompt_length
    valid_ans_len = (ans != 3).sum(dim=-1)
    out_seq = []
    for i in range(batch_size):
        if valid_ans_len[i] <= 1:  # if the answer is shorter than 1 token, drop it
            continue
        else:
            out_seq.append(seq[i:i + 1])
    out_seq = torch.cat(out_seq, dim=0)  # concate output in the batch dim

    return out_seq

class data_aug(object):
    def __init__(self):
        self.random_seed = 2023
        # self.max_length = 2048
        # self.num_beams = 5
        self.num_responses = 1
        # self.temperature = 1.5
        # self.do_sample = True
        # self.top_p = 0.95

    def process_response(self, inputs_ids, outputs):
        """来自modelling_chatglm.py，对model.generate()出来的tokens进行解码和后处理

        Args:
            inputs_ids (Tensor): batch of input_ids
            outputs (Tensor): batch of output ids

        Returns:
            list[]: list of responses
        """
        punkts = [
            [",", "，"],
            ["!", "！"],
            [":", "："],
            [";", "；"],
            ["\?", "？"],
        ]
        responses = []
        for i in range(len(inputs_ids)):
            output = outputs[i].tolist()[len(inputs_ids[i]):] # 把生成的outputs前面属于prompt的部分截掉
            response = tokenizer.decode(output)
            response = response.strip()
            response = response.replace("[[训练时间]]", "2023年")
            for item in punkts:
                response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
                response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
            responses.append(response)
        return responses
        
    def get_response_api(self, prompt):
        """ 从chatglm api获得回复

        Args:
            prompt (String): 输入的prompt，单个而非batch

        Returns:
            String: chatglm对输入prompt的回复
        """
        try:
            chat_res = requests.post(url=CHAT_URL, json={"message":prompt})
            if (chat_res.status_code == 200) and (chat_res.json()['status']):
                response = chat_res.json()['results'] 
            else:
                response = "ChatGLM接口出错！"
                print(response)
        except Exception as e:
            print(e)
            response = "ChatGLM接口出错！"
            print(response)
        return response
    
    def get_response_local(self, prompts):
        """ 从本地启动的chatglm模型获得回复。
        为了防止推理过程中显卡OOM中断进程，在主GPU和备用GPU上各启动了一个模型，当主GPU显存OOM后，
        此batch二分为两个部分，切换到备用GPU上继续进行推理，一般在备用GPU上推理3-5次后，主GPU的显存才会清除，
        此时主GPU和备用GPU角色互换，备用GPU作为主GPU进行推理，主GPU变为备用。二者之间的切换通过flag变量
        is_cuda_oom来控制。

        Args:
            prompts (List): list of prompts

        Returns:
            list[]: list of responses
        """
        global is_cuda_oom
        if len(prompts) == 0:
          return
        try:
            inputs = tokenizer(prompts, padding=True, return_tensors="pt")
            if is_cuda_oom == True:
                inputs = inputs.to("cuda:3")
                outputs = model3.generate(
                **inputs,
                max_length=2048, 
                top_p=0.7, 
                temperature=0.95, 
                num_beams=1,
                do_sample=True)
            else:
                inputs = inputs.to(device)
                outputs = model0.generate(
                    **inputs,
                    max_length=2048, 
                    top_p=0.7, 
                    temperature=0.95, 
                    num_beams=1,
                    do_sample=True)
            torch.cuda.empty_cache()
            responses = self.process_response(inputs["input_ids"], outputs)
        except Exception as e:
            is_cuda_oom = not is_cuda_oom
            print("ChatGLM接口出错！\n报错：" + str(e))
            half_len = len(prompts) // 2
            left = self.get_response_local(prompts[:half_len]) # 前半部分
            right = self.get_response_local(prompts[half_len:]) # 后半部分
            responses = left + right
        return responses
    
    def load_data(self, path=None, part=None):
        """加载原始数据

        Args:
            path (string, optional): 原始数据路径. Defaults to None.
            part (string, optional): 每个GPU对应的数据部分. Defaults to None.

        Returns:
            datatframe: 加载原始数据后，以dataframe形式返回
        """
        # df = pd.read_json(path, lines=True)
        # df = df.sample(frac=1, random_state=self.random_seed)
        # df = df[9990:]
        # df.reset_index(drop=True, inplace=True)
        filename = "GPT-4-LLM/data/reward_data/{}.pkl".format(part)
        df = pd.read_pickle(filename)
        return df
    
    def main(self, path=None, part=None, repair=False):
        """
            数据扩增主函数
        Args:
            path (string, optional): 原始数据路径. Defaults to None.
            part (string, optional): 每个GPU对应的数据部分. Defaults to None.
            repair (bool): 是否为补救生成，即对上次生成失败的prompts重新生成回复。Defaults to False.
        """
        if repair:
            df = pd.read_csv("GPT-4-LLM/data/reward_data/{}.csv".format(part))
            df_keep = df[~df.chatglm.str.contains("ChatGLM接口出错！")]
            df = df[df.chatglm.str.contains("ChatGLM接口出错！")]
        else:
            # df = self.load_data(path)
            df = self.load_data(part=part)
        # df = pd.read_excel("华院计算FAQ0428.xlsx")
        df = pd.read_excel("关于unichat.xlsx")
        # df.columns = ["id", "context", "target"]
        df["context"] = df["context"].apply(lambda x: "Instruction: {} \nAnswer: ".format(x))
        num = df.shape[0]
        bs = 16
        steps = num // bs
        # df = df[:steps*bs]
        prompts_lst = df["context"].to_list()
        responses_lst = []
        for i in tqdm(range(steps)):
            prompts = prompts_lst[i*bs : (i+1)*bs]
            responses = self.get_response_local(prompts)
            responses_lst.extend(responses)
            if i != 0 and i % 50 == 0:
                tmp = df[:i+1]
                tmp['chatglm'] = responses_lst
                tmp.to_csv("GPT-4-LLM/data/reward_data/{}.csv".format(part, i), index=False)
        if df.shape[0] != steps*bs:
            prompts = prompts_lst[steps*bs:]
            responses = self.get_response_local(prompts)
            responses_lst.extend(responses)
        if df.shape[0] != len(responses_lst):
            print("responses_lst长度和df不匹配！")
            df = df[:len(responses_lst)]
        df['chatglm'] = responses_lst
        if repair:
            df = pd.concat([df, df_keep])
            df = df.sort_index()
        df.to_csv("GPT-4-LLM/data/reward_data/{}.csv".format(part), index=False)
        return

if __name__ == "__main__":
    model_path = "/data/renma/unigpt/chatglm-6b"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    my_map = {
        "0": "df9990_2w",
        "1": "df2_3w",
        "2": "df3_4w",
        "3": "df4w_",
        }
    data_aug = data_aug()
    prompt = "美国的首都是哪里？"
    prompt_tok = tokenizer(prompt, max_length=512, padding="max_length", truncation=True, return_tensors="pt").to("cuda:2")
    model3 = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda(2)
    seq = _generate_sequence(model3, prompt_tok['input_ids'], prompt_tok['attention_mask'])
    st = time.time()
    # path = "belle/Belle_open_source_0.5M.jsonl"
    path = "GPT-4-LLM/data/alpaca_gpt4_data_zh.jsonl"
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--device_id",
    #     type=str,
    #     help= "which gpu to use",
    #     required=True,
    #     default="0",
    #     )
    # args = parser.parse_args()
    # device_id = sys.argv[1]
    device_id = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model0 = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda(device_id)
    part = my_map.get(str(device_id))
    is_cuda_oom = False
    data_aug.main(path, part)
    print(time.time() - st)
    print("完成！")


