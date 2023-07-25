# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Part of the code was adopted from https://github.com/microsoft/Megatron-DeepSpeed/blob/main/megatron/data/dataset_utils.py
"""
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from datasets import load_dataset
import numpy as np
import os
import hashlib
from itertools import chain
from . import raw_datasets
from tqdm import tqdm


def get_raw_dataset(dataset_name, output_path, seed, local_rank):

    if "Dahoas/rm-static" in dataset_name:
        return raw_datasets.DahoasRmstaticDataset(output_path, seed,
                                                  local_rank, dataset_name)
    elif "GPT-4-LLM" in dataset_name:
        return raw_datasets.gpt4llm(output_path, seed, local_rank, dataset_name)
    elif "Dahoas/full-hh-rlhf" in dataset_name:
        return raw_datasets.DahoasFullhhrlhfDataset(output_path, seed,
                                                    local_rank, dataset_name)
    elif "Dahoas/synthetic-instruct-gptj-pairwise" in dataset_name:
        return raw_datasets.DahoasSyntheticinstructgptjpairwiseDataset(
            output_path, seed, local_rank, dataset_name)
    elif "yitingxie/rlhf-reward-datasets" in dataset_name:
        return raw_datasets.YitingxieRlhfrewarddatasetsDataset(
            output_path, seed, local_rank, dataset_name)
    elif "openai/webgpt_comparisons" in dataset_name:
        return raw_datasets.OpenaiWebgptcomparisonsDataset(
            output_path, seed, local_rank, dataset_name)
    elif "stanfordnlp/SHP" in dataset_name:
        return raw_datasets.StanfordnlpSHPDataset(output_path, seed,
                                                  local_rank, dataset_name)
    elif "wangrui6/Zhihu-KOL" in dataset_name:
        return raw_datasets.Wangrui6ZhihuKOLDataset(output_path, seed,
                                                    local_rank, dataset_name)
    elif "Cohere/miracl-zh-queries-22-12" in dataset_name:
        return raw_datasets.CohereMiraclzhqueries2212Dataset(
            output_path, seed, local_rank, dataset_name)
    elif "Hello-SimpleAI/HC3-Chinese" in dataset_name:
        return raw_datasets.HelloSimpleAIHC3ChineseDataset(
            output_path, seed, local_rank, dataset_name)
    elif "mkqa-Chinese" in dataset_name:
        return raw_datasets.MkqaChineseDataset(output_path, seed, local_rank,
                                               dataset_name)
    elif "mkqa-Japanese" in dataset_name:
        return raw_datasets.MkqaJapaneseDataset(output_path, seed, local_rank,
                                                dataset_name)
    elif "Cohere/miracl-ja-queries-22-12" in dataset_name:
        return raw_datasets.CohereMiracljaqueries2212Dataset(
            output_path, seed, local_rank, dataset_name)
    elif "lmqg/qg_jaquad" in dataset_name:
        return raw_datasets.LmqgQgjaquadDataset(output_path, seed, local_rank,
                                                dataset_name)
    elif "lmqg/qag_jaquad" in dataset_name:
        return raw_datasets.LmqgQagjaquadDataset(output_path, seed, local_rank,
                                                 dataset_name)
    else:
        raise RuntimeError(
            f"We do not have configs for dataset {dataset_name}, but you can add it by yourself in raw_datasets.py."
        )


def get_shuffle_idx(seed, size):
    np_rng = np.random.RandomState(seed=seed)
    dtype_ = np.uint32
    if size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64
    shuffle_idx = np.arange(start=0, stop=size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx)
    return shuffle_idx


def get_raw_dataset_split_index(local_rank, output_path, dataset_name, seed,
                                split_name, data_split, split_index,
                                data_size):
    index_file_name = f"{output_path}/{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_index}.npy"
    if not os.path.isfile(index_file_name):
        splits = [float(s) for s in data_split.split(',')]
        splits_sum = sum(splits)
        splits = [split / splits_sum for split in splits]
        splits_index = [0]
        for index, split in enumerate(splits):
            splits_index.append(splits_index[index] +
                                int(round(split * float(data_size))))
        diff = splits_index[-1] - data_size
        for index in range(1, len(splits_index)):
            splits_index[index] -= diff
        assert splits_index[-1] == data_size

        shuffle_idx = get_shuffle_idx(seed, data_size)
        for split_i in range(len(splits)):
            shuffle_idx_split_file_name = f"{output_path}/{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_i}.npy"
            shuffle_idx_split = shuffle_idx[
                splits_index[split_i]:splits_index[split_i + 1]]
            np.save(shuffle_idx_split_file_name,
                    shuffle_idx_split,
                    allow_pickle=True)
    index = np.load(index_file_name, allow_pickle=True)
    return index.tolist()

def get_masks(input_ids, pad_token_id):
    if input_ids.dim() == 2 and input_ids.shape[0] == 1:
        input_ids = input_ids.squeeze()
    first_nonPadding_idx = (input_ids != pad_token_id).nonzero()[0].item()
    input_ids_noPadding = input_ids[first_nonPadding_idx:]
    seq_length = len(input_ids_noPadding)
    bos_token_id = 130004  # chatglm tokenizer.bos_token_id
    context_length = input_ids_noPadding.tolist().index(bos_token_id)
    attention_mask = np.ones((1, seq_length, seq_length))
    attention_mask = np.tril(attention_mask)
    attention_mask[:, :, :context_length] = 1
    attention_mask = np.bool_(attention_mask < 0.5)
    difference = first_nonPadding_idx
    attention_mask = np.pad(
        attention_mask,
        pad_width=[(0, 0), (difference, 0), (difference, 0)],
        mode='constant', constant_values=True)
    attention_mask = np.expand_dims(attention_mask, 1)
    return torch.tensor(attention_mask)

class PromptDataset(Dataset):

    def __init__(self, prompt_dataset, chosen_dataset, reject_dataset,
                 pad_token_id, train_phase) -> None:
        super().__init__()
        self.prompt_dataset = prompt_dataset
        self.chosen_dataset = chosen_dataset
        self.reject_dataset = reject_dataset
        self.pad_token_id = pad_token_id
        self.train_phase = train_phase

    def __len__(self):
        length = len(self.chosen_dataset)
        if self.train_phase == 3:
            length = len(self.prompt_dataset)
        return length

    def __getitem__(self, idx):
        if self.train_phase == 1:
            return {
                "input_ids": self.chosen_dataset[idx]["input_ids"],
                "attention_mask": self.chosen_dataset[idx]["attention_mask"],
                "labels": self.chosen_dataset[idx]["input_ids"]
            }
        elif self.train_phase == 2:
            # return self.chosen_dataset[idx]["input_ids"], self.chosen_dataset[idx]["attention_mask"], \
            #     self.reject_dataset[idx]["input_ids"], self.reject_dataset[idx]["attention_mask"]
            max_seq_len = 1536
            return self.chosen_dataset[idx][:, -max_seq_len:], \
                    get_masks(self.chosen_dataset[idx][:, -max_seq_len:], self.pad_token_id), \
                    self.reject_dataset[idx][:, -max_seq_len:], \
                    get_masks(self.reject_dataset[idx][:, -max_seq_len:], self.pad_token_id)            
        elif self.train_phase == 3:
            max_prompt_seq_len = 512
            # return self.prompt_dataset[idx]["input_ids"],self.prompt_dataset[idx]["attention_mask"], self.pad_token_id
            return self.prompt_dataset[idx][: max_prompt_seq_len], \
                    get_masks(self.prompt_dataset[idx][: max_prompt_seq_len].flip(0), self.pad_token_id).squeeze(0), \
                    self.pad_token_id

def create_dataset_split_fast(current_dataset, raw_dataset, train_phase, tokenizer,
                         end_of_conversation_token, max_seq_len):
    """ create_dataset_split的加速版，用dataset.map()来进行多进程tokenizer，
    只存储input_ids，attention_mask在PromptDataset.__getitem__()时再根据input_ids生成 """
    prompt_dataset = []
    chosen_dataset = []
    reject_dataset = []
    def get_chosen_reject(examples):
        chosen_dataset = []
        reject_dataset = []
        examples["chosen_sentences"] = [prompt + chosen for prompt, chosen in 
                            zip(examples["prompt"], examples["chosen"])]
        examples["rejected_sentences"] = [prompt + chosen for prompt, chosen in 
                            zip(examples["prompt"], examples["rejected"])]
        for chosen, reject in tqdm(zip(examples["chosen_sentences"], examples["rejected_sentences"])):
            chosen_input_ids = tokenizer(chosen,
                                    max_length=max_seq_len,
                                    padding="max_length",
                                    truncation=True,
                                    return_tensors="pt")
            # tmp = get_masks(chosen_token["input_ids"])
            reject_input_ids = tokenizer(reject,
                                    max_length=max_seq_len,
                                    padding="max_length",   
                                    truncation=True,
                                    return_tensors="pt")
            chosen_dataset.append(chosen_input_ids["input_ids"])
            reject_dataset.append(reject_input_ids["input_ids"])
        return {"chosen_dataset": chosen_dataset, "reject_dataset": reject_dataset}
    
    def get_ppo_data(examples):
        prompt_dataset = [] 
        prompts = examples["prompt"]
        for prompt in tqdm(prompts):
            prompt_token = tokenizer(prompt,
                                     max_length=max_seq_len,
                                     padding="max_length",
                                     return_tensors="pt")
            flipped_prompt_inputids = prompt_token["input_ids"].squeeze(0).flip(0) # 翻转input_ids
            prompt_token["input_ids"] = flipped_prompt_inputids
            prompt_dataset.append(prompt_token["input_ids"])
        return {"prompt_dataset": prompt_dataset}
    
    if train_phase == 2:
        output = current_dataset.map(get_chosen_reject, batched=True, num_proc=48)
        chosen_dataset = torch.tensor(output["chosen_dataset"])
        reject_dataset = torch.tensor(output["reject_dataset"])
    elif train_phase == 3:
        output = current_dataset.map(get_ppo_data, batched=True, num_proc=48)
        prompt_dataset = torch.tensor(output["prompt_dataset"]) # [num, max_prompt_seq_len]
    return PromptDataset(prompt_dataset, chosen_dataset, reject_dataset,
                        tokenizer.pad_token_id, train_phase)

def create_dataset_split(current_dataset, raw_dataset, train_phase, tokenizer,
                         end_of_conversation_token, max_seq_len):
    prompt_dataset = []
    chosen_dataset = []
    reject_dataset = []
    if train_phase == 1:
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            chosen_sentence = raw_dataset.get_prompt_and_chosen(
                tmp_data)  # the accept response
            if chosen_sentence is not None:
                chosen_sentence += end_of_conversation_token
                chosen_token = tokenizer(chosen_sentence,
                                         max_length=max_seq_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
                chosen_token["input_ids"] = chosen_token["input_ids"].squeeze(0)
                chosen_token["attention_mask"] = chosen_token["attention_mask"].squeeze(0)
                chosen_dataset.append(chosen_token)

    elif train_phase == 2:
        for i, tmp_data in tqdm(enumerate(current_dataset)):
            # tokenize the text
            chosen_sentence = raw_dataset.get_prompt_and_chosen(tmp_data)  # the accept response
            reject_sentence = raw_dataset.get_prompt_and_rejected(tmp_data)  # the accept response
            if chosen_sentence is not None and reject_sentence is not None:
                chosen_sentence += end_of_conversation_token  # the accept response
                reject_sentence += end_of_conversation_token
                chosen_token = tokenizer(chosen_sentence,
                                         max_length=max_seq_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
                reject_token = tokenizer(reject_sentence,
                                         max_length=max_seq_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
                chosen_token["input_ids"] = chosen_token["input_ids"]
                chosen_token["attention_mask"] = chosen_token["attention_mask"]
                chosen_dataset.append(chosen_token)

                reject_token["input_ids"] = reject_token["input_ids"]
                reject_token["attention_mask"] = reject_token["attention_mask"]
                reject_dataset.append(reject_token)

    elif train_phase == 3:
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            prompt = raw_dataset.get_prompt(tmp_data)
            if prompt is not None:
                prompt_token = tokenizer(prompt,
                                         max_length=max_seq_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
                prompt_token["input_ids"] = prompt_token["input_ids"]
                prompt_token["attention_mask"] = prompt_token["attention_mask"]
                for key_word in ["input_ids", "attention_mask"]:
                    length = prompt_token[key_word].size()[-1]
                    if length > max_seq_len:
                        y = prompt_token[key_word].squeeze(0)[length - (max_seq_len - 1):].flip(0)
                    else: # input_ids [1, max_length], attention_mask [1, 1, max_length, max_length]
                        y = prompt_token[key_word].squeeze(0).flip(0) # input_ids反转，attention_mask未反转
                    prompt_token[key_word] = y
                prompt_dataset.append(prompt_token)
    return PromptDataset(prompt_dataset, chosen_dataset, reject_dataset,
                         tokenizer.pad_token_id, train_phase)


def create_dataset(local_rank, dataset_name, data_split, output_path,
                   train_phase, seed, tokenizer, end_of_conversation_token,
                   max_seq_len):
    raw_dataset = get_raw_dataset(dataset_name, output_path, seed, local_rank)
    train_dataset = raw_dataset.get_train_data()
    train_index = get_raw_dataset_split_index(local_rank, output_path,
                                              raw_dataset.dataset_name_clean,
                                              seed, "train", data_split,
                                              train_phase - 1,
                                              len(train_dataset))
    # train_dataset = Subset(train_dataset, train_index)
    train_dataset = train_dataset.select(train_index)
    # train_dataset = create_dataset_split(train_dataset, raw_dataset,
    train_dataset = create_dataset_split_fast(train_dataset, raw_dataset,
                                         train_phase, tokenizer,
                                         end_of_conversation_token,
                                         max_seq_len)

    eval_dataset = raw_dataset.get_eval_data()
    eval_index = get_raw_dataset_split_index(local_rank, output_path,
                                             raw_dataset.dataset_name_clean,
                                             seed, "eval",
                                             data_split, train_phase - 1,
                                             len(eval_dataset))
    # eval_dataset = Subset(eval_dataset, eval_index)
    eval_dataset = eval_dataset.select(eval_index)
    # eval_dataset = create_dataset_split(eval_dataset, raw_dataset, train_phase,
    eval_dataset = create_dataset_split_fast(eval_dataset, raw_dataset, train_phase,
                                        tokenizer, end_of_conversation_token,
                                        max_seq_len)
    return train_dataset, eval_dataset


def create_prompt_dataset(local_rank,
                          data_path,
                          data_split,
                          output_path,
                          train_phase,
                          seed,
                          tokenizer,
                          max_seq_len,
                          end_of_conversation_token="<|endoftext|>",
                          sft_only_data_path=[]):
    """
    Creates the prompt dataset
    """
    os.makedirs(output_path, exist_ok=True)
    fname = "_".join(data_path)
    sft_cache_key = "_".join(sft_only_data_path)
    tokenizer_name = tokenizer.init_kwargs["name_or_path"].replace("/", "_")
    fname = f"{fname}_split{data_split}_phase{train_phase}_seed{seed}_tokenizer{tokenizer_name}_seqlen{max_seq_len}_sft{sft_cache_key}"
    fname = "_".join(fname.split("/"))
    # 相同fname每次python重启后的hash值不一样，导致train_fname和之前save的不一致
    fname = hashlib.sha256(fname.encode()).hexdigest()  # hash the file name to avoid too long file name
    # train_fname = f"{output_path}/traindata_{fname}.pt"
    # eval_fname = f"{output_path}/evaldata_{fname}.pt"
    if train_phase == 2:
        train_fname = f"{output_path}/train_unidt_FAQ_len2048.pt"
        eval_fname = f"{output_path}/eval_unidt_FAQ_len2048.pt"
    elif train_phase == 3:
        # train_fname = f"{output_path}/traindata_unichat_len512_num44181.pt"
        # eval_fname = f"{output_path}/evaldata_unichat_len512_num4910.pt"
        train_fname = f"{output_path}/train_unichat_num212_len512.pt"
        eval_fname = f"{output_path}/eval_unichat_num212_len512.pt"
    cache_found = os.path.isfile(train_fname) and os.path.isfile(eval_fname)
    buf_create_cache = torch.ByteTensor([not cache_found]).cuda()
    torch.distributed.all_reduce(buf_create_cache)
    print(f"\n------寻找{train_fname}文件")
    if local_rank <= 0 and buf_create_cache.item() != 0:
        if len(data_path) == 1:  # Single dataset.
            train_dataset, eval_dataset = create_dataset(
                local_rank, data_path[0], data_split, output_path, train_phase,
                seed, tokenizer, end_of_conversation_token, max_seq_len)
        else:  # Blending datasets.
            train_datasets = []
            eval_datasets = []
            train_size = 0
            eval_size = 0
            for d_path in data_path:
                train_dataset, eval_dataset = create_dataset(
                    local_rank, d_path, data_split, output_path, train_phase,
                    seed, tokenizer, end_of_conversation_token, max_seq_len)
                train_datasets.append(train_dataset)
                eval_datasets.append(eval_dataset)
                train_size += len(train_dataset)
                eval_size += len(eval_dataset)
            train_dataset = ConcatDataset(train_datasets)
            shuffle_idx = get_shuffle_idx(seed, train_size)
            train_dataset = Subset(train_dataset, shuffle_idx.tolist())
            eval_dataset = ConcatDataset(eval_datasets)
            shuffle_idx = get_shuffle_idx(seed, eval_size)
            eval_dataset = Subset(eval_dataset, shuffle_idx.tolist())

        # Append the SFT-only dataset if it exists, and current phase is 1(SFT).
        if train_phase == 1 and sft_only_data_path:
            sft_train_datasets = []
            sft_eval_datasets = []
            sft_train_size = 0
            sft_eval_size = 0
            for sft_path in sft_only_data_path:
                sft_train_dataset, sft_eval_dataset = create_dataset(
                    local_rank,
                    sft_path,
                    "10,0,0",
                    output_path,
                    train_phase,
                    seed,
                    tokenizer,
                    end_of_conversation_token,
                    max_seq_len,
                )
                sft_train_datasets.append(sft_train_dataset)
                sft_eval_datasets.append(sft_eval_dataset)
                sft_train_size += len(sft_train_dataset)
                sft_eval_size += len(sft_eval_dataset)
            if sft_train_datasets:  # Check if sft_train_datasets is not empty
                sft_train_dataset = ConcatDataset(sft_train_datasets)
                train_dataset = ConcatDataset(
                    [train_dataset, sft_train_dataset])
                shuffle_idx = get_shuffle_idx(seed, len(train_dataset))
                train_dataset = Subset(train_dataset, shuffle_idx.tolist())
            if sft_eval_datasets:  # Check if sft_eval_datasets is not empty
                sft_eval_dataset = ConcatDataset(sft_eval_datasets)
                eval_dataset = ConcatDataset([eval_dataset, sft_eval_dataset])
                shuffle_idx = get_shuffle_idx(seed, len(eval_dataset))
                eval_dataset = Subset(eval_dataset, shuffle_idx.tolist())
        torch.save(train_dataset, train_fname)
        torch.save(eval_dataset, eval_fname)
    print(f"\n------找到了文件，直接返回！！")
    torch.distributed.barrier()
    return torch.load(train_fname), torch.load(eval_fname)


class DataCollatorReward:

    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.cat([f[0]
                                        for f in data] + [f[2] for f in data],
                                       dim=0)
        batch["attention_mask"] = torch.cat([f[1] for f in data] +
                                            [f[3] for f in data],
                                            dim=0)
        return batch


class DataCollatorRLHF:

    def __init__(self, max_token_len, inference_tp_size):
        self.max_token_len = max_token_len
        self.inference_tp_size = inference_tp_size

    def __call__(self, data):
        batch = {}
        pad_token_id = data[-1][-1]
        # 按照一个batch中input_ids中最长的那个的长度padding整个batch
        prompt = pad_sequence([f[0] for f in data],
                              padding_value=pad_token_id,
                              batch_first=True)
        # 按照一个batch中attention_mask中最长的那个的长度padding整个batch
        prompt_mask = pad_sequence([f[1] for f in data],
                                   padding_value=0,
                                   batch_first=True)
        # 以上两步在成batch的input_ids和attention_mask中都没影响，因为预处理后的batch数据已经是按照max_seq_len padding的
        ### make sure the final ouput is a seqence of 2**?
        length = prompt.size()[-1]
        pad_length = self.max_token_len - length
        if pad_length > 0:
            batch["prompt"] = F.pad(prompt,  # 把prompt的input_ids填充到args.max_prompt_seq_len长度， e.g., [1,512] -> [1,1536]
                                    pad=(0, pad_length),
                                    mode='constant',
                                    value=pad_token_id)
            batch["prompt_att_mask"] = F.pad(prompt_mask,
                                             pad=(0, pad_length, 
                                                  0, pad_length),
                                             mode='constant',
                                             value=0)
        else:
            batch["prompt"] = prompt
            batch["prompt_att_mask"] = prompt_mask
        # 此处再次反转回来, 对应create_dataset_split阶段3的flip(0)
        batch["prompt"] = batch["prompt"].flip(1)
         # attention_mask未变化，因为shape为[bsz, 1, max_seq_len, max_seq_len]
        batch["prompt_att_mask"] = batch["prompt_att_mask"].flip(1)
        return batch


def get_unsupervised_data(args, tokenizer):
    unsupervised_raw_datasets = load_dataset(
        args.unsupervised_dataset_name, args.unsupervised_dataset_config_name)
    column_names = unsupervised_raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = unsupervised_raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    block_size = args.max_prompt_seq_len + args.max_answer_seq_len

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: list(chain(*examples[k]))
            for k in examples.keys()
        }
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k:
            [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    train_dataset = lm_datasets["train"]

    return train_dataset


class MiniDataset:

    def __init__(self, max_size, small_batch_size):
        self.dataset = []
        self.max_size = max_size
        self.small_batch_size = small_batch_size

    def seperate(self):
        small_dataset = []
        for large_batch in self.dataset:
            if type(large_batch) == list or type(large_batch) == tuple:
                large_size = len(large_batch[0])
            elif type(large_batch) == dict:
                large_size = len(large_batch[list(large_batch.keys())[0]])
            else:
                large_size = len(large_batch)
            for i in range(0, large_size, self.small_batch_size):
                if type(large_batch) == list or type(large_batch) == tuple:
                    small_dataset.append(
                        [x[i:i + self.small_batch_size] for x in large_batch])
                elif type(large_batch) == dict:
                    small_dataset.append({
                        k: v[i:i + self.small_batch_size]
                        for k, v in large_batch.items()
                    })
                else:
                    small_dataset.append(large_batch[i:i +
                                                     self.small_batch_size])
        self.free()

        return small_dataset

    def add(self, data):
        if len(self.dataset) < self.max_size:
            self.dataset.append(data)
            if len(self.dataset) == self.max_size:
                return self.seperate()
            else:
                return None
        else:
            raise ValueError(
                "The dataset is full but we did not stop it. There is a bug in the code."
            )

    def free(self):
        self.dataset = []
