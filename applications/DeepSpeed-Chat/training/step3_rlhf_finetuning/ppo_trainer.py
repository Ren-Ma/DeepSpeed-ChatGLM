# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import re
import pandas as pd
import torch
import torch.nn.functional as F
import sys
import os
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
# from utils.data.data_utils import get_masks # todo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from utils.utils import print_rank_0


def print_all_ranks(tag, value, rank):
    world_size = torch.distributed.get_world_size()
    all_tensor = torch.zeros(world_size, dtype=torch.float32).cuda()
    all_tensor[rank] = value
    torch.distributed.all_reduce(all_tensor, op=torch.distributed.ReduceOp.SUM)
    print_rank_0(f'{tag} {all_tensor}', rank)


def get_model_norm(model):
    with torch.no_grad():
        total = 0.0
        for param in model.parameters():
            should_gather = hasattr(
                param,
                'ds_id') and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
            with deepspeed.zero.GatheredParameters(param,
                                                   enabled=should_gather):
                total += float(param.float().norm())

    return total


def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


class DeepSpeedPPOTrainer():

    def __init__(self, rlhf_engine, args):
        self.rlhf_engine = rlhf_engine
        self.actor_model = self.rlhf_engine.actor
        self.critic_model = self.rlhf_engine.critic
        self.ref_model = self.rlhf_engine.ref
        self.reward_model = self.rlhf_engine.reward
        self.tokenizer = self.rlhf_engine.tokenizer
        self.args = args
        self.max_answer_seq_len = args.max_answer_seq_len
        self.end_of_conversation_token_id = self.tokenizer(
            args.end_of_conversation_token)['input_ids'][-1]

        # Those value can be changed
        self.kl_ctl = 0.02
        self.clip_reward_value = 5
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.gamma = 1.0
        self.lam = 0.95
        self.df_unichat = pd.read_excel("/data/renma/unigpt/inference_models/rlhf0601_unichat/unichat_comparison.xlsx")
    
    def post_process(self, response):
        punkts = [
            [",", "，"],
            ["!", "！"],
            [":", "："],
            [";", "；"],
            ["\?", "？"],
        ]
        for item in punkts:
            # 中文字符后面的问号（?）替换为中文的问号（？）
            response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
            # 中文字符前面的问号（?）替换为中文的问号（？）
            response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
            # 把所有的（?）替换为中文问号（？）
            # 因为 “哪个团队负责开发UniChat?”不符合上面两个正则匹配，也需要把最后的?替换为？
            response = re.sub("\?", "？", response)
        return response

    def _generate_sequence(self, prompts, mask):

        max_length = self.max_answer_seq_len + prompts.shape[1]

        with torch.no_grad():
            # ! 这一步很慢而且GPU利用率很低，所以需要启用enable_hybrid_engine，会大大加速生成速度
            # seq.shape = [bsz, seq_len]，其中seq_len=输入prompt长度(prompts.shape[1])+生成回复的长度
            seq = self.actor_model.module.generate(
                prompts, attention_mask=mask, max_length=max_length)
                #    min_length=max_min_length)
                 # ! 如果设置min_length的话，generate时不会出现eos token，会强行生成回复一直到256长度，所以不设置此参数。
                 # ! 如果是batch generate的话，seq会被padding成最长回复的长度
            # 生成回答后手动padding到max_length. (0, max_length-seq.shape[1])代表最后一个维度左侧pad 0个，右侧pad到max_length
            seq = torch.nn.functional.pad(seq, (0, max_length-seq.shape[-1]), mode='constant', value=self.tokenizer.pad_token_id)
        # Filter out seq with no asnwers (or very short). This happens when users directly use the pre-training ckpt without supervised finetuning
        # NOTE: this will causes each GPU has different number of examples
        batch_size = seq.shape[0]
        prompt_length = prompts.shape[1]
        ans = seq[:, prompt_length:]
        self.prompt_length = prompt_length
        valid_ans_len = (ans != self.tokenizer.pad_token_id).sum(dim=-1)
        out_seq = []
        for i in range(batch_size):
            if valid_ans_len[i] <= 1:  # if the answer is shorter than 1 token, drop it
                continue
            else:
                out_seq.append(seq[i:i + 1])
        out_seq = torch.cat(out_seq, dim=0)  # concate output in the batch dim
        return out_seq

    def generate_experience(self, prompts, mask):
        # ! 当actor_model是DeepSpeedHybridEngine时，self.actor_model._total_batch_size需要指定一下，否则会报错！！
        if "DeepSpeedHybridEngine" in str(type(self.actor_model)):
            self.actor_model._total_batch_size = prompts.shape[0]
        self.eval()
        # NOTE: 只有这里用到了预处理阶段生成的attention_mask
        seq = self._generate_sequence(prompts, mask)  # [0 0 0 prompt answer 0 0]
        # print('aaaaaaaaaaaaaaaaaaaaaaaa: ', seq.shape)
        question = self.tokenizer.batch_decode(prompts)
        question = self.post_process(question[0])
        reply = self.tokenizer.batch_decode(seq[:,self.prompt_length:])
        reply = self.post_process(reply[0])
        self.train()

        pad_token_id = self.tokenizer.pad_token_id
        decoded = self.tokenizer.batch_decode(seq)
        encoded = self.tokenizer(decoded, max_length=seq.shape[1], padding="max_length", truncation=True, return_tensors="pt")
        attention_mask_chatglm = encoded["attention_mask"]
        # attention_mask = seq.not_equal(pad_token_id).long()
        # attention_mask_chatglm = get_masks(self.tokenizer, seq) # todo
        with torch.no_grad():
            output = self.actor_model(seq, attention_mask=attention_mask_chatglm)
            output_ref = self.ref_model(seq, attention_mask=attention_mask_chatglm)
            if question in self.df_unichat['prompt'].values:
                reward_score = self.df_unichat[self.df_unichat['prompt']==question]['rejected_score_ren'].values[0]
                reward_score = torch.tensor(reward_score).unsqueeze(0).to(seq.device)
                prior_reply = self.df_unichat[self.df_unichat['prompt']==question]['rejected'].values[0]
                if reply != prior_reply:
                    print_rank_0(f"\n此问题在库里，但是回答和库里的不一致。 \n库里回答: {prior_reply}")
                    if any(waterprint in reply for waterprint in ['KEG实验室', '智谱', 'chatglm']):
                        print_rank_0(f"\n生成回答包含水印，赋值-5分。")
                        reward_score = torch.tensor(-5).unsqueeze(0).to(seq.device)
            else:
                print_rank_0(f"\n此问题不在库里: {question}")
                reward_score = self.reward_model.forward_value(
                    seq, attention_mask_chatglm, prompt_length=self.prompt_length)['chosen_end_scores'].detach()
                
            values = self.critic_model.forward_value( # [1, 767]
                seq, attention_mask_chatglm, return_value_only=True).detach()[:, :-1] # 最后一个token不要

        logits = output.logits
        logits_ref = output_ref.logits
        #  [0 0 0 prompt answer 0 0] -> [0 0 0 1 1 0 0]
        attention_mask_deepspeed = seq.not_equal(pad_token_id).long() # [1, 768]
        print_rank_0(f"\n问题: {question}")
        print_rank_0(f"生成回答: {reply}")
        print_rank_0(f"reward分数: {reward_score.item():.2f}\n")
        
        return {
            'prompts': prompts,
            'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]),
            'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], seq[:,1:]),
            'value': values,
            'rewards': reward_score,
            'input_ids': seq,
            "attention_mask_chatglm": attention_mask_chatglm,
            "attention_mask_deepspeed": attention_mask_deepspeed
        }

    def compute_rewards(self, 
                        prompts, 
                        log_probs, 
                        ref_log_probs, 
                        reward_score,
                        action_mask):

        kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
        print(f"kl_divergence_estimate: {kl_divergence_estimate}")
        rewards = kl_divergence_estimate # [1, 767]
        start = prompts.shape[1] - 1  # 511, prompt倒数第二个token
        ends = start + action_mask[:, start:].sum(1) # 767, answer倒数第二个token（不包括padding）
        reward_clip = torch.clamp(
            reward_score, -self.clip_reward_value, self.clip_reward_value) # 确保reward_score绝对值不大于clip_reward_value
        batch_size = log_probs.shape[0]
        for j in range(batch_size):
            # answer最后一个token (767) 对应的rewards（也就是values）加上reward_score
            rewards[j, start:ends[j]][-1] += reward_clip[j]
        return rewards

    def train_rlhf(self, inputs):
        # train the rlhf mode here
        ### process the old outputs
        prompts = inputs['prompts'] # [1, 512]
        log_probs = inputs['logprobs'] # [1, 767]
        ref_log_probs = inputs['ref_logprobs']  # [1, 767]
        reward_score = inputs['rewards'] # scalar
        values = inputs['value']  # [1, 767]
        attention_mask_chatglm = inputs['attention_mask_chatglm'] # [1, 1, 768, 768]
        attention_mask_deepspeed = inputs['attention_mask_deepspeed'] # [1, 768]
        seq = inputs['input_ids'] # [1, 768]

        start = prompts.size()[-1] - 1 # 511
        action_mask = attention_mask_deepspeed[:, 1:] # [1, 767] [0 0 1 1 0 0]

        old_values = values
        with torch.no_grad():
            old_rewards = self.compute_rewards(
                prompts=prompts,
                log_probs=log_probs,
                ref_log_probs=ref_log_probs,
                reward_score=reward_score,
                action_mask=action_mask) # [1, 767]
            advantages, returns = self.get_advantages_and_returns(
                values=old_values,
                rewards=old_rewards,
                start=start)

        ### process the new outputs
        batch = {'input_ids': seq, "attention_mask": attention_mask_chatglm}
        actor_prob = self.actor_model(**batch, use_cache=False).logits  # [1, 768, 130528]
        actor_log_prob = gather_log_probs(actor_prob[:, :-1, :], seq[:, 1:])# [1, 767]
        actor_loss = self.actor_loss_fn(
            logprobs=actor_log_prob[:, start:],
            old_logprobs=log_probs[:, start:], 
            advantages=advantages,
            mask=action_mask[:, start:])
        self.actor_model.backward(actor_loss)
        self.actor_model.step()
        value = self.critic_model.forward_value(
            **batch, return_value_only=True, use_cache=False)[:, :-1] # 最后一个token不要
        critic_loss = self.critic_loss_fn(
            values=value[:, start:], 
            old_values=old_values[:,start:],
            returns=returns, 
            mask=action_mask[:, start:])
        self.critic_model.backward(critic_loss)
        self.critic_model.step()
        print_rank_0(f"""act_loss: {actor_loss}|
                             cri_loss: {critic_loss}""")
        return actor_loss, critic_loss

    def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        ## policy gradient loss
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)  # [1, 256]
        pg_loss1 = -advantages * ratio  # [1, 256]
        pg_loss2 = -advantages * torch.clamp(
            ratio, 1.0 - self.cliprange, 1.0 + self.cliprange) # 确保ratio在[0.8, 1.2]之间
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
        return pg_loss

    def critic_loss_fn(self, values, old_values, returns, mask):
        ## value loss
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
        vf_loss1 = (values - returns)**2
        vf_loss2 = (values_clipped - returns)**2
        vf_loss = 0.5 * torch.sum(
            torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
        return vf_loss

    def get_advantages_and_returns(self, values, rewards, start):
        # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
        lastgaelam = 0
        advantages_reversed = []
        length = rewards.size()[-1]
        for t in reversed(range(start, length)):
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1) # [1, 256]
        returns = advantages + values[:, start:] # [1, 256]
        return advantages.detach(), returns

    def _validate_training_mode(self):
        assert self.actor_model.module.training
        assert self.critic_model.module.training

    def _validate_evaluation_mode(self):
        assert not self.actor_model.module.training
        assert not self.critic_model.module.training
        assert not self.ref_model.module.training
        assert not self.reward_model.module.training

    def train(self):
        self.actor_model.train()
        self.critic_model.train()

    def eval(self):
        self.actor_model.eval()
        self.critic_model.eval()
        self.reward_model.eval()
        self.ref_model.eval()

    def dump_model_norms(self, tag):
        actor_model_norm = get_model_norm(self.actor_model)
        ref_model_norm = get_model_norm(self.ref_model)
        critic_model_norm = get_model_norm(self.critic_model)
        reward_model_norm = get_model_norm(self.reward_model)
        print_all_ranks(f'{tag} global_actor_model_norm', actor_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_ref_model_norm', ref_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_critic_model_norm', critic_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_reward_model_norm', reward_model_norm,
                        self.args.local_rank)


class DeepSpeedPPOTrainerUnsupervised(DeepSpeedPPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_unsupervised(self, inputs, unsup_coef):
        # Train the unsupervised model here
        self._validate_training_mode()

        outputs = self.actor_model(**inputs, use_cache=False)
        loss = outputs.loss
        self.actor_model.backward(unsup_coef * loss)
        self.actor_model.step()

        return loss
