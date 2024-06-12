from .base_processor import BaseProcessor
from utils.meter import AverageMeter
from utils.process_data import *
import torch
import os
import csv
import logging

class SentenceEntropyProcessor(BaseProcessor):
    def __init__(self, model, tokenizer, model_config):
        BaseProcessor.__init__(self, model, tokenizer, model_config)
        self.name = "SentenceEntropyProcessor"
        self.soft_max = True
        self.avg_head = True
        self.total_entropy = AverageMeter()

    def process_data(self, index, data, model_generate,split_words=None):
        logging.info(f"{self.name} process data")
        res = model_generate['generate']
        input_ids = model_generate['input_ids']
        encoder = get_encoder_k(self.model,-1)

        # 划分输入        
        if split_words:
            split_tokens = split_sentence(tokenizer=self.tokenizer,question=data,input_ids=input_ids,split_words=split_words)
        else:
            split_tokens = split_sentence(tokenizer=self.tokenizer,question=data,input_ids=input_ids)
            
        # 根据句子切分attention矩阵 weight权重，token_ids 权重对应token下标
        weights,token_ids = split_attn_matrix(self.model,res,split_tokens,soft_max=self.soft_max)
        
        # 加权计算embedding得到hidden_states
        hidden_states = weighted_hidden_states(weights,token_ids,res)        

        # 计算attention矩阵
        attn_matrix = get_attention_matrix(encoder,hidden_states,soft_max=self.soft_max).to(torch.float32)
        
        # 计算entropy
        with torch.no_grad():
            sentence_entropy = get_attention_entropy(attn_matrix.cpu(),soft_max=self.soft_max,avg_head=self.avg_head)
            mean_sentence_entropy = torch.mean(sentence_entropy,dim=0).squeeze()
            if self.avg_head == False:
                mean_sentence_entropy = torch.mean(mean_sentence_entropy,dim=0).squeeze()
        return torch.mean(mean_sentence_entropy[1:])

    def append_data_to_csv(self, data):
        with open(self.save_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for key, value in data:
                writer.writerow([key, value])

    def save_data(self, index, data):
        self.append_data_to_csv([(index,data)])
    
    def set(self, data_type):
        self.data_type = data_type
        self.save_dir = self.exp_dir + f"/{self.name}" + f"/{self.data_type}"
        BaseProcessor.set(self)
    
    def reset(self):
        self.total_entropy.reset()
        
class AvgHeadSoftMaxSentenceEntropyProcessor(SentenceEntropyProcessor):
    def __init__(self, model, tokenizer, model_config):
        SentenceEntropyProcessor.__init__(self, model, tokenizer, model_config)
        self.name = "AvgHeadSoftMaxSentenceEntropyProcessor"
        self.soft_max = True
        self.avg_head = True
        logging.info(f"Init {self.name} : [soft_max : {self.soft_max} avg_head : {self.avg_head}]")
        
class AvgHeadUnSoftMaxSentenceEntropyProcessor(SentenceEntropyProcessor):
    def __init__(self, model, tokenizer, model_config):
        SentenceEntropyProcessor.__init__(self, model, tokenizer, model_config)
        self.name = "AvgHeadUnSoftMaxSentenceEntropyProcessor"
        self.soft_max = False
        self.avg_head = True
        logging.info(f"Init {self.name} : [soft_max : {self.soft_max} avg_head : {self.avg_head}]")
        
class SoftMaxSentenceEntropyProcessor(SentenceEntropyProcessor):
    def __init__(self, model, tokenizer, model_config):
        SentenceEntropyProcessor.__init__(self, model, tokenizer, model_config)
        self.name = "SoftMaxSentenceEntropyProcessor"
        self.soft_max = True
        self.avg_head = False
        logging.info(f"Init {self.name} : [soft_max : {self.soft_max} avg_head : {self.avg_head}]")
        
class UnSoftMaxSentenceEntropyProcessor(SentenceEntropyProcessor):
    def __init__(self, model, tokenizer, model_config):
        SentenceEntropyProcessor.__init__(self, model, tokenizer, model_config)
        self.name = "UnSoftMaxSentenceEntropyProcessor"
        self.soft_max = False
        self.avg_head = False
        logging.info(f"Init {self.name} : [soft_max : {self.soft_max} avg_head : {self.avg_head}]")