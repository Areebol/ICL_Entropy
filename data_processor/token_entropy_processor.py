from .base_processor import BaseProcessor
from utils.meter import AverageMeter
from utils.process_data import *
import torch
import os
import csv
import logging

class SoftMaxTokenEntropyProcessor(BaseProcessor):
    def __init__(self, model, tokenizer, model_config):
        BaseProcessor.__init__(self, model, tokenizer, model_config)
        self.name = "SoftMaxTokenEntropyProcessor"
        self.total_entropy = AverageMeter()

    def process_data(self, index, data, model_generate,split_words=None):
        logging.info(f"{self.name} process data")
        res_entropy = model_generate['entropy']
        num_input_tokens = res_entropy[0].__len__()
        num_heads = res_entropy.shape[0]
        mean_entropy = res_entropy[:,1:].mean()
        self.total_entropy.update(mean_entropy)
        logging.info(mean_entropy.item())
        return mean_entropy.item()

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
        
class UnSoftMaxTokenEntropyProcessor(BaseProcessor):
    def __init__(self, model, tokenizer, model_config):
        BaseProcessor.__init__(self, model, tokenizer, model_config)
        self.name = "UnSoftMaxTokenEntropyProcessor"
        self.total_entropy = AverageMeter()

    def process_data(self, index, data, model_generate,split_words=None):
        logging.info(f"{self.name} process data")
        res = model_generate['generate']
        encoder = get_encoder_k(self.model,-1)
        hidden_state = get_hidden_state_k(res,-2)
        # 手动计算最后一层的attention weight
        attentions = get_attention_matrix(encoder,hidden_state,soft_max=False).to(torch.float32).cpu() # shape = (bs_size,#heads,len,len)
        res_entropy = get_attention_entropy(attentions,soft_max=False) # shape = (bs_size,len)
        mean_entropy = res_entropy[:,1:].mean()
        self.total_entropy.update(mean_entropy)
        logging.info(mean_entropy.item())
        return mean_entropy.item()

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
        
class AvgHeadSoftMaxTokenEntropyProcessor(BaseProcessor):
    """对多头的attention matrix求平均再计算熵值"""
    def __init__(self, model, tokenizer, model_config):
        BaseProcessor.__init__(self, model, tokenizer, model_config)
        self.name = "AvgHeadSoftMaxTokenEntropyProcessor"
        logging.info(f"Init {self.name}")
        self.total_entropy = AverageMeter()

    def process_data(self, index, data, model_generate,split_words=None):
        logging.info(f"{self.name} process data")
        res = model_generate['generate']
        attentions = res["attentions"][0][0].to(torch.float32).cpu()
        res_entropy = get_attention_entropy(attentions,avg_head=True) # shape = (bs_size,len)
        mean_entropy = res_entropy[:,:,1:].mean()
        self.total_entropy.update(mean_entropy)
        logging.info(mean_entropy.item())
        return mean_entropy.item()
    
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
        
class AvgHeadUnSoftMaxTokenEntropyProcessor(BaseProcessor):
    """对多头的attention matrix求平均再计算熵值"""
    def __init__(self, model, tokenizer, model_config):
        BaseProcessor.__init__(self, model, tokenizer, model_config)
        self.name = "AvgHeadUnSoftMaxTokenEntropyProcessor"
        logging.info(f"Init {self.name}")
        self.total_entropy = AverageMeter()

    def process_data(self, index, data, model_generate,split_words=None):
        logging.info(f"{self.name} process data")
        res = model_generate['generate']
        encoder = get_encoder_k(self.model,-1)
        hidden_state = get_hidden_state_k(res,-2)
        # 手动计算最后一层的attention weight
        attentions = get_attention_matrix(encoder,hidden_state,soft_max=False).to(torch.float32).cpu() # shape = (bs_size,#heads,len,len)
        res_entropy = get_attention_entropy(attentions,avg_head=True,soft_max=False) # shape = (bs_size,len)
        mean_entropy = res_entropy[:,:,1:].mean()
        self.total_entropy.update(mean_entropy)
        logging.info(mean_entropy.item())
        return mean_entropy.item()
    
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