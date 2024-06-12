"""
input:
model_config：对应的模型
dataloader：任务数据集
ICL_pool: ICL部分
output:
list：(ICL部分熵值，性能)
"""

import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
import argparse
import torch
from datasets import load_dataset
import argparse
from datasets import Dataset
from utils.load_config import load_config
from utils.load_model import load_model_tokenizer
from utils.process_data import get_model_generate
from data_processor.token_entropy_processor import SoftMaxTokenEntropyProcessor
from data_loader.poem_sentiment_loader import SentimentClassificationLoader
from utils.meter import AverageMeter
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="./config/llama2.yaml", help="config file path")
    parser.add_argument("--start", default = 0, type=int, help="config file path")
    parser.add_argument("--model_cfg", default="./config/models_jq.yaml", help="model config file path")
    parser.add_argument("--ICL", default="llama2_7b_poem", help="ICL dataset")
    args = parser.parse_args()
    
    # load config 
    config = load_config(args.cfg)
    model_cfg = load_config(args.model_cfg)
    model_familys = config['model_familys']
    model_configs = []
    for key in model_familys:
        model_configs += model_cfg[f"paths_{key}"]
    model_config = model_configs[args.start]
    
    # 固定随机数种子
    set_seed(42)
    print(model_config[0])
    model,tokenizer = load_model_tokenizer(model_config)
    
    data_loader = SentimentClassificationLoader()
    # 读取数据集
    ICL_pool = load_dataset('json', data_files=f"./{args.ICL}.json",split="train")

    # 保存数据
    entropy_acc_list = [["Token_Entropy","Sentence_Entropy","Acc"]]
    # 查看数据集
    for data in ICL_pool:
        ICL_test_datas,labels = data_loader.get_ICL_test_datas(data['ICL'])
        accuracy = AverageMeter()
        for ICL_test_data,label in zip(ICL_test_datas,labels):
            ICL_input_ids = tokenizer.encode(ICL_test_data, return_tensors='pt').cuda()
            with torch.no_grad():
                outputs = model(ICL_input_ids)
                logits = outputs.logits
                last_token_logits = logits[:, -1, :]
                probabilities = torch.softmax(last_token_logits, dim=-1)
            
            # 获取对应的token id
            positive_id = tokenizer.convert_tokens_to_ids("pos")
            negative_id = tokenizer.convert_tokens_to_ids("negative")
            no_impact_id = tokenizer.convert_tokens_to_ids("no")

            # 获取token id对应的概率
            positive_prob = probabilities[0, positive_id].item()
            negative_prob = probabilities[0, negative_id].item()
            no_impact_prob = probabilities[0, no_impact_id].item()
            
            # 将概率和对应的token id放在一起
            probabilities_dict = {
                "positive": positive_prob,
                "negative": negative_prob,
                "no_impact": no_impact_prob
            }
            max_token = max(probabilities_dict, key=probabilities_dict.get)
            print(f"[negative:{negative_prob},postive:{positive_prob},no_impact:{no_impact_prob}], ground truth: {label}, prediction: {max_token}")
            accuracy.update(max_token==label)
        
        print(f"[{data['token_entropy']}:{accuracy.avg}]")
        entropy_acc_list.append([data['token_entropy'],data['sentence_entropy'],accuracy.avg])
        
    import csv
    with open(f"./{args.ICL}.csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        # 逐行写入数据
        for row in entropy_acc_list:
            writer.writerow(row)