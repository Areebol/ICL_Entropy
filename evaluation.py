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
from data_loader import *
from tqdm import tqdm
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
    parser.add_argument("--ICL", default="llama2_7b_trec", help="ICL dataset")
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
    
    # data_loader = SentimentClassificationLoader()
    # data_loader = SST2Loader()
    data_loader = TRECLoader()
    # 读取数据集
    ICL_pool = load_dataset('json', data_files=f"./exp/{args.ICL}.json",split="train")

    # 保存数据
    entropy_acc_list = [["Token_Entropy","Sentence_Entropy","Acc"]]
    # 查看数据集
    for ICL_i, data in enumerate(ICL_pool):
        ICL_test_datas,labels = data_loader.get_ICL_test_datas(data['ICL'])
        accuracy = AverageMeter()
        with tqdm(total=len(ICL_test_datas), desc=f"ICL_sample {ICL_i+1}") as pbar:
            for test_i, (ICL_test_data,label) in enumerate(zip(ICL_test_datas,labels)):
                ICL_input_ids = tokenizer.encode(ICL_test_data, return_tensors='pt').cuda()
                with torch.no_grad():
                    outputs = model(ICL_input_ids)
                    logits = outputs.logits
                    last_token_logits = logits[:, -1, :]
                    probabilities = torch.softmax(last_token_logits, dim=-1)
                
                # llama7b的词表映射
                # sst2
                # label_token_dict = {
                #     "negative":"negative",
                #     "positive":"pos",
                # }
                
                # trec 
                label_token_dict = {
                    "Abbreviation":"Ab",
                    "Entity":"Entity",
                    "Description":"Description",
                    "Person":"Person",
                    "Location":"Location",
                    "Number":"Number"
                }
                token_ids = [tokenizer.convert_tokens_to_ids(label_token_dict[label]) for label in label_token_dict]
                probabilities = [probabilities[0,token_id].item() for token_id in token_ids]
                probabilities_dict = {label:probability for label,probability in zip(label_token_dict,probabilities)}
                
                max_token = max(probabilities_dict, key=probabilities_dict.get)
                accuracy.update(max_token==label)
                pbar.update(1)
                pbar.set_postfix({'Iteration': test_i+1, 'acc': accuracy.avg})
        tqdm.write(f"ICL sample {ICL_i+1} completed, acc: {accuracy.avg:.4f}")
            # topK_values,topK_indices = torch.topk(probabilities,k=20)
            # topK_tokens = [tokenizer.decode(topK_indice) for topK_indice in topK_indices.squeeze(0).tolist()]
            # for value,token in zip(topK_values.squeeze(0),topK_tokens):
            #     print(f"{token}:{value}",end="\t")
            # print("\n")         

        print(f"[{data['token_entropy']}:{accuracy.avg}]")
        entropy_acc_list.append([data['token_entropy'],data['sentence_entropy'],accuracy.avg])
        
    import csv
    with open(f"./exp/{args.ICL}.csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        # 逐行写入数据
        for row in entropy_acc_list:
            writer.writerow(row)