import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
import argparse
from datasets import load_dataset

# 读取数据集
dataset = load_dataset('json', data_files='./tmp.json')

# 查看数据集
for data in dataset["train"]:
    print(data)