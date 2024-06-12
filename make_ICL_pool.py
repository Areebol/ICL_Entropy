"""
input:
model_config：对应的模型
dataloader：ICL数据集
output:
ICL_pool：包含熵值分数的ICL数据集
"""

import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
import argparse
from datasets import Dataset
from utils.load_config import load_config
from utils.load_model import load_model_tokenizer
from utils.process_data import get_model_generate
from data_processor.token_entropy_processor import SoftMaxTokenEntropyProcessor
from data_loader.poem_sentiment_loader import SentimentClassificationLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="./config/llama2.yaml", help="config file path")
    parser.add_argument("--start", default = 0, type=int, help="config file path")
    parser.add_argument("--model_cfg", default="./config/models_jq.yaml", help="model config file path")
    args = parser.parse_args()
    
    # load config 
    config = load_config(args.cfg)
    model_cfg = load_config(args.model_cfg)

    model_familys = config['model_familys']
    model_configs = []
    for key in model_familys:
        model_configs += model_cfg[f"paths_{key}"]
    model_config = model_configs[args.start]
    print(model_config[0])
    model,tokenizer = load_model_tokenizer(model_config)
    
    data_loader = SentimentClassificationLoader()
    ICL_examples = data_loader.get_ICL_examples()
    
    ICL_pool = list()
    
    data_processor = SoftMaxTokenEntropyProcessor(model,tokenizer,model_config)
    
    for i, ICL_example in enumerate(ICL_examples):
        print(f"[example:{i}/{len(ICL_examples)}]:{ICL_example}")
        
        # pre process
        model_generate = get_model_generate(tokenizer,model,ICL_example,max_new_tokens=1)
        processed_data = data_processor.process_data(0, ICL_example, model_generate)
        # 计算熵值
        ICL_pool.append({"ICL":ICL_example,"token_entropy":processed_data})
        
    # 保存为json文件
    dataset = Dataset.from_list(ICL_pool)
    dataset.to_json("./tmp.json")