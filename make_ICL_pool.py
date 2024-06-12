"""
input:
model_config：对应的模型
dataloader：ICL数据集
output:
ICL_pool：包含熵值分数的ICL数据集
"""

import argparse
from utils.load_model import load_model_tokenizer
from utils.load_config import load_config

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