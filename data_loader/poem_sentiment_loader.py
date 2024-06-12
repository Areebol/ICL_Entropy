import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
import datasets
import numpy as np
from typing import List, Dict
import itertools
import random

class SentimentClassificationLoader():
    def __init__(self) -> None:
        self.name = 'poem_sentiment_classification'
        dataset = datasets.load_dataset("poem_sentiment")
        self.label = {
            0:"negative",
            1:"positive",
            2:"no_impact",
            #3:"mixed", # there is no `mixed` on the test set
        }
        self.train_data = [line for line in dataset['train']]
        self.test_data = [line for line in dataset['test']]
        
        self.ICL0 = [f"Text:{d['verse_text'].strip().replace("\\n", " ")}\nSentiment:{self.label[d['label']]}" for d in self.train_data if d['label']==0]
        self.ICL1 = [f"Text:{d['verse_text'].strip().replace("\\n", " ")}\nSentiment:{self.label[d['label']]}" for d in self.train_data if d['label']==1]
        self.ICL2 = [f"Text:{d['verse_text'].strip().replace("\\n", " ")}\nSentiment:{self.label[d['label']]}" for d in self.train_data if d['label']==2]
    
    def get_ICL_examples(self,num_sample:int=3,num_example:int=20,seed:int=42):
        """
        num_sample: 每一类数据采样数量
        num_example：最终得到的ICL示例数量
        """
        random.seed(seed)
        ICL_examples = []
        
        # 从每个 ICL 列表中分别选择前 num_sample 个元素
        ICL_examples.extend(self.ICL0[:num_sample])
        ICL_examples.extend(self.ICL1[:num_sample])
        ICL_examples.extend(self.ICL2[:num_sample])

        # 全排列后抽样num_example个
        ICL_examples = list(itertools.permutations(ICL_examples))
        ICL_examples = random.sample(ICL_examples, num_example)
        
        # 拼接得到ICL examples
        ICL_examples = [
            "\n".join([example for example in perm]) 
            for perm in ICL_examples
        ]
        
        return ICL_examples

    def get_ICL_test_datas(self,ICL_demo:str=None):
        """
        ICL：ICL示例部分
        """
        instruct: str = 'Classify the text into no_impact, negative, or positive'
        ICL_test_datas = [f"{instruct}\n{ICL_demo}\nText: {d['verse_text'].strip().replace("\\n", " ")}\nSentiment:" for d in self.test_data]
        labels = [self.label[d["label"]] for d in self.test_data]
        return ICL_test_datas,labels
    
if __name__ == "__main__":
    dataset = SentimentClassificationLoader()
    ICL_examples = dataset.get_ICL_examples()