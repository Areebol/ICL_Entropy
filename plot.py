import pandas as pd
import matplotlib.pyplot as plt
import os
# data_type = "poem"
# data_type = "sst2"
data_type = "trec"
data = pd.read_csv(f"./exp/llama2_7b_{data_type}.csv")
os.makedirs(name=f"./picture/{data_type}",exist_ok=True)

plt.figure(figsize=(10, 6))
plt.scatter(data['Token_Entropy'], data['Acc'], color='blue')
plt.title('Scatter Plot of Token Entropy vs Accuracy')
plt.xlabel('Token Entropy')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig(f"./picture/{data_type}/llama2_7b_{data_type}_1.png")

plt.figure(figsize=(10, 6))
plt.scatter(data['Sentence_Entropy'], data['Acc'], color='blue')
plt.title('Scatter Plot of Sentence Entropy vs Accuracy')
plt.xlabel('Sentence Entropy')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig(f"./picture/{data_type}/llama2_7b_{data_type}_2.png")

plt.figure(figsize=(10, 6))
plt.scatter(data['Token_Entropy']/data['Sentence_Entropy'], data['Acc'], color='blue')
plt.title('Scatter Plot of Token/Sentence Entropy vs Accuracy')
plt.xlabel('Token/Sentence Entropy')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig(f"./picture/{data_type}/llama2_7b_{data_type}_3.png")