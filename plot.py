import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("llama2_7b_sst2.csv")

plt.figure(figsize=(10, 6))
plt.scatter(data['Token_Entropy'], data['Acc'], color='blue')
plt.title('Scatter Plot of Token Entropy vs Accuracy')
plt.xlabel('Token Entropy')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig("llama2_7b_sst2_1.png")

plt.figure(figsize=(10, 6))
plt.scatter(data['Sentence_Entropy'], data['Acc'], color='blue')
plt.title('Scatter Plot of Sentence Entropy vs Accuracy')
plt.xlabel('Sentence Entropy')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig("llama2_7b_sst2_2.png")

plt.figure(figsize=(10, 6))
plt.scatter(data['Token_Entropy']/data['Sentence_Entropy'], data['Acc'], color='blue')
plt.title('Scatter Plot of Token/Sentence Entropy vs Accuracy')
plt.xlabel('Token/Sentence Entropy')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig("llama2_7b_sst2_3.png")