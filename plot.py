import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("output.csv")

plt.figure(figsize=(10, 6))
plt.scatter(data['Token_Entropy'], data['Acc'], color='blue')
plt.title('Scatter Plot of Token Entropy vs Accuracy')
plt.xlabel('Token Entropy')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig("tmp.png")