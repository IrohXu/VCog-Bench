import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Example data based on the image
data = {
    'Model': ['LLaVA-7B', 'Qwen-VL-10B', 'CogVLM-19B', 'InternVL-26B', 'Yi-VL-34B', 'LLaVA-35B', 'InternVL-40B', 'Claude 3 Opus', 'GPT-4o', 'Gemini Pro 1.5', 'Human'],
    'RAVEN Accuracy': [14.29, 16.07, 12.05, 14.73, 19.64, 33.93, 33.04, 27.68, 38.84, 42.86, 84.41],
    'MaRs-VQA Accuracy': [16.88, 29.58, 26.46, 22.09, 25.21, 34.38, 32.71, 33.75, 37.38, 34.79, 69.15],
    'Parameters': [7, 10, 19, 26, 34, 35, 40, 100, 100, 100, 82],
}

df = pd.DataFrame(data)

# Create the bubble plot
plt.figure(figsize=(14, 10))
sns.set(style="whitegrid")

bubble = sns.scatterplot(data=df, x='MaRs-VQA Accuracy', y='RAVEN Accuracy', size='Parameters', hue='Model',
                         palette="viridis", sizes=(50, 2500), alpha=0.7, edgecolor='k', legend=False)

for line in range(0, df.shape[0]):
    bubble.text(df.loc[line, 'MaRs-VQA Accuracy'], df.loc[line, 'RAVEN Accuracy'], df.loc[line, 'Model'], horizontalalignment='left',
                size='medium', color='black', weight='semibold')

# plt.title('Performance vs ', fontsize=20)
plt.xlabel('MaRs-VQA Accuracy (%)', fontsize=15)
plt.ylabel('RAVEN Accuracy (%)', fontsize=15)
plt.grid(True)

# Save the plot as an image file
output_path = 'accuracy_size.png'
plt.savefig(output_path, format='png')

# Show the plot
plt.show()