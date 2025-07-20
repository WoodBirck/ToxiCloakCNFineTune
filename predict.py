import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score
import pandas as pd

# 加载数据
emoji_df = pd.read_csv('emoji_full.tsv', sep='\t')
homo_df = pd.read_csv('homo_full.tsv', sep='\t')

# 假设你想从 'content' 列中提取文本，从 'toxic' 列中提取标签（根据你的实际数据列）
emoji_texts = emoji_df['content'].tolist()
emoji_labels = emoji_df['toxic'].tolist()

homo_texts = homo_df['content'].tolist()
homo_labels = homo_df['toxic'].tolist()

# 加载保存的模型和tokenizer
model = BertForSequenceClassification.from_pretrained("path_to_save_model")
tokenizer = BertTokenizer.from_pretrained("path_to_save_tokenizer")


# 准备数据集
def tokenize_function(examples):
    encoding = tokenizer(examples['content'], padding='max_length', truncation=True, max_length=512)
    return {'input_ids': encoding['input_ids'], 'attention_mask': encoding['attention_mask']}


# 将文本转换为Dataset对象
emoji_dataset = Dataset.from_dict({'content': emoji_texts, 'toxic': emoji_labels})
homo_dataset = Dataset.from_dict({'content': homo_texts, 'toxic': homo_labels})

# Tokenize the datasets
emoji_dataset = emoji_dataset.map(tokenize_function, batched=True)
homo_dataset = homo_dataset.map(tokenize_function, batched=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择 GPU 或 CPU
model = model.to(device)  # 将模型移到 GPU 或 CPU


def predict_from_content(model, content_texts):
    model.eval()  # 设置模型为评估模式
    predictions = []

    # 对每一条 content 文本进行预测
    for text in content_texts:
        # 使用 tokenizer 将文本转换为模型输入格式
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        # 将输入数据移动到设备上
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # 禁用梯度计算
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # 获取模型的输出 logits
            predicted_label = logits.argmax(dim=-1).cpu().numpy()  # 获取预测的标签
            predictions.append(predicted_label[0])  # 单条文本，取第一个值

    return predictions


# 加载你的 tsv 文件（假设包含 content 和 toxic 列）
file_path = "homo_full.tsv"
df = pd.read_csv(file_path, sep='\t')

# 提取 content 文本和 toxic 标签
content_texts = df['content'].tolist()  # 获取 content 列文本
true_labels = df['toxic'].tolist()  # 获取真实的标签

# 进行预测
emoji_preds = predict_from_content(model, content_texts)

# 计算准确度
accuracy = accuracy_score(true_labels, emoji_preds)
print(f"Accuracy: {accuracy:.4f}")

# 将预测结果保存到文件
df['predicted_label'] = emoji_preds
output_path = "predictions_output.tsv"
df.to_csv(output_path, sep='\t', index=False)

# 打印保存路径和部分结果
print(f"Predictions saved to {output_path}")
print(df.head())