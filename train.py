import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import pandas as pd
import matplotlib.pyplot as plt


# 加载数据
df = pd.read_csv("base_data.tsv", sep="\t")  # 读取tsv文件

# 数据预处理
df = df[['content', 'toxic']]  # 确保只保留content和toxic列
df['toxic'] = df['toxic'].astype(int)  # 确保标签是整数类型

# BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# 创建Dataset类
class ToxicityDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        content = row['content']
        label = row['toxic']

        encoding = self.tokenizer.encode_plus(
            content,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'toxic': torch.tensor(label, dtype=torch.long)
        }


# Split data into train and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Create datasets
train_dataset = ToxicityDataset(train_df, tokenizer, max_len=128)
val_dataset = ToxicityDataset(val_df, tokenizer, max_len=128)

# DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16)

# BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize lists to store losses
train_losses = []
val_losses = []


def train():
    model.train()
    epoch_loss = 0
    for batch in tqdm(train_dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['toxic'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_dataloader)
    train_losses.append(avg_loss)
    print(f"Training Loss: {avg_loss}")


def evaluate():
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['toxic'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            epoch_loss += outputs.loss.item()

    avg_loss = epoch_loss / len(val_dataloader)
    val_losses.append(avg_loss)
    print(f"Validation Loss: {avg_loss}")


# Run training and evaluation
for epoch in range(10):
    print(f"Epoch {epoch + 1}")
    train()
    evaluate()

# 保存模型和tokenizer
model.save_pretrained("path_to_save_model")
tokenizer.save_pretrained("path_to_save_tokenizer")

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), train_losses, label="Training Loss", marker='o')
plt.plot(range(1, 11), val_losses, label="Validation Loss", marker='s')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss over Epochs")
plt.legend()
plt.grid(True)
plt.show()
