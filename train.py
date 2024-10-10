import os
import pandas as pd

import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
import logging
from tqdm import tqdm

class SmartContractDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        labels = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # 对于多标签分类任务，labels应该是一个二进制向量，其中1表示对应的标签存在
        labels_tensor = torch.tensor(labels, dtype=torch.float)  # 使用float而不是long，因为多标签分类通常是二进制向量

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels_tensor,
        }

# 设置日志记录器
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[
                        logging.FileHandler('training.log'),
                        logging.StreamHandler()
                    ])

logger = logging.getLogger()

# 数据集路径
data_folder = 'dataset/fewshot'
csv_file_path = 'dataset/100_label.csv'

# 读取CSV文件
df = pd.read_csv(csv_file_path)

# 获取文件名和标签
filenames = df.iloc[:, 0].tolist()  # 第一列是文件名
labels = df.iloc[:, 1:].values.tolist()  # 第二到第六列是标签


# 定义缺陷类型的数量
num_defect_types = 5

# 加载预训练的BERT模型和分词器
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_defect_types)# 替换输出层以适应多标签分类
model.classifier = torch.nn.Linear(model.classifier.in_features, num_defect_types)

# 准备数据集
texts = []
for filename in filenames:
    filename_with_extension = filename.replace('0x', '') + '.sol'
    file_path = os.path.join(data_folder, filename_with_extension)
    with open(file_path, 'r', encoding='utf-8') as file:
        texts.append(file.read())

# 确保文本和标签的长度相同
assert len(texts) == len(labels)

max_len = 512  # 根据实际情况调整

train_dataset = SmartContractDataset(texts, labels, tokenizer, max_len)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 定义优化器
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
# 定义损失函数
loss_fn = BCEWithLogitsLoss()

# 初始化TensorBoard的SummaryWriter
writer = SummaryWriter()

# 训练模型
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0")

model.to(device)
model.train()

epochs = 50  # For example, set it to 3. Change this to the number of epochs you want to train for.
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

for epoch in range(epochs):
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training", unit="batch")
    for step, batch in enumerate (train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        model.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

        # 更新进度条信息
        pbar.set_postfix(loss=loss.item())
        # 使用日志记录器记录训练信息
        logger.info(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")
        # 使用TensorBoard记录损失
        writer.add_scalar('Training/Loss', loss.item(), global_step=step + epoch * len(train_loader))

# 保存模型
model.save_pretrained('./smart_contract_bert_model')
# 关闭SummaryWriter
writer.close()