import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
import logging

from SmartContractDataset import SmartContractDataset

# 准备数据集
val_data_folder = 'dataset/Dataset/Dataset'
val_csv_file_path = 'dataset/NFTContractDefects.csv'

val_df = pd.read_csv(val_csv_file_path)

val_filenames = val_df.iloc[:, 0].tolist()
val_labels = val_df.iloc[:, 1:].values.tolist()

# 准备验证数据集
val_texts = []
for filename in val_filenames:
    filename_with_extension = filename.replace('0x', '') + '.sol'
    file_path = os.path.join(val_data_folder, filename_with_extension)
    with open(file_path, 'r', encoding='utf-8') as file:
        val_texts.append(file.read())


MAX_LEN = 512

# 加载预训练的BERT模型
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_defect_types)

# 加载保存的模型参数
model.load_state_dict(torch.load('./smart_contract_bert_model/pytorch_model.bin'))

# 将模型转移到正确的设备上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

val_dataset = SmartContractDataset(
    texts=val_texts,
    labels=val_labels,
    tokenizer=tokenizer,
    max_len=MAX_LEN
)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# 配置日志记录器
def setup_logger():
    # 创建一个logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  # 设置日志级别为INFO

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler('validation.log')
    file_handler.setLevel(logging.INFO)

    # 再创建一个handler，用于将日志输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# 在你的代码中使用logger
logger = setup_logger()

# 确保模型处于评估模式
model.eval()

# 评估模型
with torch.no_grad():  # 不计算梯度，节省内存和计算资源
    val_loss = 0
    correct = 0
    total = 0
    logger.info("Starting validation")
    for batch in tqdm(val_loader, desc="Validation"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        predictions = torch.sigmoid(logits).round()  # 使用sigmoid激活函数并四舍五入到最接近的整数

        val_loss += loss.item()
        _, predicted = torch.max(predictions, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    accuracy = correct / total
    print(f"Validation Loss: {avg_val_loss:.2f}")
    print(f"Validation Accuracy: {accuracy:.2f}")

    # 也可以记录到日志或TensorBoard
    logger.info(f"Validation Loss: {avg_val_loss:.2f}")
    logger.info(f"Validation Accuracy: {accuracy:.2f}")
    logger.info("Validation complete")