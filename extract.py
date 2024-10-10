import pandas as pd
import shutil
import os

# 定义源数据集路径、目标文件夹路径和CSV文件路径
destination_folder = 'dataset/fewshot'
dataset_path = 'dataset/Dataset/Dataset'
csv_file_path = 'dataset/100_label.csv'

# 读取CSV文件
df = pd.read_csv(csv_file_path)
filenames = df.iloc[:, 0].tolist()

# 确保目标文件夹存在，如果不存在则创建
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 遍历文件名列表，复制文件
for filename in filenames:
    # 移除前缀 '0x' 并添加 '.sol' 后缀
    filename_with_extension = filename.replace('0x', '') + '.sol'

    # 构建源文件完整路径
    source_file_path = os.path.join(dataset_path, filename_with_extension)

    # 构建目标文件完整路径
    destination_file_path = os.path.join(destination_folder, filename_with_extension)

    # 复制文件
    try:
        shutil.copy(source_file_path, destination_file_path)
        print(f"Copied '{source_file_path}' to '{destination_file_path}'")
    except FileNotFoundError:
        print(f"File '{source_file_path}' not found. Skipping.")
