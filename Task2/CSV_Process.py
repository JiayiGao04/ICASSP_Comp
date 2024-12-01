import pandas as pd

# 文件路径
predictions_file = r"D:\desktop\Necessaries\predictions_10_1_smoothed3.csv"
final_results_file = r"D:\desktop\Necessaries\FINALprediction_results.csv"
output_file = r"D:\desktop\Necessaries\TASK2results_10_1.csv"

# 配置 times 和 rest 数据
times = [850, 740, 700, 750, 810, 620, 420, 790, 800, 690, 770, 850, 740, 780, 810, 870, 700, 680, 430, 690, 730, 790, 770, 810, 640, 670, 800, 740, 810, 520, 760, 860, 830, 580, 810, 690, 720, 850, 400, 630, 770, 370, 760, 840, 710]
rest = [350, 1494, 754, 919, 1948, 546, 1395, 2464, 2164, 595, 1490, 1599, 83, 509, 1017, 721, 1455, 974, 1928, 1451, 307, 64, 1351, 1314, 1273, 1522, 1316, 224, 1279, 167, 1305, 838, 509, 2224, 111, 2239, 1738, 1442, 17, 2169, 995, 1380, 1347, 403, 946]

# 读取输入文件
predictions_df = pd.read_csv(predictions_file)
final_results_df = pd.read_csv(final_results_file)

# 初始化输出数据框
output_df = final_results_df[['id', 'subject', 'timestamp']].copy()

# 初始化索引
start_idx = 0

# 遍历 times 和 rest 构造 label
labels = []
for i, (t, r) in enumerate(zip(times, rest)):
    block_labels = []
    for j in range(t):
        label = predictions_df.loc[start_idx, 'label']
        block_labels.extend([label] * 250)
        start_idx += 1

    # 获取 block 的最后一个 label 值
    block_last_label = predictions_df.loc[start_idx - 1, 'label']
    block_labels.extend([block_last_label] * r)

    # 保存到 labels
    labels.extend(block_labels)

# 确保 labels 长度和 output_df 行数一致
# labels = labels[:len(output_df)]

# 将 labels 添加到输出数据框
output_df['label'] = labels

# 保存到输出文件
output_df.to_csv(output_file, index=False)

print(f"输出文件已保存至: {output_file}")
