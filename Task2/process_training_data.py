import json
from collections import Counter

# 输入输出文件路径
input_file = './person-in-bed-streaming-detection/train.json'
output_file = './person-in-bed-streaming-detection/train_transformed.json'

# 读取 train.json 文件
with open(input_file, 'r') as f:
    train_data = json.load(f)

# 存储最终转换后的数据
output_data = []

# 处理每个 subject
for entry in train_data:
    subject = entry['subject']
    accel_data = entry['accel']
    ts_data = entry['ts']
    labels_data = entry['labels']

    # 每次处理 2500 个数据点
    chunk_size = 2500
    num_chunks = len(accel_data) // chunk_size  # 计算可以提取多少个完整的chunk

    # 遍历每个chunk
    for chunk_id in range(num_chunks):
        # 获取当前 chunk 的数据
        chunk_accel = accel_data[chunk_id * chunk_size:(chunk_id + 1) *
                                 chunk_size]
        chunk_ts = ts_data[chunk_id * chunk_size:(chunk_id + 1) * chunk_size]
        chunk_labels = labels_data[chunk_id * chunk_size:(chunk_id + 1) *
                                   chunk_size]

        # 新的标签计算逻辑
        first_label = chunk_labels[0]
        last_label = chunk_labels[-1]

        # 判断标签的情况
        if all(label == 0 for label in chunk_labels):
            chunk_label = 0
        elif all(label == 1 for label in chunk_labels):
            chunk_label = 1
        else:
            # 如果有其他混合情况，依然可以用多数值来决定
            chunk_label = Counter(chunk_labels).most_common(1)[0][0]

        # 创建转换后的数据条目
        transformed_entry = {
            "subject": subject,
            "chunk_id": chunk_id,
            "accel": chunk_accel,
            "ts": chunk_ts,
            "label": chunk_label
        }

        # 添加到输出数据列表
        output_data.append(transformed_entry)

# 将转换后的数据保存为输出文件
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=4)

print(f"数据已成功转换并保存到 {output_file}")
