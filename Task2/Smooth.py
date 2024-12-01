import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv('./smoothed_predictions10.csv')

# 提取 label 列，并转换为 numpy 数组
labels = df['label'].values

# 平滑函数
def smooth_labels(labels, threshold=3):
    labels = np.array(labels)  # 将列表转为 numpy 数组
    n = len(labels)
    smoothed_labels = labels.copy()  # 创建一个新的数组来保存平滑后的结果

    i = 0
    while i < n:
        start = i
        # 找到连续相同的值的结束位置
        while i + 1 < n and labels[i + 1] == labels[start]:
            i += 1
        end = i
        
        # 如果这个片段的长度小于阈值，并且两侧值一致
        segment_length = end - start + 1
        if segment_length < threshold:
            left_value = labels[start - 1] if start > 0 else None
            right_value = labels[end + 1] if end + 1 < n else None
            
            # 如果两边值一致，替换当前段
            if left_value is not None and right_value is not None and left_value == right_value:
                smoothed_labels[start:end + 1] = left_value
            elif left_value is not None:  # 只有左侧值存在
                smoothed_labels[start:end + 1] = left_value
            elif right_value is not None:  # 只有右侧值存在
                smoothed_labels[start:end + 1] = right_value
        
        # 移动到下一个片段
        i += 1
    
    return smoothed_labels

# 直接替换原来的 'label' 列
df['label'] = smooth_labels(labels)

# 保存结果
df.to_csv('./smoothed_predictions10_1.csv', index=False)

print("平滑处理已完成，原 'label' 列已被更新，并保存为 './smoothed_predictions10_1.csv'")

# 读取数据
df = pd.read_csv('./smoothed_predictions10_1.csv')

# 提取 label 列，并转换为 numpy 数组
labels = df['label'].values

# 平滑函数
def smooth_labels(labels, threshold=3):
    labels = np.array(labels)  # 将列表转为 numpy 数组
    n = len(labels)
    smoothed_labels = labels.copy()  # 创建一个新的数组来保存平滑后的结果

    i = 0
    while i < n:
        start = i
        # 找到连续相同的值的结束位置
        while i + 1 < n and labels[i + 1] == labels[start]:
            i += 1
        end = i
        
        # 如果这个片段的长度小于阈值，并且两侧值一致
        segment_length = end - start + 1
        if segment_length < threshold:
            left_value = labels[start - 1] if start > 0 else None
            right_value = labels[end + 1] if end + 1 < n else None
            
            # 如果两边值一致，替换当前段
            if left_value is not None and right_value is not None and left_value == right_value:
                smoothed_labels[start:end + 1] = left_value
            elif left_value is not None:  # 只有左侧值存在
                smoothed_labels[start:end + 1] = left_value
            elif right_value is not None:  # 只有右侧值存在
                smoothed_labels[start:end + 1] = right_value
        
        # 移动到下一个片段
        i += 1
    
    return smoothed_labels

# 直接替换原来的 'label' 列
df['label'] = smooth_labels(labels)

# 保存结果
df.to_csv('./smoothed_predictions10_2.csv', index=False)

print("平滑处理已完成，原 'label' 列已被更新，并保存为 './smoothed_predictions10_2.csv'")

# 读取数据
df = pd.read_csv('./smoothed_predictions10_2.csv')

# 提取 label 列，并转换为 numpy 数组
labels = df['label'].values

# 平滑函数
def smooth_labels(labels, threshold=3):
    labels = np.array(labels)  # 将列表转为 numpy 数组
    n = len(labels)
    smoothed_labels = labels.copy()  # 创建一个新的数组来保存平滑后的结果

    i = 0
    while i < n:
        start = i
        # 找到连续相同的值的结束位置
        while i + 1 < n and labels[i + 1] == labels[start]:
            i += 1
        end = i
        
        # 如果这个片段的长度小于阈值，并且两侧值一致
        segment_length = end - start + 1
        if segment_length < threshold:
            left_value = labels[start - 1] if start > 0 else None
            right_value = labels[end + 1] if end + 1 < n else None
            
            # 如果两边值一致，替换当前段
            if left_value is not None and right_value is not None and left_value == right_value:
                smoothed_labels[start:end + 1] = left_value
            elif left_value is not None:  # 只有左侧值存在
                smoothed_labels[start:end + 1] = left_value
            elif right_value is not None:  # 只有右侧值存在
                smoothed_labels[start:end + 1] = right_value
        
        # 移动到下一个片段
        i += 1
    
    return smoothed_labels

# 直接替换原来的 'label' 列
df['label'] = smooth_labels(labels)

# 保存结果
df.to_csv('./smoothed_predictions10_3.csv', index=False)

print("平滑处理已完成，原 'label' 列已被更新，并保存为 './smoothed_predictions10_3.csv'")


