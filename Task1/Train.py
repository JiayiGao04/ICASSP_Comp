import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import Counter

# Data Preprocessing: Sliding Window
def sliding_window(data, window_size, step):
    num_windows = (len(data) - window_size) // step + 1
    return [data[i:i + window_size] for i in range(0, num_windows * step, step)]

# Load and preprocess data
class SleepDataset(Dataset):
    def __init__(self, data, is_train=True, window_size=250, step=250):
        self.samples = []
        self.is_train = is_train

        # Apply sliding window and normalization
        for sample in data:
            accel = np.array(sample['accel'])
            accel = (accel - accel.mean(axis=0)) / (accel.std(axis=0) + 1e-8)  # Standardize
            windows = sliding_window(accel, window_size, step)
            
            if is_train:
                for window in windows:
                    self.samples.append({'accel': window, 'label': sample['label']})
            else:
                chunk_id = sample['chunk_id']
                for window in windows:
                    self.samples.append({'accel': window, 'chunk_id': chunk_id})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        accel = torch.tensor(sample['accel'], dtype=torch.float32).permute(1, 0)
        if self.is_train:
            label = torch.tensor(sample['label'], dtype=torch.long)
            return accel, label
        else:
            chunk_id = sample['chunk_id']
            return accel, chunk_id

def load_data(train_path, test_path, window_size=250, step=250):
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    with open(test_path, 'r') as f:
        test_data = json.load(f)

    train_dataset = SleepDataset(train_data, is_train=True, window_size=window_size, step=step)
    test_dataset = SleepDataset(test_data, is_train=False, window_size=window_size, step=step)
    return train_dataset, test_dataset

# Model definition (unchanged)
class HybridClassifier(nn.Module):
    def __init__(self, input_channels=3, num_classes=2, conv_kernel_size=7, lstm_hidden_size=64, mlp_hidden_size=128):
        super(HybridClassifier, self).__init__()
        self.conv = nn.Conv1d(input_channels, 32, kernel_size=conv_kernel_size, padding=(conv_kernel_size - 1) // 2)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=32, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.mlp = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, num_classes)
        )

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.mlp(x)
        return x

# Training function with optimizer and scheduler improvements
def train_model(model, train_loader, val_loader, epochs=1000, lr=1e-3, patience=50, device='cuda', save_path="best_model.pth"):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)

    best_val_accuracy = 0.0
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y).sum().item()
                total += y.size(0)
        val_accuracy = correct / total

        # Learning rate adjustment
        scheduler.step(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()
            patience_counter = 0
            torch.save(best_model_state, save_path)
            print(f"Model saved with validation accuracy: {best_val_accuracy:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss / len(train_loader):.4f}, "
              f"Validation Accuracy: {val_accuracy:.4f}, Best Accuracy: {best_val_accuracy:.4f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model

# Prediction function (unchanged)
def predict(model, test_loader, output_path='predictions.csv', device='cuda'):
    model.eval()
    results = []
    with torch.no_grad():
        for x, chunk_ids in test_loader:
            x = x.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            results.extend(zip(chunk_ids.tolist(), predicted.tolist()))

    df = pd.DataFrame(results, columns=['chunk_id', 'label'])
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

# Main script
if __name__ == "__main__":
    train_path = "./icassp-person-in-bed-track-1/train.json"
    test_path = "./icassp-person-in-bed-track-1/test.json"

    train_dataset, test_dataset = load_data(train_path, test_path, window_size=250, step=125)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = HybridClassifier(input_channels=3, num_classes=4)

    save_path = "best_model_task2.pth"
    trained_model = train_model(model, train_loader, val_loader, epochs=1000, patience=50, save_path=save_path)

    model.load_state_dict(torch.load(save_path))
    predict(model, test_loader, output_path='predictions10.csv')

# 读取 CSV 文件
input_file = "./predictions10.csv"  # 输入文件路径
output_file = "./predictions_unique.csv"  # 输出文件路径
df = pd.read_csv(input_file)

# 对每个 chunk_id 分组，取多数投票的 label
unique_predictions = []
for chunk_id, group in df.groupby('chunk_id'):
    labels = group['label'].tolist()
    # 使用 Counter 找到出现次数最多的 label
    most_common_label = Counter(labels).most_common(1)[0][0]
    unique_predictions.append({'chunk_id': chunk_id, 'label': most_common_label})

# 保存结果为新的 CSV 文件
unique_df = pd.DataFrame(unique_predictions)
unique_df.to_csv(output_file, index=False)

print(f"去重后的文件已保存至: {output_file}")
