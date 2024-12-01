import json

# 输入文件路径
input_file = './person-in-bed-streaming-detection/test.json'
output_file = './person-in-bed-streaming-detection/streamingtest.json'

# 每个chunk的大小
chunk_size = 2500

# 读取文件
def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# 将数据分块
def split_into_chunks(data, chunk_size):
    result = []
    for subject_data in data:
        subject = subject_data["subject"]
        accel = subject_data["accel"]
        ts = subject_data["ts"]

        # 分块处理
        for chunk_id, i in enumerate(range(0, len(accel) - chunk_size + 1, chunk_size)):
            chunk = {
                "subject": subject,
                "chunk_id": chunk_id,
                "accel": accel[i:i + chunk_size],
                "ts": ts[i:i + chunk_size]
            }
            result.append(chunk)
    return result

# 写入文件
def save_data(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

# 主流程
def main():
    data = load_data(input_file)
    chunked_data = split_into_chunks(data, chunk_size)
    save_data(output_file, chunked_data)

if __name__ == "__main__":
    main()
