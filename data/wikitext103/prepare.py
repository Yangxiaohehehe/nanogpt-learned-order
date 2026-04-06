# data/wikitext103/prepare.py
import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# 1. 准备目录
dataset_dir = os.path.dirname(__file__)

# 2. 加载 WikiText-103 (raw 版本保留了真实的标点和大小写)
print("Downloading and loading WikiText-103 dataset...")
# 注意：首次运行会从 Hugging Face 下载大约 190MB 的压缩包
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

# 3. 初始化 GPT-2 BPE 分词器
print("Initializing GPT-2 BPE Tokenizer...")
enc = tiktoken.get_encoding("gpt2")

# 4. 定义并行处理函数
def process(example):
    # 对当前行/段落进行编码
    ids = enc.encode_ordinary(example['text'])
    # 【学术细节】：加入一个特殊的结束符（<|endoftext|>）
    # 这能帮助 AO-GPT 模型理解段落或文章的边界，不会把两篇文章的逻辑混淆
    ids.append(enc.eot_token)
    return {'ids': ids, 'len': len(ids)}

# 5. 多进程并行分词 (利用 CPU 多核极速处理)
print("Tokenizing the dataset across multiple CPUs...")
# 动态获取 CPU 核心数，留一半以免把服务器卡死
num_proc = max(1, os.cpu_count() // 2) 
tokenized = dataset.map(
    process,
    remove_columns=['text'],
    desc="Tokenizing splits",
    num_proc=num_proc,
)

# 6. 将处理好的数据写入 .bin 文件 (使用 memmap 节省 RAM)
for split, dset in tokenized.items():
    # WikiText 包含 'train', 'validation', 'test'
    # 按照 nanoGPT 的习惯，我们把 validation 映射为 val.bin
    if split == 'test':
        continue # 测试集暂不参与验证循环，直接跳过
    
    bin_split = 'val' if split == 'validation' else split
    
    # 统计总 Token 数量
    arr_len = np.sum(dset['len'], dtype=np.uint64)
    print(f"\n{split} split has {arr_len:,} tokens")
    
    filename = os.path.join(dataset_dir, f'{bin_split}.bin')
    # GPT-2 的词表是 50257，完全可以用 uint16 (0~65535) 存储，节省一半硬盘空间
    dtype = np.uint16 
    
    # 创建 memmap 数组，直接与硬盘交互，内存占用几乎为 0
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    
    # 批量写入硬盘
    print(f"Writing {filename} to disk...")
    idx = 0
    for example in tqdm(dset):
        ids = example['ids']
        arr[idx : idx + len(ids)] = ids
        idx += len(ids)
    arr.flush() # 强制刷新缓冲，确保写入

print("\n🎉 WikiText-103 preparation completed successfully!")