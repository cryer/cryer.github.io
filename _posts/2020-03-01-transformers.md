---
layout: post
title: 从头开始实现一个transformer
description: 从头开始实现一个transformer

---

### 导入

2017年，Google 在论文《[Attention Is All You Need](https://arxiv.org/abs/1706.03762)》中提出 Transformer,**完全基于注意力机制，摒弃循环与卷积，实现并行化训练，建模长距离依赖**，核心优势：

| 特性        | 说明                         |
| --------- | -------------------------- |
| **并行计算**  | 不像 RNN 需顺序计算，可并行处理整个序列     |
| **长距离依赖** | 注意力机制直接建模任意位置间关系           |
| **可扩展性强** | 易堆叠、易扩展到更大模型               |
| **统一架构**  | 可用于翻译、摘要、分类、生成等几乎所有 NLP 任务 |

其中最主要的一点就是易堆叠，你想象一下，几十上百层的Transformer层堆叠在一起，配合大规模集群和大量文本数据集的训练，那效果很难想象。还有并行计算，也正好是为了前者的大规模训练做铺垫。



**其大概的结构如下**:

```
      输入序列
          │
          ▼
    ┌─────────────┐
    │  Embedding  │ ← 词嵌入 + 位置编码
    └─────────────┘
          │
          ▼
    ┌─────────────┐
    │   Encoder   │ ← N 个相同层堆叠
    └─────────────┘
          │
          ▼
    ┌─────────────┐
    │   Decoder   │ ← N 个带掩码的层堆叠
    └─────────────┘
          │
          ▼
    ┌─────────────┐
    │  Linear +   │
    │  Softmax    │ ← 输出概率分布
    └─────────────┘
```

**论文中的图更加清晰：（原论文图是分开的）**

![](https://github.com/cryer/cryer.github.io/raw/master/image/161.jpg)

熟悉RNN的读者应该知道，这种`encoder-decoder`结构最擅长的就是`seq2seq`，序列到序列的任务，比如文本翻译任务。但你知道其实只有`decoder`的结构也擅长一个领域吗？那就是生成，只有`decoder`就相当于一个**自回归模型**，通过起始token预测下一个token，然后加入到输入token序列中作为条件，继续预测下一个token，就是最基本的生成任务的自回归模型架构。

所以这篇博客，主要就实现一个只有`decoder`的文本生成任务。

### 正文

**项目结构：**

```
decoder_only
├─data 
│ └─data.txt 
├─model.py 
├─inference.py 
└─train.py 
└─make_dataset.py 
```

#### 数据集制作：

```python
# make_dataset.py
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        for i in range(0, len(tokens) - max_len, max_len // 2):
            input_ids = tokens[i:i + max_len - 1]
            target_ids = tokens[i + 1:i + max_len]
            
            input_ids += [PAD_TOKEN] * (max_len - 1 - len(input_ids))
            target_ids += [PAD_TOKEN] * (max_len - 1 - len(target_ids))
            
            self.data.append({
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'target_ids': torch.tensor(target_ids, dtype=torch.long)
            })
        
        print(f"共构建 {len(self.data)} 个训练样本")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]['input_ids'], self.data[idx]['target_ids']
```

#### 模型


```python
# model.py
import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(SinusoidalPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=False):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        
        assert embed_dim % num_heads == 0, 
        self.head_dim = embed_dim // num_heads
        
        # 总共4个线性层：Q, K, V, 输出投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout_layer = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, attn_mask=None):
        """
        输入:(if batch_first=True)
            query: [batch_size, seq_len, embed_dim] 
            key:   [batch_size, seq_len, embed_dim]
            value: [batch_size, seq_len, embed_dim]
            attn_mask: [seq_len, seq_len] 或 [batch_size, seq_len, seq_len] (可选，bool或float)
        
        返回:
            output: [batch_size, seq_len, embed_dim]
            attn_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        if not self.batch_first:
            # 如果不是 batch_first，转置为 [seq_len, batch_size, embed_dim]
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        batch_size, seq_len, embed_dim = query.size()
        
        # 线性变换
        Q = self.q_proj(query)  # [B, L, E]
        K = self.k_proj(key)    # [B, L, E]
        V = self.v_proj(value)  # [B, L, E]
        
        # 拆分多头: [B, L, E] -> [B, L, H, D] -> [B, H, L, D]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 缩放点积注意力: [B, H, L, D] @ [B, H, D, L] -> [B, H, L, L]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用注意力掩码（如因果mask）
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # [L, L] -> [1, 1, L, L] 广播到所有batch和head
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                # [B, L, L] -> [B, 1, L, L]
                attn_mask = attn_mask.unsqueeze(1)
            # 将 mask 中 True 的位置（被屏蔽）设为 -inf
            # 根据 scores 的 dtype 选择合适的 mask 值
            # 自动获取当前数据类型的最小值
            mask_value = torch.finfo(scores.dtype).min  
            scores = scores.masked_fill(attn_mask, mask_value)
        
        # softmax + dropout
        attn_weights = self.softmax(scores)
        attn_weights = self.dropout_layer(attn_weights)
        
        # 加权求和: [B, H, L, L] @ [B, H, L, D] -> [B, H, L, D]
        output = torch.matmul(attn_weights, V)
        
        # 合并多头: [B, H, L, D] -> [B, L, H, D] -> [B, L, E]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # 最终线性变换
        output = self.out_proj(output)
        
        if not self.batch_first:
            # 转回 [seq_len, batch_size, embed_dim]
            output = output.transpose(0, 1)
            attn_weights = attn_weights.transpose(0, 1)  ]，
        
        return output, attn_weights


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    # tgt: 输入张量，形状 [batch_size, seq_len, d_model]
    # tgt_mask: 注意力掩码，形状 [seq_len, seq_len]或[batch_size, seq_len, seq_len] （通常是下三角掩码）
    def forward(self, tgt, tgt_mask=None):
        # 取第一返回值（output）
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
        # 残差连接 维度一直是[B, L, E]即[batch_size, seq_len, d_model]
        # 残差连接前对注意力输出做 dropout（论文推荐，防止过拟合）
        tgt = tgt + self.dropout1(tgt2)
        # LayerNorm：每个token的d_model维度做归一化（均值0，方差1）
        # [batch_size, seq_len, d_model]
        tgt = self.norm1(tgt)
        # 前馈神经网路Linear → ReLU → Dropout → Linear
        # linear1：d_model → dim_feedforward，如 256 → 512 投影到高维
        # linear1 ：dim_feedforward → d_model 回到原维度[B, L, E] 
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        # 残差连接
        tgt = tgt + self.dropout2(tgt2)
        # LayerNorm [B, L, E]
        tgt = self.norm2(tgt)
        return tgt

class TransformerDecoderLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1, max_len=128):
        super(TransformerDecoderLM, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask.to(DEVICE)

    def forward(self, src):
        # src:[batch_size, seq_len]
        # 获取输入序列的长度（token 数量）也就是seq_len
        # 自回归掩码（causal mask） 需要知道序列长度
        # 位置编码也可能依赖长度（虽然我们的 pos_encoder 支持任意长度）
        seq_len = src.size(1)
        # token ID 映射为稠密向量（词嵌入）
        # 然后乘以 √d_model 进行缩放 
        #  [B, L] -> [B, L, E]
        src = self.embedding(src) * math.sqrt(self.d_model)
        # 位置编码（Positional Encoding）
        # [B, L, E] -> [B, L, E]
        src = self.pos_encoder(src)
        # 对词嵌入 + 位置编码后的向量应用 Dropout（随机置零一部分元素）
        # 防止过拟合，提高泛化能力
        src = self.dropout(src)
        # 生成“下三角掩码”，用于自回归生成，防止当前位置看到未来 token
        tgt_mask = self.generate_square_subsequent_mask(seq_len)
        # 逐层通过 Transformer Decoder 层
        # 每一层都进行：自注意力（带mask)+前馈网络+残差连接+LayerNorm
        for layer in self.decoder_layers:
            src = layer(src, tgt_mask=tgt_mask)
        # 每个位置的隐藏状态映射到词表空间，得到 logits
        # [B, L, E] -> [B, L, V]
        output = self.fc_out(src)
        return output
```

基本照着导入中的图就能直接编写，非常简单，本例就是4个`decoder`层的堆叠，多头注意力层分8个头，使用`sum(p.numel() for p in model.parameters())`来打印一下参数总数，结果是`12,947,080`，也就是12M的参数量，也就相当于`resnet-50`的一半，可以说很小。还需要注意这里的一个技巧，就是注意力层分头那里，采用的是直接三维分到四维张量，而不是单独用变量分开存储分头的三维张量，因为这样可以充分高效利用张量的并运算。同时注意到维度2，3维互换了，将头数放到前面，这是为了后面的矩阵乘法处理的是每个头内部的计算。

另外注意到我们并没有实现完整的`decoder`，我们只堆叠了`Masked Multi-Head Attention`这个层，而没有使用`Multi-Head Attention`,这是因为我们是自回归生成任务，只需要处理`self attention`自注意力，而`Multi-Head Attention`其实是`cross attention`交叉注意力，需要依赖`encoder`提供的**KV**向量，而我们不需要。

整体就是这样：

```

输入 tgt ───┐
            │
            ▼
     [Multihead Self-Attention]  ←─ attn_mask（因果掩码）
            │
            ▼
        [Dropout] ───┐
            │        │
            ▼        │
   [Add: tgt + tgt2] │ ←── 残差连接
            │        │
            ▼        │
      [LayerNorm]    │
            │        │
            ▼        │
[Linear → ReLU → Dropout → Linear]  ←─ FFN
            │
            ▼
        [Dropout] ───┐
            │        │
            ▼        │
   [Add: tgt + tgt2] │ ←── 残差连接
            │        │
            ▼        │
      [LayerNorm] ───┘
            │
            ▼
         输出 tgt
```

#### 训练

```python
# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
# 混合精度支持
from torch.cuda.amp import GradScaler, autocast

from model import TransformerDecoderLM
from make_dataset import TextDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")

D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 4
DIM_FF = 512
DROPOUT = 0.1
MAX_LEN = 256
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
EPOCHS = 10
PAD_TOKEN = 0

DATA_PATH = "./data/data.txt"
MODEL_SAVE_PATH = "./decoder_only.pth"

def train_model(model, dataloader, epochs=10, lr=3e-4):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    scaler = GradScaler()  # 混合精度缩放器

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (input_ids, target_ids) in enumerate(progress_bar):
            input_ids = input_ids.to(DEVICE)
            target_ids = target_ids.to(DEVICE)
            
            optimizer.zero_grad()
            
            with autocast():  # 自动混合精度
                output = model(input_ids)
                output = output.view(-1, output.size(-1))
                target = target_ids.view(-1)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} 平均损失: {avg_loss:.4f}")
    
    # 保存模型和配置，可以把一些参数设置都保存一下
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'd_model': D_MODEL,
            'nhead': NHEAD,
            'num_layers': NUM_LAYERS,
            'vocab_size': tokenizer.vocab_size if 'tokenizer' in globals() else 21128,
            'max_len': MAX_LEN
        },
        'sampling_config': {
            'temperature': 0.7,
            'top_k': 40,
            'top_p': 0.9
        }
    }, MODEL_SAVE_PATH)
    print(f"模型已保存至 {MODEL_SAVE_PATH}")



def main():
    print("正在加载中文分词器...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    vocab_size = tokenizer.vocab_size
    print(f"词表大小: {vocab_size}")
    
    if not os.path.exists(DATA_PATH):
        print(f"错误：未找到数据文件 {DATA_PATH}")
        print(f"请在 ./data/ 目录下放置 {DATA_PATH} 文件")
        return
    
    print("正在构建数据集...")
    dataset = TextDataset(DATA_PATH, tokenizer, max_len=MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print("正在初始化模型...")
    model = TransformerDecoderLM(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FF,
        dropout=DROPOUT,
        max_len=MAX_LEN
    ).to(DEVICE)
    
    print("开始训练（混合精度）...")
    train_model(model, dataloader, epochs=EPOCHS, lr=LEARNING_RATE)
    
    # 加载模型
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
if __name__ == "__main__":
    main()
```



### 推理

```python
import torch
import torch.nn as nn
import math
from transformers import BertTokenizer
import os 
from model import TransformerDecoderLM

# ========================
# 配置参数（必须与训练时一致）
# ========================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"推理设备: {DEVICE}")

# 模型结构参数（必须与训练时完全一致）
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 4
DIM_FF = 512
DROPOUT = 0.1
MAX_LEN = 256  # 模型支持的最大上下文长度

# 模型路径
MODEL_SAVE_PATH = "./decoder_only.pth"

# 默认采样参数（可运行时调整）
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 40
DEFAULT_TOP_P = 0.9 


def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens=50,  
    temperature=0.7,
    top_k=40,
    top_p=0.9,
    stop_tokens=None        # 可选：自定义停止token
):
    """
    根据提示词生成文本，支持控制生成长度和采样策略
    参数:
        max_new_tokens: 生成的新token数量（不包括输入）
        temperature: 温度值，控制随机性（值越小越确定）
        top_k: 保留概率最高的k个词
        top_p: nucleus采样阈值
        stop_tokens: 遇到这些token时提前停止（如句号、问号等）
    """
    model.eval()
    
    if stop_tokens is None:
        # 默认遇到这些特殊token停止
        stop_tokens = set([0, 102, 103])  # PAD, SEP, UNK
        # 可添加中文句号、问号等
        period_id = tokenizer.convert_tokens_to_ids('。')
        question_id = tokenizer.convert_tokens_to_ids('？')
        exclamation_id = tokenizer.convert_tokens_to_ids('！')
        if period_id != tokenizer.unk_token_id:
            stop_tokens.add(period_id)
        if question_id != tokenizer.unk_token_id:
            stop_tokens.add(question_id)
        if exclamation_id != tokenizer.unk_token_id:
            stop_tokens.add(exclamation_id)
    
    # 编码输入
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt').to(DEVICE)
    generated = input_ids.clone()
    
    with torch.no_grad():
        for i in range(max_new_tokens):  # 控制最大生成长度
            current_len = generated.size(1)
            if current_len >= MAX_LEN:   # 超过模型最大支持长度
                print("⚠️  警告：达到模型最大长度限制，停止生成")
                break
            
            # 获取模型输出
            output = model(generated)  # [1, seq_len, vocab_size]
            next_token_logits = output[:, -1, :] / temperature  # 取最后一个位置
            
            # Top-k 过滤
            if top_k > 0:
                top_k_vals, _ = torch.topk(next_token_logits, top_k)
                kth_val = top_k_vals[:, -1].unsqueeze(-1)
                indices_to_remove = next_token_logits < kth_val
                next_token_logits[indices_to_remove] = -float('inf')
            
            # Top-p 过滤
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('inf')
            
            # 采样
            if (next_token_logits == -float('inf')).all():
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            # 检查是否遇到停止token
            if next_token.item() in stop_tokens:
                # 如果是句号类，可以生成后停止；如果是PAD则直接停止
                if next_token.item() in [0, 102, 103]:
                    break
                else:
                    # 如果是句号，生成后再停止
                    generated = torch.cat([generated, next_token], dim=1)
                    break
            
            generated = torch.cat([generated, next_token], dim=1)
    
    # 解码为文本
    generated_text = tokenizer.decode(generated.squeeze().tolist(), skip_special_tokens=True)
    return generated_text


# ========================
# 主推理函数（交互式）
# ========================

def main():
    print("正在加载中文分词器...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    # 从保存文件中读取模型配置
    if os.path.exists(MODEL_SAVE_PATH):
        print("正在加载模型...")
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
        
        # 尝试从checkpoint获取vocab_size，否则使用默认
        vocab_size = checkpoint.get('config', {}).get('vocab_size', tokenizer.vocab_size)
        
        # 初始化模型
        model = TransformerDecoderLM(
            vocab_size=vocab_size,
            d_model=D_MODEL,
            nhead=NHEAD,
            num_layers=NUM_LAYERS,
            dim_feedforward=DIM_FF,
            dropout=DROPOUT,
            max_len=MAX_LEN
        ).to(DEVICE)
        
        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ 模型加载成功！")
        
        # 读取默认采样参数
        sampling_config = checkpoint.get('sampling_config', {})
        temperature = sampling_config.get('temperature', DEFAULT_TEMPERATURE)
        top_k = sampling_config.get('top_k', DEFAULT_TOP_K)
        top_p = sampling_config.get('top_p', DEFAULT_TOP_P)
        
    else:
        print(f"❌ 错误：未找到模型文件 {MODEL_SAVE_PATH}")
        return
    
    print("\n" + "="*60)
    print("🎉 中文文本生成推理系统已启动！")
    print("输入中文提示词，模型将自动续写。输入 'quit' 退出。")
    print(f"默认参数: temperature={temperature}, top_k={top_k}, top_p={top_p}")
    print("="*60)
    
    while True:
        try:
            # 获取用户输入
            prompt = input("\n📝 请输入提示词（支持中文）: ").strip()
            if prompt.lower() == 'quit':
                print("👋 再见！")
                break
            if not prompt:
                print("❌ 提示词不能为空，请重新输入。")
                continue
            
            # 询问生成长度
            try:
                max_len_input = input("📏 请输入希望生成的最大长度（默认50，直接回车使用默认）: ").strip()
                max_new_tokens = int(max_len_input) if max_len_input else 50
                if max_new_tokens < 1:
                    max_new_tokens = 50
            except ValueError:
                max_new_tokens = 50
                print("⚠️  输入无效，使用默认值 50")
            
            # 生成文本
            print("🤖 正在生成...")
            generated_text = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens, 
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
            print(f"\n✅ 生成结果:")
            print("-" * 40)
            print(f"输入: {prompt}")
            print(f"输出: {generated_text}")
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，程序退出。")
            break
        except Exception as e:
            print(f"❌ 发生错误: {str(e)}")
            continue


if __name__ == "__main__":
    main()
```

交互式的推理，使用温度和`top_k top_p`来控制输出采样，`top_k`表示取概率最高的k个词来进行采样，`top_p`表示选择累积概率大于等于p的前几个词来采样，比如概率最高的四个token分别是0.3，0.2，0.14，0.11, 那么`top_p=0.6`就表示只选前三个token采样，因为他们的累计概率是0.64，大于0.6了。可以选择输出的token长度，最大256，这是由训练时候的参数决定的，遇到`。?!`会直接停止生成，读者可以自行设置。

### 效果

这里我使用的是一部网络小说来进行模型训练，大概`6MB`的斗破苍穹小说前600章，分词器使用`transformers`库中的`bert-base-chinese`中文分词器，词表大小是21128，使用数据集处理模块处理过之后， 一共是15432 个训练样本。使用批大小16，最大长度256，训练10个epoch，显存占用3.5GB左右，不大，训练时间也不长，半小时不到。读者们可以根据显存自由的修改这些参数，最重要的当然还有多头注意力层的堆叠的层数，还有词向量维度，也可以优先增加。

```
推理设备: cuda
正在加载中文分词器...
正在加载模型...
✅ 模型加载成功！

============================================================
🎉 中文文本生成推理系统已启动！
输入中文提示词，模型将自动续写。输入 'quit' 退出。
默认参数: temperature=0.7, top_k=40, top_p=0.9
============================================================

📝 请输入提示词（支持中文）: 萧炎
📏 请输入希望生成的最大长度（默认50，直接回车使用默认）: 200
🤖 正在生成...

✅ 生成结果:
----------------------------------------
输入: 萧炎
输出: 萧 炎 的 目 光 ， 缓 缓 的 在 周 围 扫 了 扫 ， 将 近 几 个 小 时 ， 
萧 炎 脚 步 顿 在 半 空  之 上 ， 身 体 凌 空 翻 滚 ， 双 手 紧 紧 握 ，
 一 股 劲 气 自 袖 间 猛 然 暴 涌 而 出 ， 然 后 狠  狠 地 砸 在 了 萧 炎
 的 胸 膛 之 上 。
----------------------------------------

📝 请输入提示词（支持中文）: 你若敢伤他一根毫毛
📏 请输入希望生成的最大长度（默认50，直接回车使用默认）: 200
🤖 正在生成...

✅ 生成结果:
----------------------------------------
输入: 你若敢伤他一根毫毛
输出: 你 若 敢 伤 他 一 根 毫 毛 骨 头 ， 你 的 实 力 ， 也 是 没 有 那 种
 让 人 心 中 的 感 觉 ，  当 然 ， 萧 炎 也 不 敢 再 理 会 这 话 ， 他 便 是
 瞧 见 一 名 斗 皇 强 者 的 实 力 ， 虽 然 斗 皇  强 者 ， 可 这 对 付 敖 
这 位 年 轻 人 的 实 力 ， 也 是 极 为 严 厉 。
----------------------------------------
```

简单的效果演示，对于模型参数不大，训练时间也短的程序来说，效果还是不错的，至少说的话是大体通顺，能够阅读的。读者可以自己选择更大的数据集和训练更多的轮次，当然还有设置更深的注意力层数和更大的词向量维度，你就能得到一个非常不错的文本生成模型，比如如果喜欢诗词的话，可以用诗词训练诗词生成模型，喜欢红楼梦，可以用红楼梦训练红楼梦风格文字生成模型。

### 总结

再总结一下生成模型和翻译模型（当然不只是翻译，而是任意的`seq2seq任务`）：

- 生成模型

```
Input
  │
  ▼
[Masked Self-Attention]  ←─ 使用 tgt_mask
  │
  ▼
[FFN]
  │
  ▼
Output


```

- 翻译模型

```
Input
  │
  ▼
[Masked Self-Attention]  ←─ 使用 tgt_mask
  │
  ▼
[Cross-Attention]        ←─ 使用 encoder 输出作为 K, V
  │
  ▼
[FFN]
  │
  ▼
Output
```
还有，虽然说单`decoder`模型适合生成任务，但是不代表不能做序列到序列的任务，比如翻译，也是可以做的。只是换个思路，将原始语料当成式输入的提示词。比如“我爱你”翻译成“I love you”，把我爱你作为提示词，词向量化+位置编码后输入mask多头注意力模块，更高效的方式是再加一个标志位，比如<翻译>,或者更具体的<中英翻译>,这样模型可以更高效的训练，然后下一个词的概率分布和“I”这个label做交叉熵损失，然后更新参数，接着输入mask后移变成“我爱你 I”，继续训练，下个词概率分布再和“love”做交叉熵损失，一直重复，就可以进行训练。推理同理，只需要把原始语料+<翻译>标签作为输入，得到下一词分布后采样，结果加入输入序列，重复，直到停止位即可。










