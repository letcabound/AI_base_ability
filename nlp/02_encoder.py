# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
QA:
1. einsum 规则
    0) -> 左侧的字母是随意写的，只做张量的占位符和标识。但是 相同的字母必须表示相同大小的维度
    1) -> 右侧有的维度一律保留
    2) -> 右侧没有的维度 点积再求和
2. einsum 规则理解：当过于复杂时，根据右侧的维度挨个 q=0  q=1 枚举。
3. 写 einsum：只要清楚两个张量的维度含义和之间的变化过程，准确表达写出来，einsum自然就是正确的。
"""

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.embed_size = embed_size
        self.head_dim = embed_size // heads

        assert (self.head_dim * self.heads == self.embed_size), "Embedding size must be divisible by number of heads"

        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(self.head_dim * self.heads, self.embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0] # batch
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # 分别将 q k v 切分多个Head. [batch, len, embedding] => [batch, len, heads, head_dim]
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # 切分完成之后，对每个头都自己线性变换，自己学习一下
        # q k v 分别经过了不同的Linear进行线性变换，所以 输入参数q k v 可以是完全相同的张量
        queries = self.queries(queries)
        keys = self.keys(keys)
        values = self.values(values)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) # nqhd,nkhd是输入的shape，nhqk是输出的shape

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # energy shape: [batch, head, q_len, k_len]
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=-1)

        out = torch.einsum("nhqk,nkhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(), # 激活函数的作用，主要是 引入非线性，所以在 中间的这一层layer上加 激活。但是 最后输出的一层layer就不加
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # 自注意力模块之后的 Add + Norm
        x = self.dropout(self.norm1(query + attention))

        # FFN模块以及之后的 Add + Norm
        ffn_out = self.feed_forward(x)
        out = self.dropout(self.norm2(x + ffn_out))

        return out


class Encoder(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 embed_size,
                 num_layers,
                 heads,
                 device,
                 forward_expansion,
                 dropout,
                 max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        # 通过extend 广播，生成 位置编码用的索引矩阵
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

