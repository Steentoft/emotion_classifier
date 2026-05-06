import math
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, d_model, d_key):
        super().__init__()
        self.w_q = nn.Linear(d_model, d_key)
        self.w_k = nn.Linear(d_model, d_key)
        self.w_v = nn.Linear(d_model, d_model)

    def forward(self, x):
        w_q = self.w_q(x)
        w_k = self.w_k(x)
        w_v = self.w_v(x)
        qk = w_q @ torch.transpose(w_k, -2, -1)
        qk = qk / math.sqrt(w_k.shape[-1])
        return torch.softmax(qk, -1) @ w_v


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, d_key, n_heads):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(d_model, d_key) for _ in range(n_heads)])
        self.w_o = nn.Linear(n_heads * d_model, d_model)

    def forward(self, x):
        heads = [head(x) for head in self.heads]
        return self.w_o(torch.cat(heads, -1))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_key, n_heads, mlp_factor=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, d_key, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_factor * d_model),
            nn.SiLU(),
            nn.Linear(mlp_factor * d_model, d_model),
        )

    def forward(self, x):
        ln1 = self.ln1(x)
        attn = self.attn(ln1) + x
        x = self.ln2(attn)
        return self.mlp(x) + attn


class TransformerClassifier(nn.Module):
    def __init__(self, n_embeds, n_classes, d_model=256, d_key=64, n_heads=4, mlp_factor=4, n_layers=2):
        super().__init__()
        self.token_embedding = nn.Embedding(n_embeds, d_model)
        self.transformer_model = nn.Sequential(
            *[TransformerBlock(d_model, d_key, n_heads, mlp_factor) for _ in range(n_layers)]
        )
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x):
        x = self.token_embedding(x)
        x = self.transformer_model(x)
        x = self.final_layer_norm(x)
        return self.classifier(torch.mean(x, -2))