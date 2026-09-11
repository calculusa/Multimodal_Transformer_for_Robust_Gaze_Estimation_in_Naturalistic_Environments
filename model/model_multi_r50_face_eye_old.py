"""
model_multi_face_eye_resnet50.py  (L1 loss)
多视图输入: face(448), left/right eye(112) [, frame(可选)]
输出: yaw, pitch (degree)

要点：
- backbone: resnet50(pretrained=True, maps=<C>)，和你的 resnet50 实现完全对齐
- eyes 共享同一个 resnet50 实例
- 一次性分配 pos_embed；类型嵌入区分 face/eye(/frame)
- L1 损失（度）
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet50_for_old import resnet50  # 使用你给我的这份 resnet50(…, maps=)

# --------------------- Transformer --------------------- #
def _clones(m, N):
    return nn.ModuleList([copy.deepcopy(m) for _ in range(N)])

class EncoderLayer(nn.Module):
    def __init__(self, d, heads, ff=512, p=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=p, batch_first=False)
        self.lin1, self.lin2 = nn.Linear(d, ff), nn.Linear(ff, d)
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop, self.act = nn.Dropout(p), nn.ReLU(inplace=True)

    def forward(self, x, pos):  # x:(S,B,C), pos:(S,C)
        q = k = x + pos.unsqueeze(1)
        attn_out, _ = self.attn(q, k, value=x)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.lin2(self.drop(self.act(self.lin1(x))))))
        return x

class Encoder(nn.Module):
    def __init__(self, layer, N, norm=None):
        super().__init__()
        self.layers = _clones(layer, N)
        self.norm = norm

    def forward(self, x, pos):
        for l in self.layers:
            x = l(x, pos)
        return self.norm(x) if self.norm else x

# ----------------------- Model ------------------------- #
class ModelMulti(nn.Module):
    def __init__(self,
                 face_size: int = 448,
                 eye_size:  int = 112,
                 maps:      int = 64,
                 layers:    int = 6,
                 max_seq_hint: int = 1024,
                 use_frame: bool = False):
        super().__init__()
        self.use_frame = use_frame

        # CNN 分支 (ResNet-50)
        self.back_face  = resnet50(pretrained=True, maps=maps)
        self.back_eye   = resnet50(pretrained=True, maps=maps)   # 左右共享
        if self.use_frame:
            self.back_frame = resnet50(pretrained=True, maps=maps)

        C = maps

        # CLS / 位置 / 类型嵌入
        self.cls_token = nn.Parameter(torch.randn(1, 1, C))
        self.pos_embed = nn.Embedding(max_seq_hint + 1, C)  # +1 for CLS
        type_vocab = 3 if self.use_frame else 2             # 0 face | 1 eye | [2 frame]
        self.type_embed = nn.Embedding(type_vocab, C)

        # Transformer
        enc_layer = EncoderLayer(C, heads=8, ff=512, p=0.1)
        self.encoder = Encoder(enc_layer, layers, norm=nn.LayerNorm(C))

        # 回归头（yaw, pitch in degree）
        self.feed = nn.Linear(C, 2)

        # 训练损失（与标注一致：度）
        self.loss_op = nn.L1Loss()

    @staticmethod
    def _enc(img, backbone):
        feat = backbone(img)                     # (B,C,H',W')  448->14x14, 112->4x4
        return feat.flatten(2).permute(2, 0, 1)  # (S,B,C)

    def forward(self, x):
        B = x["face"].size(0)
        seqs, types = [], []

        def push(tens, tid):
            seqs.append(tens)
            types.append(torch.full((tens.size(0),), tid,
                                    dtype=torch.long, device=tens.device))

        # face + eyes
        push(self._enc(x["face"],  self.back_face), 0)
        push(self._enc(x["eye_l"], self.back_eye),  1)
        push(self._enc(x["eye_r"], self.back_eye),  1)

        # frame (可选)
        if self.use_frame and ("frame" in x):
            push(self._enc(x["frame"], self.back_frame), 2)

        tok = torch.cat(seqs, 0)      # (S,B,C)
        typ = torch.cat(types)        # (S,)
        tok = tok + self.type_embed(typ).unsqueeze(1)

        S = tok.size(0)
        if S + 1 > self.pos_embed.num_embeddings:
            raise RuntimeError(f"pos_embed too small: need {S+1}, have {self.pos_embed.num_embeddings}")
        pos = self.pos_embed(torch.arange(S, device=tok.device))  # (S,C)

        cls = self.cls_token.expand(-1, B, -1)                    # (1,B,C)
        tok = torch.cat([cls, tok], 0)                            # (S+1,B,C)
        pos = torch.cat([self.pos_embed.weight[:1], pos], 0)      # (S+1,C)

        out = self.encoder(tok, pos)          # (S+1,B,C)
        cls_vec = out[0]                      # (B,C)
        return self.feed(cls_vec)             # (B,2) yaw,pitch (deg)
