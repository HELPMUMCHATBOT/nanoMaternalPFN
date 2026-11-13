"""Missingness-aware alternating-attention PFN, for research only."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class ModelConfig:
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 3
    mlp_hidden: int = 192
    max_classes: int = 3
    architecture: str = 'post_ln_v1'
    feature_encoder_type: str = 'linear'
    normalization: str = 'standard'

class MissingnessFeatureEncoder(nn.Module):
    def __init__(self, d_model, encoder_type='linear', normalization='standard'):
        super().__init__()
        if encoder_type not in ('linear', 'gated'):
            raise ValueError('encoder_type must be linear or gated')
        if normalization not in ('standard', 'robust'):
            raise ValueError('normalization must be standard or robust')
        self.encoder_type = encoder_type
        self.normalization = normalization
        self.proj = nn.Linear(2, d_model)
        if encoder_type == 'gated':
            # Shared across features: permutation-equivariant representation.
            self.gate = nn.Sequential(nn.Linear(2, d_model),
                                      nn.GELU(), nn.Linear(d_model, d_model))

    @staticmethod
    def encode_inputs(x, n_train, normalization='standard'):
        if x.ndim != 3 or n_train < 1 or n_train > x.shape[1]:
            raise ValueError('x must be (batch, rows, features) with >=1 context row')
        if not torch.isfinite(torch.nan_to_num(x, nan=0.0)).all():
            raise ValueError('input contains infinity')
        context = x[:, :n_train]
        observed = ~torch.isnan(context)
        counts = observed.sum(dim=1, keepdim=True).clamp(min=1)
        clean = torch.nan_to_num(context, nan=0.0)
        if normalization == 'robust':
            # Context-only median/MAD. No query observations enter statistics.
            valid = torch.where(observed, context, torch.full_like(context, float('nan')))
            mean = torch.nanmedian(valid, dim=1, keepdim=True).values
            mean = torch.nan_to_num(mean, nan=0.0)
            dev = torch.where(observed, (context-mean).abs(), torch.full_like(context, float('nan')))
            std = 1.4826 * torch.nanmedian(dev, dim=1, keepdim=True).values
            std = torch.nan_to_num(std, nan=1.0).clamp_min(1e-3)
        elif normalization == 'standard':
            mean = clean.sum(dim=1, keepdim=True) / counts
            var = torch.where(observed, context - mean, torch.zeros_like(context)).square().sum(dim=1, keepdim=True) / counts
            std = torch.sqrt(var + 1e-6)
        else:
            raise ValueError('unknown normalization')
        missing = torch.isnan(x)
        normalized = torch.where(missing, torch.zeros_like(x), (torch.nan_to_num(x, nan=0.0) - mean) / std)
        normalized = normalized.clamp(-10.0, 10.0)
        return torch.stack([normalized, missing.to(normalized.dtype)], dim=-1)

    def forward(self, x, n_train):
        encoded = self.encode_inputs(x, n_train, self.normalization)
        tokens = self.proj(encoded)
        if self.encoder_type == 'gated':
            tokens = tokens * (1 + torch.tanh(self.gate(encoded)))
        return tokens

class TargetEncoder(nn.Module):
    def __init__(self, d_model, max_classes):
        super().__init__()
        self.max_classes = max_classes
        self.proj = nn.Linear(max_classes, d_model)

    def forward(self, y_train, n_query):
        if y_train.ndim != 2 or n_query < 0:
            raise ValueError('y_train must be (batch, context rows)')
        if not torch.all((y_train >= 0) & (y_train < self.max_classes)):
            raise ValueError('class label outside configured max_classes')
        onehot = F.one_hot(y_train.long(), self.max_classes).float()
        mean = onehot.mean(dim=1, keepdim=True).expand(-1, n_query, -1)
        return self.proj(torch.cat([onehot, mean], dim=1)).unsqueeze(2)

class BiAttentionBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.architecture = cfg.architecture
        self.feat_attn = nn.MultiheadAttention(d, cfg.n_heads, batch_first=True)
        self.pat_attn = nn.MultiheadAttention(d, cfg.n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm3 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, cfg.mlp_hidden), nn.GELU(), nn.Linear(cfg.mlp_hidden, d))

    def forward(self, e, n_train):
        b, n, cols, d = e.shape
        cells = e.reshape(b * n, cols, d)
        if self.architecture == 'pre_ln_v2':
            h = self.norm1(cells)
            feat, _ = self.feat_attn(h, h, h, need_weights=False)
            e = (cells + feat).reshape(b, n, cols, d)
            pat = e.transpose(1, 2).reshape(b * cols, n, d)
            h = self.norm2(pat)
            update, _ = self.pat_attn(h, h[:, :n_train], h[:, :n_train], need_weights=False)
            pat = pat + update
            pat = pat + self.mlp(self.norm3(pat))
            return pat.reshape(b, cols, n, d).transpose(1, 2)
        # Restore exact Post-LN semantics for every old v1 checkpoint.
        feat, _ = self.feat_attn(cells, cells, cells, need_weights=False)
        e = self.norm1(cells + feat).reshape(b, n, cols, d)
        pat = e.transpose(1, 2).reshape(b * cols, n, d)
        context = pat[:, :n_train]
        update, _ = self.pat_attn(pat, context, context, need_weights=False)
        e = self.norm2(pat + update).reshape(b, cols, n, d).transpose(1, 2)
        return self.norm3(e + self.mlp(e))

class NanoMaternalPFN(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        if self.cfg.d_model % self.cfg.n_heads:
            raise ValueError('d_model must be divisible by n_heads')
        if self.cfg.architecture not in ('post_ln_v1', 'pre_ln_v2'):
            raise ValueError('Unknown model architecture version')
        self.feature_encoder = MissingnessFeatureEncoder(
            self.cfg.d_model, self.cfg.feature_encoder_type, self.cfg.normalization
        )
        self.target_encoder = TargetEncoder(self.cfg.d_model, self.cfg.max_classes)
        self.layers = nn.ModuleList([BiAttentionBlock(self.cfg) for _ in range(self.cfg.n_layers)])
        # Zero added trainable parameters; old checkpoint keys remain compatible.
        self.final_norm = (nn.LayerNorm(self.cfg.d_model, elementwise_affine=False)
                           if self.cfg.architecture == 'pre_ln_v2' else nn.Identity())
        self.decoder = nn.Sequential(nn.Linear(self.cfg.d_model, self.cfg.mlp_hidden), nn.GELU(), nn.Linear(self.cfg.mlp_hidden, self.cfg.max_classes))

    def forward(self, x, y_train, n_classes=None):
        if x.shape[0] != y_train.shape[0] or x.shape[1] <= y_train.shape[1]:
            raise ValueError('need at least one query and matching batch sizes')
        n_ctx = y_train.shape[1]
        n_classes = n_classes or self.cfg.max_classes
        if not 2 <= n_classes <= self.cfg.max_classes:
            raise ValueError('n_classes must be in [2, max_classes]')
        if torch.any(y_train >= n_classes):
            raise ValueError('labels exceed n_classes')
        e = torch.cat([self.feature_encoder(x, n_ctx), self.target_encoder(y_train, x.shape[1] - n_ctx)], dim=2)
        for layer in self.layers:
            e = layer(e, n_ctx)
        return self.decoder(self.final_norm(e[:, n_ctx:, -1, :]))[..., :n_classes]

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def predict_proba(self, x, y_train, n_classes=None, temperature=1.0):
        self.eval()
        if temperature <= 0:
            raise ValueError('temperature must be >0')
        return F.softmax(self(x, y_train, n_classes) / temperature, dim=-1)
