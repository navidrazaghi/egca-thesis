"""Attention operators: kernel-based linear cross-attention (Eq. 3.16-3.17)
and standard full softmax cross-attention (used only in ablations).

Linear attention (Katharopoulos et al., 2020):
    Att(Q,K,V)_i = phi(q_i)^T (sum_j phi(k_j) v_j^T) / (phi(q_i)^T sum_j phi(k_j))
with phi(x) = elu(x) + 1, giving O((N_q + N_k) d d_h) cost for the whole
multi-head layer (d_h = d / H) — see Appendix A of the thesis.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _phi(x):
    return F.elu(x) + 1.0


class LinearCrossAttention(nn.Module):
    """Kernel-based linear cross-attention.

    The key/value projections share one GEMM, and the two contractions are
    plain batched matmuls, so the whole layer costs O((N_q + N_k) d d_h) and —
    unlike softmax attention — never materializes an N_q x N_k matrix.

    The K/V summation adds N_k strictly positive terms, which overflows the
    fp16 range for large N_k.  Averaging instead of summing (the 1/N_k factor
    below) cancels between numerator and denominator, leaves the operator
    mathematically unchanged, and keeps the accumulator O(1) so that the layer
    is safe under mixed precision.
    """

    def __init__(self, dim, num_heads=4):
        super().__init__()
        assert dim % num_heads == 0
        self.h = num_heads
        self.dh = dim // num_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wkv = nn.Linear(dim, 2 * dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, q_tokens, kv_tokens):
        b, nq, d = q_tokens.shape
        nk = kv_tokens.shape[1]
        q = _phi(self.wq(q_tokens).view(b, nq, self.h, self.dh).transpose(1, 2))
        k, v = self.wkv(kv_tokens).view(b, nk, 2, self.h, self.dh) \
                   .permute(2, 0, 3, 1, 4)
        k = _phi(k)
        # S = (1/N_k) sum_j phi(k_j) v_j^T : B x H x d_h x d_h
        S = torch.matmul(k.transpose(-2, -1), v) / nk
        u = k.mean(dim=2)                                   # B x H x d_h
        num = torch.matmul(q, S)                            # B x H x N_q x d_h
        den = (q * u.unsqueeze(2)).sum(-1, keepdim=True).clamp(min=1e-6)
        out = (num / den).transpose(1, 2).reshape(b, nq, d)
        return self.wo(out)


class FullCrossAttention(nn.Module):
    """Standard O(N_q N_k) softmax attention — ablation baseline (Eq. 3.11)."""

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, q_tokens, kv_tokens):
        out, _ = self.attn(q_tokens, kv_tokens, kv_tokens, need_weights=False)
        return out


def make_attention(kind, dim, num_heads):
    if kind == "linear":
        return LinearCrossAttention(dim, num_heads)
    if kind == "full":
        return FullCrossAttention(dim, num_heads)
    raise ValueError(f"unknown attention kind: {kind}")
