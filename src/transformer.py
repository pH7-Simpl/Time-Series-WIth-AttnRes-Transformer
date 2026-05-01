import torch
import torch.nn as nn
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════════════════════
# FULL ATTENTION RESIDUALS (AttnRes) — Aligned with arXiv:2603.15031v1
# Section 3.1: Full Attention Residuals
# ═══════════════════════════════════════════════════════════════════════════════

def full_attn_res(sources, proj, norm):
    """
    Full Attention Residual operation.

    Each layer attends over ALL preceding individual layer outputs.

    Args:
        sources: list[Tensor] — all preceding source representations
                 sources[0] = token embedding (v_0 = h_1)
                 sources[i] = output of layer i (v_i = f_i(h_i)) for i >= 1
                 Each tensor has shape [B, T, D]
        proj:    nn.Parameter — pseudo-query vector w_l in R^d (learnable)
        norm:    nn.RMSNorm   — normalization applied to keys

    Returns:
        h: Tensor of shape [B, T, D] — weighted aggregation of sources
    """
    V = torch.stack(sources, dim=0)

    K = norm(V)

    logits = torch.einsum('d,lbtd->lbt', proj, K)

    weights = logits.softmax(dim=0)

    h = torch.einsum('lbt,lbtd->btd', weights, V)

    return h


class FullAttnResTransformerLayer(nn.Module):
    """
    One Transformer layer with Full Attention Residuals.

    Each layer l:
      1. Computes h_attn = AttnRes([v_0, v_1, ..., v_{l-1}])
      2. Applies Self-Attention: attn_out = SelfAttn(LayerNorm(h_attn))
      3. Accumulates: v_l_attn = v_{l-1} + attn_out  (standard residual)
      4. Computes h_mlp = AttnRes([v_0, ..., v_{l-1}, v_l_attn])
      5. Applies FFN: mlp_out = FFN(LayerNorm(h_mlp))
      6. Accumulates: v_l = v_l_attn + mlp_out
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(d_model)

        self.w_attn = nn.Parameter(torch.zeros(d_model))
        self.norm_attn = nn.RMSNorm(d_model)

        self.w_ffn = nn.Parameter(torch.zeros(d_model))
        self.norm_ffn = nn.RMSNorm(d_model)

    def forward(self, sources, v_prev):
        """
        Args:
            sources: list[Tensor] — [v_0, v_1, ..., v_{l-1}], each [B, T, D]
            v_prev:  Tensor       — v_{l-1}, immediate predecessor [B, T, D]

        Returns:
            v_l: Tensor — output of this layer [B, T, D]
        """
        h_attn = full_attn_res(sources, self.w_attn, self.norm_attn)
        T = h_attn.size(1)
        causal_mask = torch.triu(torch.ones(T, T, device=h_attn.device), diagonal=1).bool()

        attn_out, _ = self.self_attn(
            self.attn_norm(h_attn),
            self.attn_norm(h_attn),
            self.attn_norm(h_attn),
            attn_mask=causal_mask
        )

        v_l_attn = v_prev + attn_out

        sources_with_current = sources + [v_l_attn]
        h_mlp = full_attn_res(sources_with_current, self.w_ffn, self.norm_ffn)

        mlp_out = self.ffn(self.ffn_norm(h_mlp))

        v_l = v_l_attn + mlp_out

        return v_l


class FullAttnResTimeSeriesTransformer(nn.Module):
    """
    Time Series Transformer with Full Attention Residuals.

    Architecture:
      - Input projection to d_model
      - L Transformer layers with Full AttnRes
      - Output projection to output_dim (e.g., pred_horizon=24)

    Each layer l attends over ALL l preceding individual layer outputs.
    """
    def __init__(self, input_dim, d_model, n_heads, num_layers,
                 d_ff, dropout, max_len, output_dim):
        """
        Args:
            input_dim:  int — number of input features (e.g., 7 for ETTh1)
            d_model:    int — embedding dimension
            n_heads:    int — attention heads
            num_layers: int — number of transformer layers
            d_ff:       int — feed-forward hidden dimension
            output_dim: int — number of output values (e.g., 24 for 24h forecast)
            dropout:    float — dropout rate
        """
        super().__init__()

        self.max_len = max_len
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model))
        self.input_proj = nn.Linear(input_dim, d_model)

        self.layers = nn.ModuleList([
            FullAttnResTransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(d_model, output_dim)
        self.num_layers = num_layers

    def forward(self, x):
        """
        Args:
            x: Tensor [B, T, input_dim] — input time series

        Returns:
            out: Tensor [B, output_dim] — predicted values
        """
        h = self.input_proj(x)
        h = h + self.pos_emb[:, :h.size(1), :]

        sources = [h]
        v_prev = h

        for layer in self.layers:
            v_l = layer(sources, v_prev)
            sources.append(v_l)
            v_prev = v_l

        out = v_prev[:, -1, :]
        return self.fc_out(out)

if __name__ == "__main__":
    BATCH_SIZE = 4
    SEQ_LEN = 96          
    INPUT_DIM = 7         
    D_MODEL = 64
    N_HEADS = 4
    NUM_LAYERS = 4
    D_FF = 256
    OUTPUT_DIM = 24

    model = FullAttnResTimeSeriesTransformer(
        input_dim=INPUT_DIM,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        output_dim=OUTPUT_DIM,
        dropout=0.1
    )

    x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM)
    out = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Expected:     [{BATCH_SIZE}, {OUTPUT_DIM}]")

    for i, layer in enumerate(model.layers):
        assert torch.allclose(layer.w_attn, torch.zeros_like(layer.w_attn)), \
            f"Layer {i} w_attn not zero-initialized!"
        assert torch.allclose(layer.w_ffn, torch.zeros_like(layer.w_ffn)), \
            f"Layer {i} w_ffn not zero-initialized!"
    print("\n✓ All pseudo-queries are zero-initialized (correct!)")

    total_params = sum(p.numel() for p in model.parameters())
    attnres_params = sum(
        layer.w_attn.numel() + layer.w_ffn.numel()
        for layer in model.layers
    )
    print(f"\nTotal parameters: {total_params:,}")
    print(f"AttnRes parameters: {attnres_params:,} ({attnres_params/total_params*100:.3f}% of total)")
    print("  (negligible overhead — exactly as paper claims)")