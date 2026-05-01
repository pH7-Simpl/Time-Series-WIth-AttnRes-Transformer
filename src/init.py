import torch.nn as nn
from src.rnn import RNNModel
from src.lstm import LSTMModel
from src.gru import GRUModel
from src.transformer import FullAttnResTimeSeriesTransformer

def init_rnn(model:RNNModel) -> RNNModel:
    """
    Orthogonal for recurrent weights — preserves gradient magnitude across timesteps.
    Xavier for input weights — balances input/output variance.
    """
    for name, param in model.named_parameters():
        if 'weight_hh' in name:
            nn.init.orthogonal_(param)
        elif 'weight_ih' in name:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)
    
    return model

def init_lstm(model: LSTMModel) -> LSTMModel:
    """
    Same as RNN + forget gate bias = 1.
    Forget gate bias trick makes LSTM remember by default early in training.
    """
    for name, param in model.named_parameters():
        if 'weight_hh' in name:
            nn.init.orthogonal_(param)
        elif 'weight_ih' in name:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)
            hidden = param.shape[0] // 4
            param.data[hidden:2*hidden].fill_(1.0)
    
    return model

def init_gru(model:GRUModel) -> GRUModel:
    """GRU same as RNN — no gate bias trick needed."""
    init_rnn(model)
    return model

def init_fullattnres(model:FullAttnResTimeSeriesTransformer) -> FullAttnResTimeSeriesTransformer:
    """
    Xavier uniform for linear projections.
    Small normal for attention projections in MultiheadAttention.
    w_attn & w_ffn stay zero — already correct per paper.
    RMSNorm weight stays at 1 (default).
    """
    for name, param in model.named_parameters():
        if 'w_attn' in name or 'w_ffn' in name:
            pass

        elif 'pos_emb' in name:
            nn.init.normal_(param, mean=0.0, std=0.02)

        elif 'weight' in name and param.dim() == 2:
            nn.init.xavier_uniform_(param)

        elif 'bias' in name:
            nn.init.zeros_(param)

        elif param.dim() == 1:
            pass
    
    return model