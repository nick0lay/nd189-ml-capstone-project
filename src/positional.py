from this import s
import torch, math
import torch.nn as nn

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int, batch_first: bool = False, dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        if self.batch_first:
            x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0)]
        if self.batch_first:
            x = x.transpose(0, 1)
        return self.dropout(x)

class Time2Vector(nn.Module):
    
    def __init__(self, feature_num: int, seq_len: int):
        """
        Apply time to vector embedding.

        Args:
            sequnce_size: size of event sequence in time
            feature_num: number of features in sequence
        """
        super().__init__()
        
        self.seq_len = seq_len
        self.feature_num = feature_num
        self.weights_linear = torch.nn.Parameter(torch.empty((seq_len, feature_num)))
        self.bias_linear = torch.nn.Parameter(torch.empty(feature_num))
        self.weights_periodic = torch.nn.Parameter(torch.empty((seq_len, feature_num)))
        self.bias_periodic = torch.nn.Parameter(torch.empty(feature_num))
        self.func_periodic = torch.sin
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize component weights
        """
        # init linear weights
        nn.init.uniform_(self.weights_linear, -1, 1)
        nn.init.uniform_(self.bias_linear, -1, 1)
        
        # init periodic weights
        nn.init.uniform_(self.weights_periodic, -1, 1)
        nn.init.uniform_(self.bias_periodic, -1, 1)

    def forward(self, x):
        weights_linear = self.weights_linear.expand(x.shape[0], self.seq_len, self.feature_num)
        linear = torch.mul(weights_linear, x) + self.bias_linear
        weights_periodic = self.weights_periodic.expand(x.shape[0], self.seq_len, self.feature_num)
        periodic = self.func_periodic(torch.mul(weights_periodic, x) + self.bias_periodic)
        return torch.cat([linear, periodic], 2)