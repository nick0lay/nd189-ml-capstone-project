import torch.nn as nn
import torch
import torch.nn.functional as F
from positional import PositionalEncoding
from positional import Time2Vector

class Transformer(nn.Module):

    def __init__(self, feature_size=2, num_layers=6, dropout=0.2, use_mask=True):
        super(Transformer, self).__init__()

        self.embedding_size = 256
        self.embedding = nn.Linear(feature_size, self.embedding_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_size, nhead=8, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.decoder = nn.Linear(self.embedding_size*7, 1) #TODO change to seq_len
        self.dropout = nn.Dropout(dropout)
        self.use_mask = use_mask
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.encoder_layer.linear1.bias.data.zero_()
        self.encoder_layer.linear1.weight.data.uniform_(-initrange, initrange)
        self.encoder_layer.linear2.bias.data.zero_()
        self.encoder_layer.linear2.weight.data.uniform_(-initrange, initrange)
        self.encoder_layer.norm1.bias.data.zero_()
        self.encoder_layer.norm1.weight.data.uniform_(-initrange, initrange)
        self.encoder_layer.norm2.bias.data.zero_()
        self.encoder_layer.norm2.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        output = F.relu(self.embedding(src))
        output = self.dropout(output)
        mask = None
        if self.use_mask:
            mask = self._generate_square_subsequent_mask(len(output))
        output = self.transformer_encoder(output, mask)
        output = self.dropout(output)
        # print(f"Encoder out: {output.shape}")
        # print(f"Encoder out fl: {output.flatten(start_dim=1).shape}")
        output = self.decoder(output.flatten(start_dim=1))
        return output


class TransformerPos(nn.Module):
    """
    Transformer with positional encoder
    """

    def __init__(self, feature_size=2, seq_len=7, num_layers=6, dropout=0.2, use_mask=True):
        super(TransformerPos, self).__init__()
        self.embedding_size = 256
        self.posEncoder = PositionalEncoding(feature_size, seq_len, True)
        self.embedding = nn.Linear(feature_size, self.embedding_size)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_size,
            nhead=8,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(self.embedding_size*seq_len, 1) #TODO change to seq_len
        self.dropout = nn.Dropout(dropout)
        self.use_mask = use_mask
        self.seq_len = seq_len
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.encoder_layer.linear1.bias.data.zero_()
        self.encoder_layer.linear1.weight.data.uniform_(-initrange, initrange)
        self.encoder_layer.linear2.bias.data.zero_()
        self.encoder_layer.linear2.weight.data.uniform_(-initrange, initrange)
        self.encoder_layer.norm1.bias.data.zero_()
        self.encoder_layer.norm1.weight.data.uniform_(-initrange, initrange)
        self.encoder_layer.norm2.bias.data.zero_()
        self.encoder_layer.norm2.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, src):
        output = self.posEncoder(src)
        output = F.relu(self.embedding(output))
        output = self.dropout(output)
        mask = None
        if self.use_mask:
            mask = self._generate_square_subsequent_mask(self.seq_len)
        output = self.transformer_encoder(output, mask)
        output = self.dropout(output)
        output = self.decoder(output.flatten(start_dim=1))
        return output

class TransformerTime2Vec(nn.Module):
    """
    Transformer with time to vector encoder
    """

    # TODO fix time 2 vector
    def __init__(self, feature_size=2, num_layers=6, dropout=0.2, use_mask=True):
        super(TransformerTime2Vec, self).__init__()
        self.embedding_size = 256
        self.time2vecEncoder = Time2Vector(feature_size, 7)
        self.embedding = nn.Linear(feature_size*2+feature_size, self.embedding_size)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_size,
            nhead=8,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(self.embedding_size*7, 1) #TODO change to seq_len
        self.dropout = nn.Dropout(dropout)
        self.use_mask = use_mask
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.encoder_layer.linear1.bias.data.zero_()
        self.encoder_layer.linear1.weight.data.uniform_(-initrange, initrange)
        self.encoder_layer.linear2.bias.data.zero_()
        self.encoder_layer.linear2.weight.data.uniform_(-initrange, initrange)
        self.encoder_layer.norm1.bias.data.zero_()
        self.encoder_layer.norm1.weight.data.uniform_(-initrange, initrange)
        self.encoder_layer.norm2.bias.data.zero_()
        self.encoder_layer.norm2.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, mask):
        # print(f"Input shape: {src.shape}")
        output = self.time2vecEncoder(src)
        # print(f"Time embedding shape: {output.shape}")
        output = torch.cat([src, output], 2)
        # print(f"Concatenated shape: {output.shape}")
        output = F.relu(self.embedding(output))
        output = self.dropout(output)
        # mask = None
        # if self.use_mask:
        #     mask = self._generate_square_subsequent_mask(7)
        output = self.transformer_encoder(output, mask)
        output = self.dropout(output)
        output = self.decoder(output.flatten(start_dim=1))
        return output

class TransformerPosConv(nn.Module):
    """
    Transformer with positional encoder
    """

    def __init__(self, feature_size=2, num_layers=6, dropout=0.2, use_mask=True):
        super(TransformerPos, self).__init__()
        self.embedding_size = 256
        self.posEncoder = PositionalEncoding(feature_size, 7, True)
        self.embedding = nn.Linear(feature_size, self.embedding_size)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_size,
            nhead=8,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(self.embedding_size*7, 1) #TODO change to seq_len
        self.dropout = nn.Dropout(dropout)
        self.use_mask = use_mask
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, src):
        output = self.posEncoder(src)
        output = F.relu(self.embedding(output))
        output = self.dropout(output)
        mask = None
        if self.use_mask:
            mask = self._generate_square_subsequent_mask(7)
        output = self.transformer_encoder(output, mask)
        output = self.dropout(output)
        output = self.decoder(output.flatten(start_dim=1))
        return output