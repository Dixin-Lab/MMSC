import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o

class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim),
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        # linear_out = self.linear_net(x)
        # x = x + self.dropout(linear_out)
        # x = self.norm2(x)

        return x


class _TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderBlock(**block_args) for _ in range(num_layers)]
        )

    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps


class Self_attention(nn.Module):
    def __init__(self,
                 input_dim=None,
                 model_dim=None,
                 num_layers=2,
                 num_heads=4,
                 dropout=0.2):
        super(Self_attention, self).__init__()

        # Transformer
        self.transformer = _TransformerEncoder(
            num_layers=num_layers,
            input_dim=model_dim,
            dim_feedforward=2 * model_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(self, x, mask=None):
        x = self.transformer(x, mask=mask)
        return x
    

class Cross_attention_Encoder_Layer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.2):
        super(Cross_attention_Encoder_Layer, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, Q, KV, src_mask=None, src_key_padding_mask=None):
        src, attention_map = self.cross_attn(Q, KV, KV)
        Q2 = Q + self.dropout1(src)
        Q = self.norm1(Q2)
        
        src2 = self.linear2(self.dropout(F.relu(self.linear1(Q))))
        Q2 = Q + self.dropout2(src2)
        Q = self.norm2(Q2)
        
        return Q, attention_map



class MMSC_model(nn.Module):
    def __init__(self, output_dim, num_layers, heads, dropout=0.2):
        super(MMSC_model, self).__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout

        # Text-Vision encoder
        self.tm_cross_attention = Cross_attention_Encoder_Layer(d_model=1024, nhead=self.heads)

        # Acoustic context encoder
        audio_encoder_layers = nn.TransformerEncoderLayer(d_model=512, nhead=self.heads, dropout=self.dropout)
        self.acoustic_context_encoder = nn.TransformerEncoder(audio_encoder_layers, num_layers=num_layers)

        # Visual context encoder
        video_encoder_layers = nn.TransformerEncoderLayer(d_model=1024, nhead=self.heads, dropout=self.dropout)
        self.visual_context_encoder = nn.TransformerEncoder(video_encoder_layers, num_layers=num_layers)

        # projectors
        self.audio_proj = nn.Sequential(
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Linear(1024, self.output_dim)
        )
        self.video_proj = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, self.output_dim)
        )

        # Multi-modal fusion
        self.tma_cross_attention = Cross_attention_Encoder_Layer(d_model=self.output_dim, nhead=self.heads, dropout=self.dropout)
        
        # trailerness prediction head
        self.trailerness_proj = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.GELU(),
            nn.Linear(self.output_dim, 1)
        )
        self.trailerness_sigmoid = nn.Sigmoid()

        # emotional prediection head
        self.emotional_proj = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.GELU(),
            nn.Linear(self.output_dim, 1)
        )
        self.emotional_sigmoid = nn.Sigmoid()


    
    def forward(self, v_I, a_A, w_label, w_plot, label_exist, plot_exist):

        # concat text and video
        if label_exist == True and plot_exist == True:
            w = torch.cat((w_label, w_plot), dim=0)
            # Text-Vision encoder
            wv_I, attention_map_1 = self.tm_cross_attention(v_I, w)
        elif label_exist == True and plot_exist == False:
            w = w_label
            # Text-Vision encoder
            wv_I, attention_map_1 = self.tm_cross_attention(v_I, w)
        elif label_exist == False and plot_exist == True:
            w = w_plot
            # Text-Vision encoder
            wv_I, attention_map_1 = self.tm_cross_attention(v_I, w)
        else:
            wv_I = v_I

        # Acoustic context encoder
        a_A = self.acoustic_context_encoder(a_A)
        # Visual context encoder
        wv_I = self.visual_context_encoder(wv_I)
        
        # through projectors
        a_A_hat = self.audio_proj(a_A)
        wv_I_hat = self.video_proj(wv_I)

        # Multi-modal fusion
        wva_I_hat, attention_map_2 = self.tma_cross_attention(wv_I_hat, a_A_hat)

        # trailerness prediction head
        trailerness_linear = self.trailerness_proj(wva_I_hat)
        trailerness_pre = self.trailerness_sigmoid(trailerness_linear)

        # emotion prediction head
        emotional_linear = self.emotional_proj(wva_I_hat)
        emotional_pre = self.emotional_sigmoid(emotional_linear)

        return trailerness_pre, emotional_pre