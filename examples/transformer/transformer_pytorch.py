# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from transformer_utils import *


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos_table = np.array(
            [
                (
                    [
                        pos / np.power(10000, 2 * i / d_model)
                        for i in range(d_model)
                    ]
                    if pos != 0
                    else np.zeros(d_model)
                )
                for pos in range(max_len)
            ]
        )
        pos_table[1:, 0::2] = np.sin(
            pos_table[1:, 0::2]
        )  # word embedding dimmentions is even
        pos_table[1:, 1::2] = np.cos(
            pos_table[1:, 1::2]
        )  # word embedding dimmentions is odd
        self.pos_table = torch.FloatTensor(
            pos_table
        ).cuda()  # enc_inputs: [seq_len, d_model]

    def forward(self, enc_inputs):  # enc_inputs: [batch_size, seq_len, d_model]
        enc_inputs += self.pos_table[: enc_inputs.size(1), :]
        return self.dropout(enc_inputs.cuda())


def get_attn_pad_mask(
    seq_q, seq_k
):  # seq_q: [batch_size, seq_len] ,seq_k: [batch_size, seq_len]
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(
        1
    )  # mark seq_k that contain P (=0) with 1 ,[batch_size, 1, len_k]
    return pad_attn_mask.expand(
        batch_size, len_q, len_k
    )  # expand into multiple dimensions


def get_attn_subsequence_mask(seq):  # seq: [batch_size, tgt_len]
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(
        np.ones(attn_shape), k=1
    )  # generate an upper triangular matrix, [batch_size, tgt_len, tgt_len]
    subsequence_mask = torch.from_numpy(
        subsequence_mask
    ).byte()  # [batch_size, tgt_len, tgt_len]
    return subsequence_mask


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.weight_1 = nn.Parameter(torch.FloatTensor(d_model, d_ff))
        self.weight_2 = nn.Parameter(torch.FloatTensor(d_ff, d_model))

    # inputs: [batch_size, seq_len, d_model]
    def forward(self, inputs):
        output = torch.matmul(
            inputs, self.weight_1
        )  # [batch_size, seq_len, d_ff]
        output = nn.ReLU()(output)
        output = torch.matmul(
            output, self.weight_2
        )  # [batch_size, seq_len, d_model]
        output = nn.LayerNorm(d_model)(
            output + inputs
        )  # [batch_size, seq_len, d_model]
        return output

    def init_model(self, param, prefix=""):
        self.weight_1.data.copy_(torch.from_numpy(param[prefix + "weight_1"]))
        self.weight_2.data.copy_(torch.from_numpy(param[prefix + "weight_2"]))


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(
        self, Q, K, V, attn_mask=None
    ):  # Q: [batch_size, n_heads, len_q, d_k]
        # K: [batch_size, n_heads, len_k, d_k]
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        # scores : [batch_size, n_heads, len_q, len_k]
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    # input_Q: [batch_size, len_q, d_model]
    def forward(self, input_Q, input_K, input_V, attn_mask=None):
        # input_K: [batch_size, len_k, d_model]
        # input_V: [batch_size, len_v(=len_k), d_model]
        # attn_mask: [batch_size, seq_len, seq_len]
        residual, batch_size = input_Q, input_Q.size(0)
        # Q: [batch_size, n_heads, len_q, d_k]
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # K: [batch_size, n_heads, len_k, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        # context: [batch_size, n_heads, len_q, d_v]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        # attn: [batch_size, n_heads, len_q, len_k]
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)

        # context: [batch_size, len_q, n_heads * d_v]
        # [batch_size, len_q, d_model]
        output = self.fc(context)
        layer_norm = nn.LayerNorm(d_model)
        # return output + residual, attn
        return nn.LayerNorm(d_model)(output + residual), attn

    def init_model(self, param, prefix=""):
        self.W_Q.weight.data.copy_(
            torch.from_numpy(param[prefix + "W_Q"].transpose(1, 0))
        )
        self.W_K.weight.data.copy_(
            torch.from_numpy(param[prefix + "W_K"].transpose(1, 0))
        )
        self.W_V.weight.data.copy_(
            torch.from_numpy(param[prefix + "W_V"].transpose(1, 0))
        )
        self.fc.weight.data.copy_(
            torch.from_numpy(param[prefix + "fc"].transpose(1, 0))
        )


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = (
            MultiHeadAttention()
        )  # Multi-Head Attention mechanism
        self.pos_ffn = PoswiseFeedForwardNet()  # FeedForward neural networks

    def forward(
        self, enc_inputs, enc_self_attn_mask
    ):  # enc_inputs: [batch_size, src_len, d_model]
        # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(
            enc_inputs,
            enc_inputs,
            enc_inputs,
            # enc_outputs: [batch_size, src_len, d_model],
            enc_self_attn_mask,
        )  # attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs = self.pos_ffn(
            enc_outputs
        )  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn

    def init_model(self, param, prefix=""):
        self.enc_self_attn.init_model(param, prefix + "enc_self_attn.")
        self.pos_ffn.init_model(param, prefix + "pos_ffn.")


# TODO: test the Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # self.EmbeddingLyaer = EmbeddingLyaer()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(
        self, enc_outputs, enc_self_attn_mask
    ):  # enc_inputs: [batch_size, src_len]
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(
                enc_outputs, enc_self_attn_mask
            )  # enc_outputs :   [batch_size, src_len, d_model],
            # enc_self_attn : [batch_size, n_heads, src_len, src_len]
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

    def init_model(self, param, prefix=""):
        for i, layer in enumerate(self.layers):
            layer.init_model(param, prefix + "layers." + str(i) + ".")
