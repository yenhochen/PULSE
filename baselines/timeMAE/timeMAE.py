# from .layers import TransformerBlock, PositionalEmbedding, CrossAttnTRMBlock

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from pulse.encoder import ContextEncoder
from utils.dataset import get_label_names
from torch.nn.init import xavier_normal_, uniform_, constant_
from baselines.timeMAE.layers import (
    TransformerBlock,
    PositionalEmbedding,
    CrossAttnTRMBlock,
)


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        d_model = args.d_model
        attn_heads = args.attn_heads
        d_ffn = 4 * d_model
        layers = args.layers
        dropout = args.dropout
        enable_res_parameter = args.enable_res_parameter
        # TRMs
        self.TRMs = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, attn_heads, d_ffn, enable_res_parameter, dropout
                )
                for i in range(layers)
            ]
        )

    def forward(self, x):
        for TRM in self.TRMs:
            x = TRM(x, mask=None)
        return x


class Tokenizer(nn.Module):
    def __init__(self, rep_dim, vocab_size):
        super(Tokenizer, self).__init__()
        self.center = nn.Linear(rep_dim, vocab_size)

    def forward(self, x):
        bs, length, dim = x.shape
        probs = self.center(x.view(-1, dim))
        ret = F.gumbel_softmax(probs)
        indexes = ret.max(-1, keepdim=True)[1]
        return indexes.view(bs, length)


class Regressor(nn.Module):
    def __init__(self, d_model, attn_heads, d_ffn, enable_res_parameter, layers):
        super(Regressor, self).__init__()
        self.layers = nn.ModuleList(
            [
                CrossAttnTRMBlock(d_model, attn_heads, d_ffn, enable_res_parameter)
                for i in range(layers)
            ]
        )

    def forward(self, rep_visible, rep_mask_token):
        for TRM in self.layers:
            rep_mask_token = TRM(rep_visible, rep_mask_token)
        return rep_mask_token


class TimeMAE(nn.Module):
    def __init__(self, config, encoder=None):
        super(TimeMAE, self).__init__()

        label_names = get_label_names(config)

        args = config.model_args
        args.num_class = len(label_names)
        self.config = config

        d_model = args.d_model

        self.momentum = args.momentum
        self.linear_proba = True
        self.device = config.device
        # self.data_shape = args.data_shape
        # self.max_len = int(self.data_shape[0] / args.wave_length)
        self.max_len = int(self.config.data_args.subseq_size / args.wave_length)
        # print(self.max_len)
        self.mask_len = int(args.mask_ratio * self.max_len)
        self.position = PositionalEmbedding(self.max_len, d_model)

        self.mask_token = nn.Parameter(
            torch.randn(
                d_model,
            )
        )

        if encoder is None:
            self.input_projection = nn.Conv1d(
                config.data_args.input_dims,
                d_model,
                kernel_size=args.wave_length,
                stride=args.wave_length,
            )

            self.projector = nn.Identity()
            self.encoder_mode = "default"

        else:  # use ts2vec encoder

            self.input_projection = encoder
            self.projector = nn.Linear(config.encoder_args.emb_dim, args.d_model)
            self.encoder_mode = "conv"

        self.encoder = Encoder(args)
        self.momentum_encoder = Encoder(args)
        self.tokenizer = Tokenizer(d_model, args.vocab_size)
        self.reg = Regressor(d_model, args.attn_heads, 4 * d_model, 1, args.reg_layers)
        self.predict_head = nn.Linear(d_model, args.num_class)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)

    def copy_weight(self):
        with torch.no_grad():
            for param_a, param_b in zip(
                self.encoder.parameters(), self.momentum_encoder.parameters()
            ):
                param_b.data = param_a.data

    def momentum_update(self):
        with torch.no_grad():
            for param_a, param_b in zip(
                self.encoder.parameters(), self.momentum_encoder.parameters()
            ):
                param_b.data = (
                    self.momentum * param_b.data + (1 - self.momentum) * param_a.data
                )

    def pretrain_forward(self, x):
        # x: b t c -> b c t

        if self.encoder_mode == "default":
            # print("x HERE", x.shape)
            # x = x[:,:128]
            x = self.input_projection(x.transpose(1, 2)).transpose(1, 2).contiguous()

        else:
            x, _, _ = self.input_projection(x)  # b t n
            x = self.projector(x).transpose(1, 2)  # b d t

        tokens = self.tokenizer(x)

        x += self.position(x)
        rep_mask_token = self.mask_token.repeat(
            x.shape[0], x.shape[1], 1
        ) + self.position(x)

        index = np.arange(x.shape[1])
        random.shuffle(index)
        v_index = index[: -self.mask_len]
        m_index = index[-self.mask_len :]
        visible = x[:, v_index, :]
        mask = x[:, m_index, :]
        tokens = tokens[:, m_index]
        rep_mask_token = rep_mask_token[:, m_index, :]

        rep_visible = self.encoder(visible)
        with torch.no_grad():
            # rep_mask = self.encoder(mask)
            rep_mask = self.momentum_encoder(mask)
        rep_mask_prediction = self.reg(rep_visible, rep_mask_token)
        token_prediction_prob = self.tokenizer.center(rep_mask_prediction)

        return [rep_mask, rep_mask_prediction], [token_prediction_prob, tokens]

    def forward(self, x):
        # x: (b, t, c)
        if self.linear_proba:
            with torch.no_grad():
                x = (
                    self.input_projection(x.transpose(1, 2))
                    .transpose(1, 2)
                    .contiguous()
                )
                x += self.position(x)
                x = self.encoder(x)
                return torch.mean(x, dim=1)
        else:
            x = self.input_projection(x.transpose(1, 2)).transpose(1, 2).contiguous()
            x += self.position(x)
            x = self.encoder(x)
            return self.predict_head(torch.mean(x, dim=1))

    def get_tokens(self, x):
        x = self.input_projection(x.transpose(1, 2)).transpose(1, 2).contiguous()
        tokens = self.tokenizer(x)
        return tokens


#  ============ losses ============


class CE:
    def __init__(self, model):
        self.model = model
        self.ce = nn.CrossEntropyLoss()
        self.ce_pretrain = nn.CrossEntropyLoss(ignore_index=0)

    def compute(self, batch):
        seqs, labels = batch
        outputs = self.model(seqs)  # B * N
        labels = labels.view(-1).long()
        loss = self.ce(outputs, labels)
        return loss


class Align:
    def __init__(self):
        self.mse = nn.MSELoss(reduction="mean")
        self.ce = nn.CrossEntropyLoss()

    def compute(self, rep_mask, rep_mask_prediction):
        align_loss = self.mse(rep_mask, rep_mask_prediction)
        return align_loss


class Reconstruct:
    def __init__(self):
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.2)

    def compute(self, token_prediction_prob, tokens):
        hits = torch.sum(torch.argmax(token_prediction_prob, dim=-1) == tokens)
        NDCG10 = recalls_and_ndcgs_for_ks(
            token_prediction_prob.view(-1, token_prediction_prob.shape[-1]),
            tokens.reshape(-1, 1),
            10,
        )
        reconstruct_loss = self.ce(
            token_prediction_prob.view(-1, token_prediction_prob.shape[-1]),
            tokens.view(-1),
        )
        return reconstruct_loss, hits, NDCG10


def recalls_and_ndcgs_for_ks(scores, answers, k):
    answers = answers.tolist()
    labels = torch.zeros_like(scores).to(scores.device)
    for i in range(len(answers)):
        labels[i][answers[i]] = 1
    answer_count = labels.sum(1)

    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank
    cut = cut[:, :k]
    hits = labels_float.gather(1, cut)
    position = torch.arange(2, 2 + k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits * weights.to(hits.device)).sum(1)
    idcg = torch.Tensor([weights[: min(int(n), k)].sum() for n in answer_count]).to(
        dcg.device
    )
    ndcg = (dcg / idcg).mean()
    ndcg = ndcg.cpu().item()
    return ndcg
