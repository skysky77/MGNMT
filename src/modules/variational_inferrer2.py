# -*- coding: UTF-8 -*-
#
# MIT License
#
# Copyright (c) 2018 the xnmt authors.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#       author: Zaixiang Zheng
#       contact: zhengzx@nlp.nju.edu.cn
#           or zhengzx.142857@gmail.com
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn

from src.encoder.rnn_encoder import RNNEncoder
from src.modules.attention import ScaledDotProductAttention
from src.modules.embeddings import Embeddings
from src.modules.sublayers import MultiHeadedAttention


class _VariationalInferrer(nn.Module):
    def __init__(
        self, vocab, d_word_vec, d_model, d_latent, embed=None,
    ):
        super().__init__()

        self.vocab = vocab
        self.d_word_vec = d_word_vec
        self.d_model = d_model
        self.d_latent = d_latent

        self.embed = Embeddings(num_embeddings=vocab, embedding_dim=d_word_vec)
        if embed is not None:
            self.embed.embeddings.weight = embed.embeddings.weight

        # rnn is bidirectional
        self.infer_latent2mean = nn.Linear(d_model * 2, d_latent)
        self.infer_latent2logv = nn.Linear(d_model * 2, d_latent)

        self.should_swap = False

    def forward(self, x, is_sampling=True, stop_grad_input=True):
        batch_size = x.size(0)

        monolingual_inf = self.encode(x, stop_grad_input)

        mean = self.infer_latent2mean(monolingual_inf)
        logv = self.infer_latent2logv(monolingual_inf)

        if is_sampling:
            std = torch.exp(0.5 * logv)
            z = mean.new_tensor(torch.randn([batch_size, self.d_latent]))
            z = z * std + mean
        else:
            z = mean

        return {"mean": mean, "logv": logv, "latent": z}

    def encode(self, x, stop_grad_emb=True):
        raise NotImplementedError

    def share_parameters(self, reverse_inferrer, swap=True):
        raise NotImplementedError


class RNNInferrer(_VariationalInferrer):
    def __init__(
        self, vocab, d_word_vec, d_model, d_latent, embed=None,
    ):
        super().__init__(vocab, d_word_vec, d_model, d_latent, embed)

        self.infer_enc = RNNEncoder(vocab, d_word_vec, d_model, embeddings=self.embed)
        # self.infer_enc_x.embeddings = self.src_embed
        # self.infer_enc_y.embeddings = self.tgt_embed

    @staticmethod
    def _pool(_h, _m):
        _no_pad_mask = 1.0 - _m.float()
        _ctx_mean = (_h * _no_pad_mask.unsqueeze(2)).sum(1) / _no_pad_mask.unsqueeze(2).sum(1)
        return _ctx_mean

    def encode(self, x, stop_grad_input=True):
        x_emb = self.embed(x)
        if stop_grad_input:
            x_emb = x_emb.detach()

        enc_x, x_mask = self.infer_enc(x, x_emb)

        monolingual_inf = self._pool(enc_x, x_mask)

        return monolingual_inf

    def share_parameters(self, reverse_inferrer: _VariationalInferrer, swap=True):
        self.should_swap = swap
        self.src_embed, self.tgt_embed = reverse_inferrer.src_embed, reverse_inferrer.tgt_embed
        self.infer_enc_x, self.infer_enc_y = (
            reverse_inferrer.infer_enc_x,
            reverse_inferrer.infer_enc_y,
        )
        self.infer_latent2mean, self.infer_latent2logv = (
            reverse_inferrer.infer_latent2mean,
            reverse_inferrer.infer_latent2logv,
        )

