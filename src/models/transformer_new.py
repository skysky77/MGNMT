# MIT License

# Copyright (c) 2018 the NJUNMT-pytorch authors.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.vocabulary import PAD
from src.decoding.utils import tile_batch, tensor_gather_helper
from src.models.base import NMTModel
from src.modules.basic import BottleLinear as Linear
from src.modules.embeddings import Embeddings
from src.modules.sublayers import PositionwiseFeedForward, MultiHeadedAttention
from src.utils import nest


def get_attn_causal_mask(seq):
    """ Get an attention mask to avoid using the subsequent info.

    :param seq: Input sequence.
        with shape [batch_size, time_steps, dim]
    """
    assert seq.dim() == 3
    attn_shape = (seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    subsequent_mask = torch.from_numpy(subsequent_mask)
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()
    return subsequent_mask


class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_inner_hid, n_head, dim_per_head, dropout=0.1):
        super(EncoderBlock, self).__init__()

        self.layer_norm = nn.LayerNorm(d_model)

        self.slf_attn = MultiHeadedAttention(
            head_count=n_head, model_dim=d_model, dropout=dropout, dim_per_head=dim_per_head
        )

        self.pos_ffn = PositionwiseFeedForward(
            size=d_model, hidden_size=d_inner_hid, dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        input_norm = self.layer_norm(enc_input)
        context, _, _ = self.slf_attn(input_norm, input_norm, input_norm, slf_attn_mask)
        out = self.dropout(context) + enc_input

        return self.pos_ffn(out)


class Encoder(nn.Module):
    def __init__(
        self,
        n_layers=6,
        n_head=8,
        d_word_vec=512,
        d_model=512,
        d_inner_hid=1024,
        dropout=0.1,
        dim_per_head=None,
    ):
        super().__init__()

        self.num_layers = n_layers

        self.block_stack = nn.ModuleList(
            [
                EncoderBlock(
                    d_model=d_model,
                    d_inner_hid=d_inner_hid,
                    n_head=n_head,
                    dropout=dropout,
                    dim_per_head=dim_per_head,
                )
                for _ in range(n_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, src_emb, src_mask):
        # enc_mask = src_mask.detach().eq(PAD)
        batch_size, src_len = src_mask.size()
        self_attn_mask = src_mask.unsqueeze(1).expand(batch_size, src_len, src_len)

        out = src_emb
        for i in range(self.num_layers):
            out = self.block_stack[i](out, self_attn_mask)

        out = self.layer_norm(out)

        return out


class DecoderBlock(nn.Module):
    """ Compose with three layers """

    def __init__(self, d_model, d_inner_hid, n_head, dim_per_head, dropout=0.1):
        super(DecoderBlock, self).__init__()

        self.slf_attn = MultiHeadedAttention(
            head_count=n_head, model_dim=d_model, dropout=dropout, dim_per_head=dim_per_head
        )
        self.ctx_attn = MultiHeadedAttention(
            head_count=n_head, model_dim=d_model, dropout=dropout, dim_per_head=dim_per_head
        )
        self.pos_ffn = PositionwiseFeedForward(size=d_model, hidden_size=d_inner_hid)

        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def compute_cache(self, enc_output):
        return self.ctx_attn.compute_cache(enc_output, enc_output)

    def forward(
        self,
        dec_input,
        enc_output,
        slf_attn_mask=None,
        dec_enc_attn_mask=None,
        enc_attn_cache=None,
        self_attn_cache=None,
    ):
        # Args Checks
        input_batch, input_len, _ = dec_input.size()

        contxt_batch, contxt_len, _ = enc_output.size()

        input_norm = self.layer_norm_1(dec_input)
        all_input = input_norm

        query, _, self_attn_cache = self.slf_attn(
            all_input, all_input, input_norm, mask=slf_attn_mask, self_attn_cache=self_attn_cache
        )

        query = self.dropout(query) + dec_input

        query_norm = self.layer_norm_2(query)
        mid, attn, enc_attn_cache = self.ctx_attn(
            enc_output,
            enc_output,
            query_norm,
            mask=dec_enc_attn_mask,
            enc_attn_cache=enc_attn_cache,
        )

        output = self.pos_ffn(self.dropout(mid) + query)

        return output, attn, self_attn_cache, enc_attn_cache


class Decoder(nn.Module):
    """ A decoder model with self attention mechanism. """

    def __init__(
        self,
        n_layers=6,
        n_head=8,
        d_word_vec=512,
        d_model=512,
        d_inner_hid=1024,
        dim_per_head=None,
        dropout=0.1,
    ):

        super(Decoder, self).__init__()

        self.n_head = n_head
        self.num_layers = n_layers
        self.d_model = d_model

        self.block_stack = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=d_model,
                    d_inner_hid=d_inner_hid,
                    n_head=n_head,
                    dropout=dropout,
                    dim_per_head=dim_per_head,
                )
                for _ in range(n_layers)
            ]
        )

        self.out_layer_norm = nn.LayerNorm(d_model)

        self._dim_per_head = dim_per_head

    @property
    def dim_per_head(self):
        if self._dim_per_head is None:
            return self.d_model // self.n_head
        else:
            return self._dim_per_head

    def forward(
        self, tgt_emb, tgt_mask, enc_output, src_mask, enc_attn_caches=None, self_attn_caches=None
    ):

        (batch_size, tgt_len), src_len = tgt_mask.size(), src_mask.size(1)

        query_len = tgt_len
        key_len = tgt_len

        # # Run the forward pass of the TransformerDecoder.
        # emb = self.embeddings(tgt_seq)

        if self_attn_caches is not None:
            tgt_emb = tgt_emb[:, -1:].contiguous()
            query_len = 1

        # Decode mask
        dec_slf_attn_pad_mask = tgt_mask.unsqueeze(1).expand(batch_size, query_len, key_len)
        dec_slf_attn_sub_mask = get_attn_causal_mask(tgt_emb)

        try:
            dec_slf_attn_mask = dec_slf_attn_pad_mask + dec_slf_attn_sub_mask.bool()
        except Exception():
            dec_slf_attn_mask = torch.gt(dec_slf_attn_pad_mask + dec_slf_attn_sub_mask, 0)

        dec_enc_attn_mask = src_mask.unsqueeze(1).expand(batch_size, query_len, src_len)

        output = tgt_emb
        new_self_attn_caches = []
        new_enc_attn_caches = []
        for i in range(self.num_layers):
            output, attn, self_attn_cache, enc_attn_cache = self.block_stack[i](
                output,
                enc_output,
                dec_slf_attn_mask,
                dec_enc_attn_mask,
                enc_attn_cache=enc_attn_caches[i] if enc_attn_caches is not None else None,
                self_attn_cache=self_attn_caches[i] if self_attn_caches is not None else None,
            )

            new_self_attn_caches += [self_attn_cache]
            new_enc_attn_caches += [enc_attn_cache]

        output = self.out_layer_norm(output)

        return output, new_self_attn_caches, new_enc_attn_caches


class Generator(nn.Module):
    def __init__(self, n_words, hidden_size, shared_weight=None, padding_idx=-1):
        super(Generator, self).__init__()

        self.n_words = n_words
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx

        self.proj = Linear(self.hidden_size, self.n_words, bias=False)

        if shared_weight is not None:
            self.proj.linear.weight = shared_weight

    def _pad_2d(self, x):

        if self.padding_idx == -1:
            return x
        else:
            x_size = x.size()
            x_2d = x.view(-1, x.size(-1))

            mask = x_2d.new(1, x_2d.size(-1)).zero_()
            mask[0][self.padding_idx] = float("-inf")
            x_2d = x_2d + mask

            return x_2d.view(x_size)

    def forward(self, input, log_probs=True):
        """
        input == > Linear == > LogSoftmax
        """

        logits = self.proj(input)

        logits = self._pad_2d(logits)

        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)


class Transformer(NMTModel):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
        self,
        n_src_vocab,
        n_tgt_vocab,
        src_embed=None,
        tgt_embed=None,
        n_layers=6,
        n_head=8,
        d_word_vec=512,
        d_model=512,
        d_inner_hid=1024,
        dim_per_head=None,
        dropout=0.1,
        tie_input_output_embedding=True,
        tie_source_target_embedding=False,
        **kwargs
    ):

        super(Transformer, self).__init__()

        self._build_embeddings(
            n_src_vocab,
            n_tgt_vocab,
            src_embed=src_embed,
            tgt_embed=tgt_embed,
            d_word_vec=d_word_vec,
            dropout=dropout,
        )

        self.encoder = Encoder(
            n_layers=n_layers,
            n_head=n_head,
            d_word_vec=d_word_vec,
            d_model=d_model,
            d_inner_hid=d_inner_hid,
            dropout=dropout,
            dim_per_head=dim_per_head,
        )

        self.decoder = Decoder(
            n_layers=n_layers,
            n_head=n_head,
            d_word_vec=d_word_vec,
            d_model=d_model,
            d_inner_hid=d_inner_hid,
            dropout=dropout,
            dim_per_head=dim_per_head,
        )

        self.dropout = nn.Dropout(dropout)

        assert (
            d_model == d_word_vec
        ), "To facilitate the residual connections, \
             the dimensions of all module output shall be the same."

        if tie_source_target_embedding:
            assert (
                n_src_vocab == n_tgt_vocab
            ), "source and target vocabulary should have equal size when tying source&target embedding"
            self.src_embed.embeddings.weight = self.tgt_embed.embeddings.weight

        if tie_input_output_embedding:
            self.generator = Generator(
                n_words=n_tgt_vocab,
                hidden_size=d_word_vec,
                shared_weight=self.tgt_embed.embeddings.weight,
                padding_idx=PAD,
            )

        else:
            self.generator = Generator(n_words=n_tgt_vocab, hidden_size=d_word_vec, padding_idx=PAD)

    def _build_embeddings(
        self, n_src_vocab, n_tgt_vocab, src_embed=None, tgt_embed=None, d_word_vec=512, dropout=0.1
    ):
        self.src_embed = (
            Embeddings(
                num_embeddings=n_src_vocab,
                embedding_dim=d_word_vec,
                dropout=dropout,
                add_position_embedding=True,
            )
            if src_embed is None
            else src_embed
        )
        self.tgt_embed = (
            Embeddings(
                num_embeddings=n_tgt_vocab,
                embedding_dim=d_word_vec,
                dropout=dropout,
                add_position_embedding=True,
            )
            if tgt_embed is None
            else tgt_embed
        )

    def forward(self, src_seq, tgt_seq, src_emb=None, tgt_emb=None, log_probs=True):
        if src_emb is None:
            src_emb = self.src_embed(src_seq)
        if tgt_emb is None:
            tgt_emb = self.tgt_embed(tgt_seq)
        src_mask, tgt_mask = src_seq.eq(PAD), tgt_seq.eq(PAD)

        enc_output = self.encoder(src_emb, src_mask=src_mask)
        dec_output, _, _ = self.decoder(
            tgt_emb, tgt_mask=tgt_mask, enc_output=enc_output, src_mask=src_mask
        )

        return self.generator(dec_output, log_probs=log_probs)

    def encode(self, src_seq, src_emb=None):
        assert hasattr(self, "latent")
        if src_emb is None:
            # src_emb = self.src_embed(src_seq)
            src_emb = self.forward_embedding(src_seq, self.latent, lang="src")
        src_mask = src_seq.eq(PAD)

        ctx = self.encoder(src_emb, src_mask=src_mask)

        return {"ctx": ctx, "ctx_mask": src_mask}

    def init_decoder(self, enc_outputs, expand_size=1):

        ctx = enc_outputs["ctx"]

        ctx_mask = enc_outputs["ctx_mask"]

        assert hasattr(self, "latent")
        latent = self.latent

        if expand_size > 1:
            ctx = tile_batch(ctx, multiplier=expand_size)
            ctx_mask = tile_batch(ctx_mask, multiplier=expand_size)
            latent = tile_batch(latent, multiplier=expand_size)

        return {
            "ctx": ctx,
            "ctx_mask": ctx_mask,
            "enc_attn_caches": None,
            "slf_attn_caches": None,
            "latent": latent,
        }

    def decode(self, tgt_seq, dec_states, tgt_emb=None, log_probs=True):
        ctx = dec_states["ctx"]
        ctx_mask = dec_states["ctx_mask"]
        enc_attn_caches = dec_states["enc_attn_caches"]
        slf_attn_caches = dec_states["slf_attn_caches"]

        latent = dec_states["latent"]

        # assert hasattr(self, "latent")
        if tgt_emb is None:
            # tgt_emb = self.tgt_embed(tgt_seq)
            tgt_emb = self.forward_embedding(tgt_seq, latent, lang="tgt")

        tgt_mask = tgt_seq.eq(PAD)

        dec_output, slf_attn_caches, enc_attn_caches = self.decoder(
            tgt_emb=tgt_emb,
            tgt_mask=tgt_mask,
            enc_output=ctx,
            src_mask=ctx_mask,
            enc_attn_caches=enc_attn_caches,
            self_attn_caches=slf_attn_caches,
        )

        next_scores = self.generator(dec_output[:, -1].contiguous(), log_probs=log_probs)

        dec_states["enc_attn_caches"] = enc_attn_caches
        dec_states["slf_attn_caches"] = slf_attn_caches

        return next_scores, dec_states

    def reorder_dec_states(self, dec_states, new_beam_indices, beam_size):

        slf_attn_caches = dec_states["slf_attn_caches"]

        batch_size = slf_attn_caches[0][0].size(0) // beam_size

        n_head = self.decoder.n_head
        dim_per_head = self.decoder.dim_per_head

        slf_attn_caches = nest.map_structure(
            lambda t: tensor_gather_helper(
                gather_indices=new_beam_indices,
                gather_from=t,
                batch_size=batch_size,
                beam_size=beam_size,
                gather_shape=[batch_size * beam_size, n_head, -1, dim_per_head],
            ),
            slf_attn_caches,
        )

        dec_states["slf_attn_caches"] = slf_attn_caches

        return dec_states

    def forward_embedding(self, seq, latent, lang="src", add_pos=True):
        if lang == "src":
            emb = self.src_embed(seq)
        elif lang == "tgt":
            emb = self.tgt_embed(seq)
        if add_pos:
            emb = self.pos_embed(emb)

        # concat embeddings with latent and do a linear transformation.
        _latent = latent.unsqueeze(1).repeat(1, emb.size(1), 1)
        var_emb = self.var_inp_maps[lang](torch.cat([emb, _latent], -1))
        return var_emb
