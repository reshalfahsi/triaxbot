# MIT License
#
# Copyright (c) 2022 Resha Dwika Hefni Al-Fahsi
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Modified from:
#    - https://github.com/keras-team/keras-io/blob/master/examples/audio/transformer_asr.py
# ==============================================================================

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, num_vocab=1000, maxlen=100, num_hid=64):
        super(TokenEmbedding, self).__init__()

        self.pos_emb = nn.Embedding(maxlen, num_hid)
        self.emb = nn.Embedding(num_vocab, num_hid)

    def forward(self, x):
        x = self.emb(x)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        positions = torch.arange(
            start=0, end=x.shape[1], step=1, device=device
        ).unsqueeze(0)
        positions = self.pos_emb(positions)
        return x + positions


class SpeechFeatureEmbedding(nn.Module):
    def __init__(self, num_hid=64):
        super(SpeechFeatureEmbedding, self).__init__()

        KERNEL_SIZE = 11
        PADDING = int(KERNEL_SIZE // 2)

        self.emb = nn.Sequential(
            nn.LazyConv1d(
                num_hid,
                KERNEL_SIZE,
                stride=2,
                padding=PADDING,
            ),
            nn.GELU(),
            nn.Conv1d(
                num_hid,
                num_hid,
                KERNEL_SIZE,
                stride=2,
                padding=PADDING,
            ),
            nn.GELU(),
            nn.Conv1d(
                num_hid,
                num_hid,
                KERNEL_SIZE,
                stride=2,
                padding=PADDING,
            ),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.emb(x)
        return x.permute(0, 2, 1)


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_dim),
            nn.GELU(),
            nn.Linear(feed_forward_dim, embed_dim),
        )
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, inputs):
        attn_output, _ = self.att(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super(TransformerDecoder, self).__init__()
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(embed_dim, eps=1e-6)

        self.self_att = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.enc_att = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.self_dropout = nn.Dropout(0.5)
        self.enc_dropout = nn.Dropout(0.1)
        self.ffn_dropout = nn.Dropout(0.1)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_dim),
            nn.GELU(),
            nn.Linear(feed_forward_dim, embed_dim),
        )

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1).to(
            device=device
        )

    def forward(self, enc_out, target):

        seq_len = target.shape[1]
        src_mask = self.generate_square_subsequent_mask(seq_len)

        target_att, _ = self.self_att(target, target, target, attn_mask=src_mask)
        target_norm = self.layernorm1(target + self.self_dropout(target_att))

        enc_out, _ = self.enc_att(target_norm, enc_out, enc_out)
        enc_out_norm = self.layernorm2(self.enc_dropout(enc_out) + target_norm)

        ffn_out = self.ffn(enc_out_norm)
        ffn_out_norm = self.layernorm3(enc_out_norm + self.ffn_dropout(ffn_out))

        return ffn_out_norm


class Transformer(nn.Module):
    def __init__(
        self,
        num_hid=64,
        num_head=2,
        num_feed_forward=128,
        target_maxlen=100,
        num_layers_enc=4,
        num_layers_dec=1,
        num_classes=10,
    ):
        super(Transformer, self).__init__()

        self.target_maxlen = target_maxlen

        self.encoder_input = SpeechFeatureEmbedding(num_hid)

        self.encoder_transformer = nn.ModuleList(
            [
                TransformerEncoder(num_hid, num_head, num_feed_forward)
                for _ in range(num_layers_enc)
            ]
        )

        self.decoder_input = TokenEmbedding(num_classes, target_maxlen, num_hid)
        self.decoder_transformer = nn.ModuleList(
            [
                TransformerDecoder(num_hid, num_head, num_feed_forward)
                for _ in range(num_layers_dec)
            ]
        )

        self.classifier = nn.Linear(num_hid, num_classes)

    def encoder(self, audio):
        x = self.encoder_input(audio)
        for m in self.encoder_transformer:
            x = m(x)
        return x

    def decoder(self, audio, text):
        y = self.decoder_input(text)
        for m in self.decoder_transformer:
            y = m(audio, y)
        return y

    def forward(self, audio, text):
        x = self.encoder(audio)
        y = self.decoder(x, text)
        return self.classifier(y)

    def generate(self, source, target_start_token_idx):
        """Performs inference over one batch of inputs using greedy decoding."""
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        bs = source.shape[0]
        enc = self.encoder(source)
        dec_input = torch.ones((bs, 1), dtype=torch.int32) * target_start_token_idx
        dec_input = dec_input.to(device=device)
        dec_logits = []

        for i in range(self.target_maxlen - 1):
            dec_out = self.decoder(enc, dec_input)
            logits = self.classifier(dec_out)
            logits = torch.argmax(logits, dim=-1)
            last_logit = torch.unsqueeze(logits[:, -1], dim=-1)
            dec_logits.append(last_logit)
            dec_input = torch.cat([dec_input, last_logit], dim=-1)
        return dec_input
