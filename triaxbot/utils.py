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
import torchaudio
import numpy as np
import torch.utils.data as data

from .constant import N_FFT, HOP_LENGTH, WIN_LENGTH, MAX_SEQ_T


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(
            torch.stack(self.losses[np.maximum(len(self.losses) - self.num, 0) :])
        )


def get_data(wavs, id_to_text, maxlen=50):
    """returns mapping of audio paths and transcription texts"""
    dataset = []
    for w in wavs:
        id = w.split("/")[-1].split(".")[0]
        if len(id_to_text[id]) < maxlen:
            dataset.append({"audio": w, "text": id_to_text[id]})
    return dataset


class VectorizeChar(object):
    def __init__(self, max_len=50):
        self.vocab = (
            ["-", "#", "<", ">"]
            + [chr(i + 96) for i in range(1, 27)]
            + [" ", ".", ",", "?"]
        )
        self.max_len = max_len
        self.char_to_idx = {}
        for i, ch in enumerate(self.vocab):
            self.char_to_idx[ch] = i

    def __call__(self, text):
        text = text.lower()
        text = text[: self.max_len - 2]
        text = "<" + text + ">"
        pad_len = self.max_len - len(text)
        return [self.char_to_idx.get(ch, 1) for ch in text] + [0] * pad_len

    def get_vocabulary(self):
        return self.vocab


class AudioDataset(data.Dataset):
    def __init__(self, dataset, vectorizer):
        self.vectorizer = vectorizer
        self.dataset = dataset

    def __getitem__(self, index):
        audio = self.dataset[index]["audio"]
        text = self.dataset[index]["text"]

        audio = torchaudio.load(audio)[0]
        audio = torch.view_as_real(
            torch.stft(
                audio,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                win_length=WIN_LENGTH,
                return_complex=True,
            )
        )[..., 0]
        audio = torch.sqrt(torch.abs(audio))
        mu = audio.mean(1, keepdims=True)
        std = audio.std(1, keepdims=True)
        audio = (audio - mu) / std

        padding = nn.ConstantPad1d((0, MAX_SEQ_T), 0)
        audio = padding(audio)[..., :MAX_SEQ_T].squeeze(0)

        text = torch.LongTensor(self.vectorizer(text))

        return audio, text

    def __len__(self):
        return len(self.dataset)
