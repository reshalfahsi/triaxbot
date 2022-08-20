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


from torch.autograd import Variable
import torch
import torch.nn as nn
import torchaudio

import urllib.request

from .model import Transformer
from .utils import VectorizeChar
from .constant import (
    MAX_LEN,
    EMBED_DIM,
    N_HEAD,
    N_FFN,
    ENC_LAYER,
    DEC_LAYER,
    MODEL_URL,
    N_FFT,
    HOP_LENGTH,
    WIN_LENGTH,
    MAX_SEQ_T,
    START_TOKENID,
    END_TOKENID,
)


if not torch.cuda.is_available():
    torch.set_num_threads(12)
    torch.set_num_interop_threads(12)
    print(f"threads: {torch.get_num_threads()} {torch.get_num_interop_threads()}")


class TriAXBot(object):
    def __init__(self):
        urllib.request.urlretrieve(MODEL_URL, "triaxbot.pth")

        self.vectorizer = VectorizeChar(MAX_LEN)
        self.model = Transformer(
            num_hid=EMBED_DIM,
            num_head=N_HEAD,
            num_feed_forward=N_FFN,
            target_maxlen=MAX_LEN,
            num_layers_enc=ENC_LAYER,
            num_layers_dec=DEC_LAYER,
            num_classes=len(self.vectorizer.get_vocabulary()),
        )
        self.model.load_state_dict(
            torch.load("triaxbot.pth", map_location=torch.device("cpu"))
        )

    def transcribe(self, filename: str):
        audio = torchaudio.load(filename)[0]
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
        audio = padding(audio)[..., :MAX_SEQ_T]
        audio = Variable(audio)

        preds = self.model.generate(audio, START_TOKENID)
        preds = preds.data.cpu().numpy()

        prediction = ""
        idx_to_char = self.vectorizer.get_vocabulary()
        for idx in preds[0, :]:
            prediction += idx_to_char[idx]
            if idx == END_TOKENID:
                break

        return prediction.replace(idx_to_char[START_TOKENID], "").replace(
            idx_to_char[END_TOKENID], ""
        )
