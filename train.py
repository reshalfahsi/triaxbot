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
# ==============================================================================


import torch
import torch.nn as nn
from torch.autograd import Variable

from glob import glob
from dtrx.dtrx import ExtractorApplication

import os
import urllib.request
import shutil

from datetime import datetime

from triaxbot import Transformer
from triaxbot.utils import AvgMeter, get_data
from triaxbot.constant import (
    MAX_LEN,
    EMBED_DIM,
    N_HEAD,
    N_FFN,
    ENC_LAYER,
    DEC_LAYER,
    TRAIN_SAVE,
    BATCH_SIZE,
    LR,
    STEP_LR,
    EPOCH,
)


def listdir(dirpath):
    """lists directories and sub-directories recursively. 
    """
    paths=[]
    paths.append(dirpath)
    for path in os.listdir(dirpath):
        rpath = os.path.join(dirpath, path)
        if os.path.isdir(rpath):
            paths.append(rpath)
    return paths


def prepare():
    dataset_name = "LJSpeech-1.1"
    raw_dataset = os.path.join(os.getcwd(), "{}.tar.bz2".format(dataset_name))
    dataset_url = "https://data.keithito.com/data/speech/{}.tar.bz2".format(
        dataset_name
    )
    dataset_path = os.path.join(os.getcwd(), dataset_name + "/")

    if not os.path.exists(dataset_path):
        urllib.request.urlretrieve(dataset_url, raw_dataset)
        ExtractorApplication([raw_dataset]).run()
        print("++++++Complete++++++")

        if not os.path.exists(dataset_path):
            temp = os.path.join(
                [x for x in listdir("./") if ".dtrx" in x][0], dataset_name
            )
            print(temp)
            shutil.move(temp, os.path.dirname(dataset_path))

    if os.path.exists(raw_dataset):
        os.remove(raw_dataset)

    wavs = glob("{}/**/*.wav".format(dataset_path), recursive=True)

    id_to_text = {}
    with open(os.path.join(dataset_path, "metadata.csv"), encoding="utf-8") as f:
        for line in f:
            id = line.strip().split("|")[0]
            text = line.strip().split("|")[2]
            id_to_text[id] = text

    print(dataset_path)
    print(os.path.exists(dataset_path))
    print(len(id_to_text))

    return wavs, id_to_text


def train_epoch(model, optimizer, criterion, train_loader):
    model.train()

    loss_record = AvgMeter()

    for i, pack in enumerate(train_loader, start=1):

        optimizer.zero_grad()

        # ---- data prepare ----
        audio, text = pack
        audio = Variable(audio).cuda()
        text_train = Variable(text[..., :-1]).cuda()
        text_label = Variable(text[..., 1:]).cuda()

        # ---- forward ----
        pred = model(audio, text_train).permute(0, 2, 1)

        # ---- loss function ----
        loss = criterion(pred, text_label)

        # ---- backward ----
        loss.backward()

        optimizer.step()

        # ---- recording loss ----
        loss_record.update(loss.data, BATCH_SIZE)

        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print(
                "{} Epoch [{}/{}], Step [{}/{}], "
                "[loss: {}"
                "]".format(
                    datetime.now(),
                    n,
                    EPOCH,
                    i,
                    total_step,
                    loss_record.show(),
                )
            )


def val_epoch(model, criterion, test_loader, best):
    val_record = AvgMeter()
    for i, pack in enumerate(test_loader, start=1):
        model.eval()

        audio, text = pack
        audio = Variable(audio).cuda()
        text_train = Variable(text[..., :-1]).cuda()
        text_label = Variable(text[..., 1:]).cuda()

        with torch.inference_mode():
            pred = model(audio, text_train).permute(0, 2, 1)
            loss = criterion(pred, text_label)
            val_record.update(loss.data, BATCH_SIZE)

    scheduler.step(val_record.show())

    if not os.path.exists(TRAIN_SAVE):
        os.makedirs(TRAIN_SAVE)

    print("Best: ", best)
    print("Val: ", float(val_record.show()))

    if val_record.show() < best:
        best = float(val_record.show())
        torch.save(model.state_dict(), os.path.join(TRAIN_SAVE, "triaxbot_best.pth"))

    torch.save(model.state_dict(), os.path.join(TRAIN_SAVE, "triaxbot.pth"))
    return best


def main():

    wavs, id_to_text = prepare()

    vectorizer = VectorizeChar(MAX_LEN)
    dataset = get_data(wavs, id_to_text, MAX_LEN)

    split = int(len(dataset) * 0.99)
    train_data = dataset[:split]
    test_data = dataset[split:]

    train_set = AudioDataset(train_data, vectorizer)
    test_set = AudioDataset(test_data, vectorizer)

    train_loader = data.DataLoader(
        dataset=train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = data.DataLoader(
        dataset=test_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    model = Transformer(
        num_hid=EMBED_DIM,
        num_head=N_HEAD,
        num_feed_forward=N_FFN,
        target_maxlen=MAX_LEN,
        num_layers_enc=ENC_LAYER,
        num_layers_dec=DEC_LAYER,
        num_classes=len(vectorizer.get_vocabulary()),
    )

    weight_pth = os.path.join(TRAIN_SAVE, "triaxbot.pth")
    if os.path.exists(weight_pth):
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(weight_pth))
        else:
            model.load_state_dict(
                torch.load(weight_pth, map_location=torch.device("cpu"))
            )

    if torch.cuda.is_available():
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion.cuda()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=STEP_LR, cooldown=STEP_LR
    )

    total_step = len(train_loader)

    best = 999999999

    print("#" * 20, "Start Training", "#" * 20)

    for n in range(1, EPOCH + 1):
        train_epoch(model, optimizer, criterion, train_loader)
        best = val_epoch(model, criterion, test_loader, best)


if __name__ == "__main__":
    main()
