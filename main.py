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
#    - https://github.com/ohld/python-zeit-serverless-telegram-bot/blob/master/index.py
# ==============================================================================


import telegram
import os
import requests
import uuid


from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request


from triaxbot import TriAXBot


bot = TriAXBot()

app = FastAPI()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
requests.get(
    f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/setWebhook?url=https://triaxbot.vercel.app/api"
)


@app.get("/")
async def index():
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    if TELEGRAM_TOKEN is None:
        return {"status": "error", "reason": "empty token"}

    tbot = telegram.Bot(TELEGRAM_TOKEN)

    return str(tbot.get_me())


@app.get("/api")
async def api_get():
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    if TELEGRAM_TOKEN is None:
        return {"status": "error", "reason": "empty token"}

    tbot = telegram.Bot(TELEGRAM_TOKEN)

    return str(tbot.get_me())


@app.post("/api")
async def api_post(request: Request):
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    if TELEGRAM_TOKEN is None:
        return {"status": "error", "reason": "empty token"}

    tbot = telegram.Bot(TELEGRAM_TOKEN)
    json_data = await request.json()
    print(json_data)
    update = telegram.Update.de_json(dict(json_data), tbot)

    try:
        if update.message.text == "/start":
            update.message.reply_text(
                "Just send the audio it will try to transcribe the text based on the audio."
            )
            return {"status": "ok"}
    except Exception as e:
        print("Not the desired text! Detail: {}".format(e))

    try:
        file_id = update.message.voice.file_id
        audio_file = tbot.get_file(file_id)
        file_id = str(uuid.uuid4())
        filename = f"voice-{file_id}.ogg"
        audio_file.download(filename)

        update.message.reply_text(bot.transcribe(filename))

        os.remove(filename)
    except Exception as e:
        print("Can not process audio data! Detail: {}".format(e))

    return {"status": "ok"}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
