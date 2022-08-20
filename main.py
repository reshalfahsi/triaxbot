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
#    - https://github.com/liuhh02/python-telegram-bot-heroku/blob/master/bot.py
# ==============================================================================


import logging
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import os
import uuid

from triaxbot import TriAXBot


bot = TriAXBot()


# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)


# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.
def start(update, context):
    """Send a message when the command /start is issued."""
    update.message.reply_text("A simple bot that transcribe audio to text.")


def help(update, context):
    """Send a message when the command /help is issued."""
    update.message.reply_text(
        "Just send the audio it will try to transcribe the text based on the audio."
    )


def reply(update, context):
    """Reply the user message."""
    audio_file = update.message.voice.get_file()
    file_id = str(uuid.uuid4())
    filename = f"voice-{file_id}.ogg"
    audio_file.download(filename)

    update.message.reply_text(bot.transcribe(filename))

    os.remove(filename)


def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)


def main():
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    # Make sure to set use_context=True to use the new context based callbacks
    # Post version 12 this will no longer be necessary

    PORT = int(os.environ.get("PORT", "5000"))
    TOKEN = os.environ.get("TOKEN")

    updater = Updater(TOKEN, use_context=True)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # on different commands - answer in Telegram
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))

    # on noncommand i.e message - echo the message on Telegram
    dp.add_handler(MessageHandler(Filters.voice, reply))

    # log all errors
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_webhook(
        listen="0.0.0.0",
        port=int(PORT),
        url_path=TOKEN,
        webhook_url="https://triaxbot.herokuapp.com/" + TOKEN,
    )


if __name__ == "__main__":
    main()
