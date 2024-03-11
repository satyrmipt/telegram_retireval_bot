# -*- coding: utf-8 -*-
"""
This is main programm for teltgram bot 

Created on Sun Feb 11 21:32:10 2024

@author: satyr
based 100% on https://www.youtube.com/watch?v=vZtm1wuA2yc

"""
# this have to be done if you start bot from Spyder console
import nest_asyncio
nest_asyncio.apply()

import sys

from typing import Final
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackContext

import my_generative_bot as b

bot = b.Generative_ChatBot(model_checkpoint ="satyroffrost/FT_Merged_TinyLlama-1.1B-Chat-v1.0",
                tokenizer_checkpoint="satyroffrost/FT_Merged_TinyLlama-1.1B-Chat-v1.0",
                verbose=1 )

# bot = b.ChatBot(vector_db_path = "satyroffrost/eng_script_vectorized_v3",
#               context_len = 1, 
#               similarity_type = 'retrieval', 
#               vectorization_type = 'sent_encode',
#               character = 'Leonard',
#               bi_enc_path = "satyroffrost/triple-20e-1000-fit-all-mpnet-base-v2",
#               rerank_param = (None, None, None),
#               verbose=1 )

# bot = b.ChatBot(vector_db_path = "satyroffrost/eng_script_vectorized_v3",
#               context_len = 1, 
#               similarity_type = 'retrieval', 
#               vectorization_type = 'glove',
#               character = 'Leonard',
#               # bi_enc_path = "satyroffrost/triple-20e-1000-fit-all-mpnet-base-v2",
#               rerank_param = (None, None, None),
#               verbose=1 )

TOKEN: Final = '6514117454:AAH6DgqlBGx55oX71Z05-vSzQWXxmNBconw'
BOT_USERNAME: Final = '@TBB_chat_bot'

# Commands
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello I'm Leonard from 'The big Bang Theory' Series. Please type something in English so I can respond!")
    

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Print 'hello' for dummy test. Print 'stop' to shutdown the bot (for developer only.")
                                    
async def custom_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('This custom command do nothing right now.')
    
# Responses

def handle_response(text: str) -> str:
    '''
    Main function to handle conversation. This function get and log questions,
    store context and create answers.It uses my_bot module for all tasks
    '''
    processed: str = text.lower()
    # simple response for debugging
    if 'hello' in processed:
        return 'Hey there!'
    # developers trick to stop the conversation and cancel current session
    elif processed == 'stop':
        return "Stoppig the bot..."
        raise KeyboardInterrupt('bot stopped')
        global app
        app.stop_running()
        sys.exit(0)
    # main conversation part
    else:
        return bot.qa_iter(text)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    '''
    Wrapper for conversation . Please use handle_response function to 
    develop the logic of replies.
    '''
    message_type: str = update.message.chat.type
    text: str = update.message.text
    
    print(f"User ({update.message.chat.id}) in {message_type}: '{text}'" )
    # in case of group, bot replies only to messages addressed to it otherwise ignore the message
    if message_type == 'group':
        if BOT_USERNAME in text:
            new_text: str = text.replace(BOT_USERNAME, '').strip()
            response: str = handle_response(new_text)
        else:
            return "Bot can not participate in groups"
    else:
        response: str = handle_response(text) 
    print('Bot:', response)
    await update.message.reply_text(response)
    
    
async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"Update {update}caused error {context.error}")
    
    
if __name__ == '__main__':
    print('Startin bot...')
    app = Application.builder().token(TOKEN).build()
    
    # Commands
    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('help', help_command))
    app.add_handler(CommandHandler('custom', custom_command))
    
    # Messages
    app.add_handler(MessageHandler(filters.TEXT, handle_message))
    
    # Error
    app.add_error_handler(error)
    
    # Polls the bot
    print("Polling...")
    app.run_polling(poll_interval=3, drop_pending_updates = True) #close_loop = True
