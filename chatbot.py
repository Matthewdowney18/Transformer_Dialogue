import argparse
import os
import json
import torch
import logging
import numpy as np
import random
from nltk.tokenize import TweetTokenizer

# the telegram package holds everything we need for this tutorial
import telegram
from telegram.ext import Updater
from telegram.ext import CommandHandler
from telegram.ext import MessageHandler, Filters

import transformer
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
from dataset import DialogueDataset, Vocab
from transformer.Translator import Chatbot


class TelegramBot:
    def __init__(self, token):
        """This chatbot takes a Telegram authorization toke and a handler function,
        and deploys a Telegram chatbot to respond to user messages with that token.

        token - a string authorization token provided by @BotFather on Telegram
        handle_func - a function taking update, context parameters which responds to user inputs
        """
        self.token = token
        self.bot = telegram.Bot(token=token)
        self.updater = Updater(token=token)
        self.dispatcher = self.updater.dispatcher
        self.updater.start_polling()

    def stop(self):
        """Stop the Telegram bot"""
        self.updater.stop()

    def add_handler(self, handler):
        """Add a handler function to extend bot functionality"""
        self.dispatcher.add_handler(handler)


class ChatBot:
    def __init__(self, args):
        # get the dir with pre-trained model
        load_dir = os.path.join(args.experiment_dir, args.old_model_dir)

        # initialize, and load vocab
        self.vocab = Vocab()
        vocab_filename = os.path.join(load_dir, "vocab.json")
        self.vocab.load_from_dict(vocab_filename)

        # load configuration
        with open(os.path.join(load_dir, "config.json"), "r") as f:
            config = json.load(f)

        args.response_len = config["response_len"]
        args.history_len = config["history_len"]

        # initialize an empty dataset. used to get input features
        self.dataset = DialogueDataset(None,
                                       history_len=config["history_len"],
                                       response_len=config["response_len"],
                                       vocab=self.vocab,
                                       update_vocab=False)

        # set device
        self.device = torch.device(args.device)

        # initialize model
        model = Transformer(
            config["vocab_size"],
            config["vocab_size"],
            config["history_len"],
            config["response_len"],
            d_word_vec=config["embedding_dim"],
            d_model=config["model_dim"],
            d_inner=config["inner_dim"],
            n_layers=config["num_layers"],
            n_head=config["num_heads"],
            d_k=config["dim_k"],
            d_v=config["dim_v"],
            dropout=config["dropout"],
            pretrained_embeddings=None
        ).to(self.device)

        # load checkpoint
        checkpoint = torch.load(os.path.join(load_dir, "model.bin"),
                                map_location=self.device)
        model.load_state_dict(checkpoint['model'])

        # create chatbot
        self.chatbot = Chatbot(args, model)

        self.args = args

    def run(self):
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.INFO)

        # initialize history that is a list of sentences
        history = list()

        def greeting(bot, update):
            """Greet the user and ask for their name."""
            bot.send_message(update.message.chat_id,
                                    "Hello there! What is your name?")

        def respond(bot, update):
            """Whatever the user says, reverse their
             message and repeat it back to them."""
            message = update.message.text
            history.append(message)
            response = self._print_response(history)
            history.append(response)
            bot.send_message(update.message.chat_id,
                                    response)

        # queries sent to: https://api.telegram.org/bot<token>/METHOD_NAME
        TOKEN = "773295820:AAFF5_lCi4FdWCLd8YRbBJ9AeH2MzZWhhpw"

        bot = TelegramBot(TOKEN)
        bot.add_handler(MessageHandler(Filters.text, respond))
        bot.add_handler(CommandHandler('start', greeting))

    # print the response from the input
    def _print_response(self, history):
        # get query, add to the end of history

        # generate responses
        responses, scores = self._generate_responses(history)
        # chose response
        if self.args.choose_best:
            response = responses[0][0]
        else:
            # pick a random result from the n_best
            idx = random.randint(0, max(self.args.n_best,
                                        self.args.beam_size) - 1)
            response = responses[0][idx]

        # uncomment this line to see all the scores
        # print("scores in log prob: {}\n".format(scores[0]))

        # create output string
        output = ""
        for idx in response[:-1]:
            token = self.vocab.id2token[idx]
            output += "{} ".format(token)
        print(f'{output}')
        history.append(output)
        return output

    def _generate_responses(self, history):
        # get input features for the dialogue history
        h_seq, h_pos, h_seg = self.dataset.get_input_features(history)

        # get response from model
        response = self.chatbot.translate_batch(h_seq, h_pos, h_seg)
        return response