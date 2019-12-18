import os
import json
import torch
import logging
import re
import random
import nltk.data

# the telegram package holds everything we need for this tutorial
import telegram
from telegram.ext import Updater
from telegram.ext import CommandHandler
from telegram.ext import MessageHandler, Filters

from src.transformer import Transformer
from dataset import DialogueDataset, Vocab
from src.transformer import Chatbot


def clean_response(response):
    substitutions = [(r' </s>', r''),
                     (r' \.', '.'),
                     (r' !', r'!'),
                     (r' \?', r'?'),
                     (r' ,', r'?'),
                     (r' i ', r' I ')]
                     #('[.|!|?] *[a-z]', )]

    for (pattern, replacement) in substitutions:
        response = re.sub(pattern, replacement, response)

    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sent_tokenizer.tokenize(response)
    sentences = [sent.capitalize() for sent in sentences]
    response = ' '.join(sentences)
    return response

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
        checkpoint = torch.load(os.path.join(load_dir, args.old_model_name),
                                map_location=self.device)
        model.load_state_dict(checkpoint['model'])

        # create chatbot
        self.chatbot = Chatbot(args, model)

        self.args = args

    def run(self):
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.INFO)

        greeting_text = "Hello! I am the Hawkbot! Let me tell you about myself" \
            "... please dont hurt my feelings!" \
                        "  If you would like to reset "\
                        "the conversation, please type '/reset'. "

        # initialize history dictionary for each chat id
        history = dict()

        def greeting(bot, update):
            # reset history, or create new history for chat id
            if update.message.chat_id in history:
                id = "{}_history".format(update.message.chat_id)
                if id in history:
                    history[id].append(history[update.message.chat_id])
                else:
                    history[id] = [history[update.message.chat_id]]
                history[update.message.chat_id].clear()
            else:
                history[update.message.chat_id] = list()

            # send a message
            bot.send_message(update.message.chat_id, greeting_text)

        def respond(bot, update):
            # initialize history for chat if it doesnt exist
            if update.message.chat_id not in history:
                greeting(bot, update)
            else:
                # get message, and add to history
                message = update.message.text
                history[update.message.chat_id].append(message)
                # get response, and add to history
                response = self._print_response(history[update.message.chat_id])
                history[update.message.chat_id].append(response)
                # send response from user
                bot.send_message(update.message.chat_id, clean_response(response))

                with open(self.args.save_filename, 'w') as f:
                    json.dump({"history": history, "args": vars(self.args)},
                              f, indent=4)

        # queries sent to: https://api.telegram.org/bot<token>/METHOD_NAME
        TOKEN = self.args.token

        bot = TelegramBot(TOKEN)
        bot.add_handler(MessageHandler(Filters.text, respond))
        bot.add_handler(CommandHandler('reset', greeting))

    # print the response from the input
    def _print_response(self, history):

        # generate responses
        responses, scores = self._generate_responses(history)
        # chose response
        if self.args.choose_best:
            response = responses[0][0]
        else:
            # pick a random result from the n_best
            idx = random.randint(0, min(self.args.n_best,
                                        self.args.beam_size) - 1)
            response = responses[0][idx]

        # uncomment this line to see all the scores
        # print("scores in log prob: {}\n".format(scores[0]))

        # create output string
        output = ""
        for idx in response[:-1]:
            token = self.vocab.id2token[idx]
            output += "{} ".format(token)
        print(f'{history[-1]} -> {output}')
        return output

    def _generate_responses(self, history):
        # get input features for the dialogue history
        h_seq, h_pos, h_seg = self.dataset.get_input_features(history)

        # get response from model
        response = self.chatbot.translate_batch(h_seq, h_pos, h_seg)
        return response