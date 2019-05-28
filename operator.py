import os
import torch
import time
import numpy as np
from tqdm import tqdm

from utils import ModelConfig, load_checkpoint, cal_performance
import transformer
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
from dataset import DialogueDataset, Vocab

class ModelOperator:
    def __init__(self, args):

        # set up output directory
        self.output_dir = args.output_dir
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        # initialize model config
        self.config = ModelConfig(args)

        # check if there is a model to load
        if args.old_model_dir is not None:
            self.use_old_model = True
            self.load_dir = args.old_model_dir
            self.config.load_from_file(
                os.path.join(self.load_dir, "config.json"))

            # create vocab
            self.vocab = Vocab
            self.vocab.load_from_dict(os.path.join(self.load_dir, "vocab.json"))
            self.update_vocab = False
        else:
            self.use_old_model = False

            self.vocab = None
            self.update_vocab = True

        # create data sets
        self.dataset_filename = args.dataset_filename

        # train
        self.train_dataset = DialogueDataset(
            os.path.join(self.dataset_filename, "train.csv"),
            self.config.min_count,
            self.config.history_len,
            self.config.response_len,
            self.vocab,
            self.update_vocab)
        self.data_loader_train = torch.utils.data.DataLoader(
            self.train_dataset, self.config.train_batch_size, shuffle=True)

        self.vocab = self.train_dataset.vocab

        # eval
        self.val_dataset = DialogueDataset(
            os.path.join(self.dataset_filename, "val.csv"),
            self.config.min_count,
            self.config.history_len,
            self.config.response_len,
            self.vocab,
            self.update_vocab)
        self.data_loader_val = torch.utils.data.DataLoader(
            self.val_dataset, self.config.val_batch_size, shuffle=True)

        # update, and save vocab
        self.vocab = self.val_dataset.vocab
        self.train_dataset.vocab = self.vocab
        self.vocab.save_to_dict(os.path.join(self.output_dir, "vocab.json"))
        self.vocab_size = len(self.vocab)
        self.config.vocab_size = self.vocab_size

        # print and save the config file
        self.config.print_config()
        self.config.save_config(os.path.join(self.output_dir, "config.json"))

        # set device
        self.device = torch.device('cuda')

        # create model
        self.model = Transformer(
            self.config.vocab_size,
            self.config.vocab_size,
            self.config.history_len,
            self.config.response_len,
            d_word_vec=self.config.embedding_dim,
            d_model=self.config.model_dim,
            d_inner=self.config.inner_dim,
            n_layers=self.config.num_layers,
            n_head=self.config.num_heads,
            d_k=self.config.dim_k,
            d_v=self.config.dim_v,
            dropout=self.config.dropout
        )

        # create optimizer
        self.optimizer = torch.optim.Adam(
            filter(lambda x: x.requires_grad, self.model.parameters()),
            betas=(0.9, 0.98), eps=1e-09)

        # load old model, optimizer if there is one
        if self.use_old_model:
            self.model, self.optimizer = load_checkpoint(
                self.load_dir, self.model, self.optimizer)

        # create a sceduled optimizer object
        self.optimizer = ScheduledOptim(
            self.optimizer, self.config.model_dim, self.config.warmup_steps)

        self.model = self.model.to(self.device)

    def train(self, num_epochs):
        metrics = {"best_epoch":0, "lowest_loss":100}

        for epoch in range(num_epochs):
            epoch_metrics = dict()
            epoch_metrics["train"] = self._train_single_epoch()
            epoch_metrics["val"] = self.eval()

            # save metrics
            metrics["epoch_{}".format(epoch)] = epoch_metrics
            with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
                json.dump(all_metrics, f, indent=4)

            # save checkpoint
            if epoch_metrics["val"]["loss"] < metrics["lowest_loss"]:
                self.save_checkpoint()
                metrics["lowest_loss"] = epoch_metrics["val"]["loss"]
                metrics["best_epoch"] = epoch

    def _train_single_epoch(self):
        start = time.clock()
        phase_metrics = dict()
        epoch_loss = list()
        n_word_total = 0
        n_correct = 0
        n_word_correct = 0
        average_epoch_loss = None
        for batch in tqdm(self.data_loader_train,
            mininterval=2, desc='- (Training)', leave=False):

            # prepare data
            src_seq, src_pos, src_seg, tgt_seq, tgt_pos = map(
                lambda x: x.to(self.device), batch)

            gold = tgt_seq[:, 1:]

            # forward
            self.optimizer.zero_grad()
            pred = self.model(src_seq, src_pos, src_seg, tgt_seq, tgt_pos)

            # get loss
            loss, n_correct = cal_performance(pred, gold, smoothing=
                self.config.label_smoothing)
            epoch_loss.append(float(loss))
            average_epoch_loss = np.mean(epoch_loss)

            non_pad_mask = gold.ne(transformer.Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

            # backward
            loss.backward()

            # update parameters
            self.optimizer.step_and_update_lr()

        phase_metrics["loss"] = average_epoch_loss
        phase_metrics["token_accuracy"] = n_correct / n_word_total

        perplexity = np.exp(average_epoch_loss)
        phase_metrics["perplexity"] = perplexity

        time_taken = time.clock() - start
        string = ' Train loss: {:.3f} | time: {:.3f}'.format(
            average_epoch_loss, time_taken)
        print(string, end='\n')
        return phase_metrics

    def eval(self):
        start = time.clock()
        phase_metrics = dict()
        epoch_loss = list()
        n_word_total = 0
        n_correct = 0
        n_word_correct = 0
        average_epoch_loss = None
        for batch in tqdm(self.data_loader_train,
                          mininterval=2, desc='- (Training)', leave=False):
            # prepare data
            src_seq, src_pos, src_seg, tgt_seq, tgt_pos = map(
                lambda x: x.to(self.device), batch)

            gold = tgt_seq[:, 1:]

            # forward
            pred = self.model(src_seq, src_pos, src_seg, tgt_seq, tgt_pos)

            # get loss
            loss, n_correct = cal_performance(pred, gold,
                smoothing=self.config.label_smoothing)
            epoch_loss.append(float(loss))
            average_epoch_loss = np.mean(epoch_loss)

            non_pad_mask = gold.ne(transformer.Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct


        phase_metrics["loss"] = average_epoch_loss
        phase_metrics["token_accuracy"] = n_correct / n_word_total

        perplexity = np.exp(average_epoch_loss)
        phase_metrics["perplexity"] = perplexity

        time_taken = time.clock() - start
        string = ' Val loss: {:.3f} | time: {:.3f}'.format(
            average_epoch_loss, time_taken)
        print(string, end='\n')
        return phase_metrics

    def save_checkpoint(self):
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.optimizer.state_dict()
        }
        torch.save(state, filename)