import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np
import json
import random
from tqdm import tqdm

from utils import ModelConfig, load_checkpoint, cal_performance, bleu
import transformer
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
from dataset import DialogueDataset, Vocab

class ModelOperator:
    def __init__(self, args):

        # set up output directory
        self.output_dir = os.path.join(args.experiment_dir, args.run_name)
        if not os.path.exists(args.experiment_dir):
            os.mkdir(args.experiment_dir)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        if not os.path.exists(os.path.join(args.experiment_dir,"runs/")):
            os.mkdir(os.path.join(args.experiment_dir,"runs/"))

        # initialize tensorboard writer
        self.runs_dir = os.path.join(args.experiment_dir,"runs/",args.run_name)
        self.writer = SummaryWriter(self.runs_dir)

        # initialize global steps
        self.train_gs = 0
        self.val_gs = 0

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
        metrics = {"best_epoch":0, "lowest_loss":99999999999999}

        # output an example
        self.output_example(0)

        for epoch in range(num_epochs):
           # self.writer.add_graph(self.model)
            #self.writer.add_embedding(
            #    self.model.encoder.src_word_emb.weight, global_step=epoch)

            epoch_metrics = dict()

            # train
            epoch_metrics["train"] = self.execute_phase(epoch, "train")
            # save metrics
            metrics["epoch_{}".format(epoch)] = epoch_metrics
            with open(os.path.join(self.output_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)

            # validate
            epoch_metrics["val"] = self.execute_phase(epoch, "val")
            # save metrics
            metrics["epoch_{}".format(epoch)] = epoch_metrics
            with open(os.path.join(self.output_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)

            # save checkpoint
            if epoch_metrics["val"]["loss"] < metrics["lowest_loss"]:
                self.save_checkpoint(os.path.join(self.output_dir, "model.bin"))
                metrics["lowest_loss"] = epoch_metrics["val"]["loss"]
                metrics["best_epoch"] = epoch

            # record metrics to tensorboard
            self.writer.add_scalar("training loss total",
                epoch_metrics["train"]["loss"], global_step=epoch)
            self.writer.add_scalar("val loss total",
                epoch_metrics["val"]["loss"], global_step=epoch)

            self.writer.add_scalar("training perplexity",
                epoch_metrics["train"]["perplexity"], global_step=epoch)
            self.writer.add_scalar("val perplexity",
                epoch_metrics["val"]["perplexity"], global_step=epoch)

            self.writer.add_scalar("training time",
                epoch_metrics["train"]["time_taken"], global_step=epoch)
            self.writer.add_scalar("val time",
                epoch_metrics["val"]["time_taken"], global_step=epoch)

            self.writer.add_scalar("train_bleu_1",
                epoch_metrics["train"]["bleu_1"], global_step=epoch)
            self.writer.add_scalar("val_bleu_1",
                epoch_metrics["val"]["bleu_1"], global_step=epoch)
            self.writer.add_scalar("train_bleu_2",
                epoch_metrics["train"]["bleu_2"], global_step=epoch)
            self.writer.add_scalar("val_bleu_2",
                epoch_metrics["val"]["bleu_2"], global_step=epoch)

            # output an example
            self.output_example(epoch+1)

        self.writer.close()

    def execute_phase(self, epoch, phase):
        if phase == "train":
            self.model.train()
            dataloader = self.data_loader_train
            batch_size = self.config.train_batch_size
            train = True
        else:
            self.model.eval()
            dataloader = self.data_loader_val
            batch_size = self.config.val_batch_size
            train = False

        start = time.clock()
        phase_metrics = dict()
        epoch_loss = list()
        epoch_bleu_1 = list()
        epoch_bleu_2 = list()
        average_epoch_loss = None
        n_word_total = 0
        n_correct = 0
        n_word_correct = 0
        for i, batch in enumerate(tqdm(dataloader,
                          mininterval=2, desc=phase, leave=False)):
            # prepare data
            src_seq, src_pos, src_seg, tgt_seq, tgt_pos = map(
                lambda x: x.to(self.device), batch)

            gold = tgt_seq[:, 1:]

            # forward
            if train:
                self.optimizer.zero_grad()
            pred = self.model(src_seq, src_pos, src_seg, tgt_seq, tgt_pos)

            # get loss
            loss, n_correct = cal_performance(pred, gold,
                smoothing=self.config.label_smoothing)
            #average_loss = float(loss)/self.config.val_batch_size
            average_loss = float(loss)
            epoch_loss.append(average_loss)
            average_epoch_loss = np.mean(epoch_loss)

            if train:
                self.writer.add_scalar("train_loss",
                    average_loss, global_step=i + epoch * self.config.train_batch_size)
                # backward
                loss.backward()

                # update parameters
                self.optimizer.step_and_update_lr()

            # get_bleu
            output = torch.argmax(pred.view(-1, self.config.response_len-1, self.vocab_size), dim=2)
            epoch_bleu_1.append(bleu(gold, output, 1))
            epoch_bleu_2.append(bleu(gold, output, 2))

            # get_accuracy
            non_pad_mask = gold.ne(transformer.Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct


        phase_metrics["loss"] = average_epoch_loss
        phase_metrics["token_accuracy"] = n_correct / n_word_total

        perplexity = np.exp(average_epoch_loss)
        phase_metrics["perplexity"] = perplexity

        phase_metrics["bleu_1"] = np.mean(epoch_bleu_1)
        phase_metrics["bleu_2"] = np.mean(epoch_bleu_2)

        phase_metrics["time_taken"] = time.clock() - start
        string = ' {} loss: {:.3f} '.format(phase, average_epoch_loss)
        print(string, end='\n')
        return phase_metrics

    def save_checkpoint(self, filename):
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.optimizer.state_dict()
        }
        torch.save(state, filename)

    def output_example(self, epoch):
        random_index = random.randint(0, len(self.val_dataset))
        example = self.val_dataset[random_index]

        # prepare data
        src_seq, src_pos, src_seg, tgt_seq, tgt_pos = map(
            lambda x: torch.from_numpy(x).to(self.device).unsqueeze(0), example)

        # take out first token from target for some reason
        gold = tgt_seq[:, 1:]

        # forward
        pred = self.model(src_seq, src_pos, src_seg, tgt_seq, tgt_pos)
        output = torch.argmax(pred, dim=1)

        # get history text
        string = "history: "

        seg = -1
        for i, idx in enumerate(src_seg.squeeze()):
            if seg != idx.item():
                string+="\n"
                seg=idx.item()
            token = self.vocab.id2token[src_seq.squeeze()[i].item()]
            if token != '<blank>':
                string += "{} ".format(token)

        # get target text
        string += "\nTarget:\n"

        for idx in tgt_seq.squeeze():
            token = self.vocab.id2token[idx.item()]
            string += "{} ".format(token)

        # get prediction
        string += "\n\nPrediction:\n"

        for idx in output:
            token = self.vocab.id2token[idx.item()]
            string += "{} ".format(token)

        # print
        print("\n------------------------\n")
        print(string)
        print("\n------------------------\n")

        # add result to tensorboard
        self.writer.add_text("example_output", string, global_step=epoch)
        self.writer.add_histogram("example_vocab_ranking", pred, global_step=epoch)
        self.writer.add_histogram("example_vocab_choice", output,global_step=epoch)