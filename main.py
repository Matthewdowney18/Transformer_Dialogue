import argparse
import os
import torch
import time
import json
import numpy as np
from tqdm import tqdm

import transformer
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim

from dataset import DialogueDataset, Vocab
import utils


def main():
    parser = argparse.ArgumentParser()
    # good arguments
    parser.add_argument("-dataset_filename",
                        default="debug_data",
                        type = str,
                        required = False,
                        help = "The input data dir. Should contain the csv for the task.")
    parser.add_argument("-output_dir",
                        default="output_0/",
                        type=str,
                        required=False,
                        help="The output data dir")
    parser.add_argument("-old_model_dir",
                        default=None,
                        type=str,
                        required=False,
                        help="filename of saved model. say None to train new model")
    parser.add_argument("-num_epoch",
                        default=5,
                        type=int,
                        required=False,
                        help="The number of training epochs")

    # dataset info
    parser.add_argument("-do_eval",
                        default=True,
                        type=bool,
                        required=False,
                        help="True to train model")
    parser.add_argument("-do_train",
                        default=True,
                        type=bool,
                        required=False,
                        help="True to train model")
    parser.add_argument("-min_count",
                        default=1,
                        type=int,
                        required=False,
                        help="The minimum amount of instances to be in vocab")
    parser.add_argument("-train_batch_size",
                        default=20,
                        type=int,
                        required=False,
                        help="The batch size for training")
    parser.add_argument("-val_batch_size",
                        default=3,
                        type=int,
                        required=False,
                        help="The batch size for training")
    parser.add_argument("-history_len",
                        default=300,
                        type=int,
                        required=False,
                        help="The max length of the history")
    parser.add_argument("-response_len",
                        default=100,
                        type=int,
                        required=False,
                        help="The max length of the response")

    # transformer specs
    parser.add_argument("-pretrained_embeddings_dir",
                        default=None,
                        type=str,
                        required=False,
                        help="use pretrained embeddings")
    parser.add_argument("-embedding_dim",
                        default=512,
                        type=int,
                        required=False,
                        help="The embeddings dim will be ignored if pretrained"
                        "embeddings dir is not none")
    parser.add_argument("-model_dim",
                        default=512,
                        type=int,
                        required=False,
                        help="The hidden layer dimension")
    parser.add_argument("-inner_dim",
                        default=2048,
                        type=int,
                        required=False,
                        help="The inner dim")
    parser.add_argument("-num_layers",
                        default=6,
                        type=int,
                        required=False,
                        help="The number of layers")
    parser.add_argument("-num_heads",
                        default=8,
                        type=int,
                        required=False,
                        help="The number of attention heads")
    parser.add_argument("-dim_k",
                        default=64,
                        type=int,
                        required=False,
                        help="not really sure what k is")
    parser.add_argument("-dim_v",
                        default=64,
                        type=int,
                        required=False,
                        help="not really sure what v is")
    parser.add_argument("-dropout",
                        default=.1,
                        type=float,
                        required=False,
                        help="dropout probability")

    # optimizer specs
    parser.add_argument("-warmup_steps",
                        default=1,
                        type=int,
                        required=False,
                        help="The warmup steps for optimizer")
    parser.add_argument("-label_smoothing",
                        default=True,
                        type=bool,
                        required=False,
                        help="The batch size for training")


    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # check if to use a already trained model
    use_old_model = args.old_model_dir is not None
    if use_old_model:
        # create old model filename, and load args file
        args.old_model_filename = os.path.join(args.old_model_dir, "model.bin")
        args = utils.load_args(os.path.join(args.old_model_dir, "args.json"), args)

        # create vocab. if you are using an old model, you do not want to
        # change the vocab dictionary
        vocab = Vocab
        vocab.load_from_dict(os.path.join(args.old_model_dir, "vocab.json"))
        update_vocab = False
    else:
        vocab = None
        update_vocab = True

    # create phases for training, validation
    phases = list()
    dataloaders = list()

    # create datasets
    if args.do_train:
        train_dataset = DialogueDataset(os.path.join(args.dataset_filename, "train.csv"),
            args.min_count, args.history_len, args.response_len, vocab, update_vocab)
        data_loader_train = torch.utils.data.DataLoader(
           train_dataset, args.train_batch_size, shuffle=True)

        vocab = train_dataset.vocab

        phases.append("train")
        dataloaders.append(data_loader_train)

    if args.do_eval:
        val_dataset = DialogueDataset(os.path.join(args.dataset_filename, "train.csv"),
            args.min_count, args.history_len, args.response_len, vocab, update_vocab)
        data_loader_val = torch.utils.data.DataLoader(
            val_dataset, args.val_batch_size, shuffle=True)

        vocab = val_dataset.vocab

        phases.append("val")
        dataloaders.append(data_loader_val)

    # use same vocab mapping for both train and validation datasets
    for loader in dataloaders:
        loader.vocab = vocab
    args.vocab_size = len(vocab)

    # save vocab to dir
    vocab.save_to_dict(os.path.join(args.output_dir, "vocab.json"))

    # print info
    string = ""
    for k, v in vars(args).items():
        string += "{}: {}\n".format(k, v)

    print(string)
    output = string + '\n'

    outfile = open("{}output".format(args.output_dir), 'w')
    outfile.write(output)
    outfile.close()

    # save args to file
    utils.save_args(os.path.join(args.output_dir, "args.json"),  args)

    # create model
    device = torch.device('cuda')

    model = Transformer(
        args.vocab_size,
        args.vocab_size,
        args.history_len,
        args.response_len,
        d_word_vec=args.embedding_dim,
        d_model=args.model_dim,
        d_inner=args.inner_dim,
        n_layers=args.num_layers,
        n_head=args.num_heads,
        d_k=args.dim_k,
        d_v=args.dim_v,
        dropout=args.dropout
    )

    # initialize optimizer
    optimizer= torch.optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09)

    # load old model
    if use_old_model:
        model, optimizer = utils.load_checkpoint(
            args.old_model_filename, model, optimizer)

    optimizer = ScheduledOptim(optimizer, args.model_dim, args.warmup_steps)

    model = model.to(device)

    # initialize the best model
    best_model = model
    best_optimizer = optimizer.optimizer

    # initialize all metrics dictionary
    all_metrics = {"best_epoch": 0,
                   "lowest_loss": 100}

    # begin training
    for epoch in range(0, args.num_epoch):
        # output info
        string = 'Epoch: {}\n'.format(epoch)
        print(string, end='')
        output = output + '\n' + string

        # initialize metrics dictionary for epoch
        epoch_metrics = {}
        epoch_metrics["epoch"] = epoch

        for phase, dataloader in zip(phases, dataloaders):
            if phase == 'train':
                model.train()
                string = '--Train-- \n'
            else:
                model.eval()
                string = '--Validation-- \n'

            start = time.clock()

            print(string, end='')
            output = output + '\n' + string

            epoch_bleu = {'bleu_1': [], 'bleu_2': [], 'bleu_3': [],
                          'bleu_4': []}

            phase_metrics = {}
            epoch_loss = []
            n_word_total = 0
            n_correct = 0
            n_word_correct = 0
            average_epoch_loss = None

            for batch in tqdm(
                    dataloader, mininterval=2,
                    desc='  - (Training)   ', leave=False):
                # prepare data
                src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device),
                                                         batch)
                gold = tgt_seq[:, 1:]

                # forward
                optimizer.zero_grad()
                pred = model(src_seq, src_pos, tgt_seq, tgt_pos)

                # get loss
                loss, n_correct = utils.cal_performance(pred, gold, smoothing=
                    args.label_smoothing)
                epoch_loss.append(float(loss))
                average_epoch_loss = np.mean(epoch_loss)

                non_pad_mask = gold.ne(transformer.Constants.PAD)
                n_word = non_pad_mask.sum().item()
                n_word_total += n_word
                n_word_correct += n_correct

                if phase == "train":
                    # backward
                    loss.backward()

                    # update parameters
                    optimizer.step_and_update_lr()

            phase_metrics["loss"] = average_epoch_loss
            phase_metrics["token_accuracy"] = n_correct / n_word_total

            perplexity = np.exp(average_epoch_loss)
            phase_metrics["perplexity"] = perplexity

            time_taken = time.clock() - start
            string = ' {} loss: {:.3f} | time: {:.3f}'.format(
                phase, average_epoch_loss, time_taken)
            string += ' | lowest loss: {:.3f}'.format(
                all_metrics["lowest_loss"])
            print(string, end='\n')
            output += '\n' + string + '\n'

            if phase == 'val':
                # update best model
                if average_epoch_loss < all_metrics["lowest_loss"]:
                    best_model = model
                    best_optimizer = optimizer.optimizer
                    all_metrics["best_epoch"] = epoch
                    all_metrics["lowest_loss"] = average_epoch_loss

                epoch_metrics["val"] = phase_metrics
            else:
                epoch_metrics["train"] = phase_metrics
        # save metrics
        all_metrics[epoch] = epoch_metrics
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
            json.dump(all_metrics, f, indent=4)

        # save weights
        utils.save_checkpoint(os.path.join(args.output_dir, "model.bin"),
            all_metrics["best_epoch"], best_model, best_optimizer, epoch,
            model, optimizer.optimizer)

        outfile = open(os.path.join(args.output_dir, "output"), 'w')
        outfile.write(output)
        outfile.close()

if __name__ == '__main__':
    main()