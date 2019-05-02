import argparse
import os
import torch
from tqdm import tqdm

from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim

from dataset import DialogueDataset
import utils


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-dataset_filename",
                        default="data_1",
                        type = str,
                        required = False,
                        help = "The input data dir. Should contain the csv for the task.")
    parser.add_argument("-output_dir",
                        default="output_0/",
                        type=str,
                        required=False,
                        help="The output data dir")

    parser.add_argument("-pretrained_embeddings",
                        default=False,
                        type=bool,
                        required=False,
                        help="use pretrained embeddings")
    parser.add_argument("-old_model_name",
                        default=None,
                        type=str,
                        required=False,
                        help="filename of saved model")
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
    parser.add_argument("-num_epoch",
                        default=5,
                        type=int,
                        required=False,
                        help="The number of training epochs")
    parser.add_argument("-history_len",
                        default=100,
                        type=int,
                        required=False,
                        help="The max length of the history")
    parser.add_argument("-response_len",
                        default=300,
                        type=int,
                        required=False,
                        help="The max length of the history")
    parser.add_argument("-min_count",
                        default=1,
                        type=int,
                        required=False,
                        help="The minimum amount of instances to be in vocab")
    parser.add_argument("-model_dim",
                        default=512,
                        type=int,
                        required=False,
                        help="The hidden layer dimension")
    parser.add_argument("-warmup_steps",
                        default=1,
                        type=int,
                        required=False,
                        help="The warmup steps for optimizer")
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

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # get old model filename if it exists
    use_old_model = args.old_model_name is not None
    if use_old_model:
        args.old_model_filename = '{}/classification/{}/{}' \
                                  '/'.format(args.project_file,
                                             args.model_group,
                                             args.old_model_name)
        args = utils.load_args("{}args.json".format(args.old_model_filename), args)
    else:
        if args.pretrained_embeddings:
            args.pretrained_embeddings_file = \
                "/embeddings/embeddings_min{}_max{}.npy".format(
                    args.min_count, args.max_len)

    phases = list()
    dataloaders = list()

    #set the vocab to none
    vocab = None

    # create datasets
    if args.do_train:
        train_dataset = DialogueDataset(os.path.join(args.dataset_filename, "train.csv"),
            args.min_count, args.history_len, args.response_len, vocab)
        data_loader_train = torch.utils.data.DataLoader(
           train_dataset, args.train_batch_size, shuffle=True)

        vocab = train_dataset.vocab

        phases.append("train")
        dataloaders.append(data_loader_train)

    if args.do_eval:
        val_dataset = DialogueDataset(os.path.join(args.dataset_filename, "train.csv"),
            args.min_count, args.history_len, args.response_len, vocab)
        data_loader_val = torch.utils.data.DataLoader(
            val_dataset, args.val_batch_size, shuffle=True)

        vocab = val_dataset.vocab

        phases.append("train")
        dataloaders.append(data_loader_val)

    # use same vocab mapping for both train and validation datasets
    for loader in dataloaders:
        loader.vocab = vocab
    args.vocab_size = len(vocab)

    # print info
    string = ""
    for k, v in vars(args).items():
        string += "{}: {}\n".format(k, v)

    print(string)
    output = string + '\n'

    outfile = open("{}output".format(args.output_dir), 'w')
    outfile.write(output)
    outfile.close()

    # create model
    device = torch.device('cuda')

    model = Transformer(
        len(train_dataset.vocab),
        len(train_dataset.vocab),
        args.history_len)

    # initialize optimizer
    optimizer = ScheduledOptim(
        torch.optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        args.model_dim, args.warmup_steps)

    # load old model
    if use_old_model:
        model, optimizer = utils.load_checkpoint(
            args.old_model_filename, model, optimizer)

    # initialize the best model
    best_model = model
    best_optimizer = optimizer

    # initialize metrics dictionary
    metrics = {"token_accuracy": [],
               "sentence_accuracy": [],
               "perplexity": [],
               "bleu": {'bleu_1': [], 'bleu_2': [], 'bleu_3': [], 'bleu_4': []},
               "lowest_loss": 100,
               "average_train_epoch_losses": [],
               "train_epoch_losses": [],
               "val_loss": [],
               "best_epoch": 0}

    # begin training
    for epoch in range(0, args.num_epoch):
        for phase, dataloader in zip(phases, dataloaders):
            if phase == 'train':
                model.train()
                string = '--Train-- \n'
            else:
                model.eval()
                string = '--Validation-- \n'

            print(string, end='')
            output = output + '\n' + string

            epoch_bleu = {'bleu_1': [], 'bleu_2': [], 'bleu_3': [],
                          'bleu_4': []}
            epoch_loss = []
            epoch_sentenence_accuracy = []
            epoch_token_accuracy = []

            for batch in tqdm(
                    dataloader, mininterval=2,
                    desc='  - (Training)   ', leave=False):
                # prepare data
                src_seq, src_pos, tgt_seq, tgt_pos, _ = map(lambda x: x.to(device),
                                                         batch)
                gold = tgt_seq[:, 1:]

                optimizer.zero_grad()
                pred = model(src_seq, src_pos, tgt_seq, tgt_pos)


if __name__ == '__main__':
    main()