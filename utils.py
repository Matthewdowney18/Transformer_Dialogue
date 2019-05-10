import numpy as np
import torch
from torch.autograd import Variable
import os
import json
from torch.nn import Parameter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.translate import bleu_score
import torch.nn.functional as F
import transformer


def get_sequences_lengths(sequences, masking=0, dim=1):
    if len(sequences.size()) > 2:
        sequences = sequences.sum(dim=2)

    masks = torch.ne(sequences, masking)

    lengths = masks.sum(dim=dim)

    return lengths


def cuda(obj):
    yolo = torch.cuda.current_device()
    frig = torch.cuda.device_count()
    if torch.cuda.is_available():
        obj = obj.cuda()
        #obj = obj
    return obj


def variable(obj, volatile=False):
    if isinstance(obj, (list, tuple)):
        return [variable(o, volatile=volatile) for o in obj]

    if isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj)

    obj = cuda(obj)
    obj = Variable(obj, volatile=volatile)
    return obj


def argmax(inputs, dim=-1):
    values, indices = inputs.max(dim=dim)
    return indices


def get_sentence_from_indices(indices, vocab, eos_token, join=True):
    tokens = []
    for idx in indices:
        token = vocab.id2token[idx]

        if token == eos_token:
            break

        tokens.append(token)

    if join:
        tokens = ' '.join(tokens)

    tokens = tokens

    return tokens


def get_pretrained_embeddings(embeddings_dir):
    embeddings = np.load(embeddings_dir)
    emb_tensor = torch.FloatTensor(embeddings)
    return emb_tensor

def save_args(filename, args):
    with open(filename, 'w') as f:
        json.dump(vars(args), f, indent=4)

def load_args(filename, args):
    '''
    loads the params of a saved model
    useful for when you want to load the model, and need to create a model
    with the same parameters first
    :param filename: filename of model
    :return: parameters (dict) files (dict)
    '''
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            checkpoint = json.load(f)

        # embedding dim
        args.embedding_dim = checkpoint["embedding_dim"]
        args.pretrained_embeddings_dir = None
        # model_dim
        args.model_dim = checkpoint["model_dim"]
        # inner_dim
        args.inner_dim = checkpoint["inner_dim"]
        # num_layers
        args.num_layers = checkpoint["num_layers"]
        # num_heads
        args.num_heads = checkpoint["num_heads"]
        # dim_k
        args.dim_k = checkpoint["dim_k"]
        # dim_v
        args.dim_v = checkpoint["dim_v"]
        # dropout
        args.dropout = checkpoint["dropout"]
        # history len
        args.history_len = checkpoint["history_len"]
        # response_len
        args.response_len = checkpoint["response_len"]

    else:
        raise ValueError("no file found at {}".format(filename))
    return checkpoint

def save_checkpoint(filename, best_epoch, best_model, best_optimizer,
                    epoch, model, optimizer):
    '''
    saves model into a state dict, along with its training statistics,
    and parameters
    :param best_epoch:
    :param best_model:
    :param best_optimizer:
    :param epoch:
    :param model:
    :param optimizer:
    :return:
    '''
    state = {
        'best_model': best_model.state_dict(),
        'best_optimizer': best_optimizer.state_dict(),
        'model': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'best_epoch': best_epoch,
        'epoch' : epoch,
        }
    torch.save(state, filename)

def load_checkpoint(filename, model, optimizer, use_best_model = True):
    '''
    loads previous model
    :param filename: file name of model
    :param model: model that contains same parameters of the one you are loading
    :param optimizer:
    :param use_best_model: true if you want to use the model with the lowest
        val loss
    :return: loaded model, checkpoint
    '''
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        if use_best_model:
            model.load_state_dict(checkpoint['best_model'])
            optimizer.load_state_dict(checkpoint['best_optimizer'])
        else:
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer


def freeze_layer(layer, bool):
    '''
    freezes or unfreezes layer of pytorch model
    :param layer:
    :param bool: True to freeze, False to unfreeze
    :return:
    '''
    for param in layer.parameters():

        param.requires_grad = not bool
    layer.training = not bool
    return layer


def encoder_accuracy(targets, predicted):
    '''
    finds the token level, and sentence level accuracy
    :param targets: tensor of sentence
    :param predicted: tensor of predicted senences
    :return: sentence_accuracy, token_accuracy (float)
    '''
    batch_size = targets.size()
    sentence_acc = [1] * batch_size[0]
    token_acc = []
    for batch in range(0, batch_size[0]):
        for token in range(0, batch_size[1]):
            if targets[batch, token] != 0:
                if targets[batch, token] != predicted[batch, token]:
                    sentence_acc[batch] = 0
                    token_acc.append(0)
                else:
                    token_acc.append(1)

    sentence_accuracy = sum(sentence_acc)/len(sentence_acc)
    token_accuracy = sum(token_acc) / len(
        token_acc)
    return sentence_accuracy, token_accuracy

def bleu(targets, predicted, n_grams=4):
    '''
    calculates bleu score
    :param targets: tensor of actual sentences
    :param predicted: tensor of predicted sentences
    :param n_grams: number of n-grams for bleu score (int)
    :return: bleu score (float)
    '''
    reference = [[[str(x.item())for x in row if x.item() != 0]]
                 for row in targets]
    hypothosis = [[str(x.item()) for x in row if x.item() != 0]
                 for row in predicted]
    weights = [1/n_grams] * n_grams

    chencherry = bleu_score.SmoothingFunction()
    bleu_1 = bleu_score.corpus_bleu(
        reference, hypothosis, weights=weights,
        smoothing_function=chencherry.method1)
    return bleu_1


def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(transformer.Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct


def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(transformer.Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=transformer.Constants.PAD, reduction='sum')

    return loss