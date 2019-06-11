import csv
import argparse
import re
import os
import string
import json
from tqdm import tqdm
import random

from bs4 import BeautifulSoup

SEPERATOR_TOKEN = "<s>"

def get_history_response(dialogue):
    sentences = [" ".join(d) for d in dialogue]
    hisory = " <s> ".join(sentences[:-1])
    hisory = "<cls> " + hisory
    return hisory, "<cls> " +sentences[-1]

def get_time(soup):
    try:
        time_tags = soup.find_all("time")
        time = ((time_tags[0]["value"]).split(",")[0]).split(":")
        return int(time[0])*3600+int(time[1])*60+int(time[2])
    except:
        return -1

def get_line(soup):
    non_tokens = {'"', '\"', '\'', '\' ', '-'}
    line = list()

    for token in soup.find_all("w"):
        valid = 1
        text = token.text
        text.lower()
        text = text.replace('\' ', '')
        text = text.replace('\'', '')
        text = text.replace('"', '')
        text = text.replace('\"', '')
        if text.isdigit():
            line.append("<NUM>")
            continue
        if not text in non_tokens:
            line.append(text)
    return line

def create_dialogues(soup, max_len, max_time_interval, movie_id):
    """
    partitions the lines of the movie into dialogues based on the time
        the lines happen, and the maximum length

    :param soup(bs): the beautifulsoups object containing lines in the movie
    :param max_len(int): the maximum number of tokens in the history
    :param max_time_interval(int): the maximum time between lines in a dialogue
    :param movie_id(int): the id of the movie
    :return:
        List: [(movie id, history, response)]
    """
    history = list()
    response = list()
    dialogue = list()
    previous_time = -1

    # iterate through every line in the movie
    for line in soup.find_all('s'):
        # if the history can not  be filled any more make that an example
        while(sum([len(sentence) for sentence in dialogue[:-1]])+len(dialogue)-1 > max_len):
            dialogue = dialogue[1:]

        # decide what time to use:
        max_time = random.randrange(max_time_interval-3, max_time_interval+3, 1)

        time = get_time(line)
        # add to dialogue if its in a certain amount of time
        if time - previous_time < max_time or previous_time == -1 or time == -1:
            exerpt = get_line(line)
            dialogue.append(exerpt)
            previous_time = time

            # if the dialogue has a length 2 or greater
            # create a new history and response
            if len(dialogue) >= 2:
                h, r = get_history_response(dialogue)
                history.append(h)
                response.append(r)
        else:
            exerpt = get_line(line)
            dialogue= [exerpt]
            previous_time = time

    return [(movie_id, h, r) for h, r in zip(history, response)]


def main():
    """
    here is the plan: for each dialogue create a history sequence of sentences
    seperated by <s>. The sentences in the history must occur in a short time
    span from another so they are relevant. The last sentence becomes the response
    where the response must also be in the span
    :return:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-dataset_filename",
                        default="/home/mattd/datasets/OpenSubtitles/xml/en/",
                        type=str,
                        required=False,
                        help="The input data dir. Should contain the xml for the task.")
    parser.add_argument("-output_dir",
                        default="/home/mattd/PycharmProjects/transformer_dialogue/data/reformatted_data/romance_data/",
                        type=str,
                        required=False,
                        help="The output data dir.")
    parser.add_argument("-genres",
                        #default=["Comedy", "Romance", "Drama", "Adventure", "Western", "Action", "Film-Noir", "Thriller", "Mystery"],
                        #default=["all"],
                        default=["Romance"],
                        type=list,
                        required=False,
                        help="The genres you would like to use.")
    parser.add_argument("-max_history_tokens",
                        default=50,
                        type=int,
                        help="the maximum amout of history tokens")
    parser.add_argument("-max_time_interval",
                        default=7,
                        type=int,
                        help="the interval between responses to be considered dialogue")
    parser.add_argument("-train_val_test_split",
                        default=(.8, .1, .1),
                        type=tuple,
                        help="the ratios for train val test split")
    parser.add_argument("-a_nice_note",
                        default="yuuuge",
                        type=str,
                        required=False,
                        help="leave a nice lil note for yourself in the future")

    args = parser.parse_args()

    if args.genres == ["all"]:
        args.genres = list()
        for dir in os.listdir(args.dataset_filename):
            if not str.isdigit(dir):
                args.genres.append(dir)

    meta_data = dict()
    meta_data["args"] = dict()

    for arg in vars(args):
        meta_data["args"][arg] = getattr(args, arg)

    meta_data["genres"] = dict()

    movie_id = 0
    all_dialogues = list()

    # create dialogues for each movie
    for genre in args.genres:
        meta_data["genres"][genre] = dict()
        for year in tqdm(os.listdir("{}{}".format(args.dataset_filename, genre))):
            meta_data["genres"][genre][year] = dict()
            for movie in os.listdir("{}{}/{}".format(args.dataset_filename, genre, year)):
                if (movie.split(".")[-1] == "info"):
                    continue

                handle = open("{}{}/{}/{}".format(args.dataset_filename, genre, year, movie), 'r')
                soup = BeautifulSoup(handle)

                movie_name = "_".join([word for word in (movie.split(".")[0]).split("_") if not word.isdigit()])
                movie_data = dict()
                movie_data["id"] = movie_id

                dailogues = create_dialogues(soup.find("document"),
                    args.max_history_tokens, args.max_time_interval, movie_id)
                all_dialogues+=dailogues

                movie_data["length"] = len(dailogues)
                meta_data["genres"][genre][year][movie_name] = movie_data
                movie_id+=1

    meta_data["total_len"] = len(all_dialogues)

    # shuffle dataset
    random.shuffle(all_dialogues)

    train_end = int(len(all_dialogues)*args.train_val_test_split[0])
    test_len = int(len(all_dialogues)*args.train_val_test_split[2])

    train_set = all_dialogues[:train_end]
    val_set = all_dialogues[train_end:-test_len]
    test_set = all_dialogues[-test_len:]

    meta_data["train_len"] = len(train_set)
    meta_data["val_len"] = len(val_set)
    meta_data["test_len"] = len(test_set)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    with open(os.path.join(args.output_dir, "meta_data"), 'w') as fp:
        json.dump(meta_data, fp, indent=4, sort_keys=True)

    with open(os.path.join(args.output_dir, "train.csv"), 'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['movie_id', 'history', 'response'])
        for row in train_set:
            csv_out.writerow(row)

    with open(os.path.join(args.output_dir, "val.csv"), 'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['movie_id', 'history', 'response'])
        for row in val_set:
            csv_out.writerow(row)

    with open(os.path.join(args.output_dir, "test.csv"), 'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['movie_id', 'history', 'response'])
        for row in test_set:
            csv_out.writerow(row)

if __name__ == '__main__':
    main()