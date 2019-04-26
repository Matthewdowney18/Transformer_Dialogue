import csv
import argparse
import re
import os
import json
from tqdm import tqdm

from bs4 import BeautifulSoup

SEPERATOR_TOKEN = "<s>"

def get_history_response(dialogue):
    sentences = [" ".join(d) for d in dialogue]
    hisory = " <s> ".join(sentences[:-1])
    return hisory, sentences[-1]

def get_time(soup):
    time_tags = soup.find_all("time")
    if time_tags == []:
        return -1
    time = ((time_tags[0]["value"]).split(",")[0]).split(":")
    return int(time[0])*3600+int(time[1])*60+int(time[2])

def get_line(soup):
    line = list()
    for token in soup.find_all("w"):
        text = token.text
        text.lower()
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
        if sum([len(sentence) for sentence in dialogue])+len(dialogue)-1 > max_len:
            h, r = get_history_response(dialogue)
            history.append(h)
            response.append(r)
            previous_time = -1
            dialogue = list()


        time = get_time(line)
        # add to dialogue if its in a certain amount of time
        if time - previous_time < max_time_interval or previous_time == -1 or time == -1:
            exerpt = get_line(line)
            dialogue.append(exerpt)
            previous_time = time
        else:
            # if the dialogue has a length 2 or greater
            # create a new history and response
            if len(dialogue) >= 2:
                h, r = get_history_response(dialogue)
                history.append(h)

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
                        default="OpenSubtitles/xml/en/",
                        type=str,
                        required=False,
                        help="The input data dir. Should contain the xml for the task.")
    parser.add_argument("-output_dir",
                        default="data_1",
                        type=str,
                        required=False,
                        help="The output data dir.")
    parser.add_argument("-genres",
                        default=["Comedy", "Romance", "Drama", "adventure"],
                        type=list,
                        required=False,
                        help="The genres you would like to use.")
    parser.add_argument("-max_history_tokens",
                        default=100,
                        type=int,
                        help="the maximum amout of history tokens")
    parser.add_argument("-max_time_interval",
                        default=8,
                        type=int,
                        help="the interval between responses to be considered dialogue")

    args = parser.parse_args()

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

                movie_name = (movie.split("_")[-1]).split(".")[0]
                movie_data = dict()
                meta_data["id"] = movie_id

                dailogues = create_dialogues(soup.find("document"),
                    args.max_history_tokens, args.max_time_interval, movie_id)
                all_dialogues+=dailogues

                meta_data["length"] = len(dailogues)
                meta_data["genres"][genre][year][movie_name] = movie_data
                movie_id+=1

    meta_data["total_len"] = len(all_dialogues)

    with open(os.path.join(args.output_dir, "meta_data"), 'w') as fp:
        json.dump(meta_data, fp, indent=4, sort_keys=True)

    with open(os.path.join(args.output_dir, "data"), 'wb') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['movie_id', 'history', "response"])
        for row in all_dialogues:
            csv_out.writerow(row)

if __name__ == '__main__':
    main()