from vecto import embeddings
import numpy as np
import pickle

from classification.dataset import SentenceDataset


def get_vectors(filename):
    objects = dict()
    with (open(filename, "rb")) as openfile:
        while True:
            try:
                objects = pickle.load(openfile)
            except EOFError:
                break
    return objects


def main():
    max_len = 128
    min_count = 1

    embeddings_dir = '/home/mattd/embeddings/reddit_2/'
    #dataset_path = '/home/mattd/datasets/AskReddit/'
    dataset_path = "/home/mattd/PycharmProjects/reddit/classification/data/"
    dataset_train_filename = "{}train.csv".format(dataset_path)
    dataset_val_filename = "{}validation.csv".format(dataset_path)
    save_dir = "/home/mattd/PycharmProjects/reddit/embeddings/"

    dataset_train = SentenceDataset(dataset_train_filename, max_len, min_count)
    dataset_val = SentenceDataset(dataset_val_filename, max_len, min_count,
                                  dataset_train.vocab)
    #dataset.add_file(eng_fr_filename2)

    vectors = embeddings.load_from_dir(embeddings_dir)

    #emb = embeddings.load_from_dir(embeddings_dir)

    embs_matrix = np.zeros((len(dataset_val.vocab), len(vectors.matrix[0])))

    for i, token in enumerate(dataset_val.vocab.token2id):
        if vectors.has_word(token):
            embs_matrix[i] = vectors.get_vector(token)
    np.save('{}embeddings_min{}_max{}'.format(save_dir, min_count, max_len),
            embs_matrix)


if __name__ == '__main__':
    main()