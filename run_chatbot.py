import argparse
from chatbot import ChatBot


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-experiment_dir",
                        default="exp_6",
                        type=str,
                        required=False,
                        help="The output data dir")
    parser.add_argument("-old_model_dir",
                        default="run_1",
                        type=str,
                        required=False,
                        help="filename of saved model. say None to train new model")

    # for the beam search
    parser.add_argument("-beam_size",
                        default=4,
                        type=int,
                        required=False,
                        help="4 the beam search")
    parser.add_argument("-n_best",
                        default=4,
                        type=int,
                        required=False,
                        help="4 the beam search")
    parser.add_argument("-choose_best",
                        default=True,
                        type=bool,
                        required=False,
                        help="cuda, cpu")

    # device
    parser.add_argument("-device",
                        default="cuda",
                        type=str,
                        required=False,
                        help="cuda, cpu")

    parser.add_argument("-token",
                        default=None,
                        type=str,
                        required=False,
                        help="token for telegram chatbot")

    args = parser.parse_args()

    chatbot = ChatBot(args)
    chatbot.run()




if __name__ == "__main__":
    main()