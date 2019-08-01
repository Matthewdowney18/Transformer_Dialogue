import argparse
from chatbot import ChatBot


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-experiment_dir",
                        default="experiments/exp_8",
                        type=str,
                        required=False,
                        help="The output data dir")
    parser.add_argument("-old_model_dir",
                        default="run_8",
                        type=str,
                        required=False,
                        help="filename of saved model. say None to train new model")
    parser.add_argument("-save_filename",
                        default="chatbot_history/history_2",
                        type=str,
                        required=False,
                        help="filename of saved model. say None to train new model")


    # for the beam search
    parser.add_argument("-beam_size",
                        default=6,
                        type=int,
                        required=False,
                        help="4 the beam search")
    parser.add_argument("-n_best",
                        default=6,
                        type=int,
                        required=False,
                        help="4 the beam search")
    parser.add_argument("-choose_best",
                        default=False,
                        type=bool,
                        required=False,
                        help="weather or not to chose the highest ranked response")

    # device
    parser.add_argument("-device",
                        default="cpu",
                        type=str,
                        required=False,
                        help="cuda, cpu")

    parser.add_argument("-token",
                        default="773295820:AAFF5_lCi4FdWCLd8YRbBJ9AeH2MzZWhhpw",
                        type=str,
                        required=False,
                        help="token for telegram chatbot")

    args = parser.parse_args()

    chatbot = ChatBot(args)
    chatbot.run()


if __name__ == "__main__":
    main()