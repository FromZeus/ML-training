import argparse
import csv
import sys
import time

import matplotlib.pyplot as plt

from dummy import Dummy
from environment import LotteryEnv
from dqn import LottoNN


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("mode", choices=["predict", "learn"], type=str,
                            help=("Mode to run: learn or predict"))
        parser.add_argument("-d", "--data", dest="data_file",
                            type=str, help=("Path to data file."))
        parser.add_argument("-m", "--metadata", dest="metadata_file",
                            type=str, help=("Path to metadata file."))
        parser.add_argument("--model", dest="model_file", default=None, required='predict' in sys.argv,
                            type=str, help=("Path to model file."))
        parser.add_argument("-n", "--name", dest="model_name", default="model",
                            type=str, help=("Name of the model."))
        parser.add_argument("-s", "--steps", dest="steps", required='learn' in sys.argv or 'predict' in sys.argv,
                            type=int, help=("How many steps to run each episode."))
        parser.add_argument("-e", "--episodes", dest="episodes", required='learn' in sys.argv,
                            type=int, help=("How many episodes to execute."))
        parser.add_argument("-b", "--balls", dest="balls_number", default=69,
                            type=int, help=("Number of balls."))
        parser.add_argument("-a", "--actions", dest="actions_number",
                            type=int, help=("Number of actions."))
        parser.add_argument("-o", "--optimizer", dest="optimizer", choices=["adam", "nadam", "rmsprop"], default="adam",
                            type=str, help=("Number of actions."))
        parser.add_argument("--seq", dest="sequentially",
                            action="store_true", help=("Predict results for all data sequentially."))
        args = parser.parse_args()

        data = None
        metadata = None
        if args.data_file:
            data = []
            with open(args.data_file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    data.append(list(map(lambda x: int(x) - 1, row)))
            if args.metadata_file:
                metadata = []
                with open(args.metadata_file) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    for row in csv_reader:
                        metadata.append(list(map(lambda x: int(x) - 1, row)))

        if args.mode == "learn":
            m = LottoNN(
                args.model_name, args.model_file, args.steps, data, metadata,
                balls_number=args.balls_number, actions_number=args.actions_number,
                optimizer=args.optimizer
            )
            t_start = time.time()
            average = m.fit(args.episodes)
            print(time.time() - t_start)
            plt.plot(average)
            plt.savefig(args.model_name + ".png", dpi=400)
        elif args.mode == "predict":
            m = LottoNN(
                args.model_name, args.model_file, args.steps, data, balls_number=args.balls_number)
            if args.sequentially:
                with open("predictions.csv", "w+") as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerows(m.predict(args.sequentially))
            else:
                print(m.predict())

    except KeyboardInterrupt:
        average = m.rescue()
        plt.plot(average)
        plt.savefig(args.model_name + ".png", dpi=400)

        print('\nThe process was interrupted by the user')
        raise SystemExit


if __name__ == "__main__":
    main()
