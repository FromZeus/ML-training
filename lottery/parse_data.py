import argparse
import csv
import sys


def parse_csv(file, mode):
    data = []

    csv_reader = csv.reader(file, delimiter=',')
    for row in csv_reader:
        data.append(list(map(lambda x: int(x), row[4:9] if mode == "main" else row[9:10] )))

    return data


def get_indexes(data_file, predictions_file):
    indexes = []
    data_reader = csv.reader(data_file, delimiter=',')
    predictions_reader = csv.reader(predictions_file, delimiter=',')
    data = list(data_reader)
    predictions = list(predictions_reader)

    for i in range(len(predictions)):
        indexes.append([predictions[i].index(
            j) + 1 for j in data[i+1] if j in predictions[i]])

    return indexes


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("data", type=str,
                            help=("Path to data."))
        parser.add_argument("-o", "--output", dest="output",
                            type=str, default="parsed.csv",
                            help=("Path to save."))
        parser.add_argument("-p", "--predictions", dest="predictions",
                            type=str, default="predictions.csv",
                            help=("Path to predictions."))
        parser.add_argument("-m", "--mode", dest="mode", choices=["main", "super"],
                            default="main", help=("Mode to parse: [main, super]"))
        parser.add_argument("-i", "--indexes", dest="indexes",
                            action="store_true", help=("Calculate indexes."))
        args = parser.parse_args()

        data = []

        if args.indexes:
            with open(args.data) as data_file:
                with open(args.predictions) as predictions_file:
                    data = get_indexes(data_file, predictions_file)
        else:
            with open(args.data) as csv_file:
                data = parse_csv(csv_file, args.mode)

        with open(args.output, "w+") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerows(data)

    except KeyboardInterrupt:
        print('\nThe process was interrupted by the user')
        raise SystemExit


if __name__ == "__main__":
    main()
