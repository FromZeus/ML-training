import argparse

from keras.preprocessing.text import text_to_word_sequence

from github import GitHub


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("-r", "--repositories", dest="repos", type=str,
                            nargs="+", required=True,
                            help=("List of target repositories."))
        args = parser.parse_args()

    except KeyboardInterrupt:
        print('\nThe process was interrupted by the user')
        raise SystemExit


if __name__ == "__main__":
    main()
