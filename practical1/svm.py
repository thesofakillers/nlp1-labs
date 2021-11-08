"""Sentiment Analysis using Support Vector Machines"""
import typing as tg
import argparse
import json
from utils import extract_vocab, preprocess_reviews, split_data
import numpy.typing as npt
from sklearn.svm import LinearSVC


def encode_reviews(
    reviews: tg.List[tg.Dict], vocab: tg.Dict
) -> tg.Tuple[npt.NDArray, npt.NDArray]:
    #todo
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sentiment Analysis using Support Vector Machines"
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        help="Path to the training data",
        default="reviews.json",
    )
    parser.add_argument(
        "-pi",
        "--pos-idxs",
        nargs=4,
        type=int,
        default=[0, 900, 900, 1000],
        help="Positive train and test start and stop indices. Ignored if performing CV",
    )
    parser.add_argument(
        "-ni",
        "--neg-idxs",
        nargs=4,
        type=int,
        default=[0, 900, 900, 1000],
        help="Negative train and test start and stop indices. Ignored if performing CV",
    )
    args = parser.parse_args()
    with open(args.data, mode="r", encoding="utf-8") as f:
        reviews = json.load(f)

    lower_case_reviews = preprocess_reviews(reviews)

    vocab = extract_vocab(lower_case_reviews)
    train_reviews, test_reviews = split_data(
        lower_case_reviews,
        (args.pos_idxs[:2], args.pos_idxs[2:]),
        (args.neg_idxs[:2], args.neg_idxs[2:]),
    )

    train_X, train_Y = encode_reviews(train_reviews, vocab)
    test_X, test_Y = encode_reviews(test_reviews, vocab)

    clf = LinearSVC()
    clf.fit(train_X, train_Y)

    #todo

