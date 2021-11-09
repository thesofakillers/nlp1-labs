"""Sentiment Analysis using Support Vector Machines"""
import typing as tg
import argparse
import json
from utils import extract_vocab, preprocess_reviews, split_data, SENT_MAP, rr_cv_split
import numpy as np
import numpy.typing as npt
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.sparse import csr_matrix


def encode_reviews(
    reviews: tg.List[tg.Dict],
    codeword_map: tg.Dict,
    use_pos: bool = False,
) -> tg.Tuple[npt.NDArray, npt.NDArray]:
    """
    Encode reviews into a features matrix and labels vector.

    Returns
    -------
    feature_mat : npt.NDArray
        A matrix of features, where each row represents
        a review and each column represents the count of
        a given word from our vocab in that review.
    label_vec : npt.NDArray
        A vector of labels, where each element is either
        0 or 1, representing the sentiment (POS or NEG) of the review.
    """
    feature_mat = np.zeros((len(reviews), len(codeword_map)))
    label_vec = np.zeros(len(reviews))

    for i, review in enumerate(reviews):
        label_vec[i] = SENT_MAP[review["sentiment"]]
        for sentence in review["content"]:
            for token, pos in sentence:
                if use_pos:
                    key = (token, pos)
                else:
                    key = token
                if key in codeword_map:
                    feature_mat[i, codeword_map[key]] += 1

    return feature_mat, label_vec


def perform_rr_cv(
    data,
    n_splits,
    modulo,
    data_len: tg.Optional[int] = None,
    use_pos: bool = False,
    only_open: bool = False,
    std: bool = True,
):
    """
    Performs round robin cross validation

    Returns
    -------
    metrics : npt.NDArray
        (n_splits, ) array of accuracies
    """

    if data_len is None:
        data_len = len(data)

    train, test = rr_cv_split(data_len, n_splits, modulo)

    metrics = np.zeros(n_splits, dtype=float)

    for i, (train_idxs, test_idxs) in enumerate(zip(train, test)):
        print(f"Cross validating on split {i+1} of {n_splits}")

        train_data = [
            data_entry for data_entry in data if data_entry["cv"] in train_idxs
        ]
        test_data = [data_entry for data_entry in data if data_entry["cv"] in test_idxs]

        metrics[i] = train_eval_svm(train_data, test_data, use_pos, only_open, std)

    return metrics


def train_eval_svm(
    train_data,
    test_data,
    use_pos: bool = False,
    only_open: bool = False,
    std: bool = True,
):
    vocab = extract_vocab(train_data, use_pos, only_open)
    codeword_map = {key: idx for idx, key in enumerate(vocab.keys())}
    train_X, train_Y = encode_reviews(train_data, codeword_map, use_pos)
    test_X, test_Y = encode_reviews(test_data, codeword_map, use_pos)

    if std:
        clf = make_pipeline(StandardScaler(with_mean=False), LinearSVC(max_iter=10000))
    else:
        clf = LinearSVC(max_iter=10000)

    clf.fit(csr_matrix(train_X), train_Y)

    preds = clf.predict(csr_matrix(test_X))
    acc = np.mean(preds == test_Y)

    return acc


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
    parser.add_argument(
        "-cv",
        "--cross-validate",
        action="store_true",
        default=False,
        help="Flag to perform cross validation",
    )
    parser.add_argument(
        "-p",
        "--pos",
        action="store_true",
        default=False,
        help="Flag to use parts of speech",
    )
    parser.add_argument(
        "-s",
        "--standardise",
        action="store_true",
        default=False,
        help="Whether to standardise feature matrix before passing to SVM",
    )
    parser.add_argument(
        "-o",
        "--only-open",
        action="store_true",
        default=False,
        help="Whether to use only open-class words when using POS features",
    )
    args = parser.parse_args()
    with open(args.data, mode="r", encoding="utf-8") as f:
        reviews = json.load(f)

    lower_case_reviews = preprocess_reviews(reviews)

    if args.cross_validate:
        accuracies = perform_rr_cv(
            lower_case_reviews, 10, 10, 1000, args.pos, args.only_open, args.standardise
        )
        print(accuracies)
        print(f"Mean accuracy: {accuracies.mean()}")
        print(f"accuracy variance: {accuracies.var()}")

    else:
        train_reviews, test_reviews = split_data(
            lower_case_reviews,
            (args.pos_idxs[:2], args.pos_idxs[2:]),
            (args.neg_idxs[:2], args.neg_idxs[2:]),
        )
        acc = train_eval_svm(
            train_reviews, test_reviews, args.pos, args.only_open, args.standardise
        )
        print(acc)
