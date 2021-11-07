"""Review Sentiment Analysis using Naive Bayes"""

import typing as tg
import argparse
import json
import numpy as np
import numpy.typing as npt
from nltk.stem.porter import PorterStemmer

SENT_MAP = {
    "POS": 0,
    "NEG": 1,
}


def extract_vocab(documents: tg.List[tg.Dict], stem: bool = False):
    """
    Extracts the vocabulary from the documents,

    Parameters
    ----------
    documents : tg.List[Dict]
    stem : bool, default False
        Whether to stem the words, if False,
        words are simply lower-cased

    Returns
    -------
    dict
        The vocabulary
    """
    if stem:
        stemmer = PorterStemmer()
    vocab = {}
    for doc in documents:
        for sentence in doc["content"]:
            for word, _pos in sentence:
                if stem:
                    word = stemmer.stem(word)
                else:
                    word = word.lower()
                if word not in vocab:
                    vocab[word] = {"POS": 0, "NEG": 0}
                if doc["sentiment"] == "POS":
                    vocab[word]["POS"] += 1
                elif doc["sentiment"] == "NEG":
                    vocab[word]["NEG"] += 1
    return vocab


def train_nb(
    classes: tg.Tuple[str, ...],
    documents: tg.List[tg.Dict],
    alpha: float = 0,
    stem: bool = False,
):
    """
    Trains a Naive Bayes model
    Parameters
    ----------
    classes : tg.Tuple[str, ...]
        the classes that the model should predict
    documents : tg.List[Dict]
        the documents on which to train on
    alpha : float
        smoothing parameter, which is added to word counts.
    stem : bool, default False
        Whether to stem the words, if False,
        words are simply lower-cased

    Returns
    -------
    vocab : tg.Dict
        the computed vocabulary for the input documents
    logprior : npt.NDArray
        (n_classes, ) array of log priors
    loglikelihood : tg.List[tg.Dict]
        (n_classes, ) array of dictionary containing loglikelihood of each word
    """
    vocab: tg.Dict = extract_vocab(documents, stem)
    if alpha == 0:
        # filter vocab if we are not applying smoothing
        vocab = {k: v for (k, v) in vocab.items() if v["POS"] > 0 and v["NEG"] > 0}

    n_docs = len(documents)
    logprior = np.zeros(len(classes))
    loglikelihood: tg.List[tg.Dict] = [{} for _ in range(len(classes))]
    for c, clx in enumerate(classes):
        clx_docs = [doc for doc in documents if doc["sentiment"] == clx]
        n_docs_clx = len(clx_docs)
        logprior[c] = np.log(n_docs_clx / n_docs)
        # compute sum of word counts, which we'll use to compute loglikelihood (denom)
        word_counts_sum = (np.array([vocab[word][clx] for word in vocab]) + alpha).sum()
        for word in vocab:
            loglikelihood[c][word] = np.log(
                (vocab[word][clx] + alpha) / (word_counts_sum)
            )
    return vocab, logprior, loglikelihood


def nb_predict(
    classes: tg.Tuple[str, ...],
    vocab: tg.Dict,
    logprior: npt.NDArray[float],
    loglikelihood: tg.List[tg.Dict],
    doc: tg.Dict,
    stem: bool = False,
):
    """
    Predicts the sentiment of a document

    Parameters
    ----------
    classes : tg.Tuple[str, ...]
    vocab : tg.Dict
    prior : npt.NDArray
    loglikelihood : tg.List[tg.Dict]
    doc : tg.Dict
        the document to classify
    stem: bool, default False
        Whether to stem the words, if False,
        words are simply lower-cased

    Returns
    -------
    int
        The index of the predicted class
    """
    if stem:
        stemmer = PorterStemmer()

    doc_text = [
        stemmer.stem(word) if stem else word.lower()
        for sentence in doc["content"]
        for word, _pos in sentence
        if word in vocab
    ]
    score: npt.NDArray = np.zeros(len(classes), dtype=float)
    for c, clx in enumerate(classes):
        score[c] = logprior[c]
        for word in doc_text:
            if word in vocab:
                score[c] += loglikelihood[c][word]

    return np.argmax(score)


def split_data(
    data: tg.List[tg.Dict],
    pos_start_stop: tg.Tuple[tg.Tuple[int, int], tg.Tuple[int, int]],
    neg_start_stop: tg.Optional[
        tg.Tuple[tg.Tuple[int, int], tg.Tuple[int, int]]
    ] = None,
) -> tg.Tuple[tg.List[tg.Dict], tg.List[tg.Dict]]:
    """
    Splits data into training and testing sets,
    allowing for different starts and stops between
    negative and positive classes

    Parameters
    ----------
    data : tg.List[Dict]
        the data to split
    pos_start_stop : tg.Tuple[tg.Tuple[int, int], tg.Tuple[int, int]]
        ((train_start, train_stop), (test_start, test_stop))
        for positive class
    neg_start_stop : tg.Tuple[tg.Tuple[int, int], tg.Tuple[int, int]]
        ((train_start, train_stop), (test_start, test_stop))
        for negative class. Optional, if None, will use the same as positive

    Returns
    -------
    tg.Tuple[tg.List[Dict], tg.List[Dict]]
        (train_data, test_data)
    """
    if neg_start_stop is None:
        neg_start_stop = pos_start_stop
    starts_stops = (pos_start_stop, neg_start_stop)
    train = []
    test = []
    for i, clx in enumerate(("POS", "NEG")):
        print(
            f"Splitting data for class {clx} \n"
            f"Using train indices {starts_stops[i][0]} and "
            f"test indices {starts_stops[i][1]}"
        )

        train_idxs = range(starts_stops[i][0][0], starts_stops[i][0][1])
        test_idxs = range(starts_stops[i][1][0], starts_stops[i][1][1])
        train_data = [
            data_entry
            for data_entry in data
            if data_entry["cv"] in train_idxs and data_entry["sentiment"] == clx
        ]
        train += train_data

        test_data = [
            data_entry
            for data_entry in data
            if data_entry["cv"] in test_idxs and data_entry["sentiment"] == clx
        ]
        test += test_data
    return train, test


def perform_rr_cv(
    data,
    n_splits,
    modulo,
    alpha=0,
    data_len: tg.Optional[int] = None,
    stem: bool = False,
):
    """
    Performs round robin cross validation
    """
    if data_len is None:
        data_len = len(data)

    base_split: npt.NDArray = np.arange(0, (data_len - n_splits) + 1, modulo)
    splits: tg.List[npt.NDArray] = [base_split + i for i in range(n_splits)]

    metrics = np.zeros(n_splits, dtype=float)
    for i, test_data_idxs in enumerate(splits):
        print(f"Cross validating on split {i+1} of {n_splits}")
        train_data_idxs = np.concatenate(splits[:i] + splits[(i + 1) :])  # noqa:E203

        train_data = [
            data_entry for data_entry in data if data_entry["cv"] in train_data_idxs
        ]
        test_data = [
            data_entry for data_entry in data if data_entry["cv"] in test_data_idxs
        ]

        vocab, logprior, loglikelihood = train_nb(
            ("POS", "NEG"), train_data, alpha, stem
        )
        y_pred = np.array(
            [
                nb_predict(("POS", "NEG"), vocab, logprior, loglikelihood, doc, stem)
                for doc in test_data
            ]
        )
        y_true = np.array([SENT_MAP[doc["sentiment"]] for doc in test_data])

        accuracy = (y_pred == y_true).astype(int).sum() / len(y_true)

        metrics[i] = accuracy

    print("Cross validation complete.")

    return metrics


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Naive Bayes Sentiment Analysis")
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        help="Path to the training data",
        default="reviews.json",
    )
    parser.add_argument(
        "--alpha", "-a", type=int, help="Value to use for Smoothing", default=0
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
        "-s", "--stem", action="store_true", default=False, help="Whether to stem words"
    )
    args = parser.parse_args()
    with open(args.data, mode="r", encoding="utf-8") as f:
        reviews = json.load(f)

    if args.cross_validate:
        accuracies = perform_rr_cv(reviews, 10, 10, args.alpha, 1000, args.stem)

        print(accuracies.mean())
        print(accuracies.var())
    else:
        train_reviews, test_data = split_data(
            reviews,
            (args.pos_idxs[:2], args.pos_idxs[2:]),
            (args.neg_idxs[:2], args.neg_idxs[2:]),
        )

        vocab, logprior, loglikelihood = train_nb(
            ("POS", "NEG"), train_reviews, args.alpha, args.stem
        )
        print(f"vocab size:{len(vocab.keys())}")
        y_pred = np.array(
            [
                nb_predict(
                    ("POS", "NEG"), vocab, logprior, loglikelihood, doc, args.stem
                )
                for doc in test_data
            ]
        )
        y_true = np.array([SENT_MAP[doc["sentiment"]] for doc in test_data])
        print((y_pred == y_true).astype(int).sum() / len(y_true))
