"""Review Sentiment Analysis using Naive Bayes"""

import typing as tg
import argparse
import json
import numpy as np
import numpy.typing as npt

SENT_MAP = {
    "POS": 0,
    "NEG": 1,
}


def extract_vocab(documents: tg.List[tg.Dict]):
    """
    Extracts the vocabulary from the documents,

    Parameters
    ----------
    documents : tg.List[Dict]

    Returns
    -------
    dict
        The vocabulary
    """
    vocab = {}
    for doc in documents:
        for sentence in doc["content"]:
            for word, _pos in sentence:
                word = word.lower()
                if word not in vocab:
                    vocab[word] = {"POS": 0, "NEG": 0}
                if doc["sentiment"] == "POS":
                    vocab[word]["POS"] += 1
                elif doc["sentiment"] == "NEG":
                    vocab[word]["NEG"] += 1
    return vocab


def train_nb(
    classes: tg.Tuple[str, ...], documents: tg.List[tg.Dict], alpha: float = 0
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

    Returns
    -------
    vocab : tg.Dict
        the computed vocabulary for the input documents
    logprior : npt.NDArray
        (n_classes, ) array of log priors
    loglikelihood : tg.List[tg.Dict]
        (n_classes, ) array of dictionary containing loglikelihood of each word
    """
    vocab: tg.Dict = extract_vocab(documents)
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
):
    """
    Predicts the sentiment of a document

    Parameters
    ----------
    classes : tg.Tuple[str, ...]
    vocab : tg.Dict
    prior : npt.NDArray
    nb_model : tg.Dict
    doc : tg.Dict

    Returns
    -------
    int
        The index of the predicted class
    """
    doc_text = [
        word.lower()
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
):
    """
    Splits data into training and testing sets,
    allowing for different starts and stops between
    negative and positive classes
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
        "-pi", "--pos-idxs", nargs=4, type=int, default=[0, 900, 900, 1000]
    )
    parser.add_argument(
        "-ni", "--neg-idxs", nargs=4, type=int, default=[0, 900, 900, 1000]
    )
    args = parser.parse_args()
    with open(args.data, mode="r", encoding="utf-8") as f:
        reviews = json.load(f)

    train_reviews, test_reviews = split_data(
        reviews,
        (args.pos_idxs[:2], args.pos_idxs[2:]),
        (args.neg_idxs[:2], args.neg_idxs[2:]),
    )

    vocab, logprior, loglikelihood = train_nb(("POS", "NEG"), train_reviews, args.alpha)
    y_pred = np.array(
        [
            nb_predict(("POS", "NEG"), vocab, logprior, loglikelihood, doc)
            for doc in test_reviews
        ]
    )
    y_true = np.array([SENT_MAP[doc["sentiment"]] for doc in test_reviews])
    print((y_pred == y_true).astype(int).sum() / len(y_true))
