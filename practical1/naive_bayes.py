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
    classes: tg.Tuple[str, ...], documents: tg.List[tg.Dict], alpha: float = 1
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
    vocab: tg.Set,
    logprior: npt.NDArray[float],
    loglikelihood: tg.Dict,
    doc: tg.Dict,
):
    """
    Predicts the sentiment of a document

    Parameters
    ----------
    classes : tg.Tuple[str, ...]
    vocab : tg.Set
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
    args = parser.parse_args()
    with open(args.data, mode="r", encoding="utf-8") as f:
        reviews = json.load(f)

    train_idxs = range(0, 900)
    test_idxs = range(900, 1000)
    train_reviews = [
        review for review in reviews for i in train_idxs if review["cv"] == i
    ]
    test_reviews = [
        review for review in reviews for i in test_idxs if review["cv"] == i
    ]

    vocab, prior, nb_model = train_nb(("POS", "NEG"), train_reviews, args.alpha)
    y_pred = np.array(
        [
            nb_predict(("POS", "NEG"), vocab, prior, nb_model, doc)
            for doc in test_reviews
        ]
    )
    y_true = np.array([SENT_MAP[doc["sentiment"]] for doc in test_reviews])
    print((y_pred == y_true).astype(int).sum() / len(y_true))
