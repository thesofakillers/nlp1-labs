"""Review Sentiment Analysis using Naive Bayes"""

import typing as tg
import argparse
import json
import numpy as np
import numpy.typing as npt
from utils import extract_vocab, preprocess_reviews, split_data, SENT_MAP, rr_cv_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


def train_nb(
    classes: tg.Tuple[str, ...],
    documents: tg.List[tg.Dict],
    alpha: float = 0,
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
    loglikelihood : tg.List[tg.Dict]
    doc : tg.Dict
        the document to classify

    Returns
    -------
    int
        The index of the predicted class
    """
    doc_text = [
        token
        for sentence in doc["content"]
        for token, _pos in sentence
        if token in vocab
    ]
    score: npt.NDArray = np.zeros(len(classes), dtype=float)
    for c, clx in enumerate(classes):
        score[c] = logprior[c]
        for token in doc_text:
            if token in vocab:
                score[c] += loglikelihood[c][token]

    return np.argmax(score)


def perform_rr_cv(
    data,
    n_splits,
    modulo,
    alpha=0,
    data_len: tg.Optional[int] = None,
):
    """
    Performs round robin cross validation

    Returns
    -------
    metrics : npt.NDArray
        (2, n_splits) array of accuracies, precisions, recalls and vocab sizes
    """
    if data_len is None:
        data_len = len(data)

    train, test = rr_cv_split(data_len, n_splits, modulo)

    metrics = np.zeros((4, n_splits), dtype=float)

    for i, (train_idxs, test_idxs) in enumerate(zip(train, test)):
        print(f"Cross validating on split {i+1} of {n_splits}")

        train_data = [
            data_entry for data_entry in data if data_entry["cv"] in train_idxs
        ]
        test_data = [data_entry for data_entry in data if data_entry["cv"] in test_idxs]

        metrics[:, i] = train_eval_nb(train_data, test_data, alpha)

    print("Cross validation complete.")

    return metrics


def train_eval_nb(train_data, test_data, alpha):
    vocab, logprior, loglikelihood = train_nb(("POS", "NEG"), train_data, alpha)
    y_pred = np.array(
        [
            nb_predict(("POS", "NEG"), vocab, logprior, loglikelihood, doc)
            for doc in test_data
        ]
    )
    y_true = np.array([SENT_MAP[doc["sentiment"]] for doc in test_data])

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    vocab_size = len(vocab.keys())

    return accuracy, precision, recall, vocab_size


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
    parser.add_argument(
        "-b",
        "--bigrams",
        action="store_true",
        default=False,
        help="Whether to use bigrams as well as unigrams",
    )
    parser.add_argument(
        "-t",
        "--trigrams",
        action="store_true",
        default=False,
        help="Whether to use trigrams and bigrams as well as unigrams",
    )
    args = parser.parse_args()
    with open(args.data, mode="r", encoding="utf-8") as f:
        reviews = json.load(f)

    reviews = preprocess_reviews(
        reviews, stem=args.stem, bigrams=args.bigrams, trigrams=args.trigrams
    )

    if args.cross_validate:
        accuracies, precisions, recalls, vocab_sizes = perform_rr_cv(
            reviews, 10, 10, args.alpha, 1000
        )
        print("---")
        print(f"CV accuracies: {accuracies}")
        print(f"mean CV accuracy: {accuracies.mean()}")
        print(f"CV accuracy variance: {accuracies.var()}")
        print("---")
        print(f"CV precisions: {precisions}")
        print(f"mean CV precision: {precisions.mean()}")
        print(f"CV precision variance: {precisions.var()}")
        print("---")
        print(f"CV recalls: {recalls}")
        print(f"mean CV recall: {recalls.mean()}")
        print(f"CV recall variance: {recalls.var()}")
        print("---")
        print(f"CV vocab sizes: {vocab_sizes}")
        print(f"mean CV vocab size: {vocab_sizes.mean()}")
        print(f"CV vocab size variance: {vocab_sizes.var()}")
    else:
        train_reviews, test_reviews = split_data(
            reviews,
            (args.pos_idxs[:2], args.pos_idxs[2:]),
            (args.neg_idxs[:2], args.neg_idxs[2:]),
        )

        accuracy, precision, recall, vocab_size = train_eval_nb(
            train_reviews, test_reviews, args.alpha
        )
        print(f"accuracy: {accuracy}")
        print(f"precision: {precision}")
        print(f"recall: {recall}")
        print(f"vocab size: {vocab_size}")
