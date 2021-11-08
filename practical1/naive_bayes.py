"""Review Sentiment Analysis using Naive Bayes"""

import sys
import typing as tg
import argparse
import json
import numpy as np
import numpy.typing as npt
from nltk.stem.porter import PorterStemmer
from nltk.util import ngrams

SENT_MAP = {
    "POS": 0,
    "NEG": 1,
}


def preprocess_reviews(
    reviews: tg.List[tg.Dict],
    stem: bool = False,
    bigrams: bool = False,
    trigrams: bool = False,
    cached: bool = False,
) -> tg.List[tg.Dict]:
    """
    Preprocesses the reviews with lower-casing or stemming
    and/or bigram and trigram calculation

    Parameters
    ----------
    reviews : tg.List[Dict]
        the reviews to preprocess
    stem : bool, default False
        Whether to stem the reviews, if False, simply lower-cases
    bigram : bool, default False
        Whether to calculate bigrams
    trigrams : bool, default False
        Whether to calculate bigrams _and_ trigrams
    cached : bool, default False
        Whether the passed `reviews` are already lower-cased/stemmed

    Returns
    -------
    tg.List[Dict]
        the preprocessed reviews
    """
    if not cached:
        if stem:
            print("Stemming requested; stemming...")
            stemmer = PorterStemmer()
            reviews = [
                {
                    "cv": review["cv"],
                    "sentiment": review["sentiment"],
                    "content": [
                        [(stemmer.stem(word), pos) for word, pos in sentence]
                        for sentence in review["content"]
                    ],
                }
                for review in reviews
            ]
            print("Stemming complete.")
        else:
            reviews = [
                {
                    "cv": review["cv"],
                    "sentiment": review["sentiment"],
                    "content": [
                        [(word.lower(), pos) for word, pos in sentence]
                        for sentence in review["content"]
                    ],
                }
                for review in reviews
            ]
    if bigrams or trigrams:
        for review in reviews:
            review_words = [
                word for sentence in review["content"] for word, _pos in sentence
            ]
            print("Computing bigrams...")
            review_bigrams = list(ngrams(review_words, 2))
            bigram_pos = ["bigram" for bigram in review_bigrams]
            review["content"].append(list(zip(review_bigrams, bigram_pos)))
            if trigrams:
                print("Computing trigrams...")
                review_trigrams = list(ngrams(review_words, 3))
                trigram_pos = ["trigram" for trigram in review_trigrams]
                review["content"].append(list(zip(review_trigrams, trigram_pos)))
                print("Done.")
    return reviews


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
            for token, _pos in sentence:
                if token not in vocab:
                    vocab[token] = {"POS": 0, "NEG": 0}
                if doc["sentiment"] == "POS":
                    vocab[token]["POS"] += 1
                elif doc["sentiment"] == "NEG":
                    vocab[token]["NEG"] += 1
    return vocab


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
):
    """
    Performs round robin cross validation

    Returns
    -------
    metrics : npt.NDArray
        (2, n_splits) array of accuracies and vocab sizes
    """
    if data_len is None:
        data_len = len(data)

    base_split: npt.NDArray = np.arange(0, (data_len - n_splits) + 1, modulo)
    splits: tg.List[npt.NDArray] = [base_split + i for i in range(n_splits)]

    metrics = np.zeros((2, n_splits), dtype=float)
    for i, test_data_idxs in enumerate(splits):
        print(f"Cross validating on split {i+1} of {n_splits}")
        train_data_idxs = np.concatenate(splits[:i] + splits[(i + 1) :])  # noqa:E203

        train_data = [
            data_entry for data_entry in data if data_entry["cv"] in train_data_idxs
        ]
        test_data = [
            data_entry for data_entry in data if data_entry["cv"] in test_data_idxs
        ]

        vocab, logprior, loglikelihood = train_nb(("POS", "NEG"), train_data, alpha)
        y_pred = np.array(
            [
                nb_predict(("POS", "NEG"), vocab, logprior, loglikelihood, doc)
                for doc in test_data
            ]
        )
        y_true = np.array([SENT_MAP[doc["sentiment"]] for doc in test_data])

        accuracy = (y_pred == y_true).astype(int).sum() / len(y_true)

        metrics[0, i] = accuracy
        metrics[1, i] = len(vocab.keys())

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
        accuracies, vocab_sizes = perform_rr_cv(reviews, 10, 10, args.alpha, 1000)
        print("---")
        print(f"CV accuracies: {accuracies}")
        print(f"mean CV accuracy: {accuracies.mean()}")
        print(f"CV accuracy variance: {accuracies.var()}")
        print("---")
        print(f"CV vocab sizes: {vocab_sizes}")
        print(f"mean CV vocab size: {vocab_sizes.mean()}")
        print(f"CV vocab size variance: {vocab_sizes.var()}")
    else:
        train_reviews, test_data = split_data(
            reviews,
            (args.pos_idxs[:2], args.pos_idxs[2:]),
            (args.neg_idxs[:2], args.neg_idxs[2:]),
        )

        vocab, logprior, loglikelihood = train_nb(
            ("POS", "NEG"), train_reviews, args.alpha
        )
        print(f"vocab size:{len(vocab.keys())}")
        y_pred = np.array(
            [
                nb_predict(("POS", "NEG"), vocab, logprior, loglikelihood, doc)
                for doc in test_data
            ]
        )
        y_true = np.array([SENT_MAP[doc["sentiment"]] for doc in test_data])
        accuracy = (y_pred == y_true).astype(int).sum() / len(y_true)
        print(f"accuracy: {accuracy}")
