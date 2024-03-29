"""Utils for Sentiment Analysis"""
import copy
import typing as tg
import numpy as np
from nltk.util import ngrams
from nltk.stem import PorterStemmer


POS_MAP = {
    "JJ": "adj",
    "JJR": "adj",
    "JJS": "adj",
    "RB": "adverb",
    "RBR": "adverb",
    "RBS": "adverb",
    "NN": "noun",
    "NNS": "noun",
    "NNP": "noun",
    "NNPS": "noun",
    "VB": "verb",
    "VBD": "verb",
    "VBG": "verb",
    "VBN": "verb",
    "VBP": "verb",
    "VBZ": "verb",
    "WRB": "adverb",
}

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
            new_reviews = [
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
            new_reviews = [
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
    else:
        new_reviews = copy.deepcopy(reviews)
    if bigrams or trigrams:
        print("Computing ngrams...")
        for review in new_reviews:
            review_words = [
                word for sentence in review["content"] for word, _pos in sentence
            ]
            review_bigrams = list(ngrams(review_words, 2))
            bigram_pos = ["bigram" for bigram in review_bigrams]
            review["content"].append(list(zip(review_bigrams, bigram_pos)))
            if trigrams:
                review_trigrams = list(ngrams(review_words, 3))
                trigram_pos = ["trigram" for trigram in review_trigrams]
                review["content"].append(list(zip(review_trigrams, trigram_pos)))
        print("Done.")
    return new_reviews


def extract_vocab(
    documents: tg.List[tg.Dict], use_pos: bool = False, only_open: bool = False
):
    """
    Extracts the vocabulary from the documents,

    Parameters
    ----------
    documents : tg.List[Dict]
    use_pos : bool, default False
        Whether to use the word+POS as vocab keys
    only_open : bool, default False
        Whether to only include open-class POS-tagged words,
        ignored if `use_pos` is False

    Returns
    -------
    dict
        The vocabulary
    """
    vocab = {}
    for doc in documents:
        for sentence in doc["content"]:
            for token, pos in sentence:
                if use_pos:
                    key = (token, pos)
                    if only_open:
                        if pos not in POS_MAP:
                            # skip this token
                            continue
                else:
                    key = token
                if key not in vocab:
                    vocab[key] = {"POS": 0, "NEG": 0}
                if doc["sentiment"] == "POS":
                    vocab[key]["POS"] += 1
                elif doc["sentiment"] == "NEG":
                    vocab[key]["NEG"] += 1
    return vocab


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


def rr_cv_split(
    data_len: int,
    n_splits: int,
    modulo: int,
    alpha: float = 0,
) -> tg.Tuple[np.ndarray, np.ndarray]:
    """
    Performs round robin cross validation split

    Parameters
    ----------
    data_len : int
        the length of the data
    n_splits : int
        the number of splits
    modulo : int
        the modulo to use for round robin splitting
    alpha : float, default 0
        the smoothing parameter, which is added to word counts

    Returns
    -------
    tuple of ndarray
        tuple containing two (n_splits, -1) ndarrays
        the first containing the indices of the training splits
        and the second containing the indices of the test splits
    """

    base_split: np.ndarray = np.arange(0, (data_len - n_splits) + 1, modulo)
    splits: tg.List[np.ndarray] = [base_split + i for i in range(n_splits)]

    train_splits = np.zeros((n_splits, len(base_split) * (n_splits - 1)))
    test_splits = np.zeros((n_splits, len(base_split)))

    for i, test_data_idxs in enumerate(splits):
        train_splits[i] = np.concatenate(splits[:i] + splits[(i + 1) :])  # noqa:E203
        test_splits[i] = test_data_idxs

    return train_splits, test_splits
