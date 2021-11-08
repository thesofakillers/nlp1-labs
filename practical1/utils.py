"""Utils for Sentiment Analysis"""
import copy
import typing as tg
from nltk.utils import ngrams
from nltk.stem import PorterStemmer


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
