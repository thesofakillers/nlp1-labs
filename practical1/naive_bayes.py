"""Review Sentiment Analysis using Naive Bayes"""

import typing as tg

# from collections import Counter
import json
import numpy as np
import numpy.typing as npt

SENT_MAP = {
    "POS": 0,
    "NEG": 1,
}


def extract_vocab(documents: tg.List[tg.Dict]):
    """
    Extracts the vocabulary from the documents

    Parameters
    ----------
    documents : tg.List[Dict]

    Returns
    -------
    set
        The vocabulary
    """
    vocab = set()
    for doc in documents:
        for sentence in doc["content"]:
            for word, _pos in sentence:
                vocab.add(word.lower())
    return vocab


# def extract_vocab(documents: tg.List[tg.Dict]):
#     """
#     Extracts the vocabulary from the documents
#     i.e. the count of each word across all documents

#     Parameters
#     ----------
#     documents : tg.List[Dict]
#         tg.List of Documents

#     Returns
#     -------
#     Counter
#         The computed vocabulary
#     """
#     vocab: tg.Counter = Counter()
#     for doc in documents:
#         for sentence in doc["content"]:
#             word: str
#             for word, _pos in sentence:
#                 vocab[word.lower()] += int(1)
#     return vocab


def train_nb(classes: tg.Tuple[str, ...], documents: tg.List[tg.Dict]):
    """
    Trains the Naive Bayes model

    Parameters
    ----------
    classes : tg.Tuple[str, ...]
    documents : tg.List[Dict]

    Returns
    -------
    TODO
    not sure
    """
    vocab: tg.Set = extract_vocab(documents)
    n_docs = len(documents)
    prior = np.zeros(len(classes))

    nb_model: tg.Dict = {}
    for c, clx in enumerate(classes):
        class_docs = [doc for doc in documents if SENT_MAP[doc["sentiment"]] == clx]
        n_docs_clx = len(class_docs)
        prior[c] = n_docs_clx / n_docs

        class_text = [
            word.lower()
            for doc in class_docs
            for sentence in doc["content"]
            for word, _pos in sentence
        ]

        for word in vocab:
            if word in class_text:
                nb_model[word][c]["count"] += 1
        word_counts = np.array(
            [nb_model[word][c]["count"] for word in vocab if word in class_text]
        )
        for word in vocab:
            if word in class_text:
                nb_model[word][c]["likelihood"] = (
                    nb_model[word][c]["count"] + 1 / (word_counts + 1).sum()
                )
        # word_counts = np.array(
        #     [class_text.count(word) for word in vocab if word in class_text]
        # )
        # likelihood[:, c] = word_counts + 1 / (word_counts + 1).sum()

    return vocab, prior, nb_model


def nb_predict(
    classes: tg.Tuple[str, ...],
    vocab: tg.Set,
    prior: npt.NDArray[float],
    nb_model: tg.Dict,
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
        score[c] = np.log(prior[c])
        for word in doc_text:
            if word in nb_model:
                score[c] += np.log(nb_model[word][c]["likelihood"])

    return np.argmax(score)


if __name__ == "__main__":

    with open("reviews.json", mode="r", encoding="utf-8") as f:
        reviews = json.load(f)

    train_idxs = range(0, 900)
    test_idxs = range(900, 1000)
    train_reviews = [reviews[i] for i in train_idxs]
    test_reviews = [reviews[i] for i in test_idxs]

    vocab, prior, nb_model = train_nb(("POS", "NEG"), train_reviews)
    y_pred = np.array(
        [
            nb_predict(("POS", "NEG"), vocab, prior, nb_model, doc)
            for doc in test_reviews
        ]
    )
    y_true = np.array([SENT_MAP[doc["sentiment"]] for doc in test_reviews])
    print((y_pred == y_true).astype(int).sum() / len(y_true))
