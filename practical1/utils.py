"""Utils for Sentiment Analysis"""
import typing as tg


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
