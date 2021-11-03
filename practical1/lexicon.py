""" Lexicon-based sentiment analysis. """
from typing import Dict, List
import json
import numpy as np

# anything else --> anypos, using POS_MAP.get(pos, "anypos")
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


def compute_weighted_score(word_list: List[str], lexicon: Dict) -> int:
    """
    Computes the weighted score by summing the strength of each word in the
    word list

    Parameters
    ----------
    word_list : List[str]
        The list of words to be scored.
    lexicon : Dict
        The lexicon to be used for scoring
    Returns
    -------
    int
        The weighted score.
    """
    weighted_pol_arr = np.array(
        [
            lexicon[word]["strength"] * lexicon[word]["polarity"]
            for word in word_list
            if word in lexicon
        ]
    )
    return weighted_pol_arr.sum().astype(int)


def compute_bin_score(word_list: List[str], lexicon: Dict):
    """
    Computes the binary score by counting how many words have a positive or a
    negative label in the sentiment lexicon

    Parameters
    ----------
    word_list : List[str]
        The list of words to be scored.
    lexicon : Dict
        The lexicon to be used for scoring

    Returns
    -------
    int
        The binary score.
    """
    polarity_arr = np.array(
        [lexicon[word]["polarity"] for word in word_list if word in lexicon]
    )
    return polarity_arr.sum().astype(int)


def classify_document(
    document: Dict, lexicon: Dict, mode: str = "binary", threshold: float = 8.0
) -> int:
    """
    Classifies a document using a lexicon approach as either positive or negative.

    Parameters
    ----------
    document : dict
        The document to be classified.
        {"cv": integer, "sentiment": str, "content": list of (word, pos)}
    lexicon : dict
        The lexicon to be used for scoring
    mode : str
        The mode of classification.
        "binary" for binary classification,
        "weighted" for weighted classification
    threshold : float, default 8.0
        The threshold to be used for classification.

    Returns
    -------
    int
        1 if the document is positive, -1 if negative.
    """
    assert mode in [
        "binary",
        "weighted",
    ], "`mode` must be either 'binary' or 'weighted'"
    # flatten document
    data = [word.lower() for sentence in document["content"] for word, _pos in sentence]
    if mode == "binary":
        score_func = compute_bin_score
    else:
        score_func = compute_weighted_score
    score = score_func(data, lexicon)
    # finally, we can classify
    return 1 if score > threshold else -1


def build_lexicon(file_path: str) -> Dict:
    """
    Builds a lexicon from a file.

    Parameters
    -----------
    file_path : str The path to the file.

    Returns
    --------
    A dictionary containing the lexicon data
    """
    lexicon: Dict = {}
    with open(file_path, "r") as f:
        # parse file line by line
        for line in f:
            strength, length, word, pos, stemmed, polarity = line.split()
            # we ignore duplicate words, keep only most recent
            lexicon[word[6:].lower()] = {
                "strength": 2 if strength[5:] == "strongsubj" else 1,
                "length": int(length[4:]),
                "pos": pos[5:],
                "stemmed": stemmed[9:],
                "polarity": 1
                if polarity[14:] == "positive"
                else -1
                if polarity[14:] == "negative"
                else 0,
            }
    return lexicon


if __name__ == "__main__":
    # load data
    lexicon = build_lexicon("sent_lexicon")
    with open("reviews.json", mode="r", encoding="utf-8") as f:
        reviews = json.load(f)
    # classify
    y_pred = np.array(
        [classify_document(review, lexicon, "weighted", 12) for review in reviews]
    )
    y_true = np.array([1 if review["sentiment"] == "POS" else -1 for review in reviews])

    print((y_pred == y_true).astype(int).sum() / len(y_true))
