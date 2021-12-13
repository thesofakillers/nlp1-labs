import re

import nltk

from structs import Example


def tokens_from_treestring(s):
    """extract the tokens from a sentiment tree"""
    return re.sub(r"\([0-9] |\)", "", s).split()


def transitions_from_treestring(s):
    """
    Converts a tree string to a list of transitions
    """
    s = re.sub(r"\([0-5] ([^)]+)\)", "0", s)
    s = re.sub(r"\)", " )", s)
    s = re.sub(r"\([0-4] ", "", s)
    s = re.sub(r"\([0-4] ", "", s)
    s = re.sub(r"\)", "1", s)
    return list(map(int, s.split()))


def filereader(path):
    """
    Reads a file, yielding one line at a time
    """
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            yield line.strip().replace("\\", "")


def examplereader(path: str, lower: bool = False):
    """
    Yields all examples in a file one by one
    courtesy of University of Amsterdam
    """
    for line in filereader(path):
        line = line.lower() if lower else line
        tokens = tokens_from_treestring(line)
        tree = nltk.Tree.fromstring(line)  # use NLTK's Tree
        label = int(line[1])
        trans = transitions_from_treestring(line)
        yield Example(tokens=tokens, tree=tree, label=label, transitions=trans)


def get_SST_data(tree_path: str = "trees/", lower: bool = False):
    """
    Returns a dictionary containing for train, dev and test splits
    of the Stanford Sentiment Treebank
    """
    train_data = list(examplereader(f"{tree_path}train.txt", lower=lower))
    dev_data = list(examplereader(f"{tree_path}dev.txt", lower=lower))
    test_data = list(examplereader(f"{tree_path}test.txt", lower=lower))
    return {"train": train_data, "dev": dev_data, "test": test_data}
