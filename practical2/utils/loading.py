import re
from collections import namedtuple
from nltk import Tree
from collections import OrderedDict


# this function reads in a textfile and fixes an issue with "\\"
def filereader(path):
  with open(path, mode="r", encoding="utf-8") as f:
    for line in f:
      yield line.strip().replace("\\","")


# Let's first make a function that extracts the tokens (the leaves).
def tokens_from_treestring(s):
  """extract the tokens from a sentiment tree"""
  return re.sub(r"\([0-9] |\)", "", s).split()


# We will also need the following function, but you can ignore this for now.
# It is explained later on.
def transitions_from_treestring(s):
  s = re.sub("\([0-5] ([^)]+)\)", "0", s)
  s = re.sub("\)", " )", s)
  s = re.sub("\([0-4] ", "", s)
  s = re.sub("\([0-4] ", "", s)
  s = re.sub("\)", "1", s)
  return list(map(int, s.split()))


# A simple way to define a class is using namedtuple.
Example = namedtuple("Example", ["tokens", "tree", "label", "transitions"])

def examplereader(path, lower=False):
  """Returns all examples in a file one by one."""
  for line in filereader(path):
    line = line.lower() if lower else line
    tokens = tokens_from_treestring(line)
    tree = Tree.fromstring(line)  # use NLTK's Tree
    label = int(line[1])
    trans = transitions_from_treestring(line)
    yield Example(tokens=tokens, tree=tree, label=label, transitions=trans)

# Let's load the data into memory.
LOWER = False  # we will keep the original casing
def load_data(data_dir="trees/"):
  train_data = list(examplereader(data_dir+"train.txt", lower=LOWER))
  dev_data = list(examplereader(data_dir+"dev.txt", lower=LOWER))
  test_data = list(examplereader(data_dir+"test.txt", lower=LOWER))

  return train_data, dev_data, test_data


def load_sentiment_labels():
  # Now let's map the sentiment labels 0-4 to a more readable form
  i2t = ["very negative", "negative", "neutral", "positive", "very positive"]

  # And let's also create the opposite mapping.
  # We won't use a Vocabulary for this (although we could), since the labels
  # are already numeric.
  t2i = OrderedDict({p : i for p, i in zip(i2t, range(len(i2t)))})

  return i2t, t2i


TRAIN_DATA, DEV_DATA, TEST_DATA = load_data()


def get_subtrees_dataset(dataset):
    for example in dataset:
        subtrees = example.tree.subtrees()
        for subtree in subtrees:
            tree_string = " ".join(str(subtree).split())
            tokens = tokens_from_treestring(tree_string)
            label = int(tree_string[1])
            trans = transitions_from_treestring(tree_string)
            yield Example(
                tokens=tokens, tree=subtree, label=label, transitions=trans
            )