from collections import Counter, OrderedDict, namedtuple


Example = namedtuple("Example", ["tokens", "tree", "label", "transitions"])

i2t = ["very negative", "negative", "neutral", "positive", "very positive"]
t2i = OrderedDict({p: i for p, i in zip(i2t, range(len(i2t)))})


class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first seen"""

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


class Vocabulary:
    """A vocabulary, assigns IDs to tokens"""

    def __init__(self):
        self.freqs = OrderedCounter()
        self.w2i = {}
        self.i2w = []

    def count_token(self, t):
        self.freqs[t] += 1

    def add_token(self, t):
        self.w2i[t] = len(self.w2i)
        self.i2w.append(t)

    def build(self, min_freq=0):
        """
        min_freq: minimum number of occurrences for a word to be included
                  in the vocabulary
        """
        self.add_token("<unk>")  # reserve 0 for <unk> (unknown words)
        self.add_token("<pad>")  # reserve 1 for <pad> (discussed later)

        tok_freq = list(self.freqs.items())
        tok_freq.sort(key=lambda x: x[1], reverse=True)
        for tok, freq in tok_freq:
            if freq >= min_freq:
                self.add_token(tok)
