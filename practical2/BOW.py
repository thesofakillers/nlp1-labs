import torch
from torch import nn


class BOW(nn.Module):
  """A simple bag-of-words model"""

  def __init__(self, vocab_size, embedding_dim, vocab):
    super(BOW, self).__init__()
    self.vocab = vocab
    
    # this is a trainable look-up table with word embeddings
    self.embed = nn.Embedding(vocab_size, embedding_dim)
    
    # this is a trainable bias term
    self.bias = nn.Parameter(torch.zeros(embedding_dim), requires_grad=True)

  def forward(self, inputs):
    # this is the forward pass of the neural network
    # it applies a function to the input and returns the output

    # this looks up the embeddings for each word ID in inputs
    # the result is a sequence of word embeddings
    embeds = self.embed(inputs)
    
    # the output is the sum across the time dimension (1)
    # with the bias term added
    logits = embeds.sum(1) + self.bias

    return logits

  def reset(self, keep_embed=False):
    if not keep_embed:
      self.embed.reset_parameters()
    self.bias.data *= 0


class CBOW(nn.Module):
  """A continuous bag-of-words model"""

  def __init__(self, vocab_size, vocab, n_classes=5, embedding_dim=300):
    super(CBOW, self).__init__()
    self.vocab = vocab

    # this is a trainable look-up table with word embeddings
    self.embed = nn.Embedding(vocab_size, embedding_dim)

    self.linear = nn.Linear(embedding_dim, n_classes)

  def forward(self, inputs):
    # this is the forward pass of the neural network
    # it applies a function to the input and returns the output

    # this looks up the embeddings for each word ID in inputs
    # the result is a sequence of word embeddings
    embeds = self.embed(inputs)

    # the output is the sum across the time dimension (1)
    # fed through a linear layer
    logits = self.linear(embeds.sum(1))

    return logits

  def reset(self, keep_embed=False):
    if not keep_embed:
      self.embed.reset_parameters()
    self.linear.reset_parameters()


class DeepCBOW(nn.Module):
  """A continuous bag-of-words model"""

  def __init__(self, vocab_size, vocab, n_classes=5, dims=[300, 100, 100]):
    super(DeepCBOW, self).__init__()
    self.vocab = vocab
    
    # this is a trainable look-up table with word embeddings
    self.embed = nn.Embedding(vocab_size, dims[0])
    
    modules = []
    previous_size = dims[0]
    for size in dims[1:]:
      modules.extend([
        nn.Linear(previous_size, size),
        nn.Tanh()
      ])
      previous_size = size
    modules.append(nn.Linear(previous_size, n_classes))
    
    self.sequence = nn.Sequential(*modules)

  def forward(self, inputs):
    # this is the forward pass of the neural network
    # it applies a function to the input and returns the output

    # this looks up the embeddings for each word ID in inputs
    # the result is a sequence of word embeddings
    embeds = self.embed(inputs)

    # the output is the sum across the time dimension (1)
    # fed through a sequence of linear layers and activations
    logits = self.sequence(embeds.sum(1))

    return logits

  def reset(self, keep_embed=False):
    if not keep_embed:
      self.embed.reset_parameters()
    for module in self.sequence:
      if hasattr(module, 'reset_parameters'):
        module.reset_parameters()


# We define a new class that inherits from DeepCBOW.
class PTDeepCBOW(DeepCBOW):
  def __init__(self, vocab_size, vocab, output_dim, embedding_dim=300, hidden_dim=100):
    super(PTDeepCBOW, self).__init__(
        vocab_size, vocab, output_dim, [embedding_dim, hidden_dim, hidden_dim])