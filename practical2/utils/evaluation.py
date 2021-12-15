import torch
from .mini_batching import get_minibatch, prepare_minibatch


def evaluate(model, data, device, 
             batch_fn=get_minibatch, prep_fn=prepare_minibatch,
             batch_size=16, shuffle=False, coarse=0):
  """Accuracy of a model on given data set (using mini-batches)"""
  correct = 0
  total = 0
  model.eval()  # disable dropout

  for mb in batch_fn(data, batch_size=batch_size, shuffle=False):
    x, targets = prep_fn(mb, model.vocab, device, shuffle=shuffle)
    with torch.no_grad():
      logits = model(x)
      
    predictions = logits.argmax(dim=-1).view(-1)
    
    # add the number of correct predictions to the total correct
    exact_correct_sum = (predictions == targets.view(-1)).sum().item()
    correct += exact_correct_sum
    pred_diff = (predictions - targets.view(-1)).abs()
    correct += coarse*( (pred_diff <= 1).sum().item() - exact_correct_sum )
    total += targets.size(0)

  return correct, total, correct / float(total)


def evaluate_by_len(model, data, device, 
             batch_fn=get_minibatch, prep_fn=prepare_minibatch,
             batch_size=16):
  """Accuracy of a model on given data set by length of review (using mini-batches)"""
  correct_by_len = None
  total_by_len = None
  padded_length = None
  model.eval()  # disable dropout

  for mb in batch_fn(data, batch_size=batch_size, shuffle=False):
    x, targets = prep_fn(mb, model.vocab, device)
    with torch.no_grad():
      logits = model(x)
      
    predictions = logits.argmax(dim=-1).view(-1)

    sentences = x
    if isinstance(x, tuple): # This is the case for tree-LSTM
      sentences = x[0]
    lengths = (sentences != 1).int().sum(-1) # Sum all non-padding words
    corrects = predictions == targets.view(-1)

    if correct_by_len == None:
      padded_length = sentences.shape[1]
      correct_by_len = torch.zeros((padded_length,))
      total_by_len = torch.zeros((padded_length,))

    for i in range(len(lengths)):
      correct_by_len[lengths[i]-1] += corrects[i].item()
      total_by_len[lengths[i]-1] += 1 
    
  # add the number of correct predictions to the total correct
  correct = correct_by_len.sum().item()
  total = total_by_len.sum().item()

  return correct, total, correct / float(total), correct_by_len, total_by_len


def test_by_length(model, save_dict, data, device, prep_fn=prepare_minibatch):
  correct_by_len = None
  total_by_len = None
  runs = int(save_dict['runs'])
  accs = torch.zeros(runs)
  for run in range(runs):
    model_path = f"models/{save_dict['model_name']}_{save_dict['id']}_{run}.pt"
    model.load_state_dict(torch.load(model_path)['state_dict'])

    _,_,accs[run],cbl, tbl = evaluate_by_len(model, data, device, batch_size=1024,
      prep_fn=prep_fn)
    if correct_by_len == None:
      correct_by_len = cbl
      total_by_len = tbl
    else:
      correct_by_len += cbl
  return correct_by_len, total_by_len, accs.mean(), accs.std()


def test_model(model, save_dict, data, device, prep_fn=prepare_minibatch, shuffle=False, coarse=0):
  runs = int(save_dict['runs'])
  accs = torch.zeros(runs)    
  for run in range(runs):
    model_path = f"models/{save_dict['model_name']}_{save_dict['id']}_{run}.pt"
    model.load_state_dict(torch.load(model_path)['state_dict'])

    _,_,accs[run] = evaluate(model, data, device, batch_size=1024,
      prep_fn=prep_fn, shuffle=shuffle, coarse=coarse)
  return accs.mean(), accs.std()