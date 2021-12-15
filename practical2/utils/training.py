import torch
from torch import nn
import numpy as np
import time
import os
import json
import uuid
import csv
from .mini_batching import get_minibatch, prepare_minibatch
from .evaluation import evaluate
from .helpers import set_seed

from .loading import TRAIN_DATA, DEV_DATA, TEST_DATA


def run_train(seeds, embedding_name, **kwargs):
  id = uuid.uuid4().hex

  save_dict = {
    "model_name": kwargs['model_name'],
    "epochs":     kwargs['epochs'],
    "optimizer":  kwargs['optimizer'].__class__.__name__,
    "lr":         kwargs['optimizer'].param_groups[0]['lr'],
    "batch_size": kwargs['batch_size'],
    "embedding_name": embedding_name,
    "runs": len(seeds),
    "seeds": json.dumps(seeds),
  }

  logging_dicts = []

  for run, seed in enumerate(seeds):
    set_seed(seed)
    kwargs['model'].reset(embedding_name!="None")
    logging_dicts.append(train_custom(**kwargs, id=f"_{id}_{run}"))

  test_accs = np.array([ld['test_acc'] for ld in logging_dicts])

  save_dict.update({
    "test_acc_mean":  test_accs.mean(),
    "test_acc_std":   test_accs.std(),
    "time_for_bests": json.dumps([ld['time_for_best']  for ld in logging_dicts]),
    "best_epochs":    json.dumps([ld['best_epoch']     for ld in logging_dicts]),
    "id": id,
  })

  # Make csv file if it doesn't exist
  csv_path = "log.csv"
  if not os.path.isfile(csv_path):
    with open(csv_path, "w", newline='') as f:
      csv_writer = csv.DictWriter(f, save_dict.keys())
      csv_writer.writeheader()

  # Append new entry to csv file
  with open(csv_path, "a", newline='') as f:
    csv_writer = csv.DictWriter(f, save_dict.keys())
    csv_writer.writerow(save_dict)

  return save_dict, logging_dicts


def train_custom(model, optimizer, epochs=20, 
                batch_fn=get_minibatch, 
                prep_fn=prepare_minibatch,
                eval_fn=evaluate,
                batch_size=128, eval_batch_size=1024,
                train_data=TRAIN_DATA, dev_data=DEV_DATA, test_data=TEST_DATA,
                id='', eval_batch=None, model_name=None):
  """Train a model."""
  device = next(model.parameters()).device
  criterion = nn.CrossEntropyLoss() # loss function

  # Make folder for storing models and logs
  os.makedirs("models/", exist_ok=True)
  os.makedirs("logs/", exist_ok=True)
  if not model_name:
    model_name = model.__class__.__name__

  start = time.time()
  
  # store train loss and validation accuracy during training
  # so we can plot them afterwards
  logging_dict = {
    "id": id[1:],
    "train_losses": [],
    "train_accuracies": [],
    "val_accuracies": [],
    "best_epoch": 0,
    "time_for_best": 0,
  }

  for epoch in range(epochs):
    model.train()

    # when we run out of examples, shuffle and continue
    running_loss = []
    batch_sizes = []
    dataloader = list(batch_fn(train_data, batch_size=batch_size))
    if eval_batch == None:
      eval_batch = len(dataloader)
    for batch_nr, batch in enumerate(dataloader):

      # forward pass
      x, targets = prep_fn(batch, model.vocab, device)
      logits = model(x)

      B = targets.size(0)  # later we will use B examples per update
      
      # compute cross-entropy loss (our criterion)
      # note that the cross entropy loss function computes the softmax for us
      loss = criterion(logits.view([B, -1]), targets.view(-1))
      running_loss.append(loss.item())
      batch_sizes.append(len(targets))

      # erase previous gradients
      optimizer.zero_grad()
      
      # compute gradients
      loss.backward()

      # update weights - take a small step in the opposite dir of the gradient
      optimizer.step()

      if (batch_nr + 1) % eval_batch == 0:
        train_loss = np.average(running_loss, weights=batch_sizes)

        # evaluate
        _, _, train_acc = eval_fn(model, train_data, device, batch_size=eval_batch_size,
                                  batch_fn=batch_fn, prep_fn=prep_fn)     
        _, _, val_acc = eval_fn(model, dev_data, device, batch_size=eval_batch_size,
                                  batch_fn=batch_fn, prep_fn=prep_fn)    
        
        # save best model parameters
        message_best = ''
        if len(logging_dict['val_accuracies']) == 0 or val_acc > max(logging_dict['val_accuracies']):
          message_best = 'NEW BEST'
          logging_dict['best_epoch'] = epoch
          logging_dict['time_for_best'] = time.time()-start
          path = f"models/{model_name}{id}.pt"
          ckpt = {
              "state_dict": model.state_dict(),
              "optimizer_state_dict": optimizer.state_dict(),
              "best_eval": val_acc,
              "best_epoch": logging_dict['best_epoch'],
              "time_for_best": logging_dict['time_for_best'],
          }
          torch.save(ckpt, path)

        logging_dict['train_losses'].append(train_loss)
        logging_dict['train_accuracies'].append(train_acc)
        logging_dict['val_accuracies'].append(val_acc)
        print(f"\nEPOCH: {epoch}")
        print(f"train loss:       {train_loss:.4f}")
        print(f"train acc:        {train_acc:.4f}")   
        print(f"dev acc:          {val_acc:.4f} {message_best}")   
        print(f"time:   {time.time()-start:.2f}s")

  # done training
  print("Done training")
  
  # evaluate on train, dev, and test with best model
  print("Loading best model")
  path = f"models/{model_name}{id}.pt"        
  ckpt = torch.load(path)
  model.load_state_dict(ckpt["state_dict"])
  
  _, _, logging_dict['train_acc'] = eval_fn(
      model, train_data, device, batch_size=eval_batch_size, 
      batch_fn=batch_fn, prep_fn=prep_fn)
  _, _, logging_dict['dev_acc'] = eval_fn(
      model, dev_data, device, batch_size=eval_batch_size,
      batch_fn=batch_fn, prep_fn=prep_fn)
  _, _, logging_dict['test_acc'] = eval_fn(
      model, test_data, device, batch_size=eval_batch_size, 
      batch_fn=batch_fn, prep_fn=prep_fn)
  
  print("best model epoch {:d}: "
        "train acc={:.4f}, dev acc={:.4f}, test acc={:.4f}".format(
            logging_dict['best_epoch'], logging_dict['train_acc'], logging_dict['dev_acc'], logging_dict['test_acc']))

  with open(f"logs/{model_name}{id}.txt", "w") as f:
    f.write(json.dumps(logging_dict, indent=4))
  
  return logging_dict