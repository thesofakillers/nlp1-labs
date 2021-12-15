import torch
import numpy as np
import random
import json
import csv


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # When running on the CuDNN backend two further options must be set for reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_torch(seed=42, cuda=True):
    print("Using torch", torch.__version__) # should say 1.7.0+cu101

    if cuda:
        # PyTorch can run on CPU or on Nvidia GPU (video card) using CUDA
        # This cell selects the GPU if one is available.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Seed manually to make runs reproducible
    # You need to set this again if you do multiple runs of the same model
    set_seed(seed)

    return device


# Helper functions for batching and unbatching states
# For speed we want to combine computations by batching, but 
# for processing logic we want to turn the output into lists again
# to easily manipulate.
# Used in the Tree LSTM
def batch(states):
  """
  Turns a list of states into a single tensor for fast processing. 
  This function also chunks (splits) each state into a (h, c) pair"""
  return torch.cat(states, 0).chunk(2, 1)

def unbatch(state):
  """
  Turns a tensor back into a list of states.
  First, (h, c) are merged into a single state.
  Then the result is split into a list of sentences.
  """
  return torch.split(torch.cat(state, 1), 1, 0)



def get_saved_results(query):
    save_dict = None
    with open("log.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            found = True
            for key, value in query.items():
                if row[key] != value:
                    found = False
            if found:
                save_dict = row
                # break
    if save_dict == None:
        raise("query not found")

    model_name = save_dict['model_name']
    logging_dicts = []
    for run in range(int(save_dict['runs'])):
        with open(f"logs/{model_name}_{save_dict['id']}_{run}.txt", "r") as f:
            logging_dicts.append(json.loads(f.read()))
    
    return save_dict, logging_dicts


def bin_by_length(correct_by_len_model, total_by_len, amount_of_bins=4):      
    total = 0
    correct = 0
    bin_size = int(total_by_len.sum()/amount_of_bins)+1
    bins = []
    for i, count in enumerate(total_by_len):
        total += count
        correct += correct_by_len_model[i]
        if total > bin_size:
            bins.append(correct/total)
            total = 0
            correct = 0
    bins.append(correct/total)
    return bins, bin_size