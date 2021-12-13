"""miscellaneous functionality"""
import typing as tg
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import structs
import metrics
from copy import deepcopy


def train_model(
    model: nn.Module,
    dataset: tg.Dict,
    epochs: int,
    lr: float,
    batch_fn: tg.Callable,
    batch_size: int,
    prep_fn: tg.Callable,
    eval_fn: tg.Callable,
    device: tg.Optional[torch.device] = None,
    ckpt_path: tg.Optional[str] = None,
):
    """Train a model."""

    # infer device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initializations
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()  # loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # optimizer
    best_accuracy = 0
    logging_info = {
        "loss": {"dev": np.zeros(epochs), "train": np.zeros(epochs)},
        "accuracy": {"dev": np.zeros(epochs), "train": np.zeros(epochs)},
    }
    start_epoch = 0
    # simulate a torch dataloader
    data_loader = {
        key: list(batch_fn(dataset[key], batch_size)) for key in ("train", "dev")
    }
    # parse checkpoint if provided
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(torch.load(ckpt["state_dict"]))
        optimizer.load_state_dict(torch.load(ckpt["optimizer_state_dict"]))
        best_accuracy = ckpt["best_accuracy"]
        start_epoch = ckpt["best_epoch"]
        logging_info = ckpt["logging_info"]
    else:
        ckpt_path = f"{model.__class__.__name__}.pth"
    for epoch in range(start_epoch, epochs):
        for phase in ["train", "dev"]:
            n_batches = len(data_loader[phase])
            if phase == "train":
                model.train()
            else:
                model.eval()
            with tqdm(data_loader[phase], unit="batch") as curr_epoch:
                for batch in curr_epoch:
                    # parse batch, takes care of device placement
                    features_X, y_true = prep_fn(batch, model.vocab, device)

                    y_pred = model(features_X)
                    loss = criterion(y_pred, y_true)

                    optimizer.zero_grad()

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    logging_info["loss"][phase][epoch] += loss.item() / n_batches
                    logging_info["accuracy"][phase][epoch] += (
                        metrics.compute_accuracy(y_pred, y_true) / n_batches
                    )
            if phase == "dev":
                if logging_info["accuracy"][phase][epoch] > best_accuracy:
                    print(
                        f"New best accuracy: {logging_info['accuracy'][phase][epoch]:0.3f}"
                    )
                    best_accuracy = logging_info["accuracy"]["validation"][epoch]
                    best_model = deepcopy(model)
                    ckpt = {
                        "logs": logging_info,
                        "state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_epoch": epoch,
                        "best_accuracy": best_accuracy,
                    }
                    torch.save(ckpt, ckpt_path)
    logging_info["accuracy"]["test"] = eval_fn(best_model, dataset["test"])
    return logging_info, best_model


def set_determinism(seed: int):
    """
    Sets the random seed for all random number generators
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def print_parameters(model: nn.Module):
    """prints all parameters of a pytorch model"""
    total = 0
    for name, p in model.named_parameters():
        total += np.prod(p.shape)
        print(
            "{:24s} {:12s} requires_grad={}".format(
                name, str(list(p.shape)), p.requires_grad
            )
        )
    print("\nTotal number of parameters: {}\n".format(total))


def batch_dataset(data, batch_size, drop_last=False, shuffle=True):
    """Return batches, optional shuffling"""

    if shuffle:
        print("Shuffling training data")
        np.random.shuffle(data)  # shuffle training data each epoch

    batch = []

    # yield batches
    for example in data:
        batch.append(example)

        if len(batch) == batch_size:
            yield batch
            batch = []

    # in case there is something left and we don't want to drop it
    if not drop_last:
        if len(batch) > 0:
            yield batch


def pad(tokens, length, pad_value=1):
    """add padding 1s to a sequence to that it has the desired length"""
    return tokens + [pad_value] * (length - len(tokens))


def prepare_batch(
    batch: tg.List[structs.Example],
    vocab: structs.Vocabulary,
    device: tg.Optional[torch.device] = None,
):
    """
    batch is a list of examples.
    This function converts words to IDs and returns
    torch tensors to be used as input/targets.
    """
    # infer device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # pad batch examples so to match length of longest example
    maxlen = max([len(ex.tokens) for ex in batch])
    x = [pad([vocab.w2i.get(t, 0) for t in ex.tokens], maxlen) for ex in batch]

    x = torch.LongTensor(x)
    x = x.to(device)

    y = [ex.label for ex in batch]
    y = torch.LongTensor(y)
    y = y.to(device)

    return x, y
