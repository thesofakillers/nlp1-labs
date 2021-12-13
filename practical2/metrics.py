import typing as tg
import numpy as np
import structs


def compute_accuracy(y_pred, y_true):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Parameters
    ----------
    predictions : torch.Tensor
        2D float array of size [batch_size, n_classes]
    targets : np.ndarray
        1D int array of size [batch_size]. Ground truth labels for
        each sample in the batch

    Returns
    -------
    accuracy : float
        the accuracy of predictions,
        i.e. the average correct predictions over the whole batch
    """
    y_pred = y_pred.argmax(dim=1)
    accuracy = (y_pred == y_true).float().mean()

    return accuracy


def evaluate_model(
    model,
    data: tg.List[structs.Example],
    batch_fn: tg.Callable,
    batch_size: int,
    prep_fn: tg.Callable,
):
    """
    Performs the evaluation of the MLP model on a given dataset.
    """

    model.eval()
    accuracies = []
    for i, batch in enumerate(batch_fn(data, batch_size)):
        data, target = prep_fn(batch)
        predictions = model.forward(data)
        accuracies.append(compute_accuracy(predictions, target))

    avg_accuracy = np.mean(accuracies)

    return avg_accuracy
