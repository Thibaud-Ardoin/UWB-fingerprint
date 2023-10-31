import torch

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def accuracy(target, prediction):
    predicted = torch.argmax(prediction, dim=1)
    target_count = target.size(0)
    correct_val = (target == predicted).sum().item()
    val_acc = 100 * correct_val / target_count
    return val_acc