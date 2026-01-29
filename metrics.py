import torch.nn.functional as F

def grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item()**2
    return total_norm ** 0.5

def token_entropy(logits, labels):
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)

    mask = labels != -100
    entropy = entropy[mask]

    return entropy.mean().item()


