import torch


def to_cuda(tensor):
    # turns to cuda
    if torch.cuda.is_available():
        # return tensor.cuda()
        return tensor
    else:
        return tensor
