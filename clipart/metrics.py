from torch.nn import functional as F
def tv_loss(input):
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    return ((input[..., :-1, 1:] - input[..., :-1, :-1])**2 + (input[..., 1:, :-1] - input[..., :-1, :-1])**2).mean()
def sph_dist(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    l = (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2).mean()
    return l 