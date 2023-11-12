import torch


def meshgrid_xy(tensor1: torch.Tensor, tensor2: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """Mimick np.meshgrid(..., indexing="xy") in pytorch. torch.meshgrid only allows "ij" indexing.
    (If you're unsure what this means, safely skip trying to understand this, and run a tiny example!)

    Args:
      tensor1 (torch.Tensor): Tensor whose elements define the first dimension of the returned meshgrid.
      tensor2 (torch.Tensor): Tensor whose elements define the second dimension of the returned meshgrid.
    """
    # TESTED
    ii, jj = torch.meshgrid(tensor1, tensor2)
    return ii.transpose(-1, -2), jj.transpose(-1, -2)


def meshgrid_xy(tensor1: torch.Tensor, tensor2: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    return torch.meshgrid(tensor1, tensor2, indexing='xy')


h = torch.arange(4)
w = torch.arange(5)

a_i, a_j = meshgrid_xy(h, w)
b_i, b_j = meshgrid_xy_new(h, w)

print(a_j)
print(b_j)

print(torch.prod(torch.eq(a_i, b_i)))
print(torch.prod(torch.eq(a_j, b_j)))
