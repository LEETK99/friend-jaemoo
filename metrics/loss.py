import torch

from extensions.chamfer_distance.chamfer_distance import ChamferDistance
from extensions.earth_movers_distance.emd import EarthMoverDistance


CD = ChamferDistance()
EMD = EarthMoverDistance()


def cd_loss_L1(pcs1, pcs2):
    """
    L1 Chamfer Distance.

    Args:
        pcs1 (torch.tensor): (B, N, 3)
        pcs2 (torch.tensor): (B, M, 3)
    """
    dist1, dist2 = CD(pcs1, pcs2)
    dist1 = torch.sqrt(dist1)
    dist2 = torch.sqrt(dist2)
    return (torch.mean(dist1) + torch.mean(dist2)) / 2.0

def infocd_loss(pcs1, pcs2):
    """
    Args:
        pcs1 : (B, N, 3)
        pcs2 : (B, M, 3)
    """
    dist1, dist2 = CD(pcs1, pcs2)
    d1 = torch.sqrt(dist1)
    d2 = torch.sqrt(dist2)

    distances1 = -torch.log(torch.exp(-0.5 * d1)/(torch.sum(torch.exp(-0.5 * d1) + 1e-7,dim=-1).unsqueeze(-1))**1e-7)

    #print(torch.mean(distances1))
    distances2 = -torch.log(torch.exp(-0.5 * d2)/(torch.sum(torch.exp(-0.5 * d2) + 1e-7,dim=-1).unsqueeze(-1))**1e-7)

    return (torch.mean(distances1) + torch.mean(distances2)) / 2

    #return (distances1 + distances2) / 2


def cd_loss_L2(pcs1, pcs2):
    """
    L2 Chamfer Distance.

    Args:
        pcs1 (torch.tensor): (B, N, 3)
        pcs2 (torch.tensor): (B, M, 3)
    """
    dist1, dist2 = CD(pcs1, pcs2)
    return torch.mean(dist1) + torch.mean(dist2)


def emd_loss(pcs1, pcs2):
    """
    EMD Loss.

    Args:
        xyz1 (torch.Tensor): (b, N, 3)
        xyz2 (torch.Tensor): (b, N, 3)
    """
    dists = EMD(pcs1, pcs2)
    return torch.mean(dists)
