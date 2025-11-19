import numpy as np
import torch


def dirichlet_noise(origin_p, alpha=0.03, epison=0.3):
    if not isinstance(origin_p, list) and not isinstance(origin_p, np.ndarray) and not isinstance(origin_p,
                                                                                                  torch.Tensor):
        raise TypeError("origin_p must be list, np.ndarray or torch.tensor")
    if isinstance(origin_p, torch.Tensor):
        origin_p = origin_p.cpu().numpy()

    if isinstance(origin_p, list):
        origin_p = np.array(origin_p)

    if not torch.isclose(torch.tensor(sum(origin_p)).float(), torch.tensor(1.0).float()):
        raise ValueError(f"origin_p must sum to 1, {origin_p} sum to {sum(origin_p)}")

    noise = np.random.dirichlet([alpha] * len(origin_p))
    ret_py = noise * epison + (1 - epison) * origin_p

    if not torch.isclose(torch.tensor(sum(ret_py)).float(), torch.tensor(1.0).float()):
        raise ValueError(f"ret_py must sum to 1, {ret_py} sum to {sum(ret_py)}")
    return noise
