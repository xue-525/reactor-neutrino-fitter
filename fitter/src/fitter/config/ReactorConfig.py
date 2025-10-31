import torch

"""presumed reactor power and distances"""
reactor_power_GW = torch.tensor([100, 100, 100, 100, 100, 100, 100, 100, 100])
L = torch.tensor(
    [
        150 * 1000,
        150 * 1000,
        150 * 1000,
        150 * 1000,
        150 * 1000,
        150 * 1000,
        150 * 1000,
        150 * 1000,
        150 * 1000,
    ]
)  # in m
