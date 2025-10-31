import torch


def torch_2d_interp(x, y, xp, yp, fp):
    """
    PyTorch 2D bilinear interpolation implementation

    Args:
    x (torch.Tensor): x coordinates to interpolate
    y (torch.Tensor): y coordinates to interpolate
    xp (torch.Tensor): known grid x coordinates (1D)
    yp (torch.Tensor): known grid y coordinates (1D)
    fp (torch.Tensor): known function values with shape (len(xp), len(yp))

    Returns:
    torch.Tensor: interpolated result
    """
    # Out-of-bounds protection
    out_of_bounds_x = (x < xp[0]) | (x > xp[-1])
    out_of_bounds_y = (y < yp[0]) | (y > yp[-1])
    out_of_bounds = out_of_bounds_x | out_of_bounds_y
    # Handle out-of-bounds: return None
    if torch.any(out_of_bounds):
        print("Warning: Some points are out of bounds!")
        print(f"Number of out of bounds points: {torch.sum(out_of_bounds)}")
        print(f"Out of bounds points: {x[out_of_bounds]}, {y[out_of_bounds]}")
        print(f"Bounds: {xp[0]}, {xp[-1]}, {yp[0]}, {yp[-1]}")
        return None

    # Find insertion indices for x and y
    x_indices = torch.searchsorted(xp, x) - 1
    x_indices = torch.clamp(x_indices, 0, len(xp) - 2)
    y_indices = torch.searchsorted(yp, y) - 1
    y_indices = torch.clamp(y_indices, 0, len(yp) - 2)

    # Get the four nearest grid points and values
    x0 = xp[x_indices]
    x1 = xp[x_indices + 1]
    y0 = yp[y_indices]
    y1 = yp[y_indices + 1]

    f00 = fp[x_indices, y_indices]
    f01 = fp[x_indices, y_indices + 1]
    f10 = fp[x_indices + 1, y_indices]
    f11 = fp[x_indices + 1, y_indices + 1]

    # Compute interpolation weights
    u = (x - x0) / (x1 - x0 + 1e-20)
    v = (y - y0) / (y1 - y0 + 1e-20)

    # Perform bilinear interpolation
    return (1 - u) * (1 - v) * f00 + u * (1 - v) * f10 + (1 - u) * v * f01 + u * v * f11


def torch_interp(x, xp, fp):
    """Vectorized linear interpolation function"""
    indice = torch.searchsorted(xp, x) - 1
    indice = torch.clamp(indice, 0, len(xp) - 2)
    x_low = xp[indice]
    x_high = xp[indice + 1]
    ratio = (x - x_low) / (x_high - x_low + 1e-16)
    ratio = torch.where(x < xp[0], 0.0, ratio)  # Left boundary handling
    ratio = torch.where(x > xp[-1], 1.0, ratio)  # Right boundary handling
    return fp[indice] * (1 - ratio) + fp[indice + 1] * ratio
