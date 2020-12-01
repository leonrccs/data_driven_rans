# Python script with some utilities to read an write data as well as calculating tensors
import torch as th
import numpy as np
from scipy.interpolate import interp1d

# Default tensor type
dtype = th.DoubleTensor


def get_invariants(s0, r0):
    """function for computation of tensor basis
        Inputs:
            s0: N x 3 x 3 (N is number of data points)
            r0: N x 3 x 3
        Outputs:
            invar_sig : N x 5 (5 is number of scalar invariants)
    """
    nCells = s0.size()[0]
    invar = th.zeros(nCells, 5).type(dtype)

    s2 = s0.bmm(s0)
    r2 = r0.bmm(r0)
    s3 = s2.bmm(s0)
    r2s = r2.bmm(s0)
    r2s2 = r2.bmm(s2)

    invar[:, 0] = (s2[:, 0, 0] + s2[:, 1, 1] + s2[:, 2, 2])  # Tr(s2)
    invar[:, 1] = (r2[:, 0, 0] + r2[:, 1, 1] + r2[:, 2, 2])  # Tr(r2)
    invar[:, 2] = (s3[:, 0, 0] + s3[:, 1, 1] + s3[:, 2, 2])  # Tr(s3)
    invar[:, 3] = (r2s[:, 0, 0] + r2s[:, 1, 1] + r2s[:, 2, 2])  # Tr(r2s)
    invar[:, 4] = (r2s2[:, 0, 0] + r2s2[:, 1, 1] + r2s2[:, 2, 2])  # Tr(r2s2)

    # Scale invariants by sigmoid function
    # Can use other scaling here
    invar_sig = (1.0 - th.exp(-invar)) / (1.0 + th.exp(-invar))
    invar_sig[invar_sig != invar_sig] = 0

    return invar_sig


def get_tensor_functions(s0, r0):
    """function for computation of tensor basis
        Inputs:
            s0: N x 3 x 3 (N is number of datapoints)
            r0: N x 3 x 3
        Outputs:
            T : N x 10 x 3 x 3 (10 is number of basis tensors)
    """

    nCells = s0.size()[0]
    T = th.zeros(nCells, 10, 3, 3).type(dtype)

    s2 = s0.bmm(s0)
    r2 = r0.bmm(r0)
    sr = s0.bmm(r0)
    rs = r0.bmm(s0)

    T[:, 0] = s0
    T[:, 1] = sr - rs
    T[:, 2] = s2 - (1.0 / 3.0) * th.eye(3).type(dtype) * (s2[:, 0, 0] + s2[:, 1, 1] + s2[:, 2, 2]).unsqueeze(
        1).unsqueeze(1)
    T[:, 3] = r2 - (1.0 / 3.0) * th.eye(3).type(dtype) * (r2[:, 0, 0] + r2[:, 1, 1] + r2[:, 2, 2]).unsqueeze(
        1).unsqueeze(1)
    T[:, 4] = r0.bmm(s2) - s2.bmm(r0)
    t0 = s0.bmm(r2)
    T[:, 5] = r2.bmm(s0) + s0.bmm(r2) - (2.0 / 3.0) * th.eye(3).type(dtype) * (
            t0[:, 0, 0] + t0[:, 1, 1] + t0[:, 2, 2]).unsqueeze(1).unsqueeze(1)
    T[:, 6] = rs.bmm(r2) - r2.bmm(sr)
    T[:, 7] = sr.bmm(s2) - s2.bmm(rs)
    t0 = s2.bmm(r2)
    T[:, 8] = r2.bmm(s2) + s2.bmm(r2) - (2.0 / 3.0) * th.eye(3).type(dtype) * (
            t0[:, 0, 0] + t0[:, 1, 1] + t0[:, 2, 2]).unsqueeze(1).unsqueeze(1)
    T[:, 9] = r0.bmm(s2).bmm(r2) + r2.bmm(s2).bmm(r0)

    # Scale the tensor basis functions by the L2 norm
    l2_norm = th.zeros(T.size(0), 10)
    # l2_norm = 0   # not sure why Nick did this, introduces numerical error IMO since dtype is changed
    for (i, j), x in np.ndenumerate(np.zeros((3, 3))):
        l2_norm += th.pow(T[:, :, i, j], 2)
    T = T / th.sqrt(l2_norm).unsqueeze(2).unsqueeze(3)

    return T


def bottom_interpolation(x_new):
    """
    gives back spline interpolation for points on bottom boundary
    :return: interpolation function (takes x, gives back y)
    """
    # define bottom boundary
    x = (1.929 / 54) * np.array([0., 0.1, 9., 14., 20., 30., 40., 53.9, 54.])
    x = np.append(x, 9 - x[::-1])
    y = (1.929 / 54) * np.array([28., 28., 27., 24., 19., 11., 4., 0., 0.])
    y = np.append(y, y[::-1])

    # spline interpolation
    return interp1d(x, y, kind='cubic', fill_value='extrapolate')(x_new)


def mask_boundary_points(x, y, blthickness=0.15):
    """
    gives a mask that ensures all points that are blthickness far away form boundary are labelled True
    :param x: x coordinate dtype=float
    :param y: y coordinate dytpe=float
    :param blthickness: thickness of boundarylayer to cut dtype=float
    :return: mask dtype=np.array(bool)
    """
    mask = np.ones(x.shape, dtype=bool)
    y_interp = bottom_interpolation(x)
    mask[np.where(y < y_interp + blthickness)] = False
    mask[np.where(y > 3.035 - blthickness)] = False
    mask[np.where(x < 0. + blthickness)] = False
    mask[np.where(x > 9. - blthickness)] = False
    return mask
