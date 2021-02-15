# Python script with some utilities to read an write data as well as calculating tensors
import torch as th
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
import datetime
import os
import scripts.preProcess as pre

# Default tensor type
dtype = th.DoubleTensor


def time():
    return datetime.datetime.now().strftime("%y-%m-%d_%H-%M")


def sigmoid_scaling(tensor):
    return (1.0 - th.exp(-tensor)) / (1.0 + th.exp(-tensor))


def mean_std_scaling(tensor, cap=2., mu=None, std=None):

    if (mu is None) & (std is None):
        rescale = True

        # calculate mean and standard deviation
        mu = th.mean(tensor, 0)
        std = th.std(tensor, 0)
    else:
        rescale = False

    # normalize tensor
    tensor = (tensor - mu) / std

    # remove outliers
    tensor[tensor > cap] = cap
    tensor[tensor < -cap] = -cap

    if rescale:
        # rescale tensors and recalculate mu and std from capped tensor
        tensor = tensor * std + mu
        mu = th.mean(tensor, 0)
        std = th.std(tensor, 0)

        # renormalize tensor after capping
        tensor = (tensor - mu) / std

    return tensor, mu, std


def cap_tensor(tensor, cap):
    tensor[tensor > cap] = cap
    tensor[tensor < -cap] = -cap
    return tensor


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

    return invar


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
    T[:, 2] = s2 - (1.0 / 3.0) * th.eye(3).type(dtype) \
              * (s2[:, 0, 0] + s2[:, 1, 1] + s2[:, 2, 2]).unsqueeze(1).unsqueeze(1)
    T[:, 3] = r2 - (1.0 / 3.0) * th.eye(3).type(dtype) \
              * (r2[:, 0, 0] + r2[:, 1, 1] + r2[:, 2, 2]).unsqueeze(1).unsqueeze(1)
    T[:, 4] = r0.bmm(s2) - s2.bmm(r0)
    t0 = s0.bmm(r2)
    T[:, 5] = r2.bmm(s0) + s0.bmm(r2) - (2.0 / 3.0) * th.eye(3).type(dtype) \
              * (t0[:, 0, 0] + t0[:, 1, 1] + t0[:, 2, 2]).unsqueeze(1).unsqueeze(1)
    T[:, 6] = rs.bmm(r2) - r2.bmm(sr)
    T[:, 7] = sr.bmm(s2) - s2.bmm(rs)
    t0 = s2.bmm(r2)
    T[:, 8] = r2.bmm(s2) + s2.bmm(r2) - (2.0 / 3.0) * th.eye(3).type(dtype) \
              * (t0[:, 0, 0] + t0[:, 1, 1] + t0[:, 2, 2]).unsqueeze(1).unsqueeze(1)
    T[:, 9] = r0.bmm(s2).bmm(r2) - r2.bmm(s2).bmm(r0)

    return T


def ph_interp(x_new):
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


def cdc_interp(x_new, re='12600'):
    """
    gives back spline interpolation for points on bottom boundary
    :return: interpolation function (takes x, gives back y)
    """
    surf_path = '/home/leonriccius/PycharmProjects/data_driven_rans/scripts/files/cdc_' + re + '.dat'
    surfacePoints = np.genfromtxt(surf_path, delimiter=' ', skip_header=3,
                                  skip_footer=0, names=["X", "Y"])

    # Seperate out the points on the bump
    bumpPoints = []
    for p0 in surfacePoints:
        if not p0['Y'] <= 10 ** (-8):
            bumpPoints.append([p0['X'], p0['Y']])
    bumpPoints = np.array(bumpPoints)
    start_point = np.zeros((1, 2))

    if int(re) == 12600:
        end_point = np.array([[12.56630039, 0.0]])
    else:
        end_point = np.array([[12.500906900, 0.0]])

    bumpPoints = np.concatenate((start_point, bumpPoints, end_point))

    return interp1d(bumpPoints[:, 0], bumpPoints[:, 1], kind='linear', fill_value='extrapolate')(x_new)


def cbfs_interp(x_new):
    """
    gives back spline interpolation for points on bottom boundary
    :return: interpolation function (takes x, gives back y)
    """
    surf_path = '/home/leonriccius/PycharmProjects/data_driven_rans/scripts/files/cbfs.dat'
    surfacePoints = np.genfromtxt(surf_path, delimiter=' ', skip_header=3,
                                  skip_footer=0, names=["X", "Y"])

    # Seperate out the points on the bump
    bumpPoints = []
    for p0 in surfacePoints:
        if not p0['Y'] <= -10 ** (-8):
            bumpPoints.append([p0['X'], p0['Y']])
    bumpPoints = np.array(bumpPoints)
    start_point = np.array([[-7.34, 1.]])
    end_point = np.array([[15.4, 0.0]])

    bumpPoints = np.concatenate((start_point, bumpPoints, end_point))

    return interp1d(bumpPoints[:, 0], bumpPoints[:, 1], kind='linear', fill_value='extrapolate')(x_new)


def mask_boundary_points(x, y, blthickness=0.15):
    """
    gives a mask that ensures all points that are blthickness far away form boundary are labelled True
    :param x: x coordinate dtype=float
    :param y: y coordinate dytpe=float
    :param blthickness: thickness of boundarylayer to cut dtype=float
    :return: mask dtype=np.array(bool)
    """
    mask = np.ones(x.shape, dtype=bool)
    y_interp = ph_interp(x)
    mask[np.where(y < y_interp + blthickness)] = False
    mask[np.where(y > 3.035 - blthickness)] = False
    mask[np.where(x < 0. + blthickness)] = False
    mask[np.where(x > 9. - blthickness)] = False
    return mask


def tecplot_reader(file_path, nb_var, skiprows):
    """
    Read in Tecplot files
    :param file_path: path to tecplot file to read in
    :param nb_var: number of variables stored in file
    :param skiprows: number of rows in header to skip
    :return: tuple of arrays
    """
    arrays = []
    with open(file_path, 'r') as a:
        for idx, line in enumerate(a.readlines()):
            if idx < skiprows:
                continue
            else:
                arrays.append([float(s) for s in line.split()])
    arrays = np.concatenate(arrays)
    return np.split(arrays, nb_var)


def anisotropy(rs, k, set_nan_to_0=False):
    """calculate normalized anisotropy tensor"""

    if th.max(k == 0.0):
        print('Warning, b contains nan entries. Consider removing nan data points')

    if set_nan_to_0:
        k = th.maximum(k, th.tensor(1e-8)).unsqueeze(0).unsqueeze(0).expand(3, 3, k.size()[0]).permute(2, 0, 1)
    else:
        k = k.unsqueeze(0).unsqueeze(0).expand(3, 3, k.size()[0]).permute(2, 0, 1)

    b = rs[:] / (2 * k) - 1 / 3 * th.eye(3).unsqueeze(0).expand(k.shape[0], 3, 3)

    return b


def enforce_zero_trace(tensor):
    """
    input a set of basis tensors and get back a set of traceless basis tensors
    :param tensor: N_points x 10 x 3 x 3
    :return: tensor: N_points x 10 x 3 x 3
    """
    print('enforcing 0 trace ...')
    return tensor - 1. / 3. * th.eye(3).unsqueeze(0).unsqueeze(0).expand_as(tensor) \
           * (tensor[:, :, 0, 0] + tensor[:, :, 1, 1] + tensor[:, :, 2, 2]).unsqueeze(2).unsqueeze(3).expand_as(tensor)


def enforce_realizability(tensor, margin=0.0):
    """
    input a set of anisotropy tensors and get back a set of tensors that does not violate realizabiliy constraints.
    creates labels for branchless if clause. labels hold true where realizability contraints are violated and corrects
    entries
    :param margin: (float) set to small value larger than 0 to push eigenvalues over -1/3 instead
    asymptotically approach the boundary
    :param tensor: N_points x 3 x 3
    :return: N_points x 3 x 3
    """
    # ensure b_ii > -1/3
    diag_min = th.min(tensor[:, [0, 1, 2], [0, 1, 2]], 1)[0].unsqueeze(1)
    labels = (diag_min < th.tensor(-1. / 3.))
    tensor[:, [0, 1, 2], [0, 1, 2]] *= labels * (-1. / (3. * diag_min)) + ~labels

    # ensure 2*|b_ij| < b_ii + b_jj + 2/3
    # b_12
    labels = (2 * th.abs(tensor[:, 0, 1]) > tensor[:, 0, 0] + tensor[:, 1, 1] + 2. / 3.).unsqueeze(1)
    tensor[:, [0, 1], [1, 0]] = labels * (tensor[:, 0, 0] + tensor[:, 1, 1] + 2. / 3.).unsqueeze(1) \
                                * .5 * th.sign(tensor[:, [0, 1], [1, 0]]) + ~labels * tensor[:, [0, 1], [1, 0]]

    # b_23
    labels = (2 * th.abs(tensor[:, 1, 2]) > tensor[:, 1, 1] + tensor[:, 2, 2] + 2. / 3.).unsqueeze(1)
    tensor[:, [1, 2], [2, 1]] = labels * (tensor[:, 1, 1] + tensor[:, 2, 2] + 2. / 3.).unsqueeze(1) \
                                * .5 * th.sign(tensor[:, [1, 2], [2, 1]]) + ~labels * tensor[:, [1, 2], [2, 1]]

    # b_13
    labels = (2 * th.abs(tensor[:, 0, 2]) > tensor[:, 0, 0] + tensor[:, 2, 2] + 2. / 3.).unsqueeze(1)
    tensor[:, [0, 2], [2, 0]] = labels * (tensor[:, 0, 0] + tensor[:, 2, 2] + 2. / 3.).unsqueeze(1) \
                                * .5 * th.sign(tensor[:, [0, 2], [2, 0]]) + ~labels * tensor[:, [0, 2], [2, 0]]

    # ensure positive semidefinitness by pushing smallest eigenvalue to -1/3. Reynolds stress eigenvalues are then
    # positive semidefinite
    # ensure lambda_1 > (3*|lambda_2| - lambda_2)/2
    eigval, eigvec = th.symeig(tensor, eigenvectors=True)
    labels = (eigval[:, 2] < (3 * th.abs(eigval[:, 1]) - eigval[:, 1]) * .5).unsqueeze(1)
    eigval *= labels * ((3. * th.abs(eigval[:, 1]) - eigval[:, 1]) / (2. * eigval[:, 2])).unsqueeze(1) + ~labels
    print('Violation of condition 1: {}'.format(th.max(labels)))

    # ensure lambda_1 < 1/3 - lambda_2
    labels = (eigval[:, 2] > 1. / 3. - eigval[:, 1]).unsqueeze(1)
    eigval *= labels * ((1. / 3. - eigval[:, 1]) / (eigval[:, 2]) - margin).unsqueeze(1) + ~labels
    print('Violation of condition 2: {}'.format(th.max(labels)))

    # rotate tensor back to initial frame
    tensor = eigvec.matmul(th.diag_embed(eigval).matmul(eigvec.transpose(1, 2)))

    return tensor


def load_standardized_data(data):
    """
    function to load in all fluid data, convert it to torch tensors, and interpolate dns data on rans grid
    :param data: dict with different flow geometries
    :return:
    """

    assert (os.path.exists(data['home'])), 'dictionary specified as \'home\' does not exist'

    for i, case in enumerate(data['flowCase']):
        for j, re in enumerate(data['Re'][i]):
            print((case, re))
            # define dns, rans, and target paths
            dns_path = os.sep.join([data['home'], data['dns'], case, re, 'tensordata'])
            rans_path = os.sep.join(
                [data['home'], data['rans'], case, 'Re{}_{}_{}'.format(re, data['model'][i][j], data['ny'][i][j])])
            rans_time = data['ransTime'][i][j]
            target_path = os.sep.join([data['home'], data['target_dir'], case, re])
            print(target_path)

            # load dns data
            grid_dns = th.load(os.sep.join([dns_path, 'grid-torch.th']))
            b_dns = th.load(os.sep.join([dns_path, 'b-torch.th']))
            rs_dns = th.load(os.sep.join([dns_path, 'rs-torch.th']))
            k_dns = th.load(os.sep.join([dns_path, 'k-torch.th']))
            u_dns = th.load(os.sep.join([dns_path, 'u-torch.th']))
            if os.path.isfile(os.sep.join([dns_path, 'p-torch.th'])):
                is_p = True
                p_dns = th.load(os.sep.join([dns_path, 'p-torch.th']))
            else:
                is_p = False

            print('rans time:  ', rans_time)

            # load rans data (grid, u, k, rs, epsilon)
            grid_rans = pre.readCellCenters(rans_time, rans_path)
            u_rans = pre.readVectorData(rans_time, 'U', rans_path)
            k_rans = pre.readScalarData(rans_time, 'k', rans_path)
            if os.path.isfile(os.sep.join([rans_path, rans_time, 'turbulenceProperties:R'])):
                rs_rans = pre.readSymTensorData(rans_time, 'turbulenceProperties:R', rans_path).reshape(-1, 3, 3)
            else:
                rs_rans = pre.readSymTensorData(rans_time, 'R', rans_path).reshape(-1, 3, 3)
            if os.path.isfile(os.sep.join([rans_path, rans_time, 'gradU'])):
                grad_u_rans = pre.readTensorData(rans_time, 'gradU', rans_path)  # or 'gradU
            else:
                grad_u_rans = pre.readTensorData(rans_time, 'grad(U)', rans_path)

            # reading in epsilon, otherwise calculate from omega
            if os.path.isfile(os.sep.join([rans_path, rans_time, 'epsilon'])):
                epsilon_rans = pre.readScalarData(rans_time, 'epsilon', rans_path)
            else:
                omega_rans = pre.readScalarData(rans_time, 'omega', rans_path)  # 'epsilon' or 'omega'
                epsilon_rans = omega_rans * k_rans * 0.09  # 0.09 is beta star

            # calculate mean rate of strain and rotation tensors
            s = 0.5 * (grad_u_rans + grad_u_rans.transpose(1, 2))
            r = 0.5 * (grad_u_rans - grad_u_rans.transpose(1, 2))

            # normalize s and r
            s_hat = (k_rans / epsilon_rans).unsqueeze(1).unsqueeze(2) * s
            r_hat = (k_rans / epsilon_rans).unsqueeze(1).unsqueeze(2) * r

            # cap s and r tensors
            if data['capSandR']:
                if th.max(th.abs(s_hat)) > 6. or th.max(th.abs(r_hat)) > 6.:
                    print('capping tensors ...')
                    s_hat = cap_tensor(s_hat, 6.0)
                    r_hat = cap_tensor(r_hat, 6.0)

            # calculate invariants
            inv = get_invariants(s_hat, r_hat)
            t = get_tensor_functions(s_hat, r_hat)

            # correct invariants of cases with symmetries
            if data['correctInvariants']:
                if case != 'SquareDuct':
                    print('Setting invariants 3 and 4 to 0 ...')
                    inv[:, [2, 3]] = 0.0

            # scale invariants
            # inv = mean_std_scaling(inv)
            # print(inv.shape)

            # enfore zero trace on tensorbasis
            if data['enforceZeroTrace']:
                t = enforce_zero_trace(t)

            # compute anisotropy tensor b
            b_rans = anisotropy(rs_rans, k_rans, data['removeNan'])

            # interpolate dns data on rans grid
            print('Interpolating DNS data on RANS grid ...')
            method = data['interpolationMethod']
            if case == 'SquareDuct':
                int_grid = grid_rans[:, [1, 2, 0]]
            else:
                int_grid = grid_rans

            b_dns_interp = th.tensor(griddata(grid_dns[:, 0:2], b_dns, (int_grid[:, 0], int_grid[:, 1]),
                                              method=method))
            u_dns_interp = th.tensor(griddata(grid_dns[:, 0:2], u_dns, (int_grid[:, 0], int_grid[:, 1]),
                                              method=method))
            rs_dns_interp = th.tensor(griddata(grid_dns[:, 0:2], rs_dns, (int_grid[:, 0], int_grid[:, 1]),
                                               method=method))
            k_dns_interp = th.tensor(griddata(grid_dns[:, 0:2], k_dns, (int_grid[:, 0], int_grid[:, 1]),
                                              method=method))
            if is_p:
                p_dns_interp = th.tensor(griddata(grid_dns[:, 0:2], p_dns, (int_grid[:, 0], int_grid[:, 1]),
                                                  method=method))

            if data['saveTensors']:
                # create directories if not defined
                if not os.path.exists(target_path):
                    os.makedirs(target_path)

                # save dns
                th.save(b_dns_interp, os.sep.join([target_path, 'b_dns-torch.th']))
                th.save(rs_dns_interp, os.sep.join([target_path, 'rs_dns-torch.th']))
                th.save(k_dns_interp, os.sep.join([target_path, 'k_dns-torch.th']))
                th.save(u_dns_interp, os.sep.join([target_path, 'u_dns-torch.th']))
                if is_p:
                    th.save(p_dns_interp, os.sep.join([target_path, 'p_dns-torch.th']))

                # save rans
                th.save(grid_rans, os.sep.join([target_path, 'grid-torch.th']))
                th.save(u_rans, os.sep.join([target_path, 'u_rans-torch.th']))
                th.save(rs_rans, os.sep.join([target_path, 'rs_rans-torch.th']))
                th.save(b_rans, os.sep.join([target_path, 'b_rans-torch.th']))
                th.save(k_rans, os.sep.join([target_path, 'k_rans-torch.th']))
                th.save(inv, os.sep.join([target_path, 'inv-torch.th']))
                th.save(t, os.sep.join([target_path, 't-torch.th']))

    return 0


import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = {}
    data['home'] = '/home/leonriccius/Documents/Fluid_Data'
    data['dns'] = 'dns'
    data['rans'] = 'rans_kaandorp'
    data['target_dir'] = 'tensordata_unscaled_inv_corr'
    data['flowCase'] = ['PeriodicHills',
                        'ConvDivChannel',
                        'CurvedBackwardFacingStep',
                        'SquareDuct']
    data['Re'] = [['700', '1400', '2800', '5600', '10595'],
                  ['12600', '7900'],
                  ['13700'],
                  ['1800', '2000', '2400', '2600', '2900', '3200', '3500']]
    data['nx'] = [[140, 140, 140, 140, 140],
                  [140, 140],
                  [140],
                  [50, 50, 50, 50, 50, 50, 50]]
    data['ny'] = [[150, 150, 150, 150, 150],
                  [100, 100],
                  [150],
                  [50, 50, 50, 50, 50, 50, 50]]
    data['model'] = [['kOmega', 'kOmega', 'kOmega', 'kOmega', 'kOmega'],
                     ['kOmega', 'kOmega'],
                     ['kOmega'],
                     ['kOmega', 'kOmega', 'kOmega', 'kOmega', 'kOmega', 'kOmega', 'kOmega']]
    data['ransTime'] = [['30000', '30000', '30000', '30000', '30000'],
                        ['7000', '7000'],
                        ['3000'],
                        ['40000', '40000', '50000', '50000', '50000', '50000', '50000']]

    data['interpolationMethod'] = 'linear'
    data['enforceZeroTrace'] = True
    data['capSandR'] = True
    data['saveTensors'] = True
    data['removeNan'] = True
    data['correctInvariants'] = True

    load_standardized_data(data)

    # inv = th.load('/home/leonriccius/Documents/Fluid_Data/tensordata_unscaled_inv_corr/PeriodicHills/2800/inv-torch.th')




    # path = '/home/leonriccius/Documents/Fluid_Data/tensordata/SquareDuct/1800'
    # u = th.load(os.sep.join([path, 'u_rans-torch.th']))
    # inv = th.load(os.sep.join([path, 'inv-torch.th']))
    # grid = th.load(os.sep.join([path, 'grid-torch.th']))
    # b = th.load(os.sep.join([path, 'b_rans-torch.th']))
    #
    # fig, ax = plt.subplots()
    # ax.axis('equal')
    # plot = ax.tricontourf(grid[:, 1], grid[:, 2], b[:, 0, 0])
    # fig.colorbar(plot)
    # fig.show()


    # data['flowCase'] = ['PeriodicHills',
    #                     # 'ConvDivChannel',
    #                     'CurvedBackwardFacingStep',
    #                     'SquareDuct']
    # data['Re'] = [['700', '1400', '2800', '5600', '10595'],
    #               # ['12600', '7900'],
    #               ['13700'],
    #               ['1800']]
    # data['nx'] = [[140, 140, 140, 140, 140],
    #               # [140, 140],
    #               [140],
    #               [50]]
    # data['ny'] = [[150, 150, 150, 150, 150],
    #               # [100, 100],
    #               [150],
    #               [50]]
    # data['model'] = [['kOmega', 'kOmega', 'kOmega', 'kOmega', 'kOmega'],
    #                  # ['kOmega', 'kOmega'],
    #                  ['kOmega'],
    #                  ['kOmega']]
    # data['ransTime'] = [['30000', '30000', '30000', '30000', '30000'],
    #                     # ['7000', '7000'],
    #                     ['3000'],
    #                     ['40000']]
