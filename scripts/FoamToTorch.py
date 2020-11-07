import numpy as np
import os.path
# sys.path.insert(1, '/home/leonriccius/Documents/jupyter_notebook')
import scripts.preProcess as pre

if __name__ == '__main__':

    # setting directory structure
    rans_dir =  ['700', '1400', '2800', '5600', '10595'] #, '12600'] # ['5600']   # ['700', '1400', '2800', '5600']
    rans_path = '/home/leonriccius/OpenFOAM_build/OpenFOAM-v2006/custom_cases/periodic_hills_RANS/refined_mesh/kEpsilon/'
    rans_time = '1500'

    for i in rans_dir:

        # setting directory path and cell centers
        curr_dir = rans_path + i  # rans_dir[3]
        cell_centers = pre.readCellCenters(rans_time, curr_dir)

        # check if kEpsilon or kOmega
        is_epsilon = os.path.isfile(curr_dir + '/' + rans_time + '/epsilon')

        # Get unique x & y coords
        cell_n = cell_centers.numpy()
        # cell_coord = np.array([cell_n[:, 0], cell_n[:, 1]])
        # cell_xy = np.unique(cell_n[:, 0:2], axis=0)

        # Get index and coordinates of slice
        cell_z = cell_n[:, 2]
        cell_z_unique = np.unique(cell_z)
        slice_index = np.where(cell_z == 2.205)  # phill 2.205, convdivch 1.47/1.53
        cell_0 = cell_n[slice_index]

        # Now get averaging indexes (where x & y are the same)
        # avg_index = []
        # for j in range(cell_xy.shape[0]):
        #     if j % 500 == 0:
        #         print('Finding average indexes {}/{}'.format(j, len(cell_xy)))
        #     avg_index.append(np.where(np.all(cell_n[:, 0:2] == cell_xy[j], axis=1))[0])

        # reading in tensors and fields
        RS = pre.readSymTensorData(rans_time, 'turbulenceProperties:R', curr_dir)
        grad_U = pre.readTensorData(rans_time, 'gradU', curr_dir) # or 'gradU
        k = pre.readScalarData(rans_time, 'k', curr_dir)
        U = pre.readVectorData(rans_time, 'U', curr_dir)
        if is_epsilon:
            epsilon = pre.readScalarData(rans_time, 'epsilon', curr_dir)
        else:
            omega = pre.readScalarData(rans_time, 'omega', curr_dir) # 'epsilon' or 'omega'

        # selecting sliced fields
        RS_0 = pre.slicedField(RS, slice_index)
        grad_U_0 = pre.slicedField(grad_U, slice_index)
        U_0 = pre.slicedField(U, slice_index)
        k_0 = pre.slicedField(k, slice_index)
        if is_epsilon:
            epsilon_0 = pre.slicedField(epsilon, slice_index)
        else:
            omega_0 = pre.slicedField(omega, slice_index)

        # calculating S and R from velocity gradient
        S_0 = 0.5 * (grad_U_0 + grad_U_0.transpose(1, 2))
        R_0 = 0.5 * (grad_U_0 - grad_U_0.transpose(1, 2))

        # saving sliced fields
        pre.saveTensor(RS_0, 'RS', rans_time, curr_dir)
        pre.saveTensor(U_0, 'U', rans_time, curr_dir)
        pre.saveTensor(k_0, 'k', rans_time, curr_dir)
        pre.saveTensor(S_0, 'S', rans_time, curr_dir)
        pre.saveTensor(R_0, 'R', rans_time, curr_dir)
        if is_epsilon:
            pre.saveTensor(epsilon_0, 'epsilon', rans_time, curr_dir)
        else:
            pre.saveTensor(omega_0, 'omega', rans_time, curr_dir)

        # saving cell center coordinates
        pre.saveTensor(cell_0, 'cellCenters', rans_time, curr_dir)
