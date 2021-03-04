"""
Script for writing predicted b directly into a foam file.
Functionality currently lies in writeToFoam notebook and will be ported here when done.
"""

import os, sys
import torch as th

# own scrips
sys.path.insert(1, '/home/leonriccius/PycharmProjects/data_driven_rans/scripts/')
import preProcess as pre


# dict for boundary types
b_type = {'empty': "        {0:<16}{1}\n".format("type", "empty;"),
          'fixedValue_uniform': "        {0:<16}{1}\n".format("type", "fixedValue;")+
                                "        {0:<16}{1}\n".format("value", "uniform (0 0 0 0 0 0);"),
          'zeroGradient': "        {0:<16}{1}\n".format("type", "zeroGradient;"),
          'symmetryPlane':"        {0:<16}{1}\n".format("type", "symmetryPlane;"),
          'cyclic': "        {0:<16}{1}\n".format("type", "cyclic;"),
          'fixedValue_nonuniform': "        {0:<16}{1}\n".format("type", "fixedValue;")+
                                   "        {0:<16}{1}\n".format("value", "nonuniform List<symmTensor>")}


def writesymmtensor(tensor,
                    filename,
                    boundaries):

    n = tensor.shape[0]

    # read in header
    hf = open('/home/leonriccius/PycharmProjects/data_driven_rans/scripts/files/foam_header', 'r')
    if hf.mode == 'r':
        header = hf.read()
    hf.close()

    # open file to write
    of = open(filename, 'w')

    # check if mode is write
    if of.mode == 'w':

        # write header
        of.write(header)
        # write number of internal points
        of.write("{}\n(\n".format(tensor.shape[0]))

        # write internal points
        # for i in range(tensor.shape[0]):
        #     of.write("({:9f} {:9f} {:9f} {:9f} {:9f} {:9f})\n".format(tensor[i, 0, 0], tensor[i, 0, 1],
        #                                                               tensor[i, 0, 2], tensor[i, 1, 1],
        #                                                               tensor[i, 1, 2], tensor[i, 2, 2]))
        for point in tensor:
            of.write("({:9f} {:9f} {:9f} {:9f} {:9f} {:9f})\n".format(point[0, 0], point[0, 1],
                                                                      point[0, 2], point[1, 1],
                                                                      point[1, 2], point[2, 2]))

        # write boundary patches
        of.write(")\n;\n\nboundaryField\n{\n")

        # loop over boundaries list
        for boundary in boundaries:
            of.write("    {}\n".format(boundary[0]))
            of.write("    {\n")
            of.write(b_type[boundary[1]])

            # write boundary filed if specified
            if boundary[1] == 'fixedValue_nonuniform':
                of.write("{}\n(\n".format(boundary[2].shape[0]))
                for b_point in boundary[2]:
                    of.write("({:9f} {:9f} {:9f} {:9f} {:9f} {:9f})\n".format(b_point[0, 0], b_point[0, 1],
                                                                              b_point[0, 2], b_point[1, 1],
                                                                              b_point[1, 2], b_point[2, 2]))
                of.write(")\n;\n")
            # close boundary bracket
            of.write("    }\n")

        of.write("}")

        # write closing lines of foam
        of.write("\n\n\n// ************************************************************************* //")

    # close file
    of.close()


if __name__=='__main__':
    path = '/home/leonriccius/OpenFOAM/leonriccius-v2006/run/pitzDaily/kEpsilon'
    rans_time = '282'
    rs = pre.readSymTensorData(rans_time, 'turbulenceProperties:R', path).reshape(-1, 3, 3)
    cellCenters = pre.readCellCenters(rans_time, path)

    # compute k
    k0 = 0.5 * th.from_numpy(rs.numpy().trace(axis1=1, axis2=2))
    k = k0.unsqueeze(0).unsqueeze(0).expand(3, 3, k0.shape[0])
    k = k.permute(2, 0, 1)

    # compute b
    b0 = rs / (2 * k) - 1 / 3 * th.eye(3).unsqueeze(0).expand(k0.shape[0], 3, 3)

    rand_tensor = th.rand(5, 3, 3)
    boundary_b = 0.5 * (rand_tensor + rand_tensor.transpose(1, 2))

    # list for boundary names and corresponding type
    b_list = [('inlet', 'fixedValue_uniform'),
              ('outlet', 'zeroGradient'),
              ('upperWall', 'fixedValue_uniform'),
              ('lowerWall', 'fixedValue_nonuniform', boundary_b),
              ('frontAndBack', 'empty')]
    writesymmtensor(b0, '/home/leonriccius/Desktop/b_dd_pycharm', b_list)

