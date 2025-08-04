import os
import numpy as np
from plyfile import PlyElement, PlyData


def load_ply(file_name, with_faces=False, with_color=False, with_normal=False):
    ply_data = PlyData.read(file_name)
    vertices = ply_data['vertex']
    points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    ret_val = [points]

    if with_faces:
        faces = np.vstack(ply_data['face']['vertex_indices'])
        ret_val.append(faces)

    if with_color:
        r = np.vstack(ply_data['vertex']['red'])
        g = np.vstack(ply_data['vertex']['green'])
        b = np.vstack(ply_data['vertex']['blue'])
        color = np.hstack((r, g, b))
        ret_val.append(color)

    if with_normal:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T

    if len(ret_val) == 1:  # Unwrap the list
        ret_val = ret_val[0]

    pc = ret_val

    return (pc, normals) if with_normal else pc



def save_ply(filepath, color_pc, c=None, use_color=False, use_normal=False, verbose=False):
    # if verbose:
    #     print("Writing color pointcloud to: ", filepath)
    
    f = open(filepath, 'w')
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write('element vertex '+str(int(color_pc.shape[0])) + '\n')
    f.write('property float x\nproperty float y\nproperty float z\n')
    if use_normal:
        f.write('property float nx\nproperty float ny\nproperty float nz\n')
    f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
    f.write("end_header\n")

    if c == 'r':
        color = [255, 0, 0]
    elif c == 'g':
        color = [0, 255, 0]
    elif c == 'b':
        color = [0,0,255]
    elif isinstance(c, list):
        color = c
    else:
        color = [255,255,255]
    
    # default_normal = 0.0
    color_range = range(6,9) if use_normal else range(3,6)

    for i in range(color_pc.shape[0]):
        #f.write("v")
        row = list(color_pc[i,:])
        for j in range(3):
            f.write("%.4f " % float(row[j]))

        # ---------------- normal ---------------
        if use_normal:
            for t in range(3,6):
                f.write("%.4f " % float(row[t]))
        # else:
        #     for t in range(3):
        #         f.write("%.4f " % float(default_normal))


        # ---------------- color ----------------    
        if use_color:
            if color_pc.shape[1] == 4:
                # only intensity provided
                for t in range(3):
                    f.write(str(int(color[t] * row[3])) + ' ')
            else:
                for t in color_range:
                    f.write(str(int(row[t])) + ' ')
        else:
            for t in range(3):
                f.write(str(int(color[t])) + ' ')

        f.write("\n")
    f.write("\n")
    f.close()