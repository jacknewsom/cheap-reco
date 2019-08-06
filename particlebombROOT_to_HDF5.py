from osf.image_api import image_reader_3d
from ROOT import TFile
import numpy as np
import h5py
import sys
import os

'''
particlebomb_ROOT_to_HDF5.py

To use this file to convert 'particlebomb'-style ROOT files into HDF5, first ensure you've
booted the appropriate Docker image, i.e.

    shifter --image=deeplearnphysics/larcv2:cuda90-pytorch-dev20181015-scn

Doing so gives access to LArCV2 which is necessary to unpack Kazu's code.

Then, run

    python particlebombROOT_to_HDF5.py

to get started.
'''


def get_data_dim(filename):
    '''Return the length of a side of data tensor in ROOT file at filename'''
    try:
        neutrinos = TFile(filename)
    except:
        print('Invalid file')
        return None
    data = neutrinos.Get('sparse3d_data_tree')
    data.GetEvent(0)
    x, y, z = [int(data.GetBranch('_meta._%snum' % v).GetValue(0, 0)) for v in ['x', 'y', 'z']]
    if x == y and y == z:
        return x
    print("Bad data dimension: (%d, %d, %d)" % (x, y, z))

def load_raw_data(filename, event=0):
    '''Load index, energy deposition, label, and dimension from a Kazu-style
    ROOT file
    '''
    reader = image_reader_3d(filename)
    voxels, energies, labels = reader.get_image(event)
    return voxels, energies, labels

def load_vertex_location(tfile, event_id):
    # returns vertex location for event in voxel coordinates
    sparse3d_data_tree = tfile.Get("sparse3d_data_tree")
    monte_carlo = tfile.Get("particle_mcst_tree")

    sparse3d_data_tree.GetEvent(event_id)
    monte_carlo.GetEvent(event_id)

    xscale = sparse3d_data_tree.GetBranch("_meta._xlen").GetValue(0, 0)
    yscale = sparse3d_data_tree.GetBranch("_meta._ylen").GetValue(0, 0)
    zscale = sparse3d_data_tree.GetBranch("_meta._zlen").GetValue(0, 0)
    
    bbp1x = sparse3d_data_tree.GetBranch("_meta._p1.x").GetValue(0, 0)
    bbp1y = sparse3d_data_tree.GetBranch("_meta._p1.y").GetValue(0, 0)
    bbp1z = sparse3d_data_tree.GetBranch("_meta._p1.z").GetValue(0, 0)
    bbp2x = sparse3d_data_tree.GetBranch("_meta._p2.x").GetValue(0, 0)
    bbp2y = sparse3d_data_tree.GetBranch("_meta._p2.y").GetValue(0, 0)
    bbp2z = sparse3d_data_tree.GetBranch("_meta._p2.z").GetValue(0, 0)
    
    x_cm = monte_carlo.GetBranch("_part_v._vtx._x").GetValue(0, 1)
    y_cm = monte_carlo.GetBranch("_part_v._vtx._y").GetValue(0, 1)
    z_cm = monte_carlo.GetBranch("_part_v._vtx._z").GetValue(0, 1)

    voxelizer = lambda coord, p1coord, scale: int((coord - p1coord) / scale)
    x = voxelizer(x_cm, bbp1x, xscale)
    y = voxelizer(y_cm, bbp1y, yscale)
    z = voxelizer(z_cm, bbp1z, zscale)
    return x, y, z
    
def kazu_to_HDF5(filename, noisy=False, n_events=10000):
    '''Store Kazu-style data in an HDF5 file
    '''
    reader = image_reader_3d(filename)
    dim = get_data_dim(filename)
    dt = h5py.special_dtype(vlen=np.dtype('float32'))
    if filename[-5:] == ".root":
        filename = filename[:-5]

    if noisy:
        print("Converting %s.root to HDF5..." % filename)
        
    with h5py.File(filename+'.hdf5', 'w') as f:
        dimension = f.create_dataset('dimension', (1,), dtype=np.dtype('int32'))
        dimension[0] = dim

        # used to get vertex location
        tfile = TFile(filename + ".root")
        
        voxels_x = f.create_dataset('voxels_x', (n_events,), dtype=dt)
        voxels_y = f.create_dataset('voxels_y', (n_events,), dtype=dt)
        voxels_z = f.create_dataset('voxels_z', (n_events,), dtype=dt)
        energies = f.create_dataset('energies', (n_events,), dtype=dt)
        labels = f.create_dataset('labels', (n_events,), dtype=dt)
        vertex = f.create_dataset('vertex', (n_events, 3), dtype=np.dtype('int32'))
        
        for i in range(n_events):
            if i % 1000 == 0 and noisy:
                print("\t%d percent complete." % (100.0 * (i+1)/n_events))
            voxel, energies[i], labels[i] = reader.get_image(i)
            voxels_x[i] = voxel[:, 0]
            voxels_y[i] = voxel[:, 1]
            voxels_z[i] = voxel[:, 2]
            vertex[i] = load_vertex_location(tfile, i)

        print("\t100 percent complete.")
    
if __name__ == '__main__':
    ok = False
    while not ok:
        src_dir = raw_input("\nWhat directory would you like to convert (Kazu -> HDF5)? ")
        if src_dir[-1] != '/':
            src_dir += '/'
        print("%s contains..." % src_dir)
        for f in os.listdir(src_dir):
            print(f)
        ok = raw_input("Is this what you want? ").lower() in ['yes', 'y', 'yep', '']
    for f in os.listdir(src_dir):
        if not os.path.isfile(src_dir + f):
            continue
        elif f[-5:] != ".root":
            continue
        elif f[-5:] == ".hdf5":
            continue
        elif f[:-5] + ".hdf5" in os.listdir(src_dir):
            continue
            
        kazu_to_HDF5(src_dir + f, True)
