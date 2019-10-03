from ROOT import TFile
import numpy as np
import h5py

def chris_to_HDF5(filename):
    print("converting %s..." % filename)
    rootfile = TFile(filename)
    tree = rootfile.Get('tree')
    n_events = 50000 # number of events in each 'chris'-style ROOT file

    # create custom data type for ragged array
    dt = h5py.special_dtype(vlen=np.dtype('float32'))
    if filename[-5:] == '.root':
        filename = filename[:-5]

    # create HDF5 file
    with h5py.File(filename+'.hdf5', 'w') as f:
        voxels_x = f.create_dataset('voxels_x', (n_events,), dtype=dt)
        voxels_y = f.create_dataset('voxels_y', (n_events,), dtype=dt)
        voxels_z = f.create_dataset('voxels_z', (n_events,), dtype=dt)
        energies = f.create_dataset('energies', (n_events,), dtype=dt)
        vertex = f.create_dataset('vertex', (n_events, 3), dtype=np.dtype('int32'))

        # primary particle metadata
        pdg_codes = f.create_dataset('pdg_codes', (n_events,), dtype=dt)
        kinetic_energies = f.create_dataset('kinetic_energies', (n_events,), dtype=dt)

        for i in range(n_events):            
            tree.GetEntry(i)
            ev_x, ev_y, ev_z = list(tree.voxx), list(tree.voxy), list(tree.voxz)
            ev_energies = list(tree.voxe)
            ev_pdg_codes, ev_kinetic_energies = list(tree.fsPDG), list(tree.fsKE)
            voxels_x[i] = ev_x
            voxels_y[i] = ev_y
            voxels_z[i] = ev_z
            vertex[i] = tree.vtxx, tree.vtxy, tree.vtxz
            energies[i] = ev_energies
            pdg_codes[i] = ev_pdg_codes
            kinetic_energies[i] = ev_kinetic_energies

            if (100.0 * (i + 1) / n_events) % 10 == 0:
                print("\t {}% complete".format(100.0 * (i + 1) / n_events))

if __name__ == "__main__":
    chris_to_HDF5("/global/project/projectdirs/dune/users/marshalc/jack.root")
