# cheap-reco
A collection of various tools for reconstructing neutrino interaction events in liquid argon.

These tools have a variety of different dependencies so the easiest way to use them is inside of a conveniently prepared Docker image: https://github.com/DeepLearnPhysics/larcv2-docker/blob/build/Dockerfile.larcv1.0.0rc01-cuda90-pytorchdev20181015-scn

You'll need `PyROOT`, `numpy`, `scipy`,	and `scikit-learn` at minimum to run this code.	To draw	graphs, you'll need `plotly` as well.

You'll also need to have https://github.com/DeepLearnPhysics/dlp_opendata_api in the same directory to convert 'particlebomb' style data to HDF5.

## Usage
One typical workflow is
1. Load data from HDF5 (using functions in `hdf5_loader.py`)
2. Perform clustering (using functions in `clustering.py`)
3. Plot results (using functions in `drawing.py`)