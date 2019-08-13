'''Script to find optimal DBSCAN parameters for clustering both tracks and showers
efficiently.
'''

import json
import numpy as np
from hdf5_loader import load_and_convert_HDF5_to_sparse_np as loader
from drawing import draw_events_and_vertices as draw
from metrics import energy_metrics
from clustering import dbscan

inp = ''
tracks = []
while inp != 'n':
    inp = str(raw_input("Next event id?"))
    if inp == 'n':
        continue
    tracks.append(int(inp))

efficiencies = {}
for track in tracks:
    d, c, f, l, v = loader("data/ArCube_0000.hdf5", 1, track)
    efficiencies[track] = {}
    print("Analyzing event %d" % track)
    for epsilon in range(1, 25):
        for min_samples in range(1, 30):
            predictions = dbscan(c[0][:, :3], epsilon, min_samples).labels_
            efficiency = energy_metrics(f[0], predictions, c[0][:, -1])[0]
            efficiencies[track][str((epsilon, min_samples))] = efficiency[0][0]
            print("\t(%d, %d) has efficiency %.3f" % (epsilon, min_samples, efficiency[0][0]))

with open(str(tracks) + ".json", "w") as f:
    json.dump(efficiencies, f)

print("Done. Interpreter drop now ...")
import code
code.interact(local=locals())
            
