import numpy as np
import scipy as sp
import scipy.spatial

'''Run entire reconstruction suite.

1. Ask for data source
  - simulate_interaction
  - edep_sim

2. Load in data

3. Cluster hits

4. Associate clusters with vertices (if possible)

5. Calculate metrics

6. Save vertex-associated data to new HDF5 file and
   optionally save drawings as HTMLs
'''

data_sources = {"Simulate New Interaction": simulate_interaction, "Load from edep_sim": None}
inp = ''
while inp not in range(len(data_sources.keys())):
    print("Where would you like to load data from?\n")
    for i, data_source in enumerate(data_sources):
        print("\t{}.".format(i) + data_source)
    inp = int(raw_input("\n? "))

