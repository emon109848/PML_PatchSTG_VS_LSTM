# Spatial patches and sensor index maps

PatchSTG organizes sensor representations into groups so it can exchange spatial
information using smaller attention operations. This grouping rearranges sensors;
it does not average each patch into one token or change which sensors are targets.

Coordinate-based recursive splits produce small leaves. Training-derived traffic
similarity chooses padding positions, then leaves are merged into equal-sized
patches. For this SD configuration there are 16 patches of 64 positions:
1,024 computational positions for 716 original sensors.

`gather` maps padded positions to original sensors. `restore` chooses the one
canonical position for each original sensor. The key invariant is
`gather[restore] == arange(716)`. Without restoration, predictions can be paired
with the wrong sensor targets or padded duplicates can incorrectly affect scores.

Similarity is learned from the training interval only. Coordinates are metadata,
not future traffic. Neither validates the assumed row-to-column pairing by itself.
The saved maps preserve exact preprocessing used by the trained checkpoint.

Read [spatial preparation](../patchstg/spatial_preparation.md) and
[the source](../../src/traffic_benchmark/data/spatial.py). The project reuses its
saved maps; there is no need to rebuild them for the completed comparison.
