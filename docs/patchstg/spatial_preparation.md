# Spatial preparation: training similarity, patches and restoration

Source: [spatial.py](../../src/traffic_benchmark/data/spatial.py).
Saved maps: [spatial_maps.npz](../../runs/patchstg_sd_seed2023_50ep/spatial_maps.npz).

## Similarity from training observations only

`training_similarity` requires traffic/scaler statistics for precisely the
training interval `[0,21024)`. It normalizes training readings with the global
training mean/std, divides them into full 288-frame segments, averages the
aligned segment profiles, and calculates cosine similarity between sensors.
An exponential transform of standardized cosine scores provides padding scores.
Zero-norm profiles use zero cosine; degenerate cosine variance raises an error.

There are exactly 73 full segments. At the assumed SD cadence, 288 positions
span three days. This preserves the reviewed code convention; it is not a
96-slot daily-profile correction or a road-distance adjacency graph.

## Coordinate partitions and padding

Balanced recursive splits alternate latitude/longitude. Leaf depth 9 produces
512 leaves with capacity two; 716 sensors require up to 1,024 padded positions.
Similarity selects padding candidates, and a merge factor of 32 produces
16 patches of 64 positions. Stable sensor-index tie breaking specifies cases
that otherwise depend on sort behavior.

The map constructor checks coordinates, similarity dimensions, nonnegative
scores, and compatible patch settings. Padded duplicates supply representations
for computation; they are not additional observed sensors or scored targets.

## Gather and restore

`gather` says which original sensor fills each padded position. `restore` says
where each original sensor is recovered from that layout. The invariant is:

```python
gather[restore] == np.arange(716)
```

Every original sensor has exactly one restoration position. Model buffers store
these maps, move with the device, and are verified before loading checkpoint
weights that would otherwise overwrite them.

## Cache identity

Cache metadata includes source hashes, scaler, training boundary, segment and
partition settings, tie policy, code/version identity and payload checksum.
Mismatched/corrupt caches fail instead of silently rebuilding a different graph.
This verifies artifact identity, not independent metadata-column alignment.

Integration loaded and validated the saved maps without reconstructing them or
performing model inference. The existing maps and scores are reused unchanged.
