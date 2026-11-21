# What is saved on GitHub

The source repository includes Python code, command scripts, configurations,
tests, documentation, attribution, source-provenance records, and compact
measured reports/figures. The final comparison is available at
[reports/final_comparison](../reports/final_comparison/PROJECT_REPORT.md).

It excludes local credentials/assistant state, temp/, Python caches, logs,
raw traffic data, binary model weights/spatial caches, generated run directories
and experiment archives. These remain on the local computer; uploading source
does not back up excluded assets.

## Local assets required for reproduction

- Raw SD traffic and metadata at data/raw/largest_sd/.
- Selected contextual and PatchSTG checkpoints, reports and spatial maps under
  their documented runs/ directories.
- Cached PatchSTG forecasts for exporting the comparison dashboard without
  running PatchSTG again.

These assets are retained locally. Full run JSON/HTML exports are excluded with
runs/, while compact accuracy results and figures are copied into reports/.
The standalone interactive dashboard is a generated local artifact, not a hosted
GitHub website. References to local checkpoints/dashboards in the learning guides
describe local use and may not resolve in a source-only GitHub checkout.

The dependency list is requirements.txt; actual experiment versions are recorded
in run provenance. A source checkout supplies neither raw data nor trained weights. Transfer them separately when needed.


## Dataset exclusion

`data/raw/` is ignored. Only the data instructions and integrity manifest are
tracked. No Git LFS setup is needed for this submission.
