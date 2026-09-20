# Layered sandstone Mode III fracture: PINN source and inputs

Source code and the experimental inputs used by the PINN in
*Investigating bedding effects on Mode III fracture of sandstone using physics
informed neural networks*, by Pengcheng Zhu and Tielin Chen, Beijing Jiaotong
University.

## Package contents

- Measured peak loads for the 27 notched specimens, the identification/assessment
  split, and the measured DIC strip-gauge observations used by the inverse model.
- Specimen geometry, bulk properties, documented postfracture frame exclusions,
  and five starting cohesive-law coefficients for each bedding orientation.
- The elastic-operator builder, neural solvers, continuation, shared-law inverse,
  numerical exporter and their Python dependencies.
- Configurations for the three selected orientation fits and the initialization
  stages they require. The execution order is in `reproduction.json`.

The gauge table is a processed experimental input, with displacement in
micrometres, load fractions and recorded gauge locations. It is not a simulated
response. The starting cohesive-law coefficients are the fixed initialization
values from the preceding calibration; they are not the final PINN estimates.
The fixed 8.9 micrometre gauge scale was estimated from pooled experimental
gauges, including available replicate-3 gauges. Replicate-3 peak loads are not
used in neural fitting.

No trained weights, operator caches, run logs, computed fields, numerical
results, historical analyses or finished figures are included. Full image
recordings, photographs and full-field DIC exports are outside this minimal
model-input package.

## Check and run

Use Python 3.13. From the repository root (or the extracted `CG_Data_Code/` directory):

```console
python verify_release.py
python -m pip install -r requirements.txt
python reproduce.py
```

The last command only prints the ordered commands. To build the three elastic
operators and then train all required neural stages:

```console
python reproduce.py --all
```

Alternatively, run `python reproduce.py --operators` and the individual
`--run STAGE` commands printed by the listing. Configurations use CUDA,
double precision and seed 17. The recorded training machine had a 32 GB GPU.
Operator assembly runs on the CPU. Existing output directories are preserved;
use a fresh extraction to start another complete calculation.

This package runs the current implementation from experimental inputs.
Recomputing operators and retraining are necessary because the saved solutions
are omitted. Numerical trajectories can differ from those of the original
runs, including the earlier 45-degree reference convention.

After training, the selected results are in:

| Bedding (degrees) | Directory under `C-G/experiments/interface_pinn/runs/` |
| --- | --- |
| 0 | `b0_family_seed17_c2_initialized_inverse` |
| 45 | `b45_family_seed17_reference` |
| 90 | `b90_family_seed17_inverse` |

The optional replay/export scripts under `C-G/src/` consume these newly
computed results. Run them with `C-G/src` and the archive-root `src` on
`PYTHONPATH`.

## Physical scope

The neural formulation jointly identifies two tangential interface-slip fields
and a shared cohesive law across three notch depths. The bulk uses isotropic
elasticity with E = 14.3 GPa and Poisson's ratio 0.28. The orientation-dependent
laws are effective interface estimates under these common bulk properties.
The 0-degree law uses a compact C2 closure over the last 5% of its softening
interval and local Gaussian field enrichment.

The retained gauge/frame labels are the experimental inputs used by the
model. The 0-4-1 frame-1016 label has not been independently established as
prepeak. The documented postfracture exclusion begins at frame 1339. The
loading rate is 0.150 mm/min. Recomputing the model does not resolve the
unverified acquisition timing.
