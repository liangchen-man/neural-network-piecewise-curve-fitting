# Neural Network Piecewise Modeling and Curve Fitting

This repository contains the reproducible source code for one-dimensional ReLU
piecewise analysis, network composition, clipping functions, matrix-form neural
networks, and smooth curve-fitting experiments.

## Included

- Three executed teaching notebooks covering network composition, clipping, and
  matrix representations.
- Exact one-dimensional piecewise-affine analysis and unit tests.
- Data-audit and curve-fitting source code.
- A Python port of the small MATLAB ReLU demonstrations.

## Deliberately excluded

No raw measurement CSV, images, annotations, derived real-data slices, model
outputs, local paths, student identifiers, or report-writing materials are in
this public repository. To run a real-data audit, provide only data that you are
authorized to use locally; generated `data_small/` and `outputs/` are ignored by
Git.

## Core environment and checks

```bash
python -m pip install -r requirements-core.txt
python -m pytest -q
jupyter nbconvert --to notebook --execute notebooks/4_1_Composing_Networks.ipynb --output executed_4_1_Composing_Networks.ipynb
```

Run the other two notebooks the same way. The core notebooks require only NumPy
and Matplotlib in addition to the Jupyter execution packages.

## Optional local curve-fitting workflow

```bash
python -m pip install -r requirements-extension.txt
python extension/inspect_data.py --csv path/to/authorized_measurement.csv --out .
python extension/run_curve_fitting.py --config configs/curve_fitting.json
```

The audit script expects the wide alternating-coordinate CSV schema documented in
its source. The curve-fitting script is intended for locally authorized data and
does not ship any measurement data or pretrained outputs.

## Source attribution

The notebook headers retain their links to the original *Understanding Deep
Learning* teaching materials. Please follow the relevant course and source
material licensing terms when redistributing or adapting them.
