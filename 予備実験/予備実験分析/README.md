# Steering Experiment Analysis

This folder collects the pasted notebook-style analysis into one script.

## Folder Layout

- `analysis.py`: main analysis script
- `input/`: put experiment CSV files here
- `output/`: generated tables, figures, and `summary.txt`

The script reads every `.csv` file under `input/`, including files inside participant folders such as `input/ID001/*.csv`.

## Usage

```bash
python3 analysis.py
```

To recreate the output folder from scratch:

```bash
python3 analysis.py --clean-output
```

Useful options:

```bash
python3 analysis.py --block-size 25
python3 analysis.py --outlier-sigma 3
python3 analysis.py --diagonal-angle-deg 30
```

## Outputs

Tables are written to `output/tables/`.
Figures are written to `output/figures/`.
The short run summary is written to `output/summary.txt`.

## Dependencies

```bash
python3 -m pip install -r requirements.txt
```

`scipy` is used for learning-curve fitting. If it is not installed, the script still writes the other tables and plots, but fit parameters are left blank.
