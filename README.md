# SSMR-FS temperature-profile design

This directory contains the code for generating candidate storage-temperature profiles, selecting SSMR-FS profiles, producing four comparator designs, and evaluating the resulting designs. The measured temperature data are read from `measured_storage_temperature_records.xlsx`. The workbook has English sheet and column labels; all sample identifiers and measured data cells are preserved. Generated tables also use English labels.

## Reproduce the analysis

Use Python 3.9.24 and set this `code` directory as the project root. Install the directly used dependencies with `python -m pip install -r requirements.txt`, then run:

```text
scripts/07_run_all.py
```

The script writes results to the sibling `results` directory. It produces the main comparison at N=300, 30 repeated comparisons, and comparisons at N=200, 500, 800, 1000, 1300, and 1500. The numerical analyses use the configured master seed of 42 and distinct deterministic streams for methods and repetitions.

## Results layout

- `01_configuration_and_rules`: run configuration, named random streams, and path-generation rules.
- `02_candidate_profiles_and_selection_space`: the three candidate components and their merged unique candidate set.
- `03_ssmr_fs_sequential_selection`: selected profiles and selection details.
- `04_measured_records_and_evaluation_space`: measured records and evaluation features.
- `05_N300_main_comparison`: main-run profiles, metrics, and comparator details.
- `06_N300_repeated_runs`: profiles and metrics for the 30 runs.
- `07_sample_size_comparison`: profiles, features, selection details, and metrics by design size.
- `08_manuscript_figures`: Figures 2–8 as 600 dpi PNG and vector PDF. Figure 1 is prepared separately for the manuscript.
- `09_run_logs`: run logs.
- `10_manuscript_tables_and_supplements`: Table 6–7 export and single-worksheet Supplementary Tables S1–S16.

The supplementary tables are paired by component: S1–S2 for typical storage scenarios, S3–S4 for stratified exploration, S5–S6 for risk exposure, S7 for the 300 SSMR-FS profiles, S8 for the measured records, and S9–S16 for the four comparators' profiles and rules.

The 210 measured temperature records are used for the common evaluation transformation and evaluation. Candidate generation, the SSMR-FS selection transformation, and sequential selection use the generated candidate set. Numeric results should be reported from the files produced by the current local run.
