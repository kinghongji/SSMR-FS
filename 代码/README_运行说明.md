# Reproducing the SSMR-FS Temperature-Path Design Study

This repository contains the complete workflow used to reproduce the SSMR-FS temperature-path design results reported in the paper. The workflow covers candidate-path construction, candidate-constrained sequential energy selection, four baseline methods, evaluation using 210 laboratory refrigerator storage-temperature records, the four design-quality metrics, 30 independent repetitions, and output-size validation.

## 1. Repository layout

Keep the distributed directory structure unchanged:

```text
repository-root/
├─ 代码/
│  ├─ 210条实验室冰箱储藏温度记录.xlsx
│  ├─ config/
│  ├─ scripts/
│  ├─ src/
│  └─ README_运行说明.md
└─ 结果/                         # created automatically when the code is run
```

The input workbook is resolved relative to the `代码` directory. No machine-specific absolute path is required. All outputs are written to the sibling directory `../结果`.

## 2. Software requirements

Use a Python environment containing the following packages:

```text
numpy
pandas
scipy
scikit-learn
matplotlib
openpyxl
```

The workflow does not install, upgrade, or remove packages automatically.

## 3. Input data

The distributed input file is:

```text
./210条实验室冰箱储藏温度记录.xlsx
```

The records were obtained under laboratory refrigerator conditions and are used only to construct the common evaluation space. They do not participate in SSMR-FS candidate generation, selection-PCA fitting, or sequential energy selection, and they are not interpreted as the probability distribution of commercial cold-chain temperatures.

The workbook must retain the following structure:

- worksheet: `实际每日温度`;
- identifier column: `样本名称`;
- temperature columns: `第1天实际温度` to `第10天实际温度`.

The program checks identifier uniqueness, numeric validity, and internal missing values. Blank cells after the final valid daily temperature indicate that the storage record has ended.

## 4. Methods reproduced

The comparison includes five methods:

1. simple random sampling (SRS);
2. Sobol sequence;
3. orthogonal-array Latin hypercube sampling (OA-LHS);
4. maximum projection design (MaxPro);
5. SSMR-FS.

All methods use the same output size, nominal temperature range, storage-duration range, 24 path features, evaluation transformation, and four evaluation metrics.

SSMR-FS contains two stages. First, typical-storage, stratified-exploration, and risk-exposure paths are generated and deduplicated. Second, the merged candidate set is represented by 24 path features, standardized, projected by PCA, whitened, and reduced through candidate-constrained sequential energy selection.

## 5. Evaluation metrics

The four metrics are reported separately:

| Metric | Abbreviation | Preferred direction |
|---|---|---|
| Path coverage error | PCE | Lower |
| Key temperature-zone coverage error | KCE | Lower |
| Path separation | PS | Higher |
| Log feature volume | LFV | Higher |

## 6. Running the complete workflow

Open a terminal in the `代码` directory and run:

```bash
python scripts/07_run_all.py
```

This command sequentially performs:

1. the five-method comparison at `N = 300`;
2. 30 independent repetitions at `N = 300`;
3. validation at `N = 200, 500, 800, 1000, 1300, 1500`.

The global base random seed is fixed at 42. The data file, result path, and module imports are resolved from the script locations, so the command can be run after cloning or downloading the repository without editing local paths.

## 7. Optional scripts

Individual parts of the workflow can be run separately:

```text
scripts/01_generate_baselines.py
scripts/02_generate_ssmr_fs.py
scripts/03_generate_evaluation_sets.py
scripts/04_main_comparison.py
scripts/05_repeated_analysis.py
scripts/06_sample_size_validation.py
```

## 8. Generated results

The workflow creates the following directories under `../结果`:

```text
01_运行配置与方法参数
02_SSMR候选路径与选择空间
03_SSMR序贯能量选择结果
04_储藏温度记录与评价空间
05_N300五种方法主比较
06_N300五种方法30次重复
07_五种方法不同输出数量验证
08_论文图
09_运行日志
```

`01_运行配置与方法参数` contains the portable runtime configuration. The paper figures in `08_论文图` are generated as both 600-dpi PNG files and vector PDF files. Re-running the workflow updates files with the same names in these result directories.

## 9. Paper figures

The workflow directly generates the final figure layouts used in the paper:

- storage-temperature record heatmap;
- five-method temperature-path heatmaps;
- six-panel distribution in the common evaluation space;
- cumulative explained-variance curve for the evaluation PCA;
- two-dimensional design-quality comparison;
- 30-run stability curves;
- output-size performance curves.

All figure text is in English, and simple random sampling is displayed as `SRS`.
