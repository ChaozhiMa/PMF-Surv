# PMF-Surv

PMF-Surv estimates overall survival (OS) and progression-free survival (PFS) in patients with unresectable hepatocellular carcinoma (HCC) receiving immune checkpoint inhibitor (ICI)-based combination therapy. The model uses pretreatment arterial-phase (AP), portal venous-phase (PVP), and delayed-phase (DP) CT together with 17 clinical variables.

## Model

1. **Cross-phase fusion.** A shared 3D Swin Transformer encodes the three CT phases. Regional feature stability ranks govern the strength of feature exchange across phases.
2. **Clinical-information-conditioned aggregation.** One-sided unbalanced optimal transport associates cross-phase regional CT features with individual clinical variables. The model retains a separate imaging summary for each variable.
3. **Endpoint-specific survival readout.** OS and PFS interval queries attend to clinical features, clinically conditioned imaging summaries, and global imaging features. Separate multitask logistic regression (MTLR) heads estimate the two survival distributions over 12 intervals per endpoint.

## External validation

The retrospective study included 839 patients from four centers. The development cohort comprised 508 training and 127 internal validation patients. Three external cohorts comprised 204 patients.

| External cohort | Patients | OS C-index | PFS C-index |
| --- | ---: | ---: | ---: |
| Center 2 | 126 | 0.7951 ± 0.0038 | 0.7365 ± 0.0050 |
| Center 3 | 47 | 0.8126 ± 0.0206 | 0.7390 ± 0.0073 |
| Center 4 | 31 | 0.7471 ± 0.0183 | 0.7269 ± 0.0334 |

Values are the mean ± sample SD of the Harrell-type C-index across five training seeds (2026–2030), as reported in Table 1 of the manuscript. PMF-Surv achieved the highest mean C-index among the six comparators for both endpoints in each external cohort. The manuscript also reports time-dependent AUC, risk stratification, calibration, and integrated Brier scores.

## Setup

Use Python 3.8 or later and a CUDA-capable NVIDIA GPU. If you downloaded the source archive, extract it and run the commands from the folder containing `pyproject.toml`.

```bash
python -m pip install -r requirements.txt
```

| Path | Contents |
| --- | --- |
| `config/hard_top1.yaml` | Model, data, and training configuration |
| `scripts/train.py` | Model training |
| `scripts/evaluate.py` | Checkpoint evaluation |
| `src/restot_surv/` | Model, data loading, training, and evaluation code |

## Prepare the inputs

- **CT volumes.** Supply one preprocessed 4D volume per patient, with phases ordered AP, PVP, and DP. The model uses a `3 × 128 × 128 × 128` input. The study segmented the liver, registered AP and DP to PVP, resampled to 1 mm isotropic resolution, clipped intensities to −200 to 250 HU, applied contrast-limited adaptive histogram equalization (CLAHE), cropped to the liver bounding box, and resized the volume.
- **Clinical and outcome tables.** Set the paths to the training, internal validation, and test CSVs in `config/hard_top1.yaml`. Each CSV needs a unique patient `ID`, the 17 configured clinical variables, and the endpoint columns `OS_time`, `OS_status`, `PFS_time`, and `PFS_status`. Event indicators are 1 for observed events and 0 for censored observations. Configure the center membership CSVs for center-level external evaluation.

## Train and evaluate

Update the paths in `config/hard_top1.yaml` to match your preprocessed volumes and CSV files.

```bash
python scripts/train.py --config config/hard_top1.yaml --output-dir outputs/pmf-surv
python scripts/evaluate.py --config config/hard_top1.yaml --checkpoint outputs/pmf-surv/best_os.pth --split test
```

Training selects `best_os.pth` using the internal validation OS C-index. Use `--split val` to evaluate the internal validation cohort. Both commands use the configured train, validation, and test splits.

## Data and code availability

The patient-level data used in the study are available from the corresponding author upon reasonable request. The source code will be made publicly available upon acceptance of the manuscript.

## Citation

Chaozhi Ma, Junjie Wang, Ziming Zong, Zhengqin Huang, Shaobin Wang, Tengfei Wang, Hai Li, and Jun Xu. “PMF-Surv: Progressive Multimodal Fusion with Clinical-Information-Conditioned Aggregation of Multiphase CT for Survival Prediction in Immunotherapy-Treated Unresectable HCC.” Manuscript.
