# Implementation and reproducibility notes

This is a portfolio fork of the [team repository](https://github.com/jiwoong218/text-deepfake-detection). The competition outcome and combined pipeline are team achievements. A component-level account of Seong-U IM's individual contribution has not yet been documented here.

## What the checked-in code supports

| Component | Evidence | Scope |
| --- | --- | --- |
| KoELECTRA classifier | `src/modeling.py`, `src/train.py` | Standard classification and a custom hard-first loss wrapper |
| Curriculum-style loss weighting | `src/superloss.py` | Includes `HardFirstSuperLoss`; this does not establish each variant's contribution to competition performance |
| Class subsampling | `BalancedSampler` in `src/train.py` | Randomly retains 40% of class 0 and all class 1 by default; not a verified boundary-based sampling implementation |
| Long-text inference | `create_overlapping_windows` in `src/inference.py` | Optional overlapping windows with maximum class-1 probability aggregation |
| Ensemble | `src/ensemble.py` | Arithmetic mean of probabilities; assumes identical row order across input CSVs |
| Tokenization | `ElectraTokenizer` in preprocessing and inference | The release does not provide a separate custom BPE training pipeline |

## Reproduction status

The scripts preserve a competition workflow, but the README's original end-to-end commands are **not currently validated as a complete working recipe**. Specific issues visible in the source are:

1. `data_preparation.py` writes a `DatasetDict` with `save_to_disk` and `train` / `eval` splits. `train.py` uses `load_dataset` and then requests a `test` split. These interfaces need to be reconciled before using the generic training path.
2. `inference.py` checks `os.path.exists(checkpoint)` before loading. Its current implementation requires a **local checkpoint directory**, even though the historical README gives a Hugging Face model ID example.
3. Inference tokenization pads to the longest sequence within each dataset mapping batch, while the DataLoader uses default collation. Variable sequence lengths spanning mapping batches can therefore fail to stack. This has not been corrected or used to regenerate results in this documentation pass.
4. `src/superloss.py` directly imports SciPy. The dependency is now explicitly listed; the environment remains unpinned and an original competition environment lock is unavailable.
5. Preprocessing splits paragraphs after exploding documents. Paragraphs from the same original document may appear in both training and validation. This validation protocol must not be described as a held-out-document evaluation.
6. Training retains only sequences whose tokenized length is below 512, while long-text inference is a separate option. The effects of filtering and windowing need their own experiment records.
7. `train_super_ada.py` is an experiment script with top-level execution. Inspect its paths and settings before running or importing it.

The training loop and experiment semantics have been preserved. Resolving data interfaces, validating dynamic padding, and adding a small data-to-inference smoke run are the next engineering steps. A full competition reproduction additionally requires permitted data, the exact split, model checkpoints, environment versions, and the final ensemble membership.

## Data, artifacts, and attribution

The raw competition data, processed datasets, and model weights are not included in this checkout. The `models/` subdirectories contain placeholder files, not released checkpoints. Obtain data and models through their authorized distribution channels; do not infer redistribution permission from their mention in this repository.

The portfolio owner's account reports **7th place out of 279 teams and a Sponsored Company Award** in the 2025 SW-Centered University Digital Competition AI Track. No leaderboard or award document is included here, so these are project-context claims rather than metrics reproduced by this checkout.

No top-level license is present. Licensing and ownership should be resolved with the original team before adding a redistribution license. This fork does not assign the full system or award to one contributor.
