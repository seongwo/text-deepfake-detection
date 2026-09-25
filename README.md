# Korean AI-Generated Text Detection

2025 SW-Centered University Digital Competition, AI Track — **7th of 279 teams, Sponsored Company Award**.

This is my fork of the [team repository](https://github.com/jiwoong218/text-deepfake-detection). We worked on distinguishing AI-generated Korean text from human writing, with particular attention to class imbalance, long documents, and noisy labels. The competition result and combined pipeline are team achievements.

[한국어 설명](ko_README.md)

## Approach

| Problem | Component in this repository |
| --- | --- |
| Text classification | KoELECTRA |
| Uneven example difficulty | Hard-first loss weighting |
| Class imbalance | Class-based subsampling |
| Long inputs | Overlapping windows with maximum-score aggregation |
| Combining models | Mean of prediction probabilities |

The [implementation notes](docs/REPRODUCIBILITY.md) map each component to its source and describe the limits of the available experiment records. Individual component ownership has not yet been documented.

## Code

| File | Purpose |
| --- | --- |
| `src/data_preparation.py` | Paragraph processing, train/validation split, and tokenization |
| `src/train.py` | General training loop and class sampler |
| `src/modeling.py`, `src/superloss.py` | KoELECTRA loss wrapper and loss-weighting variants |
| `src/electra.py`, `src/train_9000.py`, `src/train_super_ada.py` | Experiment-specific training scripts |
| `src/inference.py` | Classification with optional overlapping windows |
| `src/ensemble.py` | Average scores from submission CSVs |
| `notebooks/3rd super_Ada.ipynb` | Original experiment notebook |

Competition data, processed datasets, and weights are not included. The folders under `models/` contain placeholders.

## Running the code

Install dependencies in a separate Python environment:

```bash
python -m pip install -r requirements.txt
```

The scripts preserve the competition workflow, but the generic training path needs two corrections before it can run: preprocessing saves a dataset with `save_to_disk`, while training calls `load_dataset`; preprocessing names the validation split `eval`, while training requests `test`. The original environment is also not pinned. See [reproduction notes](docs/REPRODUCIBILITY.md) before starting training.

Inference currently requires a **local checkpoint directory**, along with the test CSV and sample submission. A Hugging Face model ID alone is rejected by the current path check. Variable-length batching is another known issue; an end-to-end inference run remains unverified.

Existing prediction CSVs can be combined with:

```bash
python src/ensemble.py \
  --csvs sub_ada.csv sub_extra.csv sub_42_4000.csv sub_9000.csv \
  --output final_ensemble_submission.csv
```

Each file must have the same rows in the same order and a `generated` probability column.

## Evaluation notes

Preprocessing splits paragraphs after separating them from documents. Paragraphs from one document can therefore appear in both training and validation. This split should not be interpreted as an evaluation on unseen documents.

The competition ranking is a team result. Reproducing it requires the original data split, trained models, final ensemble selection, and evaluation setup. The public code alone does not contain all of these records.
