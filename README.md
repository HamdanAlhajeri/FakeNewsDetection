# Fake News Detection

A 6-class political statement truthfulness classifier built on the LIAR2 dataset.
Uses the Tinker API to LoRA fine-tune **NVIDIA Nemotron 30B** on remote GPU infrastructure —
no local GPU required.

---

## How It Works

### Overview

The system classifies political statements into one of six truthfulness labels:

| Label | Meaning |
|---|---|
| `true` | Accurate and fully supported |
| `mostly-true` | Mostly accurate with minor omissions |
| `half-true` | Partially accurate but leaves out key facts |
| `barely-true` | Contains a grain of truth but is mostly misleading |
| `false` | Factually inaccurate |
| `pants-fire` | Egregiously false ("pants on fire") |

### Dataset — LIAR2

[LIAR2](https://www.cs.ucsb.edu/~william/liar.html) is a benchmark of ~23,000 professionally
fact-checked political statements sourced from PolitiFact. Each record contains:

- The statement text
- Speaker name, job title, party affiliation
- Topic subject and geographic context
- Historical credibility counts for that speaker (how many times they have been rated each label)

The dataset ships as three TSV files with no header row:

```
data/train.tsv   (~10,269 samples)
data/valid.tsv   (~1,284  samples)
data/test.tsv    (~1,283  samples)
```

### Feature Engineering

Each sample is transformed into a single rich text string before being fed to the model.
Six text features are concatenated:

```
Statement: <cleaned statement text>
Speaker:   <speaker name>
Party:     <party affiliation>
Job:       <job title>
Subject:   <topic area>
Context:   <location / context>
```

On top of that, a **speaker credibility score** is derived from the historical count columns
and appended automatically:

```
Speaker history: 473 prior claims — mostly-true: 34%, half-true: 34%,
false: 15%, barely-true: 15%, pants-fire: 2%. Most common rating: mostly-true
```

This gives the model explicit signal about each speaker's track record rather than
requiring it to infer credibility from a name alone. Speakers with no history are
silently skipped.

**Text cleaning** applied to all fields before concatenation:
- Lowercase
- Strip URLs and email addresses
- Remove special characters (keep alphanumeric and spaces)
- Remove English stopwords (NLTK)

### Model — Nemotron 30B + LoRA via Tinker

The model is `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, a 30-billion-parameter
instruction-tuned LLM. Fine-tuning is done using **LoRA** (Low-Rank Adaptation) via the
Tinker API, which runs on Tinker's remote GPU infrastructure.

**How the classification works:**

Each sample is formatted as a completion prompt:

```
Classify the truthfulness of the following political statement and its context.

Statement: hillary clinton agrees john mccain voting give george bush ...
Speaker: barack obama  Party: democrat  Job: president  Subject: foreign-policy
Context: denver
Speaker history: 473 prior claims — mostly-true: 34%, half-true: 34%, ...

Choose exactly one label: barely-true, false, half-true, mostly-true, pants-fire, true

Label:
```

The model is trained to complete this prompt with the correct label (e.g. `" half-true"`).
During inference, the model generates a label token and it is matched to the nearest
known label.

**LoRA training setup (v3 config — current best):**

| Parameter | Value |
|---|---|
| Base model | nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 |
| LoRA rank | 8 |
| Learning rate | 2e-5 |
| Batch size | 8 |
| Epochs | 3 |
| Optimizer | AdamW |

### Training Flow

```
LIAR2 TSV files
    ↓
DataProcessor.load_data()        — parse TSV with named columns
    ↓
DataProcessor.process()          — clean text + build_input_text + encode labels
    ↓
DataProcessor._speaker_history() — derive credibility stats from count columns
    ↓
TinkerClassifier.connect()       — connect to Tinker API
    ↓
create_training_client()         — allocate LoRA training resources (rank=8)
    ↓
DataProcessor.prepare_tinker_dataset() — tokenize + build Datum objects
    ↓
TinkerTrainer.train()            — mini-batch forward-backward + AdamW steps
    ↓
TinkerClassifier.save_for_inference() — save LoRA weights, get URI
    ↓
Evaluate on test set             — predict each sample, compute metrics
    ↓
Save artifacts                   — results.json, loss_curve.png, confusion_matrix.png
```

---

## Preprocessing

### Why It Is a Separate Step

Preprocessing is run once and its output is saved to disk. Every subsequent training run
reads those saved files directly, skipping all cleaning and feature engineering. This means
you can retrain with different hyperparameters as many times as you like without repeating
the same work. If you change the features or cleaning logic, you re-run preprocessing once
and all future training runs automatically use the new version.

```
python src/preprocessing.py   ← run once
python src/train.py           ← run as many times as needed
```

### What Preprocessing Produces

Three tab-separated files, one per data split:

```
artifacts/preprocessed_train.tsv   (10,269 rows)
artifacts/preprocessed_val.tsv     (1,284  rows)
artifacts/preprocessed_test.tsv    (1,283  rows)
```

Each file has exactly two columns with a header row:

| Column | Content |
|---|---|
| `text` | The fully cleaned and combined input string ready for the model |
| `label` | The truthfulness label string, e.g. `false` |

### Step-by-Step Walkthrough

The entire preprocessing pipeline lives in `src/data_processor.py` inside the
`DataProcessor` class. `src/preprocessing.py` calls into it and handles saving.

---

#### Step 1 — Load the raw TSV (`DataProcessor.load_data`)

The LIAR2 TSV files have no header row, so column names are assigned explicitly from
`TSV_COLUMNS` in `config.py`. The 14 columns in order are:

```
id, label, statement, subject, speaker, job_title, state_info,
party_affiliation, barely_true_count, false_count, half_true_count,
mostly_true_count, pants_on_fire_count, context
```

Malformed lines (wrong number of columns) are silently skipped. Any row missing a
`label` or `statement` value is dropped immediately before any further processing.
The `label` column is lowercased and stripped of whitespace.

**Raw input example (one row of train.tsv):**
```
2635.json  false  Says the Annies List political group supports
           third-trimester abortions on demand.  abortion
           dwayne-bohac  State representative  Texas  republican
           0  1  0  0  0  a mailer
```

---

#### Step 2 — Clean each text field (`DataProcessor.clean_text`)

`clean_text` is applied independently to every text field before they are combined.
The transformations happen in this exact order:

1. **Lowercase** — `"Says the Annies List..."` → `"says the annies list..."`
2. **Strip URLs** — removes anything matching `http://`, `https://`, or `www.`
3. **Strip email addresses** — removes anything matching `word@word` patterns
4. **Remove special characters** — replaces everything that is not a letter, digit,
   or space with a space. Punctuation, quotes, hyphens, apostrophes are all removed.
5. **Remove English stopwords** — common words like `the`, `is`, `a`, `on`, `of`
   are removed using the NLTK English stopwords list (179 words). This reduces noise
   and shortens sequences without losing meaning-bearing content.

If a field is null, not a string, or becomes empty after cleaning, it is silently
skipped — no error, no placeholder.

**Example — cleaning the statement field:**
```
Before: "Says the Annies List political group supports third-trimester abortions on demand."
After:  "says annies list political group supports third trimester abortions demand"
```

---

#### Step 3 — Derive speaker credibility history (`DataProcessor._speaker_history`)

The LIAR2 dataset contains five columns that record how many times each speaker has
previously received each truthfulness rating:

| Column | Rating |
|---|---|
| `barely_true_count` | barely-true |
| `false_count` | false |
| `half_true_count` | half-true |
| `mostly_true_count` | mostly-true |
| `pants_on_fire_count` | pants-fire |

These counts are summed to get the speaker's total number of prior fact-checked claims.
Each rating is then expressed as a percentage of that total. The result is sorted
from most frequent to least frequent so the most characteristic pattern appears first.

This gives the model explicit, structured signal about each speaker's credibility track
record. Without it, the model can only infer trustworthiness from a name — with it,
the model is directly told that this speaker is rated `false` 40% of the time.

**Example — Barack Obama row:**
```
Counts:  mostly_true=163, half_true=160, false=71, barely_true=70, pants_on_fire=9
Total:   473 claims

Output:  "Speaker history: 473 prior claims — mostly-true: 34%, half-true: 34%,
          false: 15%, barely-true: 15%, pants-fire: 2%. Most common rating: mostly-true"
```

Speakers with all-zero counts (no prior history) produce an empty string and the
speaker history segment is omitted entirely from that sample's input.

---

#### Step 4 — Combine all features (`DataProcessor.build_input_text`)

All six text feature columns are cleaned (Step 2) and concatenated into a single
string with a descriptive prefix for each field:

```
Statement: <cleaned statement>
Speaker:   <cleaned speaker name>
Party:     <cleaned party affiliation>
Job:       <cleaned job title>
Subject:   <cleaned subject>
Context:   <cleaned context>
```

The speaker history string from Step 3 is then appended at the end. Fields that are
null or empty after cleaning are simply omitted — there is no blank placeholder.

**Full combined output for the example row:**
```
Statement: says annies list political group supports third trimester abortions demand
Speaker: dwayne bohac Party: republican Job: state representative Subject: abortion
Context: mailer Speaker history: 1 prior claims — false: 100%. Most common rating: false
```

This single string is what gets written to the `text` column of the preprocessed TSV
and what the model receives as its input at training and inference time.

---

#### Step 5 — Encode labels (`DataProcessor.encode_labels`)

Label strings are mapped to integer indices using the fixed alphabetical ordering
defined in `config.py`:

| Index | Label |
|---|---|
| 0 | barely-true |
| 1 | false |
| 2 | half-true |
| 3 | mostly-true |
| 4 | pants-fire |
| 5 | true |

The integer indices are used internally for metric computation (confusion matrix,
F1 scores). The `label` column in the preprocessed TSV keeps the original string
form so the files remain human-readable.

---

#### Step 6 — Save to TSV (`preprocessing.py`)

After all rows have been processed, the combined text and label string are written
to a tab-separated file with a header row. Any row whose combined text is still
empty after all cleaning steps is dropped at this point.

The three output files mirror the original train/val/test split structure so that
`train.py` can load them with a single `pd.read_csv(path, sep='\t')` call.

---

### Concrete Before / After Example

**Raw TSV row (train.tsv):**
```
324.json  mostly-true  Hillary Clinton agrees with John McCain "by voting
to give George Bush the benefit of the doubt on Iran."  foreign-policy
barack-obama  President  Illinois  democrat  70  71  160  163  9  Denver
```

**After preprocessing (one row of preprocessed_train.tsv):**
```
text                                                                    label
Statement: hillary clinton agrees john mccain voting give george bush   mostly-true
benefit doubt iran Subject: foreign policy Speaker: barack obama
Party: democrat Job: president Context: denver Speaker history: 473
prior claims — mostly-true: 34%, half-true: 34%, false: 15%,
barely-true: 15%, pants-fire: 2%. Most common rating: mostly-true
```

---

## Results

Trained with the v3 configuration (rank=8, lr=2e-5, batch=8, epochs=3):

| Metric | Value |
|---|---|
| Test Accuracy | **71.59%** |
| Macro F1 | **0.717** |
| Weighted F1 | — |

**Loss progression over 3 epochs:**

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 0.732 | 0.606 |
| 2 | 0.420 | 0.440 |
| 3 | 0.117 | 0.474 |

The gap between train and val loss at epoch 3 indicates mild overfitting — the model
has learned the training distribution well but generalises with some degradation. The
v4 experiment (smaller batch size = noisier gradients) was designed to address this.

---

## Project Structure

```
FakeNewsDetection/
├── src/
│   ├── config.py           — all constants: labels, Tinker config, paths, feature columns
│   ├── data_processor.py   — DataProcessor: load, clean, feature engineering, Tinker datums
│   ├── preprocessing.py    — run once: processes raw TSVs and saves preprocessed TSVs
│   ├── models.py           — TinkerClassifier: connect, train, save, predict
│   ├── training_utils.py   — TinkerTrainer, MetricsCalculator, plotting, save_results
│   ├── train.py            — training entry point (loads preprocessed TSVs, trains)
│   └── predict.py          — inference entry point + TinkerPredictor class
│
├── data/
│   ├── train.tsv           — ~10,269 raw training samples
│   ├── valid.tsv           — ~1,284  raw validation samples
│   └── test.tsv            — ~1,283  raw test samples
│
├── artifacts/
│   ├── preprocessed_train.tsv   — cleaned + combined training data (text, label)
│   ├── preprocessed_val.tsv     — cleaned + combined validation data
│   ├── preprocessed_test.tsv    — cleaned + combined test data
│   ├── runs/                    — one JSON file per training run
│   ├── runs_registry.json       — index of all training runs with metrics
│   ├── tinker_weights_uri.txt   — URI of the most recent training run
│   ├── training_results.json    — loss history + test metrics (latest run)
│   ├── loss_curve.png           — train/val loss per epoch
│   └── confusion_matrix.png     — per-class prediction breakdown
│
├── notebooks/
│   ├── 01_data_loading_cleaning_encoding.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_training_execution.ipynb
│
├── requirements.txt         — core dependencies (pandas, sklearn, nltk, matplotlib)
└── requirements_models.txt  — model dependencies (torch, transformers, tinker)
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install -r requirements_models.txt
```

### 2. Set your Tinker API key

Create a `.env` file in the project root:

```env
TINKER_API_KEY=your_key_here
```

`train.py` and `predict.py` load this file automatically via `python-dotenv` before
connecting to Tinker. No need to set environment variables manually.

### 3. Verify the data files exist

```
data/train.tsv
data/valid.tsv
data/test.tsv
```

---

## How to Train

### Step 1 — Preprocess (once)

```bash
python src/preprocessing.py
```

Reads the raw TSV files, applies all cleaning and feature engineering, and saves
three preprocessed TSVs to `artifacts/`. Only needs to be re-run if you change
the raw data or modify the cleaning / feature logic.

### Step 2 — Train (as many times as needed)

```bash
python src/train.py
```

What this does:
1. Loads the preprocessed TSVs from `artifacts/` (no cleaning or feature work)
2. Connects to the Tinker API and allocates a LoRA training session
3. Tokenises all samples into Tinker Datum objects
4. Runs 3 epochs of mini-batch LoRA fine-tuning
5. Saves the LoRA weights and records the URI
6. Predicts all test samples and computes metrics
7. Saves `training_results.json`, `loss_curve.png`, `confusion_matrix.png`
8. Appends this run to `artifacts/runs_registry.json`

Training takes roughly **1–3 hours** depending on Tinker queue and dataset size.
Progress is printed every 50 batches and at the end of each epoch.

---

## How to Predict

After training, the weights URI is stored in `artifacts/tinker_weights_uri.txt`.
The predict script reads it automatically.

**Single statement:**
```bash
python src/predict.py --text "The unemployment rate dropped to 3 percent under my administration."
```

**With per-label log-probabilities:**
```bash
python src/predict.py --text "..." --proba
```

**Predict from a CSV or TSV file:**
```bash
python src/predict.py --file statements.csv --col statement
python src/predict.py --file statements.tsv --col text --out results.tsv
```

**List all saved training runs:**
```bash
python src/predict.py --list-runs
```

**Use a specific past run:**
```bash
python src/predict.py --run run_20260407_143022 --text "..."
```

**Explicit URI:**
```bash
python src/predict.py --uri 1731a955-6aca-5d6a-a18f-b3914a1e4b47:train:0 --text "..."
```

**Example output:**
```
Loaded model from URI: 1731a955-6aca-5d6a-a18f-b3914a1e4b47:train:0

Text:       The unemployment rate dropped to 3 percent under my administration.
Prediction: mostly-true

Per-label log-probabilities:
  mostly-true    -0.3821
  half-true      -1.2044
  true           -1.8732
  barely-true    -3.1209
  false          -4.5512
  pants-fire     -6.2103
```

---

## Configuration

All tunable parameters live in `src/config.py`.

```python
# Tinker / LoRA
TINKER_CONFIG = {
    'base_model':   'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16',
    'lora_rank':    8,       # higher = more trainable params
    'learning_rate': 2e-5,
    'batch_size':   8,
    'epochs':       3,
}

# Prompt template — {text} is the full combined feature string
PROMPT_TEMPLATE = (
    "Classify the truthfulness of the following political statement "
    "and its context.\n\n"
    "{text}\n\n"
    "Choose exactly one label: barely-true, false, half-true, "
    "mostly-true, pants-fire, true\n\n"
    "Label:"
)

# Which text columns to concatenate into the model input
FEATURE_COLS = ['statement', 'speaker', 'job_title',
                'party_affiliation', 'subject', 'context']
```

---

## Module Reference

### `src/preprocessing.py`

| Function | Description |
|---|---|
| `run_and_save()` | Runs the full preprocessing pipeline and writes the three preprocessed TSVs |
| `_process_and_save(raw_path, out_path)` | Processes one split: loads, cleans, combines features, saves |

Run as a script: `python src/preprocessing.py`

---

### `src/data_processor.py` — DataProcessor

| Method | Description |
|---|---|
| `load_data(filepath)` | Load a LIAR2 TSV file into a DataFrame |
| `clean_text(text)` | Lowercase, strip URLs/emails/special chars, remove stopwords |
| `_speaker_history(row)` | Derive credibility stats from count columns |
| `build_input_text(row)` | Combine all features + speaker history into one string |
| `encode_labels(labels)` | Map label strings to 0–5 integer indices |
| `process(df)` | Full pipeline: returns `(texts, label_indices)` |
| `prepare_tinker_datum(text, label, tokenizer, template)` | Build a single Tinker Datum |
| `prepare_tinker_dataset(texts, labels, tokenizer, template)` | Build all Tinker Datums |

### `src/models.py` — TinkerClassifier

| Method | Description |
|---|---|
| `connect()` | Connect to Tinker API (reads `TINKER_API_KEY` from env) |
| `create_training_client(lora_rank)` | Allocate LoRA training session |
| `get_tokenizer()` | Get model tokenizer (cached) |
| `train_step(datums, learning_rate)` | One forward-backward + AdamW step |
| `save_for_inference(name)` | Save LoRA weights, returns URI string |
| `load_sampling_client(uri)` | Load a saved checkpoint for inference |
| `predict(text, ...)` | Predict label for one statement |
| `predict_batch(texts)` | Predict labels for a list of statements |

### `src/training_utils.py`

| Class / Function | Description |
|---|---|
| `TinkerTrainer` | Training loop: `train_epoch()`, `evaluate_loss()`, `train()` |
| `MetricsCalculator.calculate()` | Accuracy, macro F1, weighted F1, confusion matrix, report |
| `plot_curves(history)` | Save loss curve PNG |
| `plot_confusion_matrix(y_true, y_pred)` | Save confusion matrix PNG |
| `save_results(history, test_metrics)` | Write JSON results file |

---

## Troubleshooting

**`tinker SDK not installed`**
```bash
pip install tinker
```

**`No URI found at artifacts/tinker_weights_uri.txt`**
Run `python src/train.py` first. The URI is written automatically at the end of training.

**`Unknown label: 'nan'`**
A row in the TSV has a missing label. The `process()` method drops rows with null
`label` or `statement` columns — this should not occur with the standard LIAR2 files.

**Tinker connection error**
- Confirm `TINKER_API_KEY` is set in your environment
- Check your internet connection
- Verify the key is valid in the Tinker dashboard

---

## References

- LIAR2 Dataset — Wang et al. (2017), extended by Alhindi et al. (2018)
- [LIAR Dataset paper](https://arxiv.org/abs/1705.00648)
- [Tinker API](https://tinker.thinkingmachines.ai)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
