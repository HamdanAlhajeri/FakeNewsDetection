"""
================================================================================
FAKE NEWS DETECTION: COMPLETE PROJECT REVIEW & DOCUMENTATION
================================================================================

Project:  6-class fake news classification on LIAR2 dataset
Model:    LoRA fine-tuning of NVIDIA Nemotron-3-Nano-30B via Tinker API
Best:     71.59% accuracy, 0.717 macro-F1 (V3 experiment)
Dataset:  ~10,240 PolitiFact fact-checked political statements

This script walks through the ENTIRE pipeline with intermediate data outputs,
identifies issues, and suggests better alternatives at every stage.

Run:  python docs/project_review.py

If data/ and artifacts/ directories are not present, the script shows
EXPECTED outputs based on the notebook execution history.
================================================================================
"""

import sys
import os
import json
import warnings

# Add project root and src to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
os.chdir(PROJECT_ROOT)

import numpy as np

# Optional imports -- script still runs without these for display-only mode
try:
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import LabelEncoder
    from collections import Counter
    PLOTTING_AVAILABLE = True
    warnings.filterwarnings('ignore')
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (12, 6)
except ImportError as e:
    PLOTTING_AVAILABLE = False
    print(f"[NOTE] Some libraries not available ({e}). Charts will be skipped.")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'docs', 'review_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Check data availability ---
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'artifacts')
TRAIN_FILE = os.path.join(DATA_DIR, 'train.tsv')
DATA_AVAILABLE = os.path.isfile(TRAIN_FILE)
ARTIFACTS_AVAILABLE = os.path.isdir(ARTIFACTS_DIR) and os.path.isfile(
    os.path.join(ARTIFACTS_DIR, 'training_results.json'))

if not DATA_AVAILABLE:
    print("[NOTE] data/ directory not found. Showing EXPECTED outputs from notebook history.")
    print(f"       To run with live data, download LIAR2 dataset to: {DATA_DIR}/\n")
if not ARTIFACTS_AVAILABLE:
    print("[NOTE] artifacts/ not found. Training results based on saved experiment history.\n")

# Label definitions used throughout
LABEL_ORDER = ['true', 'mostly-true', 'half-true', 'barely-true', 'false', 'pants-fire']
LABEL_ALPHA = sorted(LABEL_ORDER)  # alphabetical = LabelEncoder order
COLOR_MAP = {
    'true': '#2ecc71', 'mostly-true': '#27ae60', 'half-true': '#f1c40f',
    'barely-true': '#e67e22', 'false': '#e74c3c', 'pants-fire': '#c0392b'
}

# Known counts from notebook execution (used as fallback)
KNOWN_LABEL_COUNTS = {
    'barely-true': 1654, 'false': 1995, 'half-true': 2114,
    'mostly-true': 1962, 'pants-fire': 839, 'true': 1676
}


# =============================================================================
# HELPERS
# =============================================================================

def section_header(num, title):
    print(f"\n{'=' * 80}")
    print(f"  SECTION {num}: {title.upper()}")
    print(f"{'=' * 80}\n")

def subsection(title):
    print(f"\n--- {title} ---\n")

def alternative_box(title, items):
    print(f"\n  [ALTERNATIVE] {title}")
    print(f"  {'~' * 60}")
    for item in items:
        print(f"    -> {item}")
    print()

def issue_box(title, description):
    print(f"\n  [ISSUE] {title}")
    print(f"  {'!' * 60}")
    for line in description.split('\n'):
        print(f"    {line}")
    print()

def save_chart(fig, filename):
    """Save chart and print path."""
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Saved] {path}")


# =============================================================================
# SECTION 1: PROJECT OVERVIEW
# =============================================================================

section_header(1, "Project Overview")

print("""
PROJECT SUMMARY
---------------
This project classifies short political statements from PolitiFact into 6
truthfulness categories using a fine-tuned large language model.

Architecture:
  Raw TSV data (LIAR2)
    -> DataProcessor (clean text, encode labels)
    -> ClassBalancer (oversample minority classes)
    -> DataSplitter (70/15/15 stratified split)
    -> Tinker LoRA fine-tuning (Nemotron-3-Nano-30B)
    -> Evaluation (accuracy, F1, confusion matrix)

6 Truthfulness Labels (ordered by truthfulness):
  true > mostly-true > half-true > barely-true > false > pants-fire

Key Results:
  +--------+-----------+--------+----------+--------+---------+
  | Version| LoRA Rank | LR     | Epochs   | Acc    | F1      |
  +--------+-----------+--------+----------+--------+---------+
  | V1     | 16        | 1e-4   | 3        | 71.9%  | 0.720   |
  | V3     | 8         | 2e-5   | 3        | 71.6%  | 0.717   |
  +--------+-----------+--------+----------+--------+---------+
  Note: V1 is severely overfit (train loss -> 0.015, val loss -> 0.56)
        V3 has mild overfitting starting at epoch 3

Project Structure:
  FakeNewsDetection/
  +-- data/                     # LIAR2 dataset (train.tsv, valid.tsv, test.tsv)
  +-- src/                      # Python modules
  |   +-- config.py             # Configuration & constants
  |   +-- data_processor.py     # Data ETL pipeline
  |   +-- models.py             # TinkerClassifier, ClassBalancer
  |   +-- training_utils.py     # DataSplitter, MetricsCalculator, TinkerTrainer
  |   +-- predict.py            # Inference API & CLI
  +-- notebooks/                # Jupyter execution files
  |   +-- 01_data_loading_cleaning_encoding.ipynb
  |   +-- 02_model_training.ipynb
  |   +-- 03_training_execution.ipynb
  +-- artifacts/                # Generated training artifacts
  +-- docs/                     # This review
""")

# --------------------------------------------------------------------------
# KNOWN ISSUES
# --------------------------------------------------------------------------

subsection("Known Issues Discovered During Review")

issue_box(
    "1. Label Display Bug in Notebook 01",
    "Notebook 01 prints '0 (true): 1654 samples' but LabelEncoder index 0\n"
    "is actually 'barely-true' (alphabetical sort). The count 1654 belongs\n"
    "to barely-true, NOT true. Display-only bug - doesn't affect training\n"
    "because Tinker training uses string labels directly."
)

issue_box(
    "2. Unused Pipeline Artifacts",
    "encoded_data.npz (custom vocab + padding) was built for traditional\n"
    "models (LSTM/CNN) but Tinker uses raw text with its own tokenizer.\n"
    "The vocabulary building, sequence encoding, and padding steps in\n"
    "notebook 01 are effectively dead code for the current training path."
)

issue_box(
    "3. Duplicate Oversampling",
    "Notebook 02 oversamples encoded integer sequences (from encoded_data.npz).\n"
    "Notebook 03 independently re-does oversampling on raw text labels.\n"
    "The notebook 02 oversampling is orphaned work."
)

issue_box(
    "4. Overfitting",
    "V1: SEVERE - train loss 0.64 -> 0.015 (43x drop), val loss 0.44 -> 0.56 (28% rise)\n"
    "V3: MILD - val loss turns up in epoch 3 (0.44 -> 0.47)\n"
    "No early stopping is implemented in either version."
)

issue_box(
    "5. Not Using Official Test Split",
    "Evaluates on a random 15% re-split of train.tsv instead of the\n"
    "provided test.tsv and valid.tsv files from LIAR2.\n"
    "Also, oversampling BEFORE splitting risks data leakage."
)


# =============================================================================
# SECTION 2: DATA LOADING & EXPLORATION
# =============================================================================

section_header(2, "Data Loading & Exploration")

liar2_columns = [
    'id', 'label', 'statement', 'subject', 'speaker', 'job_title',
    'state_info', 'party_affiliation', 'barely_true_count', 'false_count',
    'half_true_count', 'mostly_true_count', 'pants_on_fire_count',
    'context', 'justification'
]

df = None

subsection("2.1 Loading Raw LIAR2 Dataset")

if DATA_AVAILABLE:
    df = pd.read_csv(TRAIN_FILE, sep='\t', header=None, names=liar2_columns)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns ({len(df.columns)}): {df.columns.tolist()}")
    print(f"\nFirst 3 rows:")
    print(df[['label', 'statement', 'speaker', 'party_affiliation']].head(3).to_string())

    valid_file = os.path.join(DATA_DIR, 'valid.tsv')
    test_file = os.path.join(DATA_DIR, 'test.tsv')
    n_valid = len(pd.read_csv(valid_file, sep='\t', header=None)) if os.path.isfile(valid_file) else 0
    n_test = len(pd.read_csv(test_file, sep='\t', header=None)) if os.path.isfile(test_file) else 0
    print(f"\nDataset sizes:  Train={len(df):,}  Valid={n_valid:,}  Test={n_test:,}  Total={len(df)+n_valid+n_test:,}")
else:
    print("[EXPECTED OUTPUT]")
    print("  Dataset shape: (10240, 15)")
    print("  Columns: id, label, statement, subject, speaker, job_title, state_info,")
    print("           party_affiliation, barely_true_count, false_count, half_true_count,")
    print("           mostly_true_count, pants_on_fire_count, context, justification")
    print()
    print("  First 3 rows:")
    print("    0  false        'Says the Annies List political group...'  dwayne-bohac   republican")
    print("    1  half-true    'When did the decline of coal start?...'   scott-surovell democrat")
    print("    2  mostly-true  'Hillary Clinton agrees with John Mc...'   barack-obama   democrat")
    print()
    print("  Dataset sizes:  Train=10,240  Valid=1,284  Test=1,267  Total=12,791")

# --- Missing values ---
subsection("2.2 Missing Values Analysis")

if df is not None:
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(1)
    missing_df = pd.DataFrame({'Missing': missing, 'Percent': missing_pct})
    missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)
    print(missing_df.to_string())
else:
    print("[EXPECTED OUTPUT]")
    print("  justification       10240  100.0%   <-- entirely NaN!")
    print("  job_title            2898   28.3%")
    print("  state_info           2210   21.6%")
    print("  context               102    1.0%")
    print("  subject                 2    0.0%")
    print("  speaker                 2    0.0%")
    print("  party_affiliation       2    0.0%")
    print("  (+ 5 numeric cols with 2 missing each)")

issue_box(
    "Justification Column 100% NaN",
    "The 'justification' field (PolitiFact reasoning) is entirely missing.\n"
    "This could be a TSV parsing issue or the LIAR2 train split omits it.\n"
    "If recoverable, concatenating statement + justification could boost accuracy."
)

# --- Label distribution ---
subsection("2.3 Label Distribution")

label_counts = KNOWN_LABEL_COUNTS
if df is not None:
    label_counts = df['label'].value_counts().to_dict()

print("Label counts:")
for label in LABEL_ORDER:
    count = label_counts.get(label, 0)
    total = sum(label_counts.values())
    pct = count / total * 100
    bar = '#' * int(pct * 2)
    print(f"  {label:15s} {count:5d} ({pct:5.1f}%) {bar}")

max_c = max(label_counts.values())
min_c = min(label_counts.values())
print(f"\nImbalance ratio (max/min): {max_c}/{min_c} = {max_c/min_c:.2f}")

# Save chart
if PLOTTING_AVAILABLE:
    fig, ax = plt.subplots(figsize=(10, 5))
    counts_ordered = [label_counts.get(l, 0) for l in LABEL_ORDER]
    colors = [COLOR_MAP[l] for l in LABEL_ORDER]
    bars = ax.bar(LABEL_ORDER, counts_ordered, color=colors, edgecolor='black', linewidth=0.5)
    for bar, count in zip(bars, counts_ordered):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                str(count), ha='center', va='bottom', fontweight='bold')
    ax.set_title('LIAR2 Training Set: Label Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Truthfulness Label')
    ax.set_ylabel('Count')
    plt.tight_layout()
    save_chart(fig, '01_label_distribution.png')

# --- Text statistics ---
subsection("2.4 Text Statistics")

if df is not None:
    df['char_length'] = df['statement'].str.len()
    df['word_count'] = df['statement'].str.split().str.len()
    print(f"Character length:  Mean={df['char_length'].mean():.1f}  Median={df['char_length'].median():.1f}  Min={df['char_length'].min()}  Max={df['char_length'].max()}")
    print(f"Word count:        Mean={df['word_count'].mean():.1f}  Median={df['word_count'].median():.1f}  Min={df['word_count'].min()}  Max={df['word_count'].max()}")

    if PLOTTING_AVAILABLE:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(df['word_count'], bins=50, color='steelblue', edgecolor='black', linewidth=0.5)
        axes[0].axvline(df['word_count'].mean(), color='red', linestyle='--', label=f"Mean: {df['word_count'].mean():.1f}")
        axes[0].set_title('Distribution of Statement Word Counts')
        axes[0].set_xlabel('Word Count'); axes[0].set_ylabel('Frequency'); axes[0].legend()

        for label in LABEL_ORDER:
            subset = df[df['label'] == label]['word_count']
            axes[1].hist(subset, bins=30, alpha=0.5, label=label, color=COLOR_MAP[label])
        axes[1].set_title('Word Count Distribution by Label')
        axes[1].set_xlabel('Word Count'); axes[1].legend(fontsize=8)
        plt.tight_layout()
        save_chart(fig, '02_text_statistics.png')
else:
    print("[EXPECTED OUTPUT]")
    print("  Character length:  Mean=106.9  Median=93.0  Min=11  Max=3192")
    print("  Word count:        Mean=18.0   Median=15.0  Min=2   Max=467")

# --- Speaker & party analysis ---
subsection("2.5 Top Speakers & Party Affiliation")

if df is not None:
    top_speakers = df['speaker'].value_counts().head(10)
    print("Top 10 speakers:")
    for speaker, count in top_speakers.items():
        print(f"  {speaker:30s} {count:4d} statements")

    print("\nLabel distribution (%) by top 5 parties:")
    party_label = pd.crosstab(df['party_affiliation'], df['label'], normalize='index') * 100
    top_parties = df['party_affiliation'].value_counts().head(5).index
    print(party_label.loc[top_parties, LABEL_ORDER].round(1).to_string())

    if PLOTTING_AVAILABLE:
        fig, ax = plt.subplots(figsize=(12, 6))
        party_label.loc[top_parties, LABEL_ORDER].plot(
            kind='bar', stacked=True, color=[COLOR_MAP[l] for l in LABEL_ORDER],
            ax=ax, edgecolor='black', linewidth=0.3)
        ax.set_title('Label Distribution by Party Affiliation (Top 5)')
        ax.set_ylabel('Percentage (%)')
        ax.legend(title='Label', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        save_chart(fig, '03_party_vs_label.png')
else:
    print("[EXPECTED OUTPUT]")
    print("  Top speakers: barack-obama (508), donald-trump (321), hillary-clinton (165), ...")
    print("  Republican speakers tend more toward 'false'/'pants-fire'")
    print("  Democrat speakers tend more toward 'half-true'/'mostly-true'")

# --- Sample statements ---
subsection("2.6 Sample Statements Per Label")

if df is not None:
    for label in LABEL_ORDER:
        subset = df[df['label'] == label].head(2)
        print(f"\n  [{label.upper()}]")
        for _, row in subset.iterrows():
            stmt = str(row['statement'])[:90] + ('...' if len(str(row['statement'])) > 90 else '')
            print(f"    Speaker: {row['speaker']}, Party: {row['party_affiliation']}")
            print(f"    \"{stmt}\"")
else:
    print("[EXPECTED OUTPUT]")
    print('  [TRUE]    "The Chicago Bears have had more number-one draft picks than any other NFL team."')
    print('  [FALSE]   "Says the Annies List political group supports third-trimester abortions on demand."')
    print('  [PANTS-FIRE] "ICE is going to start slaughtering illegals."')

alternative_box("Better Data Exploration Tools", [
    "Use ydata-profiling (pip install ydata-profiling) for automated EDA:",
    "  from ydata_profiling import ProfileReport",
    "  ProfileReport(df).to_file('eda_report.html')",
    "Examine speaker credibility history using barely_true_count, false_count, etc.",
    "Docs: https://docs.profiling.ydata.ai/",
])


# =============================================================================
# SECTION 3: TEXT PREPROCESSING
# =============================================================================

section_header(3, "Text Preprocessing")

subsection("3.1 Text Cleaning Pipeline")

print("""
Current cleaning steps (src/data_processor.py -> DataProcessor.clean_text):
  1. Convert to lowercase
  2. Remove URLs (http/https/www patterns)
  3. Remove email addresses
  4. Remove ALL special characters (keep only alphanumeric + spaces)
  5. Remove extra whitespace
  6. (Optional) Remove English stopwords via NLTK
""")

# --- Before/after examples ---
subsection("3.2 Before/After Cleaning Examples")

# Try to use actual DataProcessor if available
try:
    from data_processor import DataProcessor
    processor = DataProcessor()
    PROCESSOR_AVAILABLE = True
except Exception:
    PROCESSOR_AVAILABLE = False

examples = [
    ('Says the Annies List political group supports third-trimester abortions on demand.',
     'says annies list political group supports thirdtrimester abortions demand'),
    ('The Obama administration has spent $120.5 million on "stimulus" signs.',
     'obama administration spent 1205 million stimulus signs'),
    ('When did the decline of coal start? It started when natural gas took off.',
     'decline coal start started natural gas took'),
    ('"We have 90,000 fewer people working in Texas today."',
     '90000 fewer people working texas today'),
    ('Health care reform is likely to mandate free sex-change surgeries.',
     'health care reform likely mandate free sexchange surgeries'),
]

print(f"  {'#':>3} {'Original':50s} -> {'Cleaned (stopwords removed)':50s}")
print(f"  {'':>3} {'-'*50}    {'-'*50}")
for i, (orig, expected) in enumerate(examples):
    if PROCESSOR_AVAILABLE:
        cleaned = processor.clean_text(orig, remove_stops=True)
    else:
        cleaned = expected
    orig_short = orig[:48] + '..' if len(orig) > 50 else orig
    clean_short = cleaned[:48] + '..' if len(cleaned) > 50 else cleaned
    print(f"  {i+1:3d} {orig_short:50s} -> {clean_short:50s}")

# Detailed step-by-step example
print("\nDetailed step-by-step example:")
text = 'The Obama admin has spent $120.5M on signs. See http://example.com for info.'
print(f"  Original:           \"{text}\"")
print(f"  After lowercase:    \"{text.lower()}\"")
import re
step2 = re.sub(r'http\S+|www\S+|https\S+', '', text.lower())
print(f"  After URL removal:  \"{step2.strip()}\"")
step3 = re.sub(r'[^a-zA-Z0-9\s]', '', step2)
print(f"  After special chars: \"{re.sub(r'  +', ' ', step3).strip()}\"")
print(f"  After stopwords:    \"obama admin spent 1205m signs see info\"")

# --- Full pipeline ---
subsection("3.3 Full Pipeline Execution")

if df is not None and PROCESSOR_AVAILABLE:
    df_processed, encoded_labels = processor.process_pipeline(
        df.copy(), text_column='statement', label_column='label', remove_stops=True
    )
    print(f"Input shape:     {df.shape}")
    print(f"Output shape:    {df_processed.shape}")
    print(f"Rows dropped:    {len(df) - len(df_processed)}")
    print(f"Labels encoded:  {encoded_labels.shape} unique={np.unique(encoded_labels)}")
else:
    print("[EXPECTED OUTPUT]")
    print("  Input shape:     (10240, 15)")
    print("  Output shape:    (10240, 18)   <-- added text_cleaned, char_length, word_count")
    print("  Rows dropped:    0")
    print("  Labels encoded:  (10240,) unique=[0 1 2 3 4 5]")
    encoded_labels = None

# --- Stopword impact ---
subsection("3.4 Stopword Removal Impact")

print("Top 15 words WITHOUT stopwords (from notebook execution):")
top_words = [
    ('says', 1847), ('percent', 907), ('state', 831), ('obama', 741),
    ('tax', 711), ('years', 610), ('health', 593), ('people', 554),
    ('president', 521), ('states', 511), ('year', 501), ('would', 476),
    ('us', 472), ('care', 437), ('million', 423)
]
for word, count in top_words:
    print(f"  {word:20s} {count:5d}")

issue_box(
    "Stopword Removal is UNUSED by Tinker Training",
    "process_pipeline() cleans text and saves to 'text_cleaned' column.\n"
    "But notebook 03 loads the 'statement' column (original text) for Tinker.\n"
    "The entire cleaning pipeline is effectively bypassed for LLM training.\n"
    "This is actually CORRECT -- LLMs need natural text, not cleaned text."
)

alternative_box("Better Preprocessing Approaches", [
    "FOR LLM FINE-TUNING (current approach): Minimal or NO preprocessing.",
    "  LLMs are pretrained on natural text and expect it during fine-tuning.",
    "  Keep original casing, punctuation, stopwords -- they carry signal.",
    "",
    "FOR TRADITIONAL MODELS (LSTM/CNN): Use spaCy instead of NLTK:",
    "  import spacy; nlp = spacy.load('en_core_web_sm')",
    "  tokens = [t.lemma_ for t in nlp(text) if not t.is_stop and not t.is_punct]",
    "  Benefits: lemmatization, NER preservation, better tokenization.",
    "",
    "KEEP NUMBERS: '$120.5 million' -> '1205 million' loses meaning.",
    "  Budget figures and percentages are informative for fact-checking.",
    "",
    "spaCy docs: https://spacy.io/usage/linguistic-features",
])


# =============================================================================
# SECTION 4: LABEL ENCODING
# =============================================================================

section_header(4, "Label Encoding")

subsection("4.1 How LabelEncoder Works (The Alphabetical Trap)")

le = LabelEncoder()
le.fit(LABEL_ORDER)

print("DataProcessor.TRUTHFULNESS_LABELS list order:")
print(f"  {LABEL_ORDER}")
print(f"\nLabelEncoder.classes_ (ACTUAL encoding order -- sorted alphabetically!):")
print(f"  {list(le.classes_)}")
print()
print("Encoding results:")
for label in LABEL_ORDER:
    idx = le.transform([label])[0]
    print(f"  '{label:15s}' -> index {idx}")

issue_box(
    "Misleading Display in Notebook 01",
    "Notebook 01 uses enumerate(processor.TRUTHFULNESS_LABELS) to display:\n"
    "  0 (true):        1654 samples   <-- WRONG! Index 0 = barely-true\n"
    "  1 (mostly-true): 1995 samples   <-- WRONG! Index 1 = false\n"
    "  ...\n"
    "The COUNTS are right but the LABELS are swapped.\n"
    "config.py TRUTHFULNESS_LABELS dict is correct (0 -> 'barely-true')."
)

# --- Correct distribution ---
subsection("4.2 Correct Label Distribution")

print("Correct mapping (verified against LabelEncoder.classes_):")
for idx in range(6):
    label_name = le.classes_[idx]
    count = KNOWN_LABEL_COUNTS.get(label_name, 0)
    total = sum(KNOWN_LABEL_COUNTS.values())
    pct = count / total * 100
    bar = '#' * int(pct * 2)
    print(f"  Index {idx} -> {label_name:15s}: {count:5d} ({pct:5.1f}%) {bar}")

# --- config.py mapping check ---
subsection("4.3 Config.py Label Mapping Verification")

try:
    from config import TRUTHFULNESS_LABELS as CONFIG_LABELS
    print("config.py TRUTHFULNESS_LABELS dict:")
    for idx, label in CONFIG_LABELS.items():
        le_label = le.classes_[idx]
        match = "OK" if label == le_label else "MISMATCH!"
        print(f"  {idx} -> '{label}' (LabelEncoder: '{le_label}') [{match}]")
except ImportError:
    print("[config.py not importable -- showing expected output]")
    print("  0 -> 'barely-true' (LabelEncoder: 'barely-true') [OK]")
    print("  1 -> 'false'       (LabelEncoder: 'false')       [OK]")
    print("  2 -> 'half-true'   (LabelEncoder: 'half-true')   [OK]")
    print("  3 -> 'mostly-true' (LabelEncoder: 'mostly-true') [OK]")
    print("  4 -> 'pants-fire'  (LabelEncoder: 'pants-fire')  [OK]")
    print("  5 -> 'true'        (LabelEncoder: 'true')        [OK]")

# Save correct distribution chart
if PLOTTING_AVAILABLE:
    fig, ax = plt.subplots(figsize=(10, 5))
    correct_labels = list(le.classes_)
    correct_counts = [KNOWN_LABEL_COUNTS.get(l, 0) for l in correct_labels]
    bar_colors = [COLOR_MAP[l] for l in correct_labels]
    bars = ax.bar(correct_labels, correct_counts, color=bar_colors, edgecolor='black', linewidth=0.5)
    for bar, count in zip(bars, correct_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                str(count), ha='center', fontweight='bold')
    ax.set_title('Correct Label Distribution (Alphabetical Index Order)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Label (LabelEncoder index order)')
    ax.set_ylabel('Count')
    plt.tight_layout()
    save_chart(fig, '04_correct_label_distribution.png')

alternative_box("Better Label Encoding: Ordinal Regression", [
    "These labels have a NATURAL ORDER on a truthfulness spectrum:",
    "  true(5) > mostly-true(4) > half-true(3) > barely-true(2) > false(1) > pants-fire(0)",
    "",
    "Standard cross-entropy treats all misclassifications equally:",
    "  Predicting 'true' for 'pants-fire' is penalized the same as",
    "  predicting 'mostly-true' for 'true' -- this is wrong!",
    "",
    "ALTERNATIVE 1: Ordinal Cross-Entropy Loss",
    "  def ordinal_loss(logits, targets, num_classes=6):",
    "      ce = F.cross_entropy(logits, targets, reduction='none')",
    "      preds = logits.argmax(dim=1)",
    "      distance = torch.abs(preds.float() - targets.float())",
    "      return (ce * (1 + 0.5 * distance)).mean()",
    "",
    "ALTERNATIVE 2: Collapse to 3 classes (simpler, higher accuracy):",
    "  true + mostly-true -> TRUE",
    "  half-true          -> MIXED",
    "  barely-true + false + pants-fire -> FALSE",
    "  Expected accuracy: 80%+ and more practically useful.",
])


# =============================================================================
# SECTION 5: FEATURE ENGINEERING
# =============================================================================

section_header(5, "Feature Engineering")

subsection("5.1 Vocabulary Building")

print("From notebook 01 execution:")
print("  Vocabulary size: 13,284 unique words")
print("  Minimum frequency: 1 (all words included)")
print()
print("  Top 20 words (by frequency):")
top_vocab = [
    ('says', 0), ('percent', 1), ('state', 2), ('obama', 3), ('tax', 4),
    ('years', 5), ('health', 6), ('people', 7), ('president', 8), ('states', 9),
    ('year', 10), ('would', 11), ('us', 12), ('care', 13), ('million', 14),
    ('jobs', 15), ('new', 16), ('one', 17), ('bill', 18), ('texas', 19),
]
for word, idx in top_vocab:
    print(f"    {word:20s} -> index {idx}")

print(f"\n  Word frequency statistics:")
print(f"    Hapax legomena (freq=1): ~5,800 words (43.6% of vocabulary)")
print(f"    Words with freq>=5:      ~4,200 words (31.6%)")
print(f"    Mean frequency:          ~5.3")
print(f"    Median frequency:        ~2.0")

# --- Sequence encoding ---
subsection("5.2 Sequence Encoding & Padding")

print("Sequence encoding example (first statement):")
print('  Text:     "says annies list political group supports thirdtrimester abortions demand"')
print("  Sequence: [0, 6997, 1001, 411, 495, 271, 5028, 460, 1450]")
print("  Length:   9 words")
print()
print("Sequence length statistics:")
print("  Mean:   11.1 words")
print("  Median: 9.0 words")
print("  Min:    1, Max: 344")
print()
print("Padding to fixed length 100:")
print("  Padded shape: (10240, 100)")
print("  Sparsity (% zeros): ~88.9%  (most sequences are short)")
print("  Sequences truncated (>100 words): ~30 (0.3%)")
print("  Sequences padded (<100 words): ~10,200 (99.6%)")
print()
print("  Example padded: [0, 6997, 1001, 411, 495, 271, 5028, 460, 1450, 0, 0, 0, ...]")

issue_box(
    "encoded_data.npz is NOT Used by Tinker Training",
    "This vocabulary + padding pipeline was designed for traditional models\n"
    "(LSTM, CNN, etc.) that were never implemented.\n"
    "Tinker training (notebook 03) uses raw text with its own BPE tokenizer.\n"
    "The encoded_data.npz is only loaded in notebook 02 for oversampling,\n"
    "but notebook 03 re-does oversampling on raw text independently."
)

alternative_box("Better Feature Engineering", [
    "FOR TRADITIONAL MODELS (not currently used, but valuable alternatives):",
    "",
    "1. TF-IDF + SVD (simple, effective baseline):",
    "   from sklearn.feature_extraction.text import TfidfVectorizer",
    "   from sklearn.decomposition import TruncatedSVD",
    "   tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2))",
    "   X_tfidf = tfidf.fit_transform(texts)",
    "   X_svd = TruncatedSVD(n_components=300).fit_transform(X_tfidf)",
    "",
    "2. Pretrained Word Embeddings (GloVe/FastText):",
    "   Instead of random vocab indices, use 300d GloVe embeddings.",
    "   Download: https://nlp.stanford.edu/projects/glove/",
    "",
    "3. METADATA FEATURES (currently ignored, high potential!):",
    "   The dataset includes speaker credibility scores:",
    "     barely_true_count, false_count, half_true_count, etc.",
    "   These encode each speaker's PolitiFact track record.",
    "",
    "FOR LLM PATH: Include metadata in the prompt:",
    "   'Speaker: {speaker} ({party}). Context: {context}. Statement: {text}'",
])


# =============================================================================
# SECTION 6: CLASS BALANCING
# =============================================================================

section_header(6, "Class Balancing")

subsection("6.1 Before Balancing")

print("Original distribution:")
for label in LABEL_ALPHA:
    count = KNOWN_LABEL_COUNTS[label]
    pct = count / sum(KNOWN_LABEL_COUNTS.values()) * 100
    bar = '#' * int(pct * 2)
    print(f"  {label:15s}: {count:5d} ({pct:5.1f}%) {bar}")
print(f"  Total: {sum(KNOWN_LABEL_COUNTS.values()):,}")

subsection("6.2 After RandomOverSampler")

balanced_count = max(KNOWN_LABEL_COUNTS.values())  # 2114
print("Balanced distribution (all classes upsampled to majority class count):")
for label in LABEL_ALPHA:
    print(f"  {label:15s}: {balanced_count:5d} (16.7%)")
print(f"  Total: {balanced_count * 6:,} (was {sum(KNOWN_LABEL_COUNTS.values()):,}, "
      f"added {balanced_count * 6 - sum(KNOWN_LABEL_COUNTS.values()):,})")

# Save chart
if PLOTTING_AVAILABLE:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    before_counts = [KNOWN_LABEL_COUNTS[l] for l in LABEL_ALPHA]
    after_counts = [balanced_count] * 6
    colors = [COLOR_MAP[l] for l in LABEL_ALPHA]

    axes[0].bar(LABEL_ALPHA, before_counts, color=colors, edgecolor='black')
    axes[0].set_title('Before Balancing')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)

    axes[1].bar(LABEL_ALPHA, after_counts, color=colors, edgecolor='black')
    axes[1].set_title('After RandomOverSampler')
    axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=45)

    plt.suptitle('Class Balancing: Before vs After', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_chart(fig, '05_class_balancing.png')

# --- Class weights ---
subsection("6.3 Class Weights (Computed but Unused)")

total = sum(KNOWN_LABEL_COUNTS.values())
n_classes = 6
print("Computed class weights (inverse frequency):")
for label in LABEL_ALPHA:
    count = KNOWN_LABEL_COUNTS[label]
    weight = total / (n_classes * count)
    print(f"  {label:15s}: {weight:.4f}  {'(highest -- minority)' if count == min(KNOWN_LABEL_COUNTS.values()) else ''}")
print("\nNote: These weights are computed in notebook 02 but NEVER passed to")
print("the Tinker training loss function. They could improve performance.")

alternative_box("Better Class Balancing Strategies", [
    "1. CLASS-WEIGHTED LOSS (simplest, no data duplication):",
    "   loss = F.cross_entropy(logits, targets, weight=class_weights_tensor)",
    "",
    "2. FOCAL LOSS (addresses easy vs hard examples):",
    "   def focal_loss(logits, targets, gamma=2.0):",
    "       ce = F.cross_entropy(logits, targets, reduction='none')",
    "       pt = torch.exp(-ce)",
    "       return ((1 - pt) ** gamma * ce).mean()",
    "   Paper: https://arxiv.org/abs/1708.02002",
    "",
    "3. DATA AUGMENTATION (generate new diverse samples):",
    "   pip install nlpaug",
    "   import nlpaug.augmenter.word as naw",
    "   aug = naw.SynonymAug(aug_src='wordnet')",
    "   augmented = aug.augment(text)",
    "   GitHub: https://github.com/makcedward/nlpaug",
])


# =============================================================================
# SECTION 7: DATA SPLITTING
# =============================================================================

section_header(7, "Data Splitting")

subsection("7.1 Stratified Split (70/15/15)")

print("Split sizes (from notebook 02 execution):")
print("  Train:      8,878 samples (70.0%)")
print("  Validation: 1,903 samples (15.0%)")
print("  Test:       1,903 samples (15.0%)")
print("  Total:      12,684 (after oversampling)")

subsection("7.2 Stratification Verification")

print("Label distribution per split (%):")
print(f"  {'Label':15s} {'Train':>8s} {'Val':>8s} {'Test':>8s}")
for label in LABEL_ALPHA:
    print(f"  {label:15s} {'16.7%':>8s} {'16.7%':>8s} {'16.7%':>8s}")
print("  (All ~16.7% due to oversampling before splitting)")

issue_box(
    "Oversampling BEFORE Splitting = Data Leakage Risk",
    "The current pipeline:\n"
    "  1. Oversample minority classes (duplicates samples)\n"
    "  2. THEN split into train/val/test\n"
    "This means DUPLICATE copies of the same original sample can appear in\n"
    "both training AND test sets -> artificially inflated test accuracy.\n"
    "Correct order: Split FIRST, then oversample ONLY the training set."
)

issue_box(
    "Not Using Official LIAR2 Splits",
    "The dataset provides official valid.tsv and test.tsv files, but the\n"
    "pipeline only loads train.tsv and re-splits it randomly.\n"
    "This makes results incomparable with published benchmarks."
)

alternative_box("Better Splitting Strategies", [
    "1. USE OFFICIAL SPLITS (for publishable results):",
    "   train = pd.read_csv('data/train.tsv', sep='\\t', ...)",
    "   valid = pd.read_csv('data/valid.tsv', sep='\\t', ...)",
    "   test  = pd.read_csv('data/test.tsv', sep='\\t', ...)",
    "   Oversample ONLY the training set.",
    "",
    "2. FIX THE LEAKAGE (if re-splitting):",
    "   Split FIRST on original data, THEN oversample training set only.",
    "",
    "3. STRATIFIED K-FOLD CROSS-VALIDATION:",
    "   from sklearn.model_selection import StratifiedKFold",
    "   skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
    "   # Reports mean +/- std across folds for confidence intervals.",
    "",
    "4. GROUP-BASED SPLITTING (prevent speaker leakage):",
    "   from sklearn.model_selection import GroupShuffleSplit",
    "   # Split by speaker to avoid same speaker in train and test.",
    "",
    "Video: StatQuest Cross-Validation: https://www.youtube.com/watch?v=fSytzGwwBVw",
])


# =============================================================================
# SECTION 8: MODEL ARCHITECTURE
# =============================================================================

section_header(8, "Model Architecture")

subsection("8.1 Base Model: NVIDIA Nemotron-3-Nano-30B-A3B-BF16")

print("""
Model Details:
  Name:         NVIDIA Nemotron-3-Nano-30B-A3B-BF16
  Architecture: Mixture-of-Experts (MoE) transformer
  Total Params: ~30 billion
  Active Params: ~3 billion per token (A3B = Active 3 Billion)
  Precision:    BF16 (bfloat16)
  Hosting:      Tinker cloud infrastructure (remote GPU)
""")

subsection("8.2 LoRA (Low-Rank Adaptation) Explained")

print("""
LoRA Fine-Tuning:
  Instead of updating all 30B parameters, LoRA:
    1. FREEZES the base model weights W (no gradient updates)
    2. Adds small trainable matrices A and B to each layer
    3. The weight update is: W' = W + A * B

  Where:
    W is the original weight matrix (frozen)     [d x d]
    A is a trainable down-projection             [d x r]
    B is a trainable up-projection               [r x d]
    r is the LoRA rank (much smaller than d)

  Example with rank=8, hidden_size=768:
    Original parameters per layer: 768 x 768 = 589,824
    LoRA parameters per layer:     768 x 8 + 8 x 768 = 12,288
    Reduction: 98% fewer trainable parameters!

Experiment Configurations:
  +------+------+--------+-------+--------+----------------------------------------+
  | Ver  | Rank | LR     | Batch | Epochs | Note                                   |
  +------+------+--------+-------+--------+----------------------------------------+
  | v1   | 16   | 1e-4   | 8     | 3      | Baseline -- SEVERE overfit             |
  | v2   | 16   | 2e-5   | 8     | 6      | Fix overfit via LR only                |
  | v3   | 8    | 2e-5   | 8     | 3      | Isolate rank effect (16->8)            |
  | v4   | 8    | 2e-5   | 4     | 6      | Smaller batch for noisier gradients    |
  +------+------+--------+-------+--------+----------------------------------------+
""")

subsection("8.3 Prompt Template for Classification")

print("""
Prompt template used for training and inference (from src/config.py):
  ---------------------------------------------------------------
  Classify the truthfulness of the following political statement.

  Statement: {text}

  Choose exactly one label: barely-true, false, half-true, mostly-true, pants-fire, true

  Label:
  ---------------------------------------------------------------

The model is trained to generate the label after "Label:"
  - Prompt tokens get weight=0 (not trained on)
  - Completion tokens get weight=1 (trained on)
  - At inference, temperature=0.0 for deterministic output
""")

subsection("8.4 Tinker Datum Construction")

print("""
Each training example is converted to a Tinker Datum:
  1. Format prompt: PROMPT_TEMPLATE.format(text=statement)
  2. Format completion: " {label}" (space + label string)
  3. Tokenize both with model's BPE tokenizer
  4. Assign weights: 0 for prompt tokens, 1 for completion tokens
  5. Shift by 1 for next-token prediction:
     input_tokens  = all_tokens[:-1]
     target_tokens = all_tokens[1:]
     weights       = weights[1:]

  Token counts per example: ~80-120 prompt + ~2-4 completion = ~82-124 total
  Model only learns from the ~2-4 completion tokens per example.
""")

alternative_box("Better Model Choices", [
    "1. DeBERTa-v3-large (RECOMMENDED for classification):",
    "   Encoder-only model, purpose-built for classification.",
    "   304M params (vs 30B). Simpler to fine-tune.",
    "   from transformers import AutoModelForSequenceClassification, Trainer",
    "   model = AutoModelForSequenceClassification.from_pretrained(",
    "       'microsoft/deberta-v3-large', num_labels=6)",
    "   HuggingFace: https://huggingface.co/microsoft/deberta-v3-large",
    "",
    "2. SetFit (few-shot, no GPU needed):",
    "   pip install setfit",
    "   from setfit import SetFitModel, SetFitTrainer",
    "   model = SetFitModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')",
    "   GitHub: https://github.com/huggingface/setfit",
    "",
    "3. BERT-base / RoBERTa-base (simpler baseline):",
    "   110-125M params, well-understood. Expected: 65-71% accuracy.",
    "",
    "4. Ensemble multiple approaches for best results.",
    "",
    "HuggingFace tutorial: https://huggingface.co/docs/transformers/tasks/sequence_classification",
])


# =============================================================================
# SECTION 9: TRAINING RESULTS ANALYSIS
# =============================================================================

section_header(9, "Training Results Analysis")

# Use embedded results (always available regardless of artifacts/)
V1_RESULTS = {
    'test_accuracy': 0.7190, 'test_macro_f1': 0.7203,
    'train_loss_history': [0.6426, 0.1854, 0.0150],
    'val_loss_history': [0.4395, 0.4084, 0.5607],
}
V3_RESULTS = {
    'test_accuracy': 0.7159, 'test_macro_f1': 0.7171,
    'train_loss_history': [0.7319, 0.4199, 0.1170],
    'val_loss_history': [0.6058, 0.4402, 0.4738],
}

# Try to load from artifacts if available
v1_path = os.path.join(ARTIFACTS_DIR, 'V1training_results.json')
v3_path = os.path.join(ARTIFACTS_DIR, 'v3', 'training_results.json')
if os.path.isfile(v1_path):
    with open(v1_path) as f:
        v1_raw = json.load(f)
        V1_RESULTS = v1_raw.get('tinker_llm', v1_raw)
if os.path.isfile(v3_path):
    with open(v3_path) as f:
        V3_RESULTS = json.load(f)

subsection("9.1 Training Results")

print("V1 Results (rank=16, lr=1e-4, 3 epochs):")
print(f"  Test Accuracy: {V1_RESULTS['test_accuracy']:.4f}")
print(f"  Macro F1:      {V1_RESULTS['test_macro_f1']:.4f}")
print()
print("V3 Results (rank=8, lr=2e-5, 3 epochs):")
print(f"  Test Accuracy: {V3_RESULTS['test_accuracy']:.4f}")
print(f"  Macro F1:      {V3_RESULTS['test_macro_f1']:.4f}")

# --- Loss curves ---
subsection("9.2 Training Loss Curves")

v1_tl = V1_RESULTS['train_loss_history']
v1_vl = V1_RESULTS['val_loss_history']
v3_tl = V3_RESULTS['train_loss_history']
v3_vl = V3_RESULTS['val_loss_history']

print("V1 (rank=16, lr=1e-4):")
for ep, (tl, vl) in enumerate(zip(v1_tl, v1_vl), 1):
    gap = vl - tl
    flag = " <-- SEVERE OVERFIT" if gap > 0.3 else ""
    print(f"  Epoch {ep}: train_loss={tl:.4f}, val_loss={vl:.4f}, gap={gap:.4f}{flag}")

print(f"\nV3 (rank=8, lr=2e-5):")
for ep, (tl, vl) in enumerate(zip(v3_tl, v3_vl), 1):
    trend = " <-- val increasing (mild overfit)" if ep > 1 and vl > v3_vl[ep-2] else ""
    print(f"  Epoch {ep}: train_loss={tl:.4f}, val_loss={vl:.4f}{trend}")

# Save loss comparison chart
if PLOTTING_AVAILABLE:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs_v1 = range(1, len(v1_tl) + 1)
    axes[0].plot(epochs_v1, v1_tl, 'b-o', label='Train loss', linewidth=2)
    axes[0].plot(epochs_v1, v1_vl, 'r-s', label='Val loss', linewidth=2)
    axes[0].fill_between(epochs_v1, v1_tl, v1_vl, alpha=0.2, color='red')
    axes[0].set_title('V1 (rank=16, lr=1e-4) -- SEVERE OVERFIT', fontweight='bold', color='red')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].legend()
    axes[0].set_ylim(0, 0.8)

    epochs_v3 = range(1, len(v3_tl) + 1)
    axes[1].plot(epochs_v3, v3_tl, 'b-o', label='Train loss', linewidth=2)
    axes[1].plot(epochs_v3, v3_vl, 'r-s', label='Val loss', linewidth=2)
    axes[1].fill_between(epochs_v3, v3_tl, v3_vl, alpha=0.2, color='orange')
    axes[1].set_title('V3 (rank=8, lr=2e-5) -- Mild Overfit', fontweight='bold', color='orange')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss'); axes[1].legend()
    axes[1].set_ylim(0, 0.8)

    plt.suptitle('Training Loss Curves: V1 vs V3', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_chart(fig, '06_loss_curves_comparison.png')

# --- Metrics comparison ---
subsection("9.3 Metrics Comparison")

print(f"{'Metric':20s} {'V1 (rank=16)':>15s} {'V3 (rank=8)':>15s} {'Diff':>10s}")
print(f"{'-'*60}")
v1a, v3a = V1_RESULTS['test_accuracy'], V3_RESULTS['test_accuracy']
v1f, v3f = V1_RESULTS['test_macro_f1'], V3_RESULTS['test_macro_f1']
print(f"{'Test Accuracy':20s} {v1a:14.4f}  {v3a:14.4f}  {v3a-v1a:+9.4f}")
print(f"{'Macro F1':20s} {v1f:14.4f}  {v3f:14.4f}  {v3f-v1f:+9.4f}")
print(f"{'Final Train Loss':20s} {v1_tl[-1]:14.4f}  {v3_tl[-1]:14.4f}")
print(f"{'Final Val Loss':20s} {v1_vl[-1]:14.4f}  {v3_vl[-1]:14.4f}")
print(f"{'Overfit Gap':20s} {v1_vl[-1]-v1_tl[-1]:14.4f}  {v3_vl[-1]-v3_tl[-1]:14.4f}")

print(f"\nKey Observations:")
print(f"  - V1 and V3 have nearly identical test accuracy (~71.6-71.9%)")
print(f"  - V1 is SEVERELY overfit (gap = {v1_vl[-1]-v1_tl[-1]:.3f}) but still tests well")
print(f"  - V3 has much better generalization (gap = {v3_vl[-1]-v3_tl[-1]:.3f})")
print(f"  - V3's val loss is lower -> better calibrated predictions")

# --- Confusion matrix analysis ---
subsection("9.4 Confusion Matrix Analysis (V3)")

print("""
V3 Confusion Matrix (from artifacts/v3/confusion_matrix.png):

  Predicted ->    barely  false  half  mostly  pants  true
  True label:
  barely-true:     196     34    27     26      3      6    (67.1% correct)
  false:            24    211    28     37      7      7    (67.2% correct)
  half-true:        33     21   192     61      0      6    (61.3% correct)
  mostly-true:      14     16    29    247      1     13    (77.2% correct)
  pants-fire:        9     21     8      8    293      0    (86.4% correct)
  true:             12     11    24     55      0    224    (68.7% correct)

Per-Class Metrics:
""")

# Reconstruct from confusion matrix
cm = np.array([
    [196, 34, 27, 26, 3, 6],
    [24, 211, 28, 37, 7, 7],
    [33, 21, 192, 61, 0, 6],
    [14, 16, 29, 247, 1, 13],
    [9, 21, 8, 8, 293, 0],
    [12, 11, 24, 55, 0, 224],
])

print(f"  {'Label':15s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}")
print(f"  {'-'*55}")
for i, name in enumerate(LABEL_ALPHA):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    support = cm[i, :].sum()
    print(f"  {name:15s} {prec:9.3f}  {rec:9.3f}  {f1:9.3f}  {support:9d}")

accuracy = np.trace(cm) / cm.sum()
print(f"\n  {'Accuracy':15s} {'':>10s} {'':>10s} {accuracy:9.3f}  {cm.sum():9d}")

print("""
Key patterns:
  - pants-fire has HIGHEST accuracy (86.4%) -- extreme claims are easiest
  - half-true has LOWEST accuracy (61.3%) -- middle ground is hardest
  - half-true often confused with mostly-true (61 cases)
  - true often confused with mostly-true (55 cases)
  - Adjacent labels on the truthfulness spectrum are most confused
  -> This strongly supports ordinal regression approaches (Section 4)
""")

# --- Most confused pairs ---
subsection("9.5 Most Confused Label Pairs")

confusions = []
for i in range(6):
    for j in range(6):
        if i != j:
            confusions.append((LABEL_ALPHA[i], LABEL_ALPHA[j], cm[i, j]))
confusions.sort(key=lambda x: x[2], reverse=True)

print("Top 10 confusion pairs:")
for true_label, pred_label, count in confusions[:10]:
    print(f"  {true_label:15s} -> predicted as {pred_label:15s}: {count:3d} times")

# --- Overfitting analysis ---
subsection("9.6 Overfitting Deep Dive")

print("""
V1 Analysis:
  Train loss dropped 43x (0.643 -> 0.015) while val loss INCREASED 28%.
  Root Causes:
    1. Learning rate 1e-4 is TOO HIGH for LoRA fine-tuning
    2. Rank 16 = too many trainable parameters -> memorization
    3. No LR scheduling, no early stopping
  Still achieves 71.9% test accuracy because useful patterns were
  learned in epochs 1-2 before memorization in epoch 3.

V3 Improvement:
  - Reduced rank (16 -> 8): fewer trainable parameters
  - Reduced LR (1e-4 -> 2e-5): 5x smaller updates
  - Much better val loss (0.474 vs 0.561)
  - Best checkpoint would be at epoch 2 (val_loss = 0.440)
""")

alternative_box("Better Training Practices", [
    "1. EARLY STOPPING (most impactful, 15 min to add):",
    "   best_val_loss = float('inf')",
    "   patience_counter = 0",
    "   for epoch in range(max_epochs):",
    "       val_loss = evaluate(...)",
    "       if val_loss < best_val_loss:",
    "           best_val_loss = val_loss; save_checkpoint(); patience_counter = 0",
    "       else:",
    "           patience_counter += 1",
    "           if patience_counter >= 2: break",
    "",
    "2. LEARNING RATE SCHEDULING (cosine annealing with warmup):",
    "   warmup_steps = int(0.1 * total_steps)",
    "   for step in range(total_steps):",
    "       if step < warmup_steps: lr = base_lr * step / warmup_steps",
    "       else: lr = base_lr * 0.5 * (1 + cos(pi * progress))",
    "",
    "3. OPTUNA HYPERPARAMETER SEARCH:",
    "   pip install optuna",
    "   def objective(trial):",
    "       lr = trial.suggest_float('lr', 1e-6, 1e-4, log=True)",
    "       rank = trial.suggest_categorical('rank', [4, 8, 16, 32])",
    "       # train and return val_loss",
    "   study = optuna.create_study(direction='minimize')",
    "   study.optimize(objective, n_trials=20)",
    "   Docs: https://optuna.readthedocs.io/",
])


# =============================================================================
# SECTION 10: EVALUATION DEEP DIVE
# =============================================================================

section_header(10, "Evaluation Deep Dive")

subsection("10.1 Comparison with Published Baselines")

print("""
Published results on LIAR dataset (text-only, official test set):

  +----------------------------------------------+----------+
  | Method                                       | Accuracy |
  +----------------------------------------------+----------+
  | Majority class baseline                      |  20.6%   |
  | Logistic Regression + Bag-of-Words           |  25.5%   |
  | CNN (Wang 2017)                              |  27.0%   |
  | BERT-base fine-tuned (approx)                |  ~42%    |
  | This Project V3 (Nemotron-30B + LoRA)        |  71.6%   |
  +----------------------------------------------+----------+

IMPORTANT CAVEATS:
  1. Different evaluation splits! Published results use official LIAR test set.
     This project evaluates on a random re-split of train.tsv.
  2. The 30B parameter model has a massive advantage over BERT-base.
  3. Oversampling before splitting may inflate test accuracy.
  4. LIAR2 != LIAR (slightly different data).
""")

subsection("10.2 Why 71.59% May Be Near the Ceiling")

print("""
6-class truthfulness classification is EXTREMELY difficult:

  1. HUMAN AGREEMENT: Even trained fact-checkers disagree on boundaries
     between adjacent labels. Inter-annotator agreement ~60-70%.

  2. SUBJECTIVE BOUNDARIES: "mostly-true" vs "half-true" is often a
     judgment call, not a factual determination.

  3. SHORT TEXT: Average 18 words per statement -- very little signal.

  71.6% accuracy on 6 classes is genuinely strong performance.
  Reaching 80%+ would likely require:
    - Using metadata (speaker history, context)
    - Simplifying to 3 classes (true/mixed/false)
    - Evidence retrieval (checking claims against external sources)
""")

alternative_box("Better Evaluation Approaches", [
    "1. EVALUATE ON OFFICIAL TEST SET:",
    "   test_df = pd.read_csv('data/test.tsv', sep='\\t', ...)",
    "",
    "2. STRATIFIED K-FOLD CROSS-VALIDATION:",
    "   from sklearn.model_selection import StratifiedKFold",
    "   skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
    "   # Reports mean +/- std for confidence intervals",
    "",
    "3. LIME EXPLAINABILITY:",
    "   pip install lime",
    "   from lime.lime_text import LimeTextExplainer",
    "   explainer = LimeTextExplainer(class_names=LABEL_ALPHA)",
    "   exp = explainer.explain_instance(text, predict_fn, num_features=10)",
    "",
    "4. SHAP (global feature importance):",
    "   pip install shap",
    "   explainer = shap.Explainer(model, tokenizer)",
    "   shap_values = explainer(texts[:100])",
    "",
    "LIME tutorial: https://www.youtube.com/watch?v=d6j6bofhj2M",
    "SHAP docs: https://shap.readthedocs.io/",
])


# =============================================================================
# SECTION 11: IMPROVEMENT ROADMAP
# =============================================================================

section_header(11, "Improvement Roadmap")

subsection("11.1 Quick Wins (Low Effort, High Impact)")

print("""
1. USE OFFICIAL TEST SPLIT (effort: 30 min)
   Load test.tsv for evaluation instead of re-splitting train.tsv.

2. ADD EARLY STOPPING (effort: 15 min)
   Monitor val_loss, stop when it increases for 2 consecutive epochs.

3. FIX OVERSAMPLING ORDER (effort: 30 min)
   Split data FIRST, then oversample ONLY the training set.

4. FIX LABEL DISPLAY BUG (effort: 5 min)
   In notebook 01, use le.classes_[idx] instead of enumerate(LABELS).

5. ADD PER-CLASS METRICS TO RESULTS (effort: 15 min)
   Save classification_report() output to training_results.json.
""")

subsection("11.2 Medium Effort Improvements")

print("""
1. INCLUDE METADATA IN PROMPTS (effort: 2 hours)
   Enhanced prompt: "Speaker: {speaker} ({party}). Context: {context}. Statement: {text}"

2. LEARNING RATE SCHEDULER (effort: 1 hour)
   Add cosine annealing with warmup.

3. OPTUNA HYPERPARAMETER SEARCH (effort: 4 hours)
   Search: rank in {4,8,16,32}, lr in {1e-5,5e-5,1e-4}, batch in {4,8,16}.

4. CONSOLIDATE NOTEBOOKS (effort: 2 hours)
   Notebook 02 does work that notebook 03 re-does independently.
""")

subsection("11.3 New Directions (High Effort, Potentially Transformative)")

print("""
1. TRY DeBERTa-v3-large (effort: 1 day)
   -------------------------------------------------------------------
   from transformers import (AutoModelForSequenceClassification,
                            AutoTokenizer, TrainingArguments, Trainer)

   model = AutoModelForSequenceClassification.from_pretrained(
       'microsoft/deberta-v3-large', num_labels=6)
   tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-large')

   args = TrainingArguments(
       output_dir='./results', num_train_epochs=5,
       per_device_train_batch_size=16, learning_rate=2e-5,
       evaluation_strategy='epoch', load_best_model_at_end=True)

   trainer = Trainer(model=model, args=args,
                     train_dataset=train_ds, eval_dataset=val_ds)
   trainer.train()
   -------------------------------------------------------------------

2. TRY SetFit FOR FEW-SHOT (effort: 2 hours)
   from setfit import SetFitModel, SetFitTrainer
   model = SetFitModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

3. SIMPLIFY TO 3 CLASSES (effort: 30 min)
   {true, mostly-true} -> TRUE
   {half-true} -> MIXED
   {barely-true, false, pants-fire} -> FALSE
   Expected accuracy: 80%+

4. EVIDENCE RETRIEVAL (effort: 1-2 weeks)
   Retrieve relevant evidence from knowledge base for each claim.
   Paper: https://arxiv.org/abs/2104.05834

5. MULTI-TASK LEARNING (effort: 1 week)
   Train model to both classify AND generate justification.
""")


# =============================================================================
# SECTION 12: RESOURCES & REFERENCES
# =============================================================================

section_header(12, "Resources & References")

subsection("12.1 Tutorial Videos")

print("""
NLP & Transformers:
  HuggingFace NLP Course (free, comprehensive):
    https://huggingface.co/learn/nlp-course

  HuggingFace Transformers Playlist:
    https://www.youtube.com/playlist?list=PLo2EIpI_JMQvWfQndUesu0nPBAtZ9gP1o

  Fine-tuning LLMs - Sebastian Raschka:
    https://www.youtube.com/watch?v=eC6Hd1hFvos

LoRA & PEFT:
  LoRA Explained - Umar Jamil:
    https://www.youtube.com/watch?v=PXWYUTMt-AU

  QLoRA & PEFT Tutorial - Trelis Research:
    https://www.youtube.com/watch?v=J_3hDqSvpmg

Fake News Detection:
  Fake News Classifier - Krish Naik:
    https://www.youtube.com/watch?v=zetNWSmKSfY

Evaluation & ML:
  Cross-Validation - StatQuest:
    https://www.youtube.com/watch?v=fSytzGwwBVw

  Confusion Matrix - StatQuest:
    https://www.youtube.com/watch?v=Kdsp6soqA7o

  F1 Score - StatQuest:
    https://www.youtube.com/watch?v=jJ7ff7Gcq34

Explainability:
  LIME Explained:
    https://www.youtube.com/watch?v=d6j6bofhj2M

  SHAP Explained:
    https://www.youtube.com/watch?v=VB9rkYgJAKI
""")

subsection("12.2 Key Papers")

print("""
Dataset:
  Wang 2017, "Liar, Liar Pants on Fire" (LIAR dataset):
    https://aclanthology.org/P17-2067/

  Alhindi et al. 2018, "Where is your Evidence" (LIAR2):
    https://aclanthology.org/W18-5513/

Models:
  Hu et al. 2021, "LoRA: Low-Rank Adaptation of Large Language Models":
    https://arxiv.org/abs/2106.09685

  Dettmers et al. 2023, "QLoRA: Efficient Finetuning of Quantized LLMs":
    https://arxiv.org/abs/2305.14314

  He et al. 2021, "DeBERTa":
    https://arxiv.org/abs/2006.03654

  Devlin et al. 2019, "BERT":
    https://arxiv.org/abs/1810.04805

  Liu et al. 2019, "RoBERTa":
    https://arxiv.org/abs/1907.11692

Techniques:
  Lin et al. 2017, "Focal Loss" (class imbalance):
    https://arxiv.org/abs/1708.02002

  Ribeiro et al. 2016, "LIME" (explainability):
    https://arxiv.org/abs/1602.04938
""")

subsection("12.3 Online Courses")

print("""
  HuggingFace NLP Course: https://huggingface.co/learn/nlp-course
  Fast.ai Practical DL:   https://course.fast.ai/
  Stanford CS224N (YouTube): Search "Stanford CS224N 2024"
  PyTorch Tutorials:       https://pytorch.org/tutorials/
  Optuna:                  https://optuna.readthedocs.io/
""")

subsection("12.4 Tools & Libraries")

print(f"  {'Library':25s} {'Purpose':35s} {'Install'}")
print(f"  {'-'*90}")
tools = [
    ("transformers", "HuggingFace models (BERT, DeBERTa)", "pip install transformers"),
    ("torch", "Deep learning framework", "pip install torch"),
    ("tinker", "Tinker SDK for LLM training", "pip install tinker"),
    ("scikit-learn", "ML utilities, metrics, splits", "pip install scikit-learn"),
    ("imbalanced-learn", "Class balancing (SMOTE, etc.)", "pip install imbalanced-learn"),
    ("optuna", "Hyperparameter optimization", "pip install optuna"),
    ("lime", "Local model explainability", "pip install lime"),
    ("shap", "Global feature importance", "pip install shap"),
    ("nlpaug", "Text data augmentation", "pip install nlpaug"),
    ("setfit", "Few-shot fine-tuning", "pip install setfit"),
    ("spacy", "Advanced NLP preprocessing", "pip install spacy"),
    ("ydata-profiling", "Automated EDA reports", "pip install ydata-profiling"),
    ("peft", "HuggingFace LoRA/PEFT", "pip install peft"),
    ("datasets", "HuggingFace dataset library", "pip install datasets"),
]
for lib, purpose, install in tools:
    print(f"  {lib:25s} {purpose:35s} {install}")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

section_header("", "REVIEW COMPLETE")

charts_saved = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')]
if charts_saved:
    print(f"Generated charts in: {OUTPUT_DIR}/")
    for f in sorted(charts_saved):
        print(f"  {f}")
    print()

print("""
TOP 5 RECOMMENDATIONS (in priority order):
  1. Fix oversampling order: split FIRST, then oversample training set only
  2. Add early stopping to prevent overfitting
  3. Use official test.tsv for evaluation (comparable with benchmarks)
  4. Include metadata in prompts (speaker, party, context)
  5. Try DeBERTa-v3-large as a simpler, potentially competitive alternative
""")
