"""
Generate all project diagrams and save them to artifacts/diagrams/.

Usage:
    python src/generate_diagrams.py

Output (artifacts/diagrams/):
    01_dataset_splits.png
    02_label_taxonomy.png
    03_preprocessing_pipeline.png
    04_text_cleaning_steps.png
    05_feature_assembly.png
    06_lora_architecture.png
    07_training_loop.png
    08_all_run_metrics.png
    09_lr_comparison.png
    10_loss_curves_all_runs.png
    11_qwen_best_loss_curve.png
    12_evaluation_metrics.png
    13_end_to_end_flow.png
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT      = Path(__file__).parent.parent
RUNS_DIR  = ROOT / 'artifacts' / 'runs'
OUT_DIR   = ROOT / 'artifacts' / 'diagrams'
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = ['barely-true', 'false', 'half-true', 'mostly-true', 'pants-fire', 'true']
BLUE   = '#4C72B0'
GREEN  = '#55A868'
ORANGE = '#DD8452'
RED    = '#C44E52'
PURPLE = '#8172B2'
TEAL   = '#64B5CD'

NVIDIA_COLOR = '#76B900'
QWEN_COLOR   = '#4C72B0'

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'font.family':      'DejaVu Sans',
    'font.size':        11,
})


# ── helpers ───────────────────────────────────────────────────────────────────

def save(name: str):
    path = OUT_DIR / name
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  saved {path.name}")


def flow_box(ax, x, y, text, w=1.6, h=0.5, color='#D0E8FF', fontsize=9, radius=0.08):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=f"round,pad={radius}",
                         facecolor=color, edgecolor='#555', linewidth=1.2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            wrap=True, multialignment='center')


def arrow(ax, x1, y1, x2, y2, color='#555'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))


def load_runs():
    runs = []
    for p in sorted(RUNS_DIR.glob('run_*.json')):
        data = json.loads(p.read_text())
        runs.append(data)
    return runs


# ── 01 dataset splits ─────────────────────────────────────────────────────────

def plot_dataset_splits():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle('Dataset Splits — LIAR2', fontsize=14, fontweight='bold', y=1.01)

    splits  = ['Train', 'Validation', 'Test']
    counts  = [10269, 1284, 1283]
    colors  = [BLUE, ORANGE, GREEN]

    ax = axes[0]
    bars = ax.bar(splits, counts, color=colors, edgecolor='white', linewidth=1.5, width=0.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Sample Counts per Split')
    ax.set_ylim(0, 12000)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    ax2 = axes[1]
    wedges, texts, autotexts = ax2.pie(
        counts, labels=splits, colors=colors,
        autopct='%1.1f%%', startangle=90,
        pctdistance=0.75, wedgeprops=dict(edgecolor='white', linewidth=2)
    )
    for t in autotexts:
        t.set_fontweight('bold')
    ax2.set_title('Proportion of Each Split')

    plt.tight_layout()
    save('01_dataset_splits.png')


# ── 02 label taxonomy ─────────────────────────────────────────────────────────

def plot_label_taxonomy():
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle('Label Taxonomy — Truthfulness Scale', fontsize=14, fontweight='bold')

    labels      = ['true', 'mostly-true', 'half-true', 'barely-true', 'false', 'pants-fire']
    descriptions = [
        'Accurate and fully supported',
        'Mostly accurate, minor omissions',
        'Partially accurate, key facts missing',
        'Grain of truth, mostly misleading',
        'Factually inaccurate',
        'Egregiously false ("pants on fire")',
    ]
    colors = ['#2ecc71', '#82e0aa', '#f9e79f', '#f0b27a', '#e74c3c', '#922b21']

    y_positions = list(range(len(labels), 0, -1))
    bars = ax.barh(y_positions, [1]*len(labels), color=colors,
                   edgecolor='white', linewidth=1.5, height=0.7)

    for i, (label, desc, y) in enumerate(zip(labels, descriptions, y_positions)):
        ax.text(0.02, y, label, va='center', fontweight='bold', fontsize=11, color='#111')
        ax.text(0.52, y, desc, va='center', fontsize=10, color='#333')

    ax.set_xlim(0, 1.0)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel('')

    truthfulness_gradient = ax.annotate(
        '', xy=(1.02, len(labels)+0.35), xytext=(1.02, 0.65),
        arrowprops=dict(arrowstyle='->', color='#555', lw=2)
    )
    ax.text(1.04, (len(labels)+1)/2, 'More\nTruthful', va='center',
            fontsize=9, color='#555', ha='left')

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    save('02_label_taxonomy.png')


# ── 03 preprocessing pipeline ─────────────────────────────────────────────────

def plot_preprocessing_pipeline():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    fig.suptitle('Preprocessing Pipeline', fontsize=14, fontweight='bold')

    steps = [
        (5, 9.0, 'Raw TSV Files\n(train / val / test)',         '#FADADD'),
        (5, 7.5, 'Load & Assign Column Names\n(14 columns)',    '#D0E8FF'),
        (5, 6.0, 'Drop Rows\n(missing label or statement)',     '#D0E8FF'),
        (5, 4.5, 'Clean Each Text Field\n(clean_text)',         '#D0E8FF'),
        (5, 3.0, 'Derive Speaker\nCredibility History',         '#FFE8B2'),
        (5, 1.5, 'Concatenate All Features\nInto One String',   '#D0E8FF'),
        (5, 0.2, 'Save to preprocessed_*.tsv\n(text, label)',   '#D5F5E3'),
    ]

    for x, y, text, color in steps:
        flow_box(ax, x, y, text, w=3.6, h=0.9, color=color, fontsize=9)

    for i in range(len(steps) - 1):
        _, y1, _, _ = steps[i]
        _, y2, _, _ = steps[i+1]
        arrow(ax, 5, y1 - 0.45, 5, y2 + 0.45)

    side_note = (
        "Run once:\n"
        "python src/preprocessing.py\n\n"
        "Output reused by\nevery training run"
    )
    ax.text(8.8, 5.0, side_note, ha='center', va='center', fontsize=8.5,
            color='#444', style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FAFAFA', edgecolor='#aaa'))

    plt.tight_layout()
    save('03_preprocessing_pipeline.png')


# ── 04 text cleaning steps ────────────────────────────────────────────────────

def plot_text_cleaning():
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis('off')
    fig.suptitle('Text Cleaning Steps — DataProcessor.clean_text', fontsize=13, fontweight='bold')

    boxes = [
        (1.0,  2.0, 'Raw Text\n"Says the Annies\nList..."',  '#FADADD'),
        (3.0,  2.0, '1. Lowercase\n"says the annies\nlist..."', '#D0E8FF'),
        (5.0,  2.0, '2. Strip URLs\n& Emails',                  '#D0E8FF'),
        (7.0,  2.0, '3. Remove\nSpecial Chars\n[^a-z0-9 ]',     '#D0E8FF'),
        (9.0,  2.0, '4. Remove\nStopwords\n(NLTK, 179 words)',   '#D0E8FF'),
        (11.0, 2.0, 'Clean Text\n"says annies list\n..."',       '#D5F5E3'),
    ]

    for x, y, text, color in boxes:
        flow_box(ax, x, y, text, w=1.7, h=1.6, color=color, fontsize=8.5)

    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 0.85
        x2 = boxes[i+1][0] - 0.85
        arrow(ax, x1, 2.0, x2, 2.0)

    plt.tight_layout()
    save('04_text_cleaning_steps.png')


# ── 05 feature assembly ───────────────────────────────────────────────────────

def plot_feature_assembly():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    fig.suptitle('Feature Engineering — Feature Assembly', fontsize=13, fontweight='bold')

    raw_fields = [
        (1.2, 6.8, 'statement'),
        (1.2, 5.9, 'speaker'),
        (1.2, 5.0, 'party_affiliation'),
        (1.2, 4.1, 'job_title'),
        (1.2, 3.2, 'subject'),
        (1.2, 2.3, 'context'),
        (1.2, 1.1, '5 count columns'),
    ]
    for x, y, text in raw_fields:
        color = '#FFE8B2' if 'count' in text else '#D0E8FF'
        flow_box(ax, x, y, text, w=1.9, h=0.65, color=color, fontsize=8.5)

    cleaned_fields = [
        (4.5, 6.8, 'clean statement'),
        (4.5, 5.9, 'clean speaker'),
        (4.5, 5.0, 'clean party'),
        (4.5, 4.1, 'clean job'),
        (4.5, 3.2, 'clean subject'),
        (4.5, 2.3, 'clean context'),
    ]
    for x, y, text in cleaned_fields:
        flow_box(ax, x, y, text, w=1.9, h=0.65, color='#D5F5E3', fontsize=8.5)

    for i in range(6):
        arrow(ax, raw_fields[i][0] + 0.95, raw_fields[i][1],
                  cleaned_fields[i][0] - 0.95, cleaned_fields[i][1])

    flow_box(ax, 4.5, 1.1, '_speaker_history\n(credibility stats)', w=1.9, h=0.65,
             color='#FFE8B2', fontsize=8.5)
    arrow(ax, raw_fields[6][0] + 0.95, raw_fields[6][1],
              4.5 - 0.95, 1.1)

    output_text = (
        "Statement: says annies list political\n"
        "           group supports...\n"
        "Speaker:   dwayne bohac\n"
        "Party:     republican\n"
        "Job:       state representative\n"
        "Subject:   abortion\n"
        "Context:   mailer\n"
        "Speaker history: 1 prior claims —\n"
        "  false: 100%. Most common: false"
    )
    flow_box(ax, 9.5, 4.0, output_text, w=4.5, h=5.5, color='#EAF4FF', fontsize=8)

    for x, y, _ in cleaned_fields:
        arrow(ax, x + 0.95, y, 9.5 - 2.25, 4.0)
    arrow(ax, 4.5 + 0.95, 1.1, 9.5 - 2.25, 4.0)

    ax.text(1.2, 7.5, 'Raw Fields', ha='center', fontsize=9, fontweight='bold', color='#333')
    ax.text(4.5, 7.5, 'After clean_text', ha='center', fontsize=9, fontweight='bold', color='#333')
    ax.text(9.5, 7.5, 'Combined Input String', ha='center', fontsize=9, fontweight='bold', color='#333')

    plt.tight_layout()
    save('05_feature_assembly.png')


# ── 06 LoRA architecture ──────────────────────────────────────────────────────

def plot_lora_architecture():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')
    fig.suptitle('LoRA Architecture — Frozen Base + Trainable Adapters', fontsize=13, fontweight='bold')

    flow_box(ax, 5.0, 6.2, 'Input h', w=2.0, h=0.6, color='#FADADD', fontsize=10)

    flow_box(ax, 2.5, 4.3, 'Pretrained Weights W\n(~8B params)  FROZEN', w=3.5, h=1.0, color='#E0E0E0', fontsize=9)
    flow_box(ax, 7.5, 5.2, 'Matrix A  (d × r)\ntrainable', w=2.5, h=0.7, color='#D5F5E3', fontsize=9)
    flow_box(ax, 7.5, 4.0, 'Matrix B  (r × d)\ntrainable', w=2.5, h=0.7, color='#D5F5E3', fontsize=9)

    flow_box(ax, 5.0, 2.4, 'Sum   W·h + B·A·h', w=3.5, h=0.7, color='#FFE8B2', fontsize=10)

    flow_box(ax, 5.0, 1.0, 'Output', w=2.0, h=0.6, color='#D5F5E3', fontsize=10)

    arrow(ax, 5.0, 5.9, 2.5, 4.8)
    arrow(ax, 5.0, 5.9, 7.5, 5.55)
    arrow(ax, 7.5, 4.85, 7.5, 4.35)
    arrow(ax, 2.5, 3.8, 4.2, 2.75)
    arrow(ax, 7.5, 3.65, 5.8, 2.75)
    arrow(ax, 5.0, 2.05, 5.0, 1.3)

    ax.text(1.0, 2.0,
            'rank r = 32\n\nOnly A and B are updated.\nBase model W is frozen.\n\nAdapters add a tiny\nfraction of 8B params.',
            va='top', fontsize=8.5, color='#444',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FAFAFA', edgecolor='#bbb'))

    plt.tight_layout()
    save('06_lora_architecture.png')


# ── 07 training loop ──────────────────────────────────────────────────────────

def plot_training_loop():
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 10)
    ax.axis('off')
    fig.suptitle('Training Loop — TinkerTrainer', fontsize=13, fontweight='bold')

    flow_box(ax, 4.5, 9.3, 'Connect to Tinker API\ncreate_training_client(rank=32)', w=5.0, h=0.8, color='#D0E8FF')
    flow_box(ax, 4.5, 8.1, 'Tokenize samples → Tinker Datums\n(with inverse-frequency class weights)', w=5.0, h=0.8, color='#D0E8FF')
    flow_box(ax, 4.5, 6.9, 'For each epoch (max 5):\n  shuffle training datums', w=5.0, h=0.8, color='#FFF9C4')
    flow_box(ax, 4.5, 5.7, 'For each batch of 8:\n  forward_backward → cross-entropy loss\n  optim_step with Adam', w=5.0, h=1.0, color='#FFF9C4')
    flow_box(ax, 4.5, 4.3, 'Evaluate val loss\n(forward pass, no gradient update)', w=5.0, h=0.8, color='#D0E8FF')
    flow_box(ax, 2.0, 3.0, 'Val loss improved?\nSave as best, reset patience', w=3.2, h=0.8, color='#D5F5E3')
    flow_box(ax, 7.0, 3.0, 'No improvement:\npatience counter++', w=3.2, h=0.8, color='#FADADD')
    flow_box(ax, 7.0, 1.8, 'patience = 2?\nEarly stop', w=3.2, h=0.8, color='#FADADD')
    flow_box(ax, 4.5, 0.5, 'Save LoRA weights → Tinker URI\nEvaluate on test set', w=5.0, h=0.8, color='#D5F5E3')

    arrow(ax, 4.5, 8.9, 4.5, 8.5)
    arrow(ax, 4.5, 7.7, 4.5, 7.3)
    arrow(ax, 4.5, 6.5, 4.5, 6.2)
    arrow(ax, 4.5, 5.2, 4.5, 4.7)
    arrow(ax, 4.5, 3.9, 2.0, 3.4)
    arrow(ax, 4.5, 3.9, 7.0, 3.4)
    arrow(ax, 7.0, 2.6, 7.0, 2.2)
    arrow(ax, 7.0, 1.4, 4.5, 0.9)
    arrow(ax, 2.0, 2.6, 2.0, 1.5)
    ax.annotate('', xy=(4.5, 6.5), xytext=(2.0, 1.5),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.2,
                                connectionstyle='arc3,rad=-0.5'))

    ax.text(0.2, 6.2, 'loop', fontsize=8, color='#888', style='italic')

    plt.tight_layout()
    save('07_training_loop.png')


# ── 08 all run metrics ────────────────────────────────────────────────────────

def plot_all_run_metrics():
    runs = load_runs()
    labels   = [r['run_id'].replace('run_', '') for r in runs]
    accuracy = [r['metrics']['accuracy'] for r in runs]
    macro_f1 = [r['metrics']['macro_f1'] for r in runs]
    models   = [r['config']['base_model'] for r in runs]

    colors = [NVIDIA_COLOR if 'nvidia' in m.lower() else QWEN_COLOR for m in models]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 5.5))
    bars1 = ax.bar(x - width/2, accuracy, width, label='Accuracy', color=colors, alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x + width/2, macro_f1,  width, label='Macro F1', color=colors, alpha=0.55, edgecolor='white')

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7.5)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=8)
    ax.set_ylabel('Score')
    ax.set_title('All Runs — Accuracy and Macro F1', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 0.6)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    nvidia_patch = mpatches.Patch(color=NVIDIA_COLOR, label='Nvidia Nemotron-30B')
    qwen_patch   = mpatches.Patch(color=QWEN_COLOR,   label='Qwen3-8B')
    solid_patch  = mpatches.Patch(color='grey', alpha=0.85, label='Accuracy (solid)')
    faded_patch  = mpatches.Patch(color='grey', alpha=0.55, label='Macro F1 (faded)')
    ax.legend(handles=[nvidia_patch, qwen_patch, solid_patch, faded_patch],
              fontsize=8.5, loc='lower right')

    plt.tight_layout()
    save('08_all_run_metrics.png')


# ── 09 learning rate comparison ───────────────────────────────────────────────

def plot_lr_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle('Learning Rate Comparison — Nvidia vs Qwen', fontsize=13, fontweight='bold')

    runs   = load_runs()
    nvidia = [r for r in runs if 'nvidia' in r['config']['base_model'].lower()]
    qwen   = [r for r in runs if 'Qwen'  in r['config']['base_model']]

    ax = axes[0]
    nvidia_lrs  = [r['config']['learning_rate'] for r in nvidia]
    nvidia_acc  = [r['metrics']['accuracy'] for r in nvidia]
    nvidia_f1   = [r['metrics']['macro_f1'] for r in nvidia]
    nvidia_labels = [r['run_id'].replace('run_', '') for r in nvidia]

    x = np.arange(len(nvidia))
    ax.bar(x - 0.2, nvidia_acc, 0.35, label='Accuracy', color=NVIDIA_COLOR, alpha=0.85)
    ax.bar(x + 0.2, nvidia_f1,  0.35, label='Macro F1',  color=NVIDIA_COLOR, alpha=0.50)
    ax.set_xticks(x)
    ax.set_xticklabels(nvidia_labels, rotation=20, ha='right', fontsize=8)
    ax.set_title('Nvidia Nemotron-30B\n(LR: 2e-5 → 1e-5)', fontsize=10)
    ax.set_ylim(0, 0.6)
    ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
    ax.legend(fontsize=8)
    for i, (a, f, lr) in enumerate(zip(nvidia_acc, nvidia_f1, nvidia_lrs)):
        ax.text(i, 0.02, f'lr={lr:.0e}', ha='center', fontsize=7.5, color='#222')

    ax2 = axes[1]
    qwen_lrs   = [r['config']['learning_rate'] for r in qwen]
    qwen_acc   = [r['metrics']['accuracy'] for r in qwen]
    qwen_f1    = [r['metrics']['macro_f1'] for r in qwen]
    qwen_labels = [r['run_id'].replace('run_', '') for r in qwen]

    x2 = np.arange(len(qwen))
    ax2.bar(x2 - 0.2, qwen_acc, 0.35, label='Accuracy', color=QWEN_COLOR, alpha=0.85)
    ax2.bar(x2 + 0.2, qwen_f1,  0.35, label='Macro F1',  color=QWEN_COLOR, alpha=0.50)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(qwen_labels, rotation=20, ha='right', fontsize=8)
    ax2.set_title('Qwen3-8B\n(LR: 2e-4)', fontsize=10)
    ax2.set_ylim(0, 0.6)
    ax2.yaxis.grid(True, alpha=0.4); ax2.set_axisbelow(True)
    ax2.legend(fontsize=8)
    for i, (a, f, lr) in enumerate(zip(qwen_acc, qwen_f1, qwen_lrs)):
        ax2.text(i, 0.02, f'lr={lr:.0e}', ha='center', fontsize=7.5, color='#222')

    plt.tight_layout()
    save('09_lr_comparison.png')


# ── 10 loss curves all runs ───────────────────────────────────────────────────

def plot_all_loss_curves():
    runs = [r for r in load_runs() if r.get('history') and r['history'].get('train_loss')]

    n    = len(runs)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.2))
    fig.suptitle('Train vs Validation Loss — All Runs', fontsize=13, fontweight='bold')
    axes = axes.flatten()

    for i, run in enumerate(runs):
        ax     = axes[i]
        hist   = run['history']
        model  = run['config']['base_model']
        color  = NVIDIA_COLOR if 'nvidia' in model.lower() else QWEN_COLOR
        epochs = range(1, len(hist['train_loss']) + 1)
        label  = run['run_id'].replace('run_', '')

        ax.plot(epochs, hist['train_loss'], 'o-', color=color,    label='Train', lw=2)
        ax.plot(epochs, hist['val_loss'],   's--', color='#E74C3C', label='Val',   lw=2)
        ax.set_title(label, fontsize=8.5)
        ax.set_xlabel('Epoch', fontsize=8)
        ax.set_ylabel('Loss',  fontsize=8)
        ax.legend(fontsize=7.5)
        ax.yaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)

        acc  = run['metrics']['accuracy']
        lr   = run['config']['learning_rate']
        rank = run['config']['lora_rank']
        ax.set_title(f"{label}\nacc={acc:.3f}  lr={lr:.0e}  rank={rank}", fontsize=7.5)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    save('10_loss_curves_all_runs.png')


# ── 11 Qwen best loss curve ───────────────────────────────────────────────────

def plot_qwen_best_loss():
    runs  = load_runs()
    qwen  = [r for r in runs if 'Qwen' in r['config']['base_model']
             and r.get('history') and r['history'].get('train_loss')]
    best  = max(qwen, key=lambda r: r['metrics']['accuracy'])

    hist   = best['history']
    epochs = range(1, len(hist['train_loss']) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, hist['train_loss'], 'o-', color=QWEN_COLOR,  label='Train Loss', lw=2.5, ms=8)
    ax.plot(epochs, hist['val_loss'],   's--', color='#E74C3C',   label='Val Loss',   lw=2.5, ms=8)

    best_val_ep = int(np.argmin(hist['val_loss'])) + 1
    ax.axvline(best_val_ep, color='#555', linestyle=':', lw=1.5, label=f'Best val (epoch {best_val_ep})')

    ax.fill_between(epochs, hist['train_loss'], hist['val_loss'],
                    alpha=0.08, color='#E74C3C', label='Train–Val gap')

    acc = best['metrics']['accuracy']
    ax.set_title(f"Qwen3-8B — Best Run ({best['run_id'].replace('run_','')})\n"
                 f"lr=2e-4  rank=32  accuracy={acc:.3f}", fontsize=11, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    save('11_qwen_best_loss_curve.png')


# ── 12 evaluation metrics ─────────────────────────────────────────────────────

def plot_evaluation_metrics():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Model Evaluation Metrics — Qwen3-8B Final Run (Apr 15)',
                 fontsize=13, fontweight='bold')

    metrics = {'Accuracy': 0.4544, 'Macro F1': 0.4522, 'Weighted F1': 0.4448}
    names   = list(metrics.keys())
    values  = list(metrics.values())
    colors  = [QWEN_COLOR, GREEN, ORANGE]

    ax = axes[0]
    bars = ax.bar(names, values, color=colors, edgecolor='white', linewidth=1.5, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax.axhline(1/6, color='#aaa', linestyle='--', lw=1.5, label='Random baseline (16.7%)')
    ax.set_ylim(0, 0.65)
    ax.set_ylabel('Score')
    ax.set_title('Headline Metrics')
    ax.legend(fontsize=8.5)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    ax2 = axes[1]
    metric_defs = {
        'Accuracy':    ('correct / total', 'Hides per-class failure'),
        'Macro F1':    ('unweighted mean F1\nacross 6 classes', 'Key metric — penalises\nweak rare-class performance'),
        'Weighted F1': ('support-weighted\nmean F1', 'Reflects typical\nuser experience'),
    }
    ax2.axis('off')
    y = 0.92
    for name, (defn, note) in metric_defs.items():
        ax2.text(0.02, y, name, fontsize=11, fontweight='bold', transform=ax2.transAxes)
        ax2.text(0.02, y - 0.07, defn, fontsize=9.5, color='#333', transform=ax2.transAxes)
        ax2.text(0.02, y - 0.14, note, fontsize=8.5, color='#777', style='italic', transform=ax2.transAxes)
        ax2.axhline(y - 0.17, color='#ddd', lw=1, xmin=0.02, xmax=0.98)
        y -= 0.30
    ax2.set_title('Metric Definitions', fontsize=11, fontweight='bold')

    plt.tight_layout()
    save('12_evaluation_metrics.png')


# ── 13 end-to-end flow ────────────────────────────────────────────────────────

def plot_end_to_end():
    fig, ax = plt.subplots(figsize=(10, 11))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    fig.suptitle('End-to-End Pipeline', fontsize=14, fontweight='bold')

    steps = [
        (5.0, 11.0, 'LIAR2 TSV Files\ntrain / val / test (~12,836 rows)',        '#FADADD'),
        (5.0,  9.5, 'DataProcessor.clean_text\nLowercase → strip URLs/emails\n→ remove special chars → stopwords', '#D0E8FF'),
        (5.0,  7.9, 'DataProcessor._speaker_history\nDerive credibility stats\nfrom 5 count columns',             '#FFE8B2'),
        (5.0,  6.4, 'DataProcessor.build_input_text\nConcatenate 6 fields + history\ninto one prompt string',     '#D0E8FF'),
        (5.0,  4.9, 'DataProcessor.prepare_tinker_dataset\nTokenize → Tinker Datums\n+ inverse-frequency class weights', '#D0E8FF'),
        (5.0,  3.4, 'TinkerTrainer.train\nLoRA fine-tune Qwen3-8B\nrank=32  lr=2e-4  Adam  cross-entropy\nearly stop patience=2', '#C8E6FF'),
        (5.0,  1.9, 'classifier.save_for_inference\nSave LoRA adapter weights\n→ Tinker URI',                     '#FFE8B2'),
        (5.0,  0.5, 'MetricsCalculator.calculate\nAccuracy=45.4%  Macro F1=0.452\nConfusion matrix  Loss curves', '#D5F5E3'),
    ]

    for x, y, text, color in steps:
        flow_box(ax, x, y, text, w=6.5, h=0.95, color=color, fontsize=8.5)

    for i in range(len(steps) - 1):
        _, y1, _, _ = steps[i]
        _, y2, _, _ = steps[i+1]
        arrow(ax, 5, y1 - 0.48, 5, y2 + 0.48)

    side_labels = [
        (8.8, 11.0, 'INPUT'),
        (8.8,  3.4, 'TRAIN'),
        (8.8,  0.5, 'OUTPUT'),
    ]
    label_colors = ['#FADADD', '#C8E6FF', '#D5F5E3']
    for (x, y, text), color in zip(side_labels, label_colors):
        ax.text(x, y, text, ha='center', va='center', fontsize=9,
                fontweight='bold', color='#333',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, edgecolor='#aaa'))

    plt.tight_layout()
    save('13_end_to_end_flow.png')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Saving diagrams to {OUT_DIR}\n")
    plot_dataset_splits()
    plot_label_taxonomy()
    plot_preprocessing_pipeline()
    plot_text_cleaning()
    plot_feature_assembly()
    plot_lora_architecture()
    plot_training_loop()
    plot_all_run_metrics()
    plot_lr_comparison()
    plot_all_loss_curves()
    plot_qwen_best_loss()
    plot_evaluation_metrics()
    plot_end_to_end()
    print(f"\nDone — {len(list(OUT_DIR.glob('*.png')))} diagrams saved.")


if __name__ == '__main__':
    main()
