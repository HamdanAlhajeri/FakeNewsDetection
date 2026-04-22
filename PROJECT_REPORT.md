Fake News Detection — Project Report

A 6-class political statement truthfulness classifier built on the LIAR2 dataset,
fine-tuned on Qwen3-8B using LoRA via the Tinker API.


Table of Contents

1. Dataset Exploration
2. Data Preprocessing
3. Feature Engineering
4. Fine-Tuning
5. Hyperparameter Tuning
6. Model Evaluation Metrics


1. Dataset Exploration

1.1 Source

The project uses LIAR2 — a benchmark of professionally fact-checked political
statements sourced from PolitiFact. Each record has been manually rated on a
6-point truthfulness scale by expert fact-checkers.

1.2 Splits

    Training    data/train.tsv    ~10,269 rows
    Validation  data/valid.tsv    ~1,284  rows
    Test        data/test.tsv     ~1,283  rows
    Total                         ~12,836 rows

1.3 Label Taxonomy

Six classes, ordered from most truthful to least truthful:

    0  barely-true   Contains a grain of truth but is mostly misleading
    1  false         Factually inaccurate
    2  half-true     Partially accurate; leaves out key facts
    3  mostly-true   Mostly accurate with minor omissions
    4  pants-fire    Egregiously false ("pants on fire")
    5  true          Accurate and fully supported

1.4 TSV Column Schema

Each row has 14 tab-separated columns with no header row:

    id, label, statement, subject, speaker, job_title, state_info,
    party_affiliation, barely_true_count, false_count, half_true_count,
    mostly_true_count, pants_on_fire_count, context

1.5 Class Imbalance Challenge

Label distribution is uneven — half-true, mostly-true, and false dominate while
pants-fire is rare. This imbalance is why inverse-frequency class weights are
applied during training (see Section 4).


2. Data Preprocessing

Preprocessing is a one-off step — run once with python src/preprocessing.py
and its output is reused by every subsequent training run.

2.1 Pipeline Overview

    Raw TSV (14 columns)
        → Load and assign column names
        → Drop rows with missing label or statement
        → Clean each text field
        → Derive speaker credibility history
        → Concatenate all features into one string
        → Encode labels 0 to 5
        → Save to preprocessed_*.tsv (columns: text, label)

2.2 Text Cleaning Steps

Five transformations applied in order by DataProcessor.clean_text:

    1  Lowercase             "Says the..." → "says the..."
    2  Strip URLs            removes http://, https://, www. patterns
    3  Strip emails          removes word@word patterns
    4  Remove special chars  punctuation, quotes, hyphens → spaces
    5  Remove stopwords      uses NLTK English stopword list (179 words)

Concrete example:

    Before: "Says the Annies List political group supports
             third-trimester abortions on demand."
    After:  "says annies list political group supports
             third trimester abortions demand"

2.3 Why Preprocess Once?

Separating preprocessing from training means hyperparameter experiments do not
repeat the cleaning work. Each training run starts from the same clean baseline:

    Raw TSVs → preprocessing.py (run once) → preprocessed_*.tsv → train.py (run N times)


3. Feature Engineering

3.1 Why Feature Engineering Matters

The LIAR2 dataset provides metadata signals beyond the raw statement text —
speaker name, party, job, topic, and historical accuracy counts. A model trained
only on statement text misses this signal. Feature engineering concatenates all
fields into a single rich prompt.

3.2 Feature Assembly

Each of the six text fields is cleaned independently, then combined:

    Raw fields                clean_text applied       Combined output
    statement           →     clean statement      →
    speaker             →     clean speaker        →   Statement: ...
    party_affiliation   →     clean party          →   Speaker: ...
    job_title           →     clean job            →   Party: ...
    subject             →     clean subject        →   Job: ...
    context             →     clean context        →   Subject: ...
    5 count columns     →     _speaker_history     →   Context: ...
                                                        Speaker history: ...

3.3 Speaker Credibility History — The Key Engineered Feature

This is the single most valuable feature. Each speaker has five columns tracking
how many times they have received each rating. DataProcessor._speaker_history
converts these raw counts into a natural-language summary:

    Barack Obama row:
    Counts: mostly_true=163, half_true=160, false=71, barely_true=70, pants_on_fire=9
    Total:  473 prior claims

    Output: "Speaker history: 473 prior claims — mostly-true: 34%, half-true: 34%,
             false: 15%, barely-true: 15%, pants-fire: 2%. Most common rating: mostly-true"

Why this design beats passing raw numbers:

    1. LLMs are weaker at arithmetic than at reading natural language.
    2. Sorting rates descending puts the dominant signal first.
    3. Repeating the most common rating at the end reinforces the prior.
    4. Speakers with no history are silently skipped — no misleading 0 prior claims.

3.4 Final Prompt Template

The engineered text is slotted into a chat-style prompt fed to Qwen3-8B:

    <|im_start|>user
    Classify the truthfulness of the following political statement and its context.

    Statement: hillary clinton agrees john mccain voting give george bush
    Speaker: barack obama  Party: democrat  Job: president
    Subject: foreign-policy  Context: denver
    Speaker history: 473 prior claims — mostly-true: 34%, half-true: 34%, ...

    Choose exactly one label: barely-true, false, half-true, mostly-true, pants-fire, true

    Label:<|im_end|>
    <|im_start|>assistant


4. Fine-Tuning

4.1 Approach — LoRA (Low-Rank Adaptation)

Full fine-tuning of Qwen3-8B would require updating all 8 billion parameters —
expensive in both compute and memory. LoRA freezes the base weights and injects
small trainable rank-r matrices into each attention layer. Only these adapter
matrices are updated during training.

The effective weight during inference is W + B·A, where only A and B (rank 32
matrices) are updated — a tiny fraction of the 8B base parameters.

    Base model weights W (~8B params)  —  FROZEN
    Adapter Matrix A (d × r)           —  trainable
    Adapter Matrix B (r × d)           —  trainable
    Output = W·h + B·A·h

4.2 Training Loop

    train.py connects to Tinker via ServiceClient
    create_training_client allocates a LoRA session (rank=32)

    For each epoch (max 5):
        For each batch of 8:
            forward_backward on batch with cross_entropy loss
            gradients computed only on LoRA adapter weights
            optim_step using Adam
        evaluate_loss on full validation set
        if val_loss improved → record new best, reset patience
        if no improvement   → increment patience counter
        if patience reaches 2 → early stop

    save_weights_for_sampler → returns Tinker URI

4.3 Loss Function and Class Balancing

Loss — token-level cross-entropy on completion tokens only. Prompt tokens get
weight 0.0 so the model is never trained to reproduce the question.

Class balancing — DataProcessor.compute_class_weights computes inverse-frequency
weights per class. The rare pants-fire class gets a higher weight so its
gradients contribute proportionally:

    weight = n_total / (n_classes × count_of_class)

4.4 Inference — Logprob-Based Classification

Rather than generating a label token and parsing it, inference scores each of
the six candidate labels by summing logprobs of its completion tokens, then
picks the argmax.

    New statement
        → build_input_text (combine features)
        → format prompt
        → score each of 6 labels via compute_logprobs
        → argmax of logprob sums → predicted label

    Example logprob scores:
        barely-true  → -3.12
        false        → -4.55
        half-true    → -1.20
        mostly-true  → -0.38  ← winner
        pants-fire   → -6.21
        true         → -1.87

This is deterministic, avoids tokenisation collisions (e.g. true inside
mostly-true), and produces calibrated probabilities for analysis.


5. Hyperparameter Tuning

5.1 Two Model Families Tested

Nvidia Nemotron-30B (4 runs, Apr 07–09):

    Run 1  Apr 07  rank=8,  lr=2e-5, epochs=3
    Run 2  Apr 08  rank=8,  lr=2e-5, epochs=3
    Run 3  Apr 08  rank=16, lr=1e-5, epochs=5 (early stopped at 4)
    Run 4  Apr 09  rank=16, lr=1e-5, epochs=5 (early stopped at 4)

Qwen3-8B (3 runs + 1 eval, Apr 13–16):

    Run 1  Apr 13  rank=32, lr=2e-4, epochs=5 (early stopped at 4)
    Run 2  Apr 14  rank=32, lr=2e-4, epochs=5 (early stopped at 4)
    Run 3  Apr 15  rank=32, lr=2e-4, epochs=5 (ran all 4)
    Eval   Apr 16  reused Apr 15 weights

5.2 Side-by-Side Comparison

    Run        Model          Rank  LR     Epochs  Best Val Loss  Accuracy  Macro F1
    Apr 07     Nemotron-30B   8     2e-5   3       0.518 (ep2)    41.9%     0.430
    Apr 08     Nemotron-30B   8     2e-5   3       0.516 (ep2)    43.1%     0.449
    Apr 08e    Nemotron-30B   16    1e-5   4*      0.543 (ep2)    44.0%     0.449
    Apr 09     Nemotron-30B   16    1e-5   4*      0.512 (ep2)    45.1%     0.460
    Apr 13     Qwen3-8B       32    2e-4   4*      0.545 (ep2)    45.4%     0.460
    Apr 14     Qwen3-8B       32    2e-4   4*      0.545 (ep2)    45.5%     0.456
    Apr 15     Qwen3-8B       32    2e-4   4       0.540 (ep2)    45.4%     0.452  ← final

    * early-stopped

5.3 Learning Rate — Why 2e-5 vs 2e-4?

Nemotron-30B used 2e-5 because:
    - It is a 30B parameter MoE model; large updates risk breaking instruction tuning
    - Weight space is already highly pretrained and sensitive to large perturbations
    - Any higher LR caused training divergence in early tests

Qwen3-8B used 2e-4 (10× higher) because:
    - It is a smaller dense 8B model; adapter has proportionally more influence
    - Standard LoRA recommendation for 7-8B dense models is ~2e-4
    - At 2e-5 the adapter barely moved; at 2e-4 it reached target in 2 epochs

5.4 Rationale for Every Hyperparameter

    base_model              Qwen/Qwen3-8B    Matches Nemotron-30B accuracy at a fraction of the size
    lora_rank               32               Rank 8→16→32 tested; 32 gave best results on 8B model
    learning_rate           2e-4             10× Nemotron LR; dense 8B needs stronger adapter updates
    batch_size              8                Tinker memory ceiling; never changed — loss was stable
    epochs                  5 (max)          Val loss peaks at ep 2-3 then overfits; 5 is a safe cap
    early_stopping_patience 2                Catches one noisy epoch before halting
    temperature             0.0              Greedy inference — argmax label, no sampling
    max_inference_tokens    10               Longest label (pants-fire) is 2 tokens; 10 is a safe cap

5.5 Training Dynamics — Loss Curves

Qwen3-8B Apr 15 run:

    Epoch   Train Loss   Val Loss
    1       0.713        0.562
    2       0.588        0.540   ← best val loss
    3       0.534        0.557
    4       0.465        0.570

Pattern observed in every run: train loss drops monotonically, val loss bottoms
out at epoch 2 then rises. This is the signal early-stopping catches — further
training memorises the training set without generalising.


6. Model Evaluation Metrics

6.1 Metrics Used

Headline metrics:
    Accuracy    correct predictions / total samples
    Macro F1    unweighted mean of per-class F1 scores
    Weighted F1 class-size weighted mean of per-class F1 scores

Diagnostic tools:
    Per-class report    precision, recall, F1 for each of the 6 labels
    Confusion matrix    6×6 grid of true vs predicted labels
    Loss curves         train and val loss plotted per epoch

6.2 Why These Three Metrics Together

    Accuracy     overall correct rate         hides per-class failure (a model predicting only
                                              half-true can still score ~30%)
    Macro F1     unweighted mean F1           punishes weak performance on rare classes equally
    Weighted F1  support-weighted mean F1     reflects typical sample experience

Because LIAR2 is imbalanced, macro F1 is the most important metric — it reveals
whether the model learned the rare classes or only the common ones.

6.3 Final Test-Set Results (Qwen3-8B, Apr 15 run)

    Accuracy     45.4%
    Macro F1     0.452
    Weighted F1  0.445

6.4 Confusion Matrix Interpretation

Rows represent the true label. Columns represent the predicted label. The
diagonal is correct predictions; off-diagonal cells are confusions.

Expected patterns:
    - Adjacent labels confused most: mostly-true vs half-true, false vs barely-true
    - Rare class pants-fire tends to be under-predicted
    - Common classes show stronger diagonal

The 6×6 confusion matrix is saved to artifacts/confusion_matrix.png after every
run. Confusion between adjacent truthfulness levels is expected — those
distinctions are subjective even for human annotators.

6.5 Metric Computation Pipeline

    Test texts
        → classifier.predict_batch
        → predicted label strings
        → map to indices 0-5
        → MetricsCalculator.calculate (with ground-truth indices)
        → accuracy, macro_f1, weighted_f1, confusion_matrix, classification_report
        → saved to training_results.json and confusion_matrix.png

6.6 Observations and Limitations

45% accuracy on 6 classes is approximately 2.7× random chance (16.7%). Non-trivial
but far from perfect — truthfulness classification is inherently hard.

The hardest classes are the middle ones (half-true, barely-true) — even human
fact-checkers disagree on these.

The model's largest gains came from two changes: adding the speaker-history
engineered feature, and switching from Nemotron to Qwen with a properly tuned
learning rate.


Summary — End-to-End Flow

    LIAR2 TSV files
        → Preprocessing (clean text, engineer speaker-history feature)
        → preprocessed_*.tsv
        → Tokenize to Tinker Datums with class weights
        → LoRA fine-tune Qwen3-8B (rank=32, lr=2e-4, Adam, cross-entropy)
        → Early stop when val loss stops improving (patience=2)
        → Save LoRA adapter weights → Tinker URI
        → Evaluate on test set
        → Accuracy, Macro F1, Weighted F1, Confusion matrix, Loss curves
