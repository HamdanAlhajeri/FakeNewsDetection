6. Data Splitting
=================

Splitting Code
--------------

.. code-block:: python

   from training_utils import DataSplitter

   splitter = DataSplitter()
   splits = splitter.split_data(
       X_balanced, y_balanced,
       train_size=0.7, val_size=0.15, test_size=0.15, random_state=42)

   X_train, y_train = splits['train']
   X_val, y_val = splits['val']
   X_test, y_test = splits['test']

Intermediate Output -- Split Sizes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Train:      8,878 samples (70.0%)
   Validation: 1,903 samples (15.0%)
   Test:       1,903 samples (15.0%)
   Total:      12,684 (after oversampling)

Intermediate Output -- Stratification Check
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Label distribution per split (%):
     Label            Train      Val     Test
     barely-true      16.7%    16.7%    16.7%
     false            16.7%    16.7%    16.7%
     half-true        16.7%    16.7%    16.7%
     mostly-true      16.7%    16.7%    16.7%
     pants-fire       16.7%    16.7%    16.7%
     true             16.7%    16.7%    16.7%

   All ~16.7% due to oversampling before splitting.

.. danger:: Data Leakage Risk

   **Current pipeline** oversamples BEFORE splitting:

   1. Oversample minority classes (creates duplicate samples)
   2. Then split into train/val/test

   This means **duplicate copies of the same original sample** can appear in
   both training AND test sets, leading to **artificially inflated accuracy**.

   **Correct order**: Split first, then oversample ONLY the training set.

.. warning:: Not Using Official LIAR2 Splits

   The dataset provides official ``valid.tsv`` and ``test.tsv`` files, but the
   pipeline only loads ``train.tsv`` and re-splits it randomly.

   This makes results **incomparable with published benchmarks**.

.. admonition:: Alternatives -- Better Splitting

   **1. Use official splits** (for publishable results):

   .. code-block:: python

      train = pd.read_csv('data/train.tsv', sep='\t', ...)
      valid = pd.read_csv('data/valid.tsv', sep='\t', ...)
      test  = pd.read_csv('data/test.tsv',  sep='\t', ...)
      # Oversample ONLY the training set

   **2. Fix the leakage**:

   .. code-block:: python

      # Split FIRST on original data
      X_train, X_temp, y_train, y_temp = train_test_split(X, y, ...)
      X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, ...)
      # THEN oversample training set only
      X_train, y_train = balancer.oversample_minority(X_train, y_train)

   **3. Stratified K-Fold cross-validation**:

   .. code-block:: python

      from sklearn.model_selection import StratifiedKFold
      skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
      for train_idx, val_idx in skf.split(X, y):
          # Train and evaluate on each fold
      # Report: mean +/- std for confidence intervals

   **4. Group-based splitting** (prevent speaker leakage):

   .. code-block:: python

      from sklearn.model_selection import GroupShuffleSplit
      gss = GroupShuffleSplit(n_splits=1, test_size=0.15)
      # Split by speaker to avoid same person in train and test

   Video: `StatQuest Cross-Validation <https://www.youtube.com/watch?v=fSytzGwwBVw>`_
