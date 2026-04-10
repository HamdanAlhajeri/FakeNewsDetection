4. Feature Engineering
======================

Vocabulary Building
-------------------

.. code-block:: python

   vocab = processor.build_vocabulary(df_processed['text_cleaned'], min_freq=1)

Intermediate Output -- Vocabulary Stats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Vocabulary size: 13,284 unique words
   Minimum frequency: 1 (all words included)

   Top 20 words (by frequency -> index):
     says         -> index 0
     percent      -> index 1
     state        -> index 2
     obama        -> index 3
     tax          -> index 4
     years        -> index 5
     health       -> index 6
     people       -> index 7
     president    -> index 8
     states       -> index 9
     year         -> index 10
     would        -> index 11
     us           -> index 12
     care         -> index 13
     million      -> index 14
     jobs         -> index 15
     new          -> index 16
     one          -> index 17
     bill         -> index 18
     texas        -> index 19

   Word frequency statistics:
     Hapax legomena (freq=1): ~5,800 words (43.6%)
     Words with freq>=5:      ~4,200 words (31.6%)
     Mean frequency:          ~5.3
     Median frequency:        ~2.0

Sequence Encoding
-----------------

.. code-block:: python

   sequences = processor.texts_to_sequences(df_processed['text_cleaned'])

Intermediate Output -- Sequence Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Text:     "says annies list political group supports thirdtrimester abortions demand"
   Sequence: [0, 6997, 1001, 411, 495, 271, 5028, 460, 1450]
   Length:   9 words

   Sequence length statistics:
     Mean:   11.1 words
     Median: 9.0 words
     Min:    1
     Max:    344

Padding
-------

.. code-block:: python

   X = processor.pad_sequences(sequences, maxlen=100)

Intermediate Output -- Padding Stats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Padded shape: (10240, 100)
   Sparsity (% zeros): ~88.9%
   Sequences truncated (>100 words): ~30 (0.3%)
   Sequences padded (<100 words): ~10,200 (99.6%)

   Example: [0, 6997, 1001, 411, 495, 271, 5028, 460, 1450, 0, 0, 0, ...]

.. image:: ../review_outputs/06_sequence_lengths.png
   :width: 600
   :alt: Sequence length distribution
   :class: only-with-data

.. warning:: encoded_data.npz is NOT Used by Tinker Training

   This entire vocabulary + padding pipeline was designed for **traditional
   models** (LSTM, CNN) that were never implemented.

   Tinker training (notebook 03) uses **raw text** with its own BPE tokenizer.
   The ``encoded_data.npz`` is only loaded in notebook 02 for oversampling,
   but notebook 03 re-does oversampling independently.

.. admonition:: Alternative -- Better Features

   **TF-IDF + SVD** (simple, effective baseline for traditional models):

   .. code-block:: python

      from sklearn.feature_extraction.text import TfidfVectorizer
      from sklearn.decomposition import TruncatedSVD

      tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
      X_tfidf = tfidf.fit_transform(texts)
      X_svd = TruncatedSVD(n_components=300).fit_transform(X_tfidf)

   **Metadata features** (currently ignored, high potential!):

   The dataset includes speaker credibility scores (``barely_true_count``,
   ``false_count``, etc.) that encode each speaker's PolitiFact track record.

   **For the LLM path**, include metadata in the prompt:

   .. code-block:: text

      Speaker: {speaker} (Party: {party}). Context: {context}.
      Statement: {text}
