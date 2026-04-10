2. Text Preprocessing
=====================

The ``DataProcessor.clean_text()`` method applies a 5-step cleaning pipeline:

.. code-block:: text

   1. Convert to lowercase
   2. Remove URLs (http/https/www patterns)
   3. Remove email addresses
   4. Remove ALL special characters (keep only alphanumeric + spaces)
   5. Remove extra whitespace
   6. (Optional) Remove English stopwords via NLTK

Cleaning Code
-------------

.. code-block:: python

   from data_processor import DataProcessor
   processor = DataProcessor()

   # Single text
   cleaned = processor.clean_text(text, remove_stops=True)

   # Full pipeline (clean + encode labels)
   df_processed, encoded_labels = processor.process_pipeline(
       df, text_column='statement', label_column='label', remove_stops=True)

Intermediate Output -- Before/After Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    #  Original                                            Cleaned (stopwords removed)
    -  --------------------------------------------------  --------------------------------------------------
    1  Says the Annies List political group supports th..  says annies list political group supports thirdtr..
    2  The Obama administration has spent $120.5 millio..  obama administration spent 1205 million stimulus ..
    3  When did the decline of coal start? It started w..  decline coal start started natural gas took
    4  "We have 90,000 fewer people working in Texas t..  90000 fewer people working texas today
    5  Health care reform is likely to mandate free sex..  health care reform likely mandate free sexchange ..

Intermediate Output -- Step-by-Step Transformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Input: "The Obama admin has spent $120.5M on signs. See http://example.com for info."

   Step 1 - lowercase:      "the obama admin has spent $120.5m on signs. see http://example.com for info."
   Step 2 - URL removal:    "the obama admin has spent $120.5m on signs. see  for info."
   Step 3 - special chars:  "the obama admin has spent 1205m on signs see  for info"
   Step 4 - whitespace:     "the obama admin has spent 1205m on signs see for info"
   Step 5 - stopwords:      "obama admin spent 1205m signs see info"

Intermediate Output -- Pipeline Result
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Input shape:     (10240, 15)
   Output shape:    (10240, 18)   <-- added text_cleaned, char_length, word_count
   Rows dropped:    0
   Labels encoded:  (10240,)  unique values: [0 1 2 3 4 5]

Stopword Removal Impact
------------------------

.. code-block:: text

   Top 15 words WITHOUT stopwords:
     says                  1847
     percent                907
     state                  831
     obama                  741
     tax                    711
     years                  610
     health                 593
     people                 554
     president              521
     states                 511
     year                   501
     would                  476
     us                     472
     care                   437
     million                423

.. image:: ../review_outputs/04_stopword_comparison.png
   :width: 700
   :alt: Stopword comparison
   :class: only-with-data

.. warning:: Stopword Removal is UNUSED by Tinker Training

   The ``process_pipeline()`` saves cleaned text to the ``text_cleaned`` column.
   But **notebook 03 loads the ``statement`` column** (original text) for Tinker.

   The entire cleaning pipeline (lowercase, URL removal, stopwords) is
   **bypassed** for the actual LLM training path.

   This is actually **correct** -- LLMs need natural text with original casing,
   punctuation, and stopwords intact.

.. admonition:: Alternative -- Better Preprocessing

   **For LLM fine-tuning** (current approach): Use minimal or NO preprocessing.
   LLMs are pretrained on natural text and expect it during fine-tuning.

   **For traditional models** (LSTM/CNN): Use spaCy instead of NLTK:

   .. code-block:: python

      import spacy
      nlp = spacy.load('en_core_web_sm')
      doc = nlp(text)
      tokens = [t.lemma_ for t in doc if not t.is_stop and not t.is_punct]

   Benefits: lemmatization, NER preservation, better tokenization.

   **Keep numbers**: Current regex removes ``$120.5 million`` -> ``1205 million``.
   Budget figures and percentages are informative for fact-checking.

   See: `spaCy Linguistic Features <https://spacy.io/usage/linguistic-features>`_
