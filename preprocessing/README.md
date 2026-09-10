# Preprocessing

This folder contains the preprocessing scripts used to create the three input representations in the LEAD+ dataset.

## Scripts

* `01_pdf_to_markdown.py`
  Converts the original PDF decisions to Markdown using Docling.

* `02_operative_section.py`
  Extracts the operative section from the full-text Markdown files.

* `03_sentence_selection.py`
  Selects sentences containing predefined combinations of Legal Actions and Legal Objects.

## Usage

From the repository root:

```bash
python preprocessing/01_pdf_to_markdown.py path/to/pdfs
python preprocessing/02_operative_section.py
python preprocessing/03_sentence_selection.py
```

The scripts produce the following folders:

```text
LEAD+ Dataset/
├── full_text/
├── operative_section/
└── key_sentences/
```

## Dependencies

The preprocessing requires Python and the following packages:

```text
docling
spacy
```

The sentence-selection script uses the Dutch spaCy model:

```bash
python -m spacy download nl_core_news_lg
```
