# Legal IE Input Selection

This repository accompanies a study on how input representation affects information extraction from Dutch administrative decisions (*beschikkingen*).

The extraction task covers four information types:

* **Recipient**
* **Decision-Making Authority**
* **Legal Action**
* **Legal Object**

The dataset contains 700 administrative decisions from seven Dutch public authorities. Each document is represented using three input representations: the complete document, the operative section, and selected sentences.

## Repository structure

```text
legal-ie-input-selection/
├── LEAD+ Dataset/
│   ├── full_text/
│   ├── operative_section/
│   ├── key_sentences/
│   ├── golden_truth/
│   └── document_mapping.csv
├── preprocessing/
├── extraction/
├── results/
└── evaluation/
```

The `golden_truth` folder contains the manually validated reference annotations. The `results` folder contains the model extractions for the three input representations. Evaluation results are derived from `evaluation/results_verification.xlsm`.

## Data sources

The administrative decisions were collected from the public portals of the respective Dutch authorities:

* Autoriteit Consument & Markt (ACM)
* Autoriteit Nucleaire Veiligheid en Stralingsbescherming (ANVS)
* Autoriteit Persoonsgegevens (AP)
* Kansspelautoriteit (KSA)
* Rijksoverheid / Dutch central government
* Province of Drenthe
* Municipality of Rotterdam

The source documents were publicly available through the official portals of these authorities at the time of collection. The repository does not claim ownership of the original administrative decisions. Rights and reuse conditions applicable to the original documents remain with the respective public authorities and source portals.

## Reproducing the experiment

The preprocessing scripts create the three input representations:

```bash
python preprocessing/01_pdf_to_markdown.py path/to/pdfs
python preprocessing/02_operative_section.py
python preprocessing/03_sentence_selection.py
```

The extraction script runs the information extraction task independently for each representation:

```bash
python extraction/run_extraction.py
```

Evaluation is based on the manually reviewed results in `evaluation/results_verification.xlsm`.

## Data

These documents were obtained from publicly accessible official portals of the relevant Dutch public authorities, including ACM, ANVS, Autoriteit Persoonsgegevens, Kansspelautoriteit, the Dutch central government, the Province of Drenthe, and the Municipality of Rotterdam.

The repository does not claim ownership of these source documents. Their availability on a public authority portal should not be interpreted as a transfer of copyright or other rights by this repository. Any rights, attribution requirements, or reuse conditions applying to the original documents remain those of the respective authority or source portal.
