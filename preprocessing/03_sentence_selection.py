from pathlib import Path
import re
import shutil

import spacy


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "LEAD+ Dataset" / "full_text"
OUTPUT_DIR = ROOT / "LEAD+ Dataset" / "key_sentences"
SPACY_MODEL = "nl_core_news_lg"

GAP = r"(?:\s+\S+){0,10}?\s+"

OBJECT_PATTERNS = {
    "vergunning": r"\b(?:[\w-]*vergunning)(?:en)?\b",
    "boete": r"\b(?:bestuurlijke\s+)?boete(?:s)?\b",
    "dwangsom": r"\b(?:dwangsom(?:men)?|last(?:en)?\s+onder\s+dwangsom|verbeurde\s+dwangsom(?:men)?)\b",
    "ontheffing": r"\bontheffing(?:en)?\b",
    "subsidie": r"\b(?:subsidie(?:verlening|bedrag|voucher)?|instellingssubsidie)(?:s)?\b",
    "concessie": r"\bconcessie(?:s)?\b",
    "aanwijzing": r"\b(?:bindende\s+)?aanwijzing(?:en)?\b",
    "bestuursdwang": r"\b(?:last\s+onder\s+)?bestuursdwang\b",
    "goedkeuring": r"\bgoedkeuring(?:en)?\b",
    "bezwaar": r"\b(?:bezwaar|bezwaren|bezwaarschrift(?:en)?)\b",
    "aanvraag": r"\b(?:[\w-]*aanvraag)(?:en)?\b",
    "verzoek": r"\bverzoek(?:en)?\b",
    "besluit": r"\b(?:besluit(?:en)?|beschikking(?:en)?|invorderingsbesluit(?:en)?|verleningsbeschikking(?:en)?)\b",
}

ACTION_PATTERNS = {
    "verlenen": r"\bverleen\b|\bverlenen\b|\bverleent\b" rf"|\bwordt\b{GAP}\bverleend\b" rf"|\bverleend\b{GAP}\bwordt\b" r"|\bverlening(?:en)?\b" rf"|\bworden\b{GAP}\bverleend\b",
    "intrekken": r"\bintrek(?:ken|t)?\b" rf"|\btrek(?:t)?\b{GAP}\bin\b" rf"|\bin\b{GAP}\btrek(?:ken|t)?\b" r"|\bin\s+te\s+trekken\b" rf"|\bwordt\b{GAP}\bingetrokken\b" r"|\bintrekking(?:en)?\b|\bingetrokken\b",
    "wijzigen": r"\bwijzig(?!ing)\w*\b" rf"|\bwordt?\b{GAP}\bgewijzigd\b" r"|\bwijziging(?:en)?\b|\bgewijzigd\b",
    "opleggen": r"\bopleg(?!ging)\w*\b" rf"|\bleg(?:t)?\b{GAP}\bop\b" r"|\bop\s+te\s+leggen\b" rf"|\bwordt\b{GAP}\bopgelegd\b" r"|\boplegging(?:en)?\b|\bopgelegd\b",
    "afwijzen": r"\bafwijzen\b" rf"|\bwijs(?:t)?\b{GAP}\baf\b" r"|\baf\s+te\s+wijzen\b",
    "goedkeuren": r"\bgoedkeur(?!ing)\w*\b" rf"|\bkeur(?:t)?\b{GAP}\bgoed\b" rf"|\bwordt\b{GAP}\bgoed\s*gekeurd\b",
    "toekennen": r"\btoeken(?!ning)\w*\b" rf"|\bken(?:t)?\b{GAP}\btoe\b" rf"|\btoe\b{GAP}\bken(?:nen|t)?\b" r"|\btoe\s+te\s+kennen\b" rf"|\bwordt\b{GAP}\btoegekend\b",
    "verstrekken": r"\bverstrekken\b|\bverstrek(?:t)?\b",
    "weigeren": r"\bweiger(?!ing)\w*\b|\bte\s+weigeren\b",
    "verhogen": r"\bverhoog(?!ing)\w*\b|\bverhogen\b",
    "verlagen": r"\bverlaag(?!ing)\w*\b|\bverlagen\b",
    "verlengen": r"\bverleng(?!ing)\w*\b|\bverlengen\b",
    "verkorten": r"\bverkort(?!ing)\w*\b|\bverkorten\b",
    "vervallen": r"\bvervalt\b|\bvervallen\b",
    "beeindigen": r"\b(?:beëindig|beeindig)(?:d|t|en)?\b",
    "vaststellen": r"\bvaststel(?!ling)\w*\b" rf"|\bstel(?:t)?\b{GAP}\bvast\b" rf"|\bvast\b{GAP}\bstel(?:len|t)?\b" r"|\bvast\s+te\s+stellen\b" rf"|\bwordt\b{GAP}\bvastgesteld\b",
    "afzien": r"\bafzien\w*\b" rf"|\bziet\b{GAP}\baf\b" rf"|\bwordt\b{GAP}\bafgezien\b",
    "opheffen": r"\bophef(?!fing)\w*\b" rf"|\bhef(?:t)?\b{GAP}\bop\b" r"|\bop\s+te\s+heffen\b" rf"|\bwordt\b{GAP}\bopgeheven\b",
    "opschorten": r"\bschort\b" + GAP + r"\bop\b|\bop\s+te\s+schorten\b" + rf"|\bwordt\b{GAP}\bopgeschort\b",
    "verminderen": r"\bverminder(?:d|t|en)\b",
    "invorderen": r"\bvorder(?:t)?\b" + GAP + r"\bin\b|\bin\s+te\s+vorderen\b" + rf"|\bwordt\b{GAP}\bingevorderd\b" r"|\binvordering(?:en)?\b|\bingevorderd\b",
    "geven": r"\bgeef(?:t)?\b|\bgeven\b" rf"|\bwordt\b{GAP}\bgegeven\b",
    "bindend_verklaren": rf"\bverklaar(?:t|d)?\b{GAP}\bbindend\b" rf"|\bwordt\b{GAP}\bbindend\b{GAP}\bverklaard\b",
    "gegrond_verklaren": rf"\b(?:verklaar(?:t|d)?|verklaren|verklaarde)\b{GAP}\bgegrond\b" rf"|\bgegrond\b{GAP}\b(?:verklaar(?:t|d)?|verklaren|verklaarde)\b",
    "ongegrond_verklaren": rf"\b(?:verklaar(?:t|d)?|verklaren|verklaarde)\b{GAP}\bongegrond\b" rf"|\bongegrond\b{GAP}\b(?:verklaar(?:t|d)?|verklaren|verklaarde)\b",
    "niet_ontvankelijk_verklaren": rf"\b(?:verklaar(?:t|d)?|verklaren|verklaarde)\b{GAP}\bniet[-\s]?ontvankelijk\b" rf"|\bniet[-\s]?ontvankelijk\b{GAP}\b(?:verklaar(?:t|d)?|verklaren|verklaarde)\b",
    "herroepen": r"\bherroep\w*\b",
    "handhaven": r"\b(?:handhaaf|handhaaft|handhaven|gehandhaafd)\b",
    "toewijzen": r"\btoewij\w*\b" rf"|\bwijs(?:t)?\b{GAP}\btoe\b" r"|\btoe\s+te\s+wijzen\b",
    "buiten_behandeling_stellen": rf"\bstel\w*\b{GAP}\bbuiten\s+behandeling\b" r"|\bbuiten\s+behandeling\s+gesteld\b" r"|\bbuiten\s+behandeling\b",
    "matigen": r"\bmatig\w*\b",
    "herzien": r"\bherzi(?:en|et|ene|end)\w*\b",
}

OBJECT_ACTIONS = {
    "vergunning": ["verlenen", "intrekken", "wijzigen", "verlengen", "vervallen", "beeindigen"],
    "boete": ["opleggen", "intrekken", "wijzigen", "vaststellen", "afzien", "verhogen", "verlagen", "matigen"],
    "dwangsom": ["opleggen", "intrekken", "wijzigen", "opheffen", "verlengen", "verkorten", "verlagen", "verhogen", "opschorten", "verminderen", "invorderen"],
    "ontheffing": ["verlenen", "intrekken", "wijzigen"],
    "subsidie": ["verlenen", "toekennen", "verstrekken", "intrekken", "wijzigen", "weigeren", "verhogen", "verlagen", "vaststellen", "herzien"],
    "concessie": ["verlenen", "toekennen", "intrekken", "wijzigen", "beeindigen"],
    "aanwijzing": ["opleggen", "intrekken", "geven", "opheffen", "bindend_verklaren", "wijzigen"],
    "bestuursdwang": ["opleggen", "intrekken", "wijzigen", "opheffen"],
    "goedkeuring": ["verlenen", "goedkeuren"],
    "bezwaar": ["gegrond_verklaren", "ongegrond_verklaren", "niet_ontvankelijk_verklaren"],
    "aanvraag": ["afwijzen", "toewijzen", "buiten_behandeling_stellen", "intrekken"],
    "verzoek": ["afwijzen", "toewijzen", "niet_ontvankelijk_verklaren"],
    "besluit": ["herroepen", "handhaven", "wijzigen", "intrekken"],
}

OBJECT_RE = {k: re.compile(v, re.IGNORECASE | re.UNICODE) for k, v in OBJECT_PATTERNS.items()}
ACTION_RE = {k: re.compile(v, re.IGNORECASE | re.UNICODE) for k, v in ACTION_PATTERNS.items()}
MODAL_RE = re.compile(
    r"\b(?:kan|kunnen|mag|mogen|zal|zullen|zou|zouden)\b"
    r"(?:\s+\S+){0,5}?\s+\bword(?:t|en)\b",
    re.IGNORECASE,
)


def matches(text, patterns):
    return [
        (category, match.start())
        for category, pattern in patterns.items()
        for match in pattern.finditer(text)
    ]


def select_sentence(sentence):
    objects = matches(sentence, OBJECT_RE)
    actions = matches(sentence, ACTION_RE)

    pairs = [
        (obj, action)
        for obj in objects
        for action in actions
        if action[0] in OBJECT_ACTIONS[obj[0]]
    ]

    if not pairs:
        return False

    modal_spans = list(MODAL_RE.finditer(sentence))
    if not modal_spans:
        return True

    return any(
        not any(
            modal.start() <= action[1] <= modal.end() + 160
            for modal in modal_spans
        )
        for _, action in pairs
    )


def main():
    nlp = spacy.load(SPACY_MODEL, disable=["ner"])
    nlp.max_length = max(nlp.max_length, 5_000_000)

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    for source in sorted(INPUT_DIR.rglob("*.md")):
        text = source.read_text(encoding="utf-8-sig", errors="replace")
        doc = nlp(text)
        selected = [
            sent.text.strip()
            for sent in doc.sents
            if select_sentence(sent.text)
        ]

        output = (OUTPUT_DIR / source.relative_to(INPUT_DIR)).with_suffix(".txt")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n\n".join(selected), encoding="utf-8")


if __name__ == "__main__":
    main()
