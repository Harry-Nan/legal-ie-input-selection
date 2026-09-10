import csv
import json
import time
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, ConfigDict


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "LEAD+ Dataset"
OUTPUT_DIR = HERE / "results"
MAPPING_FILE = DATA_DIR / "document_mapping.csv"

INPUT_DIRS = {
    "full_text": DATA_DIR / "full_text",
    "operative_section": DATA_DIR / "operative_section",
    "key_sentences": DATA_DIR / "key_sentences",
}

MODEL = "gpt-5.6-luna"
MAX_OUTPUT_TOKENS = 4_000

PROMPT_TEMPLATE = """Je bent een informatie-extractiesysteem voor Nederlandse bestuursrechtelijke besluiten (*beschikkingen*).

Extraheer de volgende informatie uit de aangeleverde tekst:

1. **Ontvanger**: de persoon, onderneming, organisatie of andere partij van wie de rechtspositie rechtstreeks wordt bepaald door het huidige besluit.
2. **Besluitvormend orgaan**: het bestuursorgaan of de bestuurlijke autoriteit die het huidige besluit neemt.
3. **Rechtshandeling**: de juridisch relevante handeling die door het besluitvormend orgaan wordt verricht, zoals `verleent`, `weigert`, `wijst af`, `legt op`, `trekt in` of `verklaart ongegrond`.
4. **Rechtsobject**: het juridische instrument, verzoek, de sanctie, aanspraak of ander object waarop de rechtshandeling betrekking heeft, zoals `vergunning`, `ontheffing`, `subsidie`, `boete`, `last onder dwangsom`, `verzoek` of `bezwaar`.

### Instructies

- Extraheer alleen informatie die betrekking heeft op het **huidige bestuursrechtelijke besluit**.
- Extraheer geen informatie die betrekking heeft op eerdere besluiten, aanvragen, voorgenomen besluiten, rechterlijke uitspraken, feitelijke achtergrond of handelingen van andere partijen.
- De Rechtshandeling en het Rechtsobject moeten betrekking hebben op **dezelfde beslissing**.
- Geef bij scheidbare werkwoorden de volledige rechtshandeling terug, bijvoorbeeld `wijst af`, `legt op` of `trekt in`.
- Wanneer termen zoals `aanvrager`, `betrokkene`, `overtreder`, `wij` of `u` worden gebruikt, herleid deze dan tot de expliciet genoemde partij indien dit op basis van de aangeleverde tekst mogelijk is.
- Gebruik uitsluitend de aangeleverde tekst. De tekst kan een onvolledig fragment van een groter besluit zijn.
- Als informatie niet uit de aangeleverde tekst kan worden vastgesteld, geef dan `null` of een lege lijst terug zoals hieronder aangegeven.
- Raad niet en gebruik geen externe kennis.
- Behoud namen en rechtsobjecten zoveel mogelijk zoals zij in de tekst voorkomen.

Geef **uitsluitend geldige JSON** terug in het volgende formaat:

{
"ontvanger": [],
"besluitvormend_orgaan": null,
"beslissingen": [
{
"rechtshandeling": null,
"rechtsobject": null
}
]
}

Als geen ontvanger kan worden vastgesteld, geef dan:

"ontvanger": []

Als geen combinatie van Rechtshandeling en Rechtsobject kan worden vastgesteld, geef dan:

"beslissingen": []

Als het huidige besluit meerdere ontvangers of meerdere afzonderlijke combinaties van Rechtshandeling en Rechtsobject bevat, geef deze dan allemaal terug.

TEKST:
{TEXT}"""


class Beslissing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rechtshandeling: str | None
    rechtsobject: str | None


class ExtractieResultaat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ontvanger: list[str]
    besluitvormend_orgaan: str | None
    beslissingen: list[Beslissing]


def normalize_name(name: str) -> str:
    name = Path(name).name.lower()
    while Path(name).suffix.lower() in {".txt", ".md", ".pdf", ".json"}:
        name = str(Path(name).with_suffix(""))
    return name


def load_document_ids() -> dict[str, str]:
    lookup = {}

    with MAPPING_FILE.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            document_id = row["document_id"]

            for field in ("original_filename", "original_stem"):
                value = row.get(field)
                if value:
                    lookup[normalize_name(value)] = document_id

    return lookup


def usage_value(obj, *attributes):
    for attribute in attributes:
        obj = getattr(obj, attribute, None)
        if obj is None:
            return 0
    return int(obj)


def load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}

    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["document_id"]] = record

    return records


def save_results(path: Path, records: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for document_id in sorted(records):
            file.write(json.dumps(records[document_id], ensure_ascii=False) + "\n")


def run_representation(client: OpenAI, representation: str, folder: Path, id_lookup: dict[str, str]) -> None:
    output_file = OUTPUT_DIR / f"{representation}.jsonl"
    records = load_existing(output_file)

    files = sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}
    )

    for i, path in enumerate(files, start=1):
        document_id = id_lookup.get(normalize_name(path.name))

        if document_id is None:
            raise ValueError(f"No document ID found for {path.name}")

        if records.get(document_id, {}).get("status") == "success":
            continue

        text = path.read_text(encoding="utf-8-sig").strip()
        start = time.perf_counter()

        try:
            response = client.responses.parse(
                model=MODEL,
                input=[
                    {
                        "role": "user",
                        "content": PROMPT_TEMPLATE.replace("{TEXT}", text),
                    }
                ],
                text_format=ExtractieResultaat,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                store=False,
            )

            usage = response.usage
            input_tokens = usage_value(usage, "input_tokens")
            cached_tokens = usage_value(
                usage, "input_tokens_details", "cached_tokens"
            )
            output_tokens = usage_value(usage, "output_tokens")
            reasoning_tokens = usage_value(
                usage, "output_tokens_details", "reasoning_tokens"
            )

            records[document_id] = {
                "document_id": document_id,
                "input_representation": representation,
                "input_file": path.relative_to(DATA_DIR).as_posix(),
                "model": MODEL,
                "status": "success",
                "elapsed_seconds": round(time.perf_counter() - start, 3),
                "input_tokens": input_tokens,
                "cached_input_tokens": cached_tokens,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
                "total_tokens": usage_value(usage, "total_tokens"),
                "extraction": response.output_parsed.model_dump(),
            }

        except Exception as exc:
            records[document_id] = {
                "document_id": document_id,
                "input_representation": representation,
                "input_file": path.relative_to(DATA_DIR).as_posix(),
                "model": MODEL,
                "status": "error",
                "elapsed_seconds": round(time.perf_counter() - start, 3),
                "error": str(exc),
            }

        save_results(output_file, records)
        print(f"[{representation}] {i}/{len(files)} {document_id}")


def main():
    client = OpenAI()
    id_lookup = load_document_ids()

    for representation, folder in INPUT_DIRS.items():
        run_representation(client, representation, folder, id_lookup)


if __name__ == "__main__":
    main()
