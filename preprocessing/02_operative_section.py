from pathlib import Path
import html
import shutil
import string


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "LEAD+ Dataset" / "full_text"
OUTPUT_DIR = ROOT / "LEAD+ Dataset" / "operative_section"

TARGET_WORDS = {"dictum", "besluit", "besluiten", "conclusie", "beslissing"}
TARGET_PHRASES = {"inhoud en geldigheid van"}


def normalize_spaces(text):
    return " ".join(str(text).split())


def strip_inline_markdown(text):
    text = html.unescape(str(text))
    for char in ("`", "*", "_", "~"):
        text = text.replace(char, "")
    return normalize_spaces(text)


def collapse_spaced_target_words(text):
    raw = strip_inline_markdown(text)
    letters = "".join(c.casefold() for c in raw if c.isalpha())
    return letters if letters in TARGET_WORDS else raw


def normalize_header(text):
    return normalize_spaces(
        collapse_spaced_target_words(text).strip().rstrip(":;,.!? ")
    ).casefold()


def header_words(text):
    cleaned = normalize_header(text)
    for char in string.punctuation:
        cleaned = cleaned.replace(char, " ")
    return cleaned.split()


def header_contains_target(text):
    normalized = normalize_header(text)
    return (
        any(phrase in normalized for phrase in TARGET_PHRASES)
        or any(word in TARGET_WORDS for word in header_words(text))
    )


def header_is_excluded(text):
    normalized = normalize_header(text)
    if "direct in de problemen door dit besluit" in normalized or "niet eens met d" in normalized:
        return True
    words = header_words(text)
    return any(words[i:i + 2] == ["besluit", "openbaar"] for i in range(len(words) - 1))


def standalone_besluit(line):
    raw = html.unescape(str(line)).strip()
    if not raw:
        return False
    if any(not c.isalpha() and not c.isspace() and c not in string.punctuation for c in raw):
        return False
    return "".join(c.casefold() for c in raw if c.isalpha()) == "besluit"


def body_has_words(text):
    cleaned = html.unescape(str(text)).replace("<!-- image -->", " ")
    for char in ("`", "*", "_", "~", "#", "-", ">", "|"):
        cleaned = cleaned.replace(char, " ")
    return any(c.isalnum() for c in cleaned)


def parse_header(line):
    raw = line.rstrip("\r\n")
    stripped = raw.lstrip()

    if not stripped.startswith("#"):
        return None

    level = len(stripped) - len(stripped.lstrip("#"))
    if level == 0 or level > 6:
        return None
    if level < len(stripped) and not stripped[level].isspace():
        return None

    text = stripped[level:].strip()
    while text.endswith("#"):
        text = text[:-1].rstrip()

    return level, text


def find_headers(text):
    headers = []
    offset = 0

    for line in text.splitlines(keepends=True):
        raw = line.rstrip("\r\n")
        parsed = parse_header(line)

        if parsed:
            level, heading = parsed
            headers.append({
                "start": offset,
                "end": offset + len(raw),
                "level": level,
                "text": heading,
            })
        elif standalone_besluit(raw):
            headers.append({
                "start": offset,
                "end": offset + len(raw),
                "level": None,
                "text": raw.strip(),
            })

        offset += len(line)

    return headers


def next_header_start(headers, index, text_length):
    return headers[index + 1]["start"] if index + 1 < len(headers) else text_length


def resolve_empty_header(text, headers, index):
    while True:
        end = next_header_start(headers, index, len(text))
        if body_has_words(text[headers[index]["end"]:end]) or index + 1 >= len(headers):
            return index
        index += 1


def starts_u_ontvangt(line):
    cleaned = html.unescape(line).strip()
    while cleaned and cleaned[0] in "#>*+-_`~ ":
        cleaned = cleaned[1:].lstrip()

    normalized = normalize_spaces(cleaned).casefold()
    return normalized == "u ontvangt" or normalized.startswith(("u ontvangt ", "u ontvangt:"))


def u_ontvangt_starts(text):
    starts = []
    offset = 0

    for line in text.splitlines(keepends=True):
        if starts_u_ontvangt(line.rstrip("\r\n")):
            starts.append(offset)
        offset += len(line)

    return starts


def next_header_after(headers, position, text_length):
    for header in headers:
        if header["start"] > position:
            return header["start"]
    return text_length


def merge_intervals(intervals):
    merged = []

    for start, end in sorted(intervals):
        if not merged or start >= merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    return merged


def extract_operative(text):
    headers = find_headers(text)

    dictum_present = any(
        "dictum" in header_words(h["text"])
        and not header_is_excluded(h["text"])
        for h in headers
    )

    intervals = []

    for i, header in enumerate(headers):
        heading = header["text"]

        if not header_contains_target(heading) or header_is_excluded(heading):
            continue
        if dictum_present and "dictum" not in header_words(heading):
            continue

        resolved = resolve_empty_header(text, headers, i)
        intervals.append((
            headers[resolved]["start"],
            next_header_start(headers, resolved, len(text)),
        ))

    if not dictum_present:
        for start in u_ontvangt_starts(text):
            intervals.append((start, next_header_after(headers, start, len(text))))

    return "\n\n".join(
        text[start:end].strip()
        for start, end in merge_intervals(intervals)
        if text[start:end].strip()
    )


def main():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    for source in sorted(INPUT_DIR.rglob("*.md")):
        operative = extract_operative(
            source.read_text(encoding="utf-8", errors="replace")
        )

        if operative:
            output = OUTPUT_DIR / source.relative_to(INPUT_DIR)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(operative, encoding="utf-8")


if __name__ == "__main__":
    main()
