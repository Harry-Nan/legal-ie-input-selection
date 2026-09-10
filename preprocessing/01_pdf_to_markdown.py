from pathlib import Path
import argparse
import re
import shutil
import tempfile
import unicodedata

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "LEAD+ Dataset" / "full_text"


def safe_name(name):
    name = unicodedata.normalize("NFKD", name)
    name = name.replace("–", "-").replace("—", "-").replace("“", "").replace("”", "").replace("‘", "").replace("’", "")
    name = name.encode("ascii", errors="ignore").decode("ascii")
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_dir", type=Path)
    args = parser.parse_args()

    pdf_dir = args.pdf_dir.resolve()
    pdf_files = sorted(pdf_dir.rglob("*.pdf"))

    options = PdfPipelineOptions()
    options.do_ocr = True
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options)
        },
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for pdf_path in pdf_files:
        output = (OUTPUT_DIR / pdf_path.relative_to(pdf_dir)).with_suffix(".md")
        output.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            temp_pdf = Path(tmp) / safe_name(pdf_path.name)
            shutil.copy2(pdf_path, temp_pdf)
            converter.convert(temp_pdf).document.save_as_markdown(output)


if __name__ == "__main__":
    main()
