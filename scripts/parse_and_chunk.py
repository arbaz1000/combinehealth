"""
Parse UHC policy PDFs with Docling and create section-aware chunks with metadata.

Usage:
    python scripts/parse_and_chunk.py                  # Parse all PDFs
    python scripts/parse_and_chunk.py --limit 5        # Parse first 5 only (for dev)
    python scripts/parse_and_chunk.py --single ablative-treatment-spinal-pain.pdf
"""

import argparse
import gc
import json
import re
from pathlib import Path

import yaml
from tqdm import tqdm

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "insurers" / "uhc.yaml"
PDF_DIR = PROJECT_ROOT / "data" / "raw_pdfs"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.json"
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "uhc_chunks.jsonl"

# Major sections we care about (in priority order for RAG)
# These are the top-level sections we'll use to label chunks
MAJOR_SECTIONS = [
    "Application",
    "Coverage Rationale",
    "Definitions",
    "Applicable Codes",
    "Description of Services",
    "Clinical Evidence",
    "U.S. Food and Drug Administration",
    "References",
    "Policy History/Revision Information",
    "Instructions for Use",
]

# Max tokens per chunk (rough estimate: 1 token ≈ 4 chars)
MAX_CHUNK_CHARS = 3200  # ~800 tokens
# Clinical Evidence can be very long; we chunk it more aggressively
CLINICAL_EVIDENCE_MAX_CHARS = 3200


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_manifest() -> dict:
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def extract_metadata_from_markdown(md_text: str, filename: str, manifest_entry: dict | None) -> dict:
    """Extract policy metadata from the markdown text."""
    metadata = {
        "policy_name": "",
        "policy_number": "",
        "effective_date": "",
        "source_url": manifest_entry.get("url", "") if manifest_entry else "",
        "filename": filename,
        "insurer": "UHC",
    }

    # Extract policy number
    pn_match = re.search(r"Policy Number\s*:?\s*(\S+)", md_text)
    if pn_match:
        metadata["policy_number"] = pn_match.group(1).strip()

    # Extract effective date
    ed_match = re.search(r"Effective Date\s*:?\s*(.+?)(?:\n|$)", md_text)
    if ed_match:
        metadata["effective_date"] = ed_match.group(1).strip()

    # Extract policy name from manifest or filename
    if manifest_entry:
        metadata["policy_name"] = manifest_entry.get("name", "")
    if not metadata["policy_name"]:
        metadata["policy_name"] = filename.replace(".pdf", "").replace("-", " ").title()

    return metadata


def extract_cpt_codes(text: str) -> list[str]:
    """Extract CPT/HCPCS codes from text (typically 5-digit codes)."""
    # Match standalone 5-digit codes that look like CPT codes
    codes = re.findall(r'\b(\d{5})\b', text)
    # Also match HCPCS codes (letter + 4 digits)
    hcpcs = re.findall(r'\b([A-Z]\d{4})\b', text)
    return list(set(codes + hcpcs))


def split_text_into_paragraphs(text: str) -> list[str]:
    """Split text on paragraph boundaries (double newlines or section breaks)."""
    # Split on double newlines
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]


def chunk_section(section_text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """
    Split a section into chunks, respecting paragraph boundaries.
    Never splits mid-paragraph if possible.
    """
    if len(section_text) <= max_chars:
        return [section_text]

    paragraphs = split_text_into_paragraphs(section_text)
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # If adding this paragraph would exceed limit, start a new chunk
        if current_chunk and len(current_chunk) + len(para) + 2 > max_chars:
            chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def parse_single_pdf(pdf_path: Path, manifest_entry: dict | None = None, converter=None) -> list[dict]:
    """
    Parse a single PDF with Docling and return a list of chunks with metadata.

    Strategy:
    - Use Docling's section_header detection to split by major sections
    - Keep tables intact (never split mid-table)
    - Chunk large sections on paragraph boundaries
    - Attach rich metadata to each chunk
    """
    if converter is None:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()

    result = converter.convert(str(pdf_path))
    doc = result.document

    # Get full markdown
    full_md = doc.export_to_markdown()

    # Extract document-level metadata
    metadata = extract_metadata_from_markdown(full_md, pdf_path.name, manifest_entry)

    # Extract all CPT codes from the Applicable Codes section
    all_cpt_codes = extract_cpt_codes(full_md)

    # Strategy: Split markdown by major section headers
    # We'll use regex to split on section headers that Docling produces as ## headers
    sections = split_markdown_by_sections(full_md)

    chunks = []
    for section_name, section_text in sections:
        if not section_text.strip():
            continue

        # Skip very low-value sections
        if section_name in ["Instructions for Use", "References", "Table of Contents"]:
            continue

        # Determine max chunk size based on section importance
        max_chars = MAX_CHUNK_CHARS
        if section_name == "Clinical Evidence":
            max_chars = CLINICAL_EVIDENCE_MAX_CHARS

        # Chunk the section
        section_chunks = chunk_section(section_text, max_chars)

        for i, chunk_text in enumerate(section_chunks):
            chunk = {
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "section_name": section_name,
                    "chunk_index": i,
                    "total_chunks_in_section": len(section_chunks),
                    "applicable_cpt_codes": all_cpt_codes,
                    "char_count": len(chunk_text),
                },
            }
            chunks.append(chunk)

    return chunks


def split_markdown_by_sections(md_text: str) -> list[tuple[str, str]]:
    """
    Split Docling markdown output by section headers.
    Returns list of (section_name, section_content) tuples.
    """
    # Docling outputs section headers as ## headers
    # Split on ## lines, keeping the header as the section name
    lines = md_text.split('\n')

    sections = []
    current_section = "Preamble"
    current_content = []

    for line in lines:
        # Check if this is a section header (## or ###)
        header_match = re.match(r'^#{1,3}\s+(.+)$', line)
        if header_match:
            # Save previous section
            if current_content:
                content = '\n'.join(current_content).strip()
                if content:
                    sections.append((current_section, content))
            # Determine if this header maps to a major section
            header_text = header_match.group(1).strip()
            current_section = classify_section(header_text)
            current_content = []
        else:
            current_content.append(line)

    # Save last section
    if current_content:
        content = '\n'.join(current_content).strip()
        if content:
            sections.append((current_section, content))

    # Merge consecutive chunks with the same section name
    merged = []
    for section_name, content in sections:
        if merged and merged[-1][0] == section_name:
            merged[-1] = (section_name, merged[-1][1] + "\n\n" + content)
        else:
            merged.append((section_name, content))

    return merged


def classify_section(header_text: str) -> str:
    """
    Normalize section headers to a canonical name where possible.
    Returns the original header text for sections we don't recognize —
    this is intentional; the section name is metadata for context, not a filter.
    """
    header_lower = header_text.lower().strip()

    # Preamble: UHC branding headers / policy titles
    if "unitedhealthcare" in header_lower or "medical policy" in header_lower:
        return "Preamble"

    # Normalize known sections (handles minor variations like "Reference(s)" vs "References")
    NORMALIZE_MAP = {
        "application": "Application",
        "coverage rationale": "Coverage Rationale",
        "definitions": "Definitions",
        "applicable codes": "Applicable Codes",
        "description of services": "Description of Services",
        "clinical evidence": "Clinical Evidence",
        "u.s. food and drug administration": "U.S. Food and Drug Administration",
        "references": "References",
        "reference(s)": "References",
        "policy history/revision information": "Policy History/Revision Information",
        "policy history": "Policy History/Revision Information",
        "instructions for use": "Instructions for Use",
        "benefit considerations": "Benefit Considerations",
        "background": "Background",
    }

    for key, canonical in NORMALIZE_MAP.items():
        if key in header_lower or header_lower in key:
            return canonical

    # Default: keep the original header as the section name.
    # This preserves useful specificity (e.g., "Initial Therapy", "Knee",
    # "Breast Cancer") that helps the LLM understand context.
    return header_text


def get_already_parsed(chunks_path: Path) -> set[str]:
    """Read existing JSONL to find which PDFs are already parsed (for --resume)."""
    done = set()
    if chunks_path.exists():
        with open(chunks_path) as f:
            for line in f:
                try:
                    chunk = json.loads(line)
                    done.add(chunk["metadata"]["filename"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def main():
    parser = argparse.ArgumentParser(description="Parse UHC policy PDFs and create chunks")
    parser.add_argument("--limit", type=int, help="Parse only N PDFs (for dev)")
    parser.add_argument("--single", type=str, help="Parse a single PDF by filename")
    parser.add_argument("--resume", action="store_true",
                        help="Skip PDFs already in the output file (resume after crash)")
    args = parser.parse_args()

    manifest = load_manifest()

    # Build filename → manifest entry lookup
    manifest_lookup = {}
    for entry in manifest.get("policies", []):
        manifest_lookup[entry["filename"]] = entry

    # Get list of PDFs to parse
    if args.single:
        pdf_files = [PDF_DIR / args.single]
    else:
        pdf_files = sorted(PDF_DIR.glob("*.pdf"))
        if args.limit:
            pdf_files = pdf_files[:args.limit]

    # Ensure output directory exists
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: skip already-parsed PDFs
    already_done = set()
    file_mode = "w"
    if args.resume:
        already_done = get_already_parsed(CHUNKS_PATH)
        if already_done:
            file_mode = "a"  # append to existing file
            print(f"Resuming: {len(already_done)} PDFs already parsed, skipping them.")
        pdf_files = [p for p in pdf_files if p.name not in already_done]

    print(f"Parsing {len(pdf_files)} PDFs...")

    # Create converter ONCE (loads ML models once, not 253 times)
    from docling.document_converter import DocumentConverter
    print("Loading Docling models (one-time)...")
    converter = DocumentConverter()
    print("Models loaded.")

    total_chunks = 0
    failed = []

    # Stream chunks to disk — don't hold all in memory
    with open(CHUNKS_PATH, file_mode) as out_f:
        for pdf_path in tqdm(pdf_files, desc="Parsing PDFs"):
            try:
                manifest_entry = manifest_lookup.get(pdf_path.name)
                chunks = parse_single_pdf(pdf_path, manifest_entry, converter=converter)
                for chunk in chunks:
                    out_f.write(json.dumps(chunk) + "\n")
                total_chunks += len(chunks)
                out_f.flush()  # flush after each PDF so progress is saved
            except Exception as e:
                print(f"\n  Failed: {pdf_path.name} — {e}")
                failed.append({"filename": pdf_path.name, "error": str(e)})
            finally:
                gc.collect()  # reclaim memory after each PDF

    print(f"\n=== Parsing Complete ===")
    print(f"Total new chunks: {total_chunks}")
    print(f"PDFs parsed: {len(pdf_files) - len(failed)}")
    print(f"PDFs failed: {len(failed)}")
    if already_done:
        print(f"PDFs skipped (already done): {len(already_done)}")
    print(f"Output: {CHUNKS_PATH}")

    if failed:
        print("\nFailed PDFs:")
        for f_item in failed:
            print(f"  - {f_item['filename']}: {f_item['error']}")


if __name__ == "__main__":
    main()
