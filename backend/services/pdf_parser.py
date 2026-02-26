"""
PDF parsing service using PyMuPDF (fitz).
Streams pages one by one to avoid loading the entire document into memory.
"""
import fitz  # PyMuPDF
import base64
from dataclasses import dataclass, field
from typing import Generator

from config import IMAGE_DPI


@dataclass
class PageContent:
    """Content extracted from a single PDF page."""
    page_number: int
    text: str
    has_images: bool = False
    tables: list[str] = field(default_factory=list)
    # NOTE: image_base64 is NOT stored here anymore.
    # Use render_page_image() on demand to avoid RAM bloat.


@dataclass
class ParsedPDF:
    """Lightweight PDF handle — pages are iterated on demand."""
    filename: str
    total_pages: int
    file_path: str  # Keep path for on-demand image rendering


def open_pdf(file_path: str) -> ParsedPDF:
    """
    Open a PDF and return a lightweight handle.
    Does NOT load all pages into memory.
    """
    doc = fitz.open(file_path)
    total = len(doc)
    doc.close()
    import os
    return ParsedPDF(
        filename=os.path.basename(file_path),
        total_pages=total,
        file_path=file_path,
    )


def iter_pages(pdf: ParsedPDF) -> Generator[PageContent, None, None]:
    """
    Iterate over pages one at a time (streaming).
    Each page is parsed and yielded, then discarded from memory.
    """
    doc = fitz.open(pdf.file_path)
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]

            # Extract text
            text = page.get_text("text")

            # Detect images
            has_images = len(page.get_images(full=True)) > 0

            # Extract table-like structures
            tables = _extract_table_text(page)

            yield PageContent(
                page_number=page_num + 1,
                text=text.strip(),
                has_images=has_images,
                tables=tables,
            )
    finally:
        doc.close()


def render_page_image(file_path: str, page_number: int) -> str:
    """
    Render a single PDF page as a base64 PNG image on demand.
    Keeps only one page image in memory at a time.
    
    Args:
        file_path: Path to the PDF.
        page_number: 1-indexed page number.
        
    Returns:
        Base64-encoded PNG string.
    """
    doc = fitz.open(file_path)
    try:
        page = doc[page_number - 1]
        mat = fitz.Matrix(IMAGE_DPI / 72, IMAGE_DPI / 72)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        return base64.b64encode(img_bytes).decode("utf-8")
    finally:
        doc.close()


def _extract_table_text(page: fitz.Page) -> list[str]:
    """
    Extract table-like structures from a PDF page using text block positions.
    """
    tables: list[str] = []
    blocks = page.get_text("blocks")

    if not blocks:
        return tables

    blocks = sorted(blocks, key=lambda b: (b[1], b[0]))

    rows: list[list] = []
    current_row: list = []
    current_y = None
    y_tolerance = 5

    for block in blocks:
        x0, y0, x1, y1, text, block_no, block_type = block
        if block_type != 0:
            continue
        if current_y is None or abs(y0 - current_y) < y_tolerance:
            current_row.append(text.strip())
            current_y = y0
        else:
            if len(current_row) > 1:
                rows.append(current_row)
            current_row = [text.strip()]
            current_y = y0

    if len(current_row) > 1:
        rows.append(current_row)

    if len(rows) >= 2:
        table_text = "\n".join([" | ".join(row) for row in rows])
        tables.append(table_text)

    return tables


# -- Backwards compatibility: full parse for the FastAPI endpoint ----------
# Still usable but not recommended for large files.
@dataclass
class ParsedPDFFull:
    """Complete parsed PDF document (legacy)."""
    filename: str
    total_pages: int
    pages: list[PageContent]
    full_text: str


def parse_pdf(file_path: str) -> ParsedPDFFull:
    """Legacy full-parse (used by main.py). For large PDFs, prefer iter_pages()."""
    pdf = open_pdf(file_path)
    pages = list(iter_pages(pdf))
    full_text = "\n\n".join(p.text for p in pages)
    return ParsedPDFFull(
        filename=pdf.filename,
        total_pages=pdf.total_pages,
        pages=pages,
        full_text=full_text,
    )
