"""
PDF parsing service using PyMuPDF (fitz).
Streams pages one by one to avoid loading the entire document into memory.
"""
import fitz  # PyMuPDF
import base64
import re
from dataclasses import dataclass, field
from typing import Generator

try:
    import pdfplumber  # type: ignore
except Exception:  # pragma: no cover
    pdfplumber = None

from config import IMAGE_DPI


@dataclass
class PageContent:
    """Content extracted from a single PDF page."""
    page_number: int
    text: str
    has_images: bool = False
    image_count: int = 0
    drawing_count: int = 0
    text_area_ratio: float = 0.0
    max_image_area_ratio: float = 0.0
    max_drawing_area_ratio: float = 0.0
    tables: list[str] = field(default_factory=list)
    tables_structured: list[dict] = field(default_factory=list)
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

    def _normalize_line(line: str) -> str:
        l = (line or "").strip().lower()
        l = re.sub(r"\s+", " ", l)
        # Replace digits so that "Page 1" and "Page 2" normalize together.
        l = re.sub(r"\d+", "#", l)
        return l

    def _detect_repeating_edges(texts: list[str], max_lines: int = 2, min_hits: int = 3) -> tuple[set[str], set[str]]:
        """Detect likely header/footer lines based on repeated first/last lines across pages."""
        if not texts:
            return set(), set()

        head_counts: dict[str, int] = {}
        foot_counts: dict[str, int] = {}

        for t in texts:
            lines = [ln.strip() for ln in (t or "").splitlines() if ln.strip()]
            if not lines:
                continue

            for ln in lines[:max_lines]:
                key = _normalize_line(ln)
                head_counts[key] = head_counts.get(key, 0) + 1

            for ln in lines[-max_lines:]:
                key = _normalize_line(ln)
                foot_counts[key] = foot_counts.get(key, 0) + 1

        header_keys = {k for k, c in head_counts.items() if c >= min_hits and len(k) >= 4}
        footer_keys = {k for k, c in foot_counts.items() if c >= min_hits and len(k) >= 4}
        return header_keys, footer_keys

    def _clean_page_text(text: str, header_keys: set[str], footer_keys: set[str]) -> str:
        lines = [ln.rstrip() for ln in (text or "").splitlines()]
        cleaned: list[str] = []
        for ln in lines:
            s = ln.strip()
            if not s:
                cleaned.append("")
                continue

            # Drop standalone page-number lines.
            if re.fullmatch(r"(?i)(page|p\.|seite)\s*\d+", s):
                continue

            key = _normalize_line(s)
            if key in header_keys or key in footer_keys:
                continue

            cleaned.append(ln)

        # Collapse excessive blank lines.
        out_lines: list[str] = []
        blank_run = 0
        for ln in cleaned:
            if not ln.strip():
                blank_run += 1
                if blank_run <= 2:
                    out_lines.append("")
            else:
                blank_run = 0
                out_lines.append(ln.strip())
        return "\n".join(out_lines).strip()

    buffer_pages = min(5, len(doc))
    buffered_texts: list[str] = []
    buffered_meta: list[tuple[int, bool, list[str], list[dict]]] = []

    plumber_doc = None
    if pdfplumber is not None:
        try:
            plumber_doc = pdfplumber.open(pdf.file_path)
        except Exception:
            plumber_doc = None

    def _table_to_markdown(table: list[list[object]]) -> tuple[str, list[str]]:
        """Convert a 2D table into a markdown table + row-level strings."""
        if not table:
            return "", []

        norm_rows: list[list[str]] = []
        for row in table:
            if row is None:
                continue
            norm = [str(c).strip() if c is not None else "" for c in row]
            # Drop empty rows
            if any(x for x in norm):
                norm_rows.append(norm)

        if not norm_rows:
            return "", []

        headers = norm_rows[0]
        body = norm_rows[1:] if len(norm_rows) > 1 else []

        def esc(s: str) -> str:
            return (s or "").replace("|", "\\|")

        md_lines: list[str] = []
        md_lines.append("| " + " | ".join(esc(h) for h in headers) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for r in body:
            # Pad rows to header length
            rr = (r + [""] * len(headers))[: len(headers)]
            md_lines.append("| " + " | ".join(esc(c) for c in rr) + " |")

        row_texts: list[str] = []
        for ridx, r in enumerate(body):
            rr = (r + [""] * len(headers))[: len(headers)]
            kv = [f"{headers[i]} = {rr[i]}" for i in range(len(headers)) if headers[i] or rr[i]]
            if kv:
                row_texts.append("TABLE ROW: " + " | ".join(kv))
            else:
                row_texts.append("")

        return "\n".join(md_lines).strip(), row_texts

    def _extract_tables_structured(page_index0: int) -> list[dict]:
        """Extract tables using pdfplumber when available; return list of {markdown,row_texts}."""
        if plumber_doc is None:
            return []
        try:
            p = plumber_doc.pages[page_index0]
            tables = p.extract_tables() or []
        except Exception:
            return []

        structured: list[dict] = []
        for t in tables:
            md, row_texts = _table_to_markdown(t)
            if md:
                structured.append({"markdown": md, "row_texts": row_texts})
        return structured

    def _compute_layout_metrics(page: fitz.Page) -> tuple[int, int, float, float, float]:
        """Return (image_count, drawing_count, text_area_ratio, max_image_area_ratio, max_drawing_area_ratio)."""
        try:
            pr = page.rect
            page_area = float(pr.width * pr.height) if pr else 0.0
            if page_area <= 0:
                return 0, 0, 0.0, 0.0, 0.0

            d = page.get_text("dict") or {}
            blocks = d.get("blocks") or []

            text_area = 0.0
            image_count = 0
            max_img_area = 0.0
            drawing_count = 0
            max_draw_area = 0.0
            for b in blocks:
                bt = b.get("type")
                bbox = b.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                x0, y0, x1, y1 = bbox
                area = max(0.0, float(x1 - x0)) * max(0.0, float(y1 - y0))
                if bt == 0:
                    text_area += area
                elif bt == 1:
                    image_count += 1
                    if area > max_img_area:
                        max_img_area = area

            # Vector drawings (lines/shapes) are common in datasheets and are not counted as images.
            # We approximate their visual dominance by the bounding rectangles reported by PyMuPDF.
            try:
                drawings = page.get_drawings() or []
            except Exception:
                drawings = []

            for dr in drawings:
                r = dr.get("rect")
                if not r:
                    continue
                drawing_count += 1
                area = max(0.0, float(r.x1 - r.x0)) * max(0.0, float(r.y1 - r.y0))
                if area > max_draw_area:
                    max_draw_area = area

            return image_count, drawing_count, (text_area / page_area), (max_img_area / page_area), (max_draw_area / page_area)
        except Exception:
            # Fail open: keep metrics at zero, and rely on other heuristics.
            try:
                image_count = len(page.get_images(full=True))
            except Exception:
                image_count = 0
            return image_count, 0, 0.0, 0.0, 0.0
    try:
        # --- Buffer first pages to detect repeating header/footer lines ---
        for page_num in range(buffer_pages):
            page = doc[page_num]
            text = page.get_text("text")
            image_count, drawing_count, text_area_ratio, max_image_area_ratio, max_drawing_area_ratio = _compute_layout_metrics(page)
            has_images = image_count > 0
            tables_structured = _extract_tables_structured(page_num)
            tables = _extract_table_text(page) if not tables_structured else []
            buffered_texts.append(text or "")
            buffered_meta.append((page_num + 1, has_images, image_count, drawing_count, text_area_ratio, max_image_area_ratio, max_drawing_area_ratio, tables, tables_structured))

        header_keys, footer_keys = _detect_repeating_edges(buffered_texts)

        for i, raw in enumerate(buffered_texts):
            page_number, has_images, image_count, drawing_count, text_area_ratio, max_image_area_ratio, max_drawing_area_ratio, tables, tables_structured = buffered_meta[i]
            cleaned_text = _clean_page_text(raw, header_keys, footer_keys)
            yield PageContent(
                page_number=page_number,
                text=cleaned_text,
                has_images=has_images,
                image_count=image_count,
                drawing_count=drawing_count,
                text_area_ratio=text_area_ratio,
                max_image_area_ratio=max_image_area_ratio,
                max_drawing_area_ratio=max_drawing_area_ratio,
                tables=tables,
                tables_structured=tables_structured,
            )

        # --- Stream remaining pages using detected header/footer patterns ---
        for page_num in range(buffer_pages, len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            image_count, drawing_count, text_area_ratio, max_image_area_ratio, max_drawing_area_ratio = _compute_layout_metrics(page)
            has_images = image_count > 0
            tables_structured = _extract_tables_structured(page_num)
            tables = _extract_table_text(page) if not tables_structured else []
            cleaned_text = _clean_page_text(text or "", header_keys, footer_keys)
            yield PageContent(
                page_number=page_num + 1,
                text=cleaned_text,
                has_images=has_images,
                image_count=image_count,
                drawing_count=drawing_count,
                text_area_ratio=text_area_ratio,
                max_image_area_ratio=max_image_area_ratio,
                max_drawing_area_ratio=max_drawing_area_ratio,
                tables=tables,
                tables_structured=tables_structured,
            )
    finally:
        try:
            if plumber_doc is not None:
                plumber_doc.close()
        except Exception:
            pass
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
