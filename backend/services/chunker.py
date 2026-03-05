"""
Chunking service for splitting PDF text into overlapping chunks.
"""
from dataclasses import dataclass
import re
from typing import Optional

from config import CHUNK_SIZE, CHUNK_OVERLAP, CHARS_PER_TOKEN


@dataclass
class Chunk:
    """A text chunk with metadata."""
    doc_id: str
    chunk_id: str
    text: str
    page_number: int
    chunk_index: int
    has_images: bool = False  # Whether the source page has images
    image_base64: str = ""   # Page image if relevant
    chunk_type: str = "text"  # text | table | ocr
    table_id: Optional[str] = None
    row_index: Optional[int] = None


def chunk_text(
    text: str,
    page_number: int,
    has_images: bool = False,
    image_base64: str = "",
    doc_id: str = "doc",
) -> list[Chunk]:
    """
    Split text into overlapping chunks.
    
    Uses character-based splitting with approximate token-to-char conversion.
    Tries to split at sentence boundaries when possible.
    
    Args:
        text: The text to chunk.
        page_number: Source page number.
        has_images: Whether the source page contains images.
        image_base64: Base64 image of the source page.
        doc_id: Document identifier for chunk IDs.
        
    Returns:
        List of Chunk objects.
    """
    if not text.strip():
        return []

    chunk_chars_target = CHUNK_SIZE * CHARS_PER_TOKEN
    overlap_units_target = max(0, CHUNK_OVERLAP)

    def _split_into_units(src: str) -> list[str]:
        """Split page text into chunkable units while keeping table blocks intact."""
        units: list[str] = []

        # Keep [TABLE DATA] blocks intact (they may contain line-oriented tables).
        if "[TABLE DATA]" in src:
            pre, _, table_part = src.partition("[TABLE DATA]")
            if pre.strip():
                src = pre.strip() + "\n\n[TABLE DATA]\n" + table_part.strip()

        parts = re.split(r"(\n\s*\n)", src)
        paragraph_acc: list[str] = []
        for p in parts:
            if not p:
                continue
            if re.fullmatch(r"\n\s*\n", p):
                para = "".join(paragraph_acc).strip()
                paragraph_acc = []
                if not para:
                    continue

                if "[TABLE DATA]" in para:
                    units.append(para)
                    continue

                # Split paragraph into sentences, but keep punctuation.
                sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Þ0-9])", para)
                sentences = [s.strip() for s in sentences if s and s.strip()]
                if sentences:
                    units.extend(sentences)
                else:
                    units.append(para)
            else:
                paragraph_acc.append(p)

        tail = "".join(paragraph_acc).strip()
        if tail:
            if "[TABLE DATA]" in tail:
                units.append(tail)
            else:
                sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Þ0-9])", tail)
                sentences = [s.strip() for s in sentences if s and s.strip()]
                units.extend(sentences if sentences else [tail])

        # Merge tiny units (e.g., headings) with the next unit.
        merged: list[str] = []
        for u in units:
            if not merged:
                merged.append(u)
                continue
            if len(merged[-1]) < 40 and "[TABLE DATA]" not in merged[-1]:
                merged[-1] = (merged[-1].rstrip() + " " + u.lstrip()).strip()
            else:
                merged.append(u)
        return merged

    units = _split_into_units(text)
    if not units:
        return []

    def _flush(current_units: list[str], chunk_index: int) -> Chunk:
        chunk_text_content = "\n".join(current_units).strip()
        return Chunk(
            doc_id=doc_id,
            chunk_id=f"{doc_id}_p{page_number}_c{chunk_index}",
            text=chunk_text_content,
            page_number=page_number,
            chunk_index=chunk_index,
            has_images=has_images,
            image_base64=image_base64 if has_images else "",
        )

    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0
    chunk_index = 0

    for u in units:
        u_len = len(u)

        # If a single unit is extremely large (mostly big tables), keep it as its own chunk.
        if "[TABLE DATA]" in u and u_len > chunk_chars_target:
            if current:
                chunks.append(_flush(current, chunk_index))
                chunk_index += 1
                current = []
                current_len = 0
            chunks.append(_flush([u], chunk_index))
            chunk_index += 1
            continue

        if current and (current_len + u_len + 1) > chunk_chars_target:
            chunks.append(_flush(current, chunk_index))
            chunk_index += 1

            # Overlap by last N units to preserve sentence continuity.
            if overlap_units_target > 0:
                current = current[-overlap_units_target:]
                current_len = sum(len(x) for x in current) + max(0, len(current) - 1)
            else:
                current = []
                current_len = 0

        current.append(u)
        current_len += u_len + (1 if len(current) > 1 else 0)

    if current:
        chunks.append(_flush(current, chunk_index))

    return chunks


def chunk_document(pages: list, doc_id: str = "doc") -> list[Chunk]:
    """
    Chunk an entire document (list of PageContent objects).
    
    Args:
        pages: List of PageContent from pdf_parser.
        doc_id: Document identifier.
        
    Returns:
        All chunks from the document.
    """
    all_chunks: list[Chunk] = []
    
    for page in pages:
        # Combine page text and table text
        combined_text = page.text
        if page.tables:
            combined_text += "\n\n[TABLE DATA]\n" + "\n".join(page.tables)
        
        page_chunks = chunk_text(
            text=combined_text,
            page_number=page.page_number,
            has_images=page.has_images,
            image_base64=page.image_base64,
            doc_id=doc_id,
        )
        all_chunks.extend(page_chunks)
    
    return all_chunks


def make_table_chunks(
    *,
    doc_id: str,
    page_number: int,
    table_index: int,
    table_markdown: str,
    row_texts: list[str],
    chunk_index_start: int,
) -> list[Chunk]:
    """Create table chunks (one table summary + per-row chunks).

    These chunks are stored separately from normal page text so retrieval can
    distinguish between narrative text and table content.
    """
    chunks: list[Chunk] = []
    table_id = f"p{page_number}_t{table_index}"

    # Table-level summary chunk
    chunks.append(
        Chunk(
            doc_id=doc_id,
            chunk_id=f"{doc_id}_{table_id}_summary",
            text=(table_markdown or "").strip(),
            page_number=page_number,
            chunk_index=chunk_index_start,
            has_images=False,
            image_base64="",
            chunk_type="table",
            table_id=table_id,
            row_index=None,
        )
    )

    # Row-level chunks
    for i, rt in enumerate(row_texts):
        t = (rt or "").strip()
        if not t:
            continue
        chunks.append(
            Chunk(
                doc_id=doc_id,
                chunk_id=f"{doc_id}_{table_id}_r{i}",
                text=t,
                page_number=page_number,
                chunk_index=chunk_index_start + 1 + i,
                has_images=False,
                image_base64="",
                chunk_type="table",
                table_id=table_id,
                row_index=i,
            )
        )

    return chunks
