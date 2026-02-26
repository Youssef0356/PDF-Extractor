"""
Chunking service for splitting PDF text into overlapping chunks.
"""
from dataclasses import dataclass

from config import CHUNK_SIZE, CHUNK_OVERLAP, CHARS_PER_TOKEN


@dataclass
class Chunk:
    """A text chunk with metadata."""
    chunk_id: str
    text: str
    page_number: int
    chunk_index: int
    has_images: bool = False  # Whether the source page has images
    image_base64: str = ""   # Page image if relevant


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
    
    chunk_chars = CHUNK_SIZE * CHARS_PER_TOKEN
    overlap_chars = CHUNK_OVERLAP * CHARS_PER_TOKEN
    
    chunks: list[Chunk] = []
    start = 0
    chunk_index = 0
    
    while start < len(text):
        current_start = start
        end = min(start + chunk_chars, len(text))
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence-ending punctuation near the end
            search_start = max(end - 100, start)
            best_break = end
            
            for sep in ['. ', '.\n', '!\n', '?\n', '\n\n']:
                pos = text.rfind(sep, search_start, end)
                if pos > start:
                    best_break = pos + len(sep)
                    break
            
            end = best_break
        
        chunk_text_content = text[start:end].strip()
        
        if chunk_text_content:  # Don't create empty chunks
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_p{page_number}_c{chunk_index}",
                text=chunk_text_content,
                page_number=page_number,
                chunk_index=chunk_index,
                has_images=has_images,
                image_base64=image_base64 if has_images else "",
            ))
            chunk_index += 1
        
        if end >= len(text):
            break
            
        # Move start forward, accounting for overlap
        start = end - overlap_chars
        
        # Safety: ensure we always move forward
        if start <= current_start:
            start = current_start + 1
    
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
