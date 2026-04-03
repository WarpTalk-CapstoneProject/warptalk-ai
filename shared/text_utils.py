"""Text processing utilities for NLP pipelines."""

from __future__ import annotations

import re

# Regex for sentence boundaries (split on '.', '!', '?', or newline).
# Looks for punctuation followed by space or end of string.
# Also handles Vietnamese and general Latin scripts well.
SENTENCE_SPLIT_REGEX = re.compile(r'(?<=[.!?])\s+|[\n\r]+')

def split_into_sentences(text: str, max_length: int = 150) -> list[str]:
    """Split text into sentences, falling back to commas if too long.
    
    Args:
        text: The text to split.
        max_length: If a sentence exceeds this character count, 
                   it will be further split by commas.
                   
    Returns:
        List of non-empty sentence chunks.
    """
    if not text or not text.strip():
        return []
        
    raw_sentences = SENTENCE_SPLIT_REGEX.split(text)
    chunks = []
    
    for sentence in raw_sentences:
        s = sentence.strip()
        if not s:
            continue
            
        # If sentence is reasonably short, keep it
        if len(s) <= max_length:
            chunks.append(s)
        else:
            # Fall back to splitting by comma if sentence is too long
            sub_chunks = re.split(r'(?<=,)\s+', s)
            current_chunk = ""
            
            for sub in sub_chunks:
                if not current_chunk:
                    current_chunk = sub
                elif len(current_chunk) + len(sub) + 1 <= max_length:
                    current_chunk += " " + sub
                else:
                    chunks.append(current_chunk)
                    current_chunk = sub
            
            if current_chunk:
                chunks.append(current_chunk)
                
    return chunks
