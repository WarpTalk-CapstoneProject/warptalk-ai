"""Clean transcript worker (WT-716).

Turns the raw STT stream into the stored transcript people read: one line per SENTENCE,
fillers and stutters gone, self-repairs resolved, punctuation correct. The raw segments are
never touched — they stay the record billing, retranscribe and corrections work from.
"""
