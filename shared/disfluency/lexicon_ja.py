"""Japanese disfluency lexicon (WT-716). Data only — rules live in prepass.py.

Everything is matched on WHOLE morphemes (after merging, see tokenize.py): "ええ" (yes) and
"ええと" (um) share a prefix and mean opposite things for a transcript. Keys are hiragana with
long marks unified (normalize_key), so "エート" and "えーーと" meet "えーと".

B entries are never deleted: "あの資料" is a demonstrative, "なんか飲みますか" is "something",
"ちょっと待って" is "wait a moment". Only the elongated forms (あのー, そのー) are fillers.
"""

from __future__ import annotations

from shared.disfluency.normalize import normalize_key


def _keys(*words: str) -> frozenset[str]:
    return frozenset(normalize_key(w, "ja") for w in words)


A1_FILLERS = _keys(
    "えーと", "えっと", "えーっと", "ええと", "えと", "えー", "あのー", "そのー", "んーと"
)
A2_FILLERS = _keys("うーん", "んー", "あー", "まー")
B_MARKERS = _keys(
    "あの", "その", "まあ", "なんか", "ちょっと", "やっぱり", "こう", "なんていうか"
)
C_KEEP = _keys("はい", "ええ", "うん", "そう", "ああ", "いや", "いえ", "ううん")

# 畳語 — reduplicated words. The tokenizer never splits a morpheme, so a word whose two halves
# are equal stays whole; these keys additionally glue UniDic's split forms (はい|はい) back.
PROTECTED_REDUPLICATIONS = _keys(
    "時々", "人々", "色々", "いろいろ", "我々", "どんどん", "まだまだ", "そろそろ", "だんだん",
    "ますます", "わくわく", "はいはい", "そうそう", "うんうん", "いえいえ", "もしもし",
    "どうもどうも", "ねえねえ", "まあまあ", "まーまー",
)

# Correction markers. "赤じゃなくて青がいい" is a real contrast and "月曜、じゃなくて火曜" is a
# repair; only meaning tells them apart, so both escalate and nothing is deleted.
SELF_REPAIR_MARKERS = _keys(
    "じゃなくて", "ではなくて", "じゃなく", "ではなく", "というか", "っていうか", "間違えた",
    "失礼", "いや",
)

# Multi-morpheme surfaces the tokenizer glues back into one token (UniDic splits えー|と).
MERGE_KEYS = (
    A1_FILLERS | A2_FILLERS | B_MARKERS | C_KEEP | PROTECTED_REDUPLICATIONS | SELF_REPAIR_MARKERS
)
MERGE_MAX_MORPHEMES = 5

# A sentence ending on one of these conjunctive particles is unfinished ("来週の会議ですが"), so
# no terminal 。 is added. て/で are left out on purpose: "ちょっと待って" is a complete request.
CONTINUATION_PARTICLES = frozenset(
    {"が", "けど", "けれど", "けれども", "し", "ので", "のに", "ながら", "ば", "ても", "でも"}
)

NUMERAL_CHARS = frozenset("0123456789０１２３４５６７８９〇一二三四五六七八九十百千万億")
