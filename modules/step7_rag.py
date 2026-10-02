"""
═══════════════════════════════════════════════════════════════════
STEP 6.5 — RAG (Retrieval-Augmented Generation) over the SDS sheet
═══════════════════════════════════════════════════════════════════

PURPOSE:
    After Step 6 (LLM recommendations), let the user ask free-form
    questions about the uploaded Safety Data Sheet and get grounded
    answers retrieved from the document itself.

HOW IT WORKS (no external vector DB required):
    1. Chunking      — the extracted SDS text is split into overlapping
                       sentence-based chunks (~800 chars).
    2. Retrieval     — TF-IDF cosine similarity (lightweight, offline)
                       ranks chunks against the user query; top-k are kept.
    3. Augmentation  — retrieved chunks + structured substance table are
                       injected into a system prompt.
    4. Generation    — the answer is produced by the OpenRouter LLM
                       selected in the sidebar (Qwen / Phi-4 / Mistral /
                       DeepSeek / OpenAI). If no API key or the call fails,
                       it degrades gracefully to an extractive answer that
                       simply returns the most relevant passages.
"""

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: int
    page: Optional[int]
    score: float
    text: str


@dataclass
class RAGAnswer:
    question: str
    answer: str
    sources: List[RetrievedChunk] = field(default_factory=list)
    generator: str = "extractive"   # "llm:<model>" or "extractive"
    llm_error: Optional[str] = None


# ── Chunking ─────────────────────────────────────────────────────────────────

_SENT_SPLIT = re.compile(r"(?<=[.;!?\n])\s+")


def _split_pages(full_text: str, pages: Optional[List]) -> List[tuple]:
    """Return list of (page_no_or_None, page_text). Uses per-page results
    when available so citations can name a page number."""
    if pages:
        out = []
        for i, p in enumerate(pages, start=1):
            txt = getattr(p, "raw_text", None) or getattr(p, "text", "") or ""
            if txt.strip():
                out.append((i, txt))
        if out:
            return out
    return [(None, full_text)] if full_text.strip() else []


def build_chunks(full_text: str, pages: Optional[List] = None,
                 max_chars: int = 800, overlap: int = 120) -> List[RetrievedChunk]:
    """Split document text into overlapping, sentence-aligned chunks."""
    chunks: List[RetrievedChunk] = []
    cid = 0
    for page_no, ptext in _split_pages(full_text, pages):
        sents = [s.strip() for s in _SENT_SPLIT.split(ptext) if s.strip()]
        cur = ""
        for s in sents:
            if len(cur) + len(s) + 1 <= max_chars:
                cur = f"{cur} {s}".strip()
            else:
                if cur:
                    chunks.append(RetrievedChunk(cid, page_no, 0.0, cur))
                    cid += 1
                # keep an overlap tail from the previous chunk for context
                tail = cur[-overlap:] if overlap and len(cur) > overlap else ""
                cur = f"{tail} {s}".strip() if tail else s
            # hard-split pathologically long sentences
            while len(cur) > max_chars:
                cut = cur.rfind(" ", 0, max_chars)
                cut = cut if cut > 0 else max_chars
                chunks.append(RetrievedChunk(cid, page_no, 0.0, cur[:cut].strip()))
                cid += 1
                cur = cur[cut:].strip()
        if cur:
            chunks.append(RetrievedChunk(cid, page_no, 0.0, cur))
            cid += 1
    return chunks


# ── TF-IDF retrieval ─────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-\.%]*")
_STOP = set("""a an the and or of to in for on with by is are was were be been
this that these those from at as it its you your we our they their he she i
not no nor so if then than there here what which who whom will would can could
should may might do does did have has had but about into over under per""".split())


def _tokenize(text: str) -> List[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOP and len(w) > 1]


class TfidfIndex:
    """Minimal pure-Python TF-IDF index with cosine similarity."""

    def __init__(self, chunks: List[RetrievedChunk]):
        self.chunks = chunks
        self.doc_tokens = [_tokenize(c.text) for c in chunks]
        self.doc_tf = [Counter(t) for t in self.doc_tokens]
        self.doc_len = [math.sqrt(sum(c[f] ** 2 for f in c)) or 1.0
                        for c in self.doc_tf]
        n = max(len(chunks), 1)
        df: Counter = Counter()
        for c in self.doc_tf:
            for term in c:
                df[term] += 1
        self.idf = {t: math.log(1 + n / (1 + d)) for t, d in df.items()}

    def search(self, query: str, k: int = 4) -> List[RetrievedChunk]:
        qt = Counter(_tokenize(query))
        if not qt:
            return []
        scored = []
        for i, tf in enumerate(self.doc_tf):
            dot = sum(self.idf.get(t, 0.0) ** 2 * min(tf[t], 3)
                      for t in qt if t in tf)
            if dot <= 0:
                continue
            qnorm = math.sqrt(sum(self.idf.get(t, 0.0) ** 2 for t in qt)) or 1.0
            sim = dot / (qnorm * self.doc_len[i])
            scored.append((sim, self.chunks[i]))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for sim, ch in scored[:k]:
            c = RetrievedChunk(ch.chunk_id, ch.page, round(sim, 4), ch.text)
            out.append(c)
        return out


# ── Structured context (SDS composition sheet) ───────────────────────────────

def _substances_table(structured_mds) -> str:
    if structured_mds is None:
        return ""
    rows = []
    try:
        subs = structured_mds.all_substances
    except AttributeError:
        subs = []
    for s in subs:
        rng = (f"{s.weight_ppm_min:g}-{s.weight_ppm_max:g} ppm"
               if getattr(s, "is_range", False) and s.weight_ppm_min is not None
               else f"{s.weight_ppm:g} ppm")
        rows.append("| {} | {} | {} | {} |".format(
            s.name or "", s.cas_number or "-",
            f"{s.weight_fraction_max_pct:g}%" if s.weight_fraction_max_pct is not None else "-",
            rng))
    if not rows:
        return ""
    header = ("### Structured SDS composition table\n"
              "| Substance | CAS | Max wt% | Concentration |\n"
              "|---|---|---|---|\n")
    return header + "\n".join(rows)


# ── RAG engine ───────────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = (
    "You are a precise assistant answering questions about a Safety Data "
    "Sheet (SDS) that was uploaded and processed by a compliance pipeline. "
    "Answer ONLY from the provided context excerpts and tables. If the "
    "answer is not contained in the context, say exactly: "
    "\"The requested information is not present in this SDS.\" Quote exact "
    "values (CAS numbers, concentrations, section names) as written. Keep "
    "the answer concise (max ~200 words)."
)


class SdsRagEngine:
    """Builds a retrieval index over the OCR'd SDS and answers user prompts."""

    def __init__(self, full_text: str, pages=None, structured_mds=None,
                 chunk_size: int = 800):
        self.chunks = build_chunks(full_text, pages, max_chars=chunk_size)
        self.index = TfidfIndex(self.chunks)
        self.table_ctx = _substances_table(structured_mds)

    def retrieve(self, question: str, k: int = 4) -> List[RetrievedChunk]:
        return self.index.search(question, k=k)

    def answer(self, question: str, provider: str = "openrouter",
               model: Optional[str] = None, temperature: float = 0.2,
               max_tokens: int = 512, top_k: int = 4) -> RAGAnswer:
        hits = self.retrieve(question, k=top_k)
        if not hits:
            return RAGAnswer(question,
                             "No relevant passages found in the SDS for this "
                             "question. Try rephrasing with terms from the "
                             "document (substance names, CAS numbers, sections).",
                             [], "extractive")

        context = "\n\n".join(
            f"[Excerpt {i+1}{f' — page {h.page}' if h.page else ''}]\n{h.text}"
            for i, h in enumerate(hits))
        user_prompt = (
            f"### Retrieved SDS excerpts\n{context}\n\n"
            + (f"{self.table_ctx}\n\n" if self.table_ctx else "")
            + f"### Question\n{question}"
        )

        # Try generative answer via the configured LLM first
        try:
            from modules.step6_llm import call_openrouter
            from .config import get_openrouter_api_key, get_openrouter_llm_model
            if not get_openrouter_api_key():
                raise RuntimeError("OpenRouter API key not configured.")
            ans = call_openrouter(user_prompt, RAG_SYSTEM_PROMPT,
                                  model=model or get_openrouter_llm_model(),
                                  temperature=temperature,
                                  max_tokens=max_tokens).strip()
            return RAGAnswer(question, ans, hits,
                             f"llm:{model or 'openrouter'}")
        except Exception as exc:
            logger.warning(f"RAG LLM generation failed ({exc}); using extractive answer.")
            excerpt = "\n\n".join(
                f"**Relevant passage {'(page %d)' % h.page if h.page else '(see document)'}:**\n"
                f"> {h.text[:400]}" for h in hits[:2])
            return RAGAnswer(
                question,
                "⚠️ Generative answering unavailable (set api_keys.openrouter in "
                "config.json or OPENROUTER_API_KEY). Showing the most relevant "
                f"SDS passages instead:\n\n{excerpt}",
                hits, "extractive", llm_error=str(exc))
