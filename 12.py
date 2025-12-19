#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12_fixed.py — OpenAI-only (requires OPENAI_API_KEY)

Fixes vs original:
- Actually uses --no-extract and --kw-weight (AI keyword extraction + embedding blend)
- Uses honest cosine -> match% mapping (no inflated "enhanced_sim" nonsense)
- Output 1 prints cosine, match%, est CTR, and keyword hits
- Prompt grounding fixed: provides Japanese segment text (translates EN->JA only when needed)
- Forces model to return JSON (validated) then renders markdown
- Reuses one OpenAI client (less overhead)
- No proxy-env nuking side effects

Usage:
  export OPENAI_API_KEY=sk-...
  export EMBEDDING_BACKEND=openai
  export EMBEDDING_MODEL="text-embedding-3-small"
  python3 12_fixed.py --brief "Target SMB owners buying routers and labelers"
  python3 12_fixed.py --debug
"""

import os, sys, json, argparse, re, time, traceback
from datetime import datetime, timezone
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import faiss
import httpx
from packaging import version
from openai import OpenAI, BadRequestError

from utils.embedding import get_embedding  # uses EMBEDDING_BACKEND

# ---------------------------
# Config
# ---------------------------
GEN_MODEL = os.getenv("OPENAI_GEN_MODEL", "gpt-5.2")
DEBUG = False  # set from args

def _path(*parts):
    p1 = os.path.join("data", *parts)
    p2 = os.path.join("Data", *parts)
    return p1 if os.path.exists(p1) else p2

INDEX_PATH = _path("faiss2.index")
DOCS_PATH  = _path("docs2.jsonl")
JAPAN_MAP_PATH = _path("japan.json")

# ---------------------------
# Sanity: index & docs
# ---------------------------
missing = [p for p in (INDEX_PATH, DOCS_PATH) if not os.path.exists(p)]
if missing:
    sys.exit("❌ Missing files:\n  " + "\n  ".join(missing) +
             "\nTip: re-run `ingest_index_json.py` with your current EMBEDDING_MODEL.")

index = faiss.read_index(INDEX_PATH)
docs  = [json.loads(l) for l in open(DOCS_PATH, "r", encoding="utf-8")]

# Load Japanese name mapping (optional)
japanese_names: Dict[str, str] = {}
if os.path.exists(JAPAN_MAP_PATH):
    with open(JAPAN_MAP_PATH, "r", encoding="utf-8") as f:
        japanese_names = json.load(f)
else:
    print(f"⚠️  WARNING: Japanese mapping file not found at {JAPAN_MAP_PATH}")
    print("    Segment names will remain in English.")

# Translation cache
_translation_cache: Dict[str, str] = {}

# ---------------------------
# HTTPX client helper (handles 0.27 vs 0.28+)
# ---------------------------
def _make_httpx_client(proxy: Optional[str], timeout: float = 90.0) -> httpx.Client:
    """
    Create an httpx.Client that works across httpx versions.
    httpx 0.28+ uses 'proxy='; 0.27- uses 'proxies='.
    """
    kw = "proxy" if version.parse(httpx.__version__) >= version.parse("0.28.0") else "proxies"
    kwargs = {"timeout": timeout}
    if proxy:
        kwargs[kw] = proxy
    try:
        return httpx.Client(**kwargs)
    except TypeError:
        # Fallback if the current httpx doesn't support the chosen kw
        kwargs.pop(kw, None)
        alt_kw = "proxies" if kw == "proxy" else "proxy"
        if proxy:
            kwargs[alt_kw] = proxy
        return httpx.Client(**kwargs)

# ---------------------------
# OpenAI client (singleton)
# ---------------------------
_OPENAI_CLIENT: Optional[OpenAI] = None

def _openai_client() -> OpenAI:
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is not None:
        return _OPENAI_CLIENT

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")

    proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or os.getenv("ALL_PROXY")
    http_client = _make_httpx_client(proxy, timeout=90.0)
    _OPENAI_CLIENT = OpenAI(api_key=api_key, http_client=http_client)
    return _OPENAI_CLIENT

def _chat_create(client: OpenAI, model: str, messages: List[dict], temperature: float = 0.0, max_completion_tokens: Optional[int] = None):
    """
    Compatibility wrapper using max_completion_tokens for newer models
    """
    kwargs = {"model": model, "messages": messages, "temperature": temperature}
    if max_completion_tokens is not None:
        kwargs["max_completion_tokens"] = max_completion_tokens
    return client.chat.completions.create(**kwargs)

# ---------------------------
# Helpers
# ---------------------------
def debug_print(msg: str):
    if DEBUG:
        print(msg)

def has_japanese(text: str) -> bool:
    if not text:
        return False
    return any('\u3040' <= c <= '\u309F' or
               '\u30A0' <= c <= '\u30FF' or
               '\u4E00' <= c <= '\u9FAF' for c in text)

def get_japanese_name(english_name: str) -> str:
    return japanese_names.get(english_name, english_name)

def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.array(v, dtype="float32")
    n = np.linalg.norm(v) + 1e-12
    return v / n

def _percent_from_cos(cos_val: float) -> float:
    # cosine (-1..1) → 0..100%
    return max(0.0, min(1.0, (cos_val + 1.0) / 2.0)) * 100.0

def _tokenize_lower(s: str) -> set:
    if not s:
        return set()
    english_terms = set(re.findall(r"[a-zA-Z0-9\-\+/#\.]+", s.lower()))
    japanese_terms = set(re.findall(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+", s))
    return english_terms | japanese_terms

def estimate_ctr_percent(match_pct: float, hit_count: int, base_ctr_pct: float = 1.0) -> float:
    """
    Heuristic CTR estimator:
      est_ctr% = base_ctr_pct * score_factor * kw_bonus
      score_factor = 0.5 + 0.5 * (match_pct/100)
      kw_bonus = 1 + 0.02 * hit_count, capped at 1.25
    """
    score_factor = 0.5 + 0.5 * (match_pct / 100.0)
    kw_bonus = min(1.25, 1.0 + 0.02 * hit_count)
    return round(base_ctr_pct * score_factor * kw_bonus, 2)

# ---------------------------
# Translation (cached)
# ---------------------------
def _cached_translate(key_prefix: str, text: str, sys_msg: str) -> str:
    if not text:
        return text
    key = f"{key_prefix}:{text[:200]}"
    if key in _translation_cache:
        return _translation_cache[key]

    # Skip translation for very short text to save time/cost
    if len(text.strip()) < 15:
        _translation_cache[key] = text
        return text

    try:
        client = _openai_client()
        resp = _chat_create(
            client,
            model=GEN_MODEL,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": text}
            ],
            temperature=0.0,
            max_completion_tokens=500
        )
        out = resp.choices[0].message.content.strip()
        _translation_cache[key] = out
        return out
    except Exception as e:
        debug_print(f"⚠️ Translation failed ({key_prefix}): {e}")
        _translation_cache[key] = text
        return text

def translate_japanese_to_english(text: str) -> str:
    if not has_japanese(text):
        return text
    sys_msg = (
        "Translate the following Japanese text to English. "
        "Keep the meaning accurate and preserve marketing/business terminology. "
        "Return ONLY the English translation."
    )
    return _cached_translate("ja2en", text, sys_msg)

def translate_english_to_japanese(text: str) -> str:
    # Only translate if it looks non-Japanese
    if has_japanese(text):
        return text
    sys_msg = (
        "Translate the following text to Japanese. "
        "Keep meaning accurate, preserve marketing/business terminology, and keep it concise. "
        "Return ONLY the Japanese translation."
    )
    return _cached_translate("en2ja", text, sys_msg)

# ---------------------------
# Step 1: AI keyword extraction (optional)
# ---------------------------
def extract_keywords_ai(brief: str, max_terms: int = 20) -> List[str]:
    """
    Extract compact keywords (English and/or Japanese).
    Returns list of strings.
    """
    client = _openai_client()
    sys_msg = (
        f"Extract up to {max_terms} concise keywords (English and/or Japanese) from the campaign brief that are "
        "useful for matching Amazon audience/product segments. "
        "Return ONLY a JSON array of strings. Include both English and Japanese terms when relevant."
    )
    try:
        resp = _chat_create(
            client,
            model=GEN_MODEL,
            messages=[{"role": "system", "content": sys_msg},
                      {"role": "user", "content": brief}],
            temperature=0.0,
            max_completion_tokens=220
        )
        content = resp.choices[0].message.content.strip()
        arr = json.loads(content)
        out, seen = [], set()
        for x in arr:
            if isinstance(x, str):
                t = x.strip()
                if t and t.lower() not in seen:
                    seen.add(t.lower())
                    out.append(t)
        return out[:max_terms]
    except Exception as e:
        debug_print(f"Keyword extraction failed, falling back: {e}")
        toks = [t for t in _tokenize_lower(brief) if len(t) > 1]
        out, seen = [], set()
        for t in list(toks)[:max_terms]:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

# ---------------------------
# Step 2: Retrieval
# ---------------------------
def retrieve_segments_detailed(
    brief: str,
    top_k: int = 10,
    use_extract: bool = True,
    kw_weight: float = 0.4,
    min_cos: float = 0.20,
    base_ctr_pct: float = 1.0
) -> Tuple[List[dict], List[str], Optional[str]]:
    """
    Returns: (rows, ai_kws, error_msg)
    Each row includes:
      keyword, jp_name, text, cosine, match_pct, hits, est_ctr
    """
    try:
        if not brief or len(brief.strip()) < 10:
            return [], [], f"キャンペーンの詳細は10文字以上で入力してください。現在の文字数: {len(brief.strip()) if brief else 0}文字"
        if top_k < 1:
            return [], [], "top_k must be >= 1"

        debug_print(f"Brief: {brief[:120]}")

        # Use English for embedding if brief is JP (better embedding match)
        emb_text = translate_japanese_to_english(brief)
        emb_brief = _normalize(get_embedding(emb_text))

        ai_kws: List[str] = []
        emb_kw: Optional[np.ndarray] = None
        if use_extract:
            ai_kws = extract_keywords_ai(brief, max_terms=20)
            if ai_kws:
                # Embed keywords: join them; translate JP->EN so embedding space consistent
                kws_text = " | ".join(ai_kws)
                kws_text_en = translate_japanese_to_english(kws_text)
                emb_kw = _normalize(get_embedding(kws_text_en))

        # Blend query vectors if keyword embedding exists
        q_vec = emb_brief
        if emb_kw is not None:
            w = float(max(0.0, min(1.0, kw_weight)))
            q_vec = _normalize((1.0 - w) * emb_brief + w * emb_kw)

        # Search
        search_size = min(max(top_k * 8, 50), len(docs))
        D, I = index.search(np.array([q_vec], dtype="float32"), search_size)

        # Token sets for hits
        brief_terms = _tokenize_lower(brief)
        kws_terms = set()
        for k in ai_kws:
            kws_terms |= _tokenize_lower(k)

        rows: List[dict] = []
        seen = set()

        # Primary pass: min_cos
        for rank, idx in enumerate(I[0]):
            if len(rows) >= top_k:
                break
            rec = docs[idx]
            key = rec.get("keyword") or f"seg_{idx}"
            if key in seen:
                continue
            seen.add(key)

            text = (rec.get("text") or rec.get("answer") or "")
            cos = float(D[0][rank])

            if cos < min_cos:
                continue

            seg_terms = _tokenize_lower(text) | _tokenize_lower(key)
            hits = sorted((brief_terms | kws_terms) & seg_terms)
            match_pct = _percent_from_cos(cos)
            est_ctr = estimate_ctr_percent(match_pct, hit_count=len(hits), base_ctr_pct=base_ctr_pct)

            rows.append({
                "keyword": key,
                "jp_name": get_japanese_name(key),
                "text": text,
                "cosine": cos,
                "match_pct": match_pct,
                "hits": hits[:20],
                "est_ctr_pct": est_ctr
            })

        # Fallback: if not enough, just take top remaining (lower threshold) but keep ranking honest
        if len(rows) < top_k:
            for rank, idx in enumerate(I[0]):
                if len(rows) >= top_k:
                    break
                rec = docs[idx]
                key = rec.get("keyword") or f"seg_{idx}"
                if key in seen:
                    continue
                seen.add(key)

                text = (rec.get("text") or rec.get("answer") or "")
                cos = float(D[0][rank])

                seg_terms = _tokenize_lower(text) | _tokenize_lower(key)
                hits = sorted((brief_terms | kws_terms) & seg_terms)
                match_pct = _percent_from_cos(cos)
                est_ctr = estimate_ctr_percent(match_pct, hit_count=len(hits), base_ctr_pct=base_ctr_pct)

                rows.append({
                    "keyword": key,
                    "jp_name": get_japanese_name(key),
                    "text": text,
                    "cosine": cos,
                    "match_pct": match_pct,
                    "hits": hits[:20],
                    "est_ctr_pct": est_ctr
                })

        # Sort by cosine desc (FAISS already mostly does, but fallback pass can append)
        rows.sort(key=lambda r: r["cosine"], reverse=True)

        if not rows:
            return [], ai_kws, "No matching segments found. Try different keywords or rebuild the index."

        return rows[:top_k], ai_kws, None

    except Exception as e:
        traceback.print_exc()
        return [], [], f"Retrieval error: {str(e)}"

# ---------------------------
# Step 3: Generation (JSON -> validated -> markdown)
# ---------------------------
def _build_generation_prompt_json(campaign_brief: str, rows: List[dict]) -> str:
    """
    Provide per-segment text in Japanese (translate only if needed),
    force JSON output for validation.
    """
    blocks = []
    allowed_names = []
    for i, r in enumerate(rows, 1):
        name = r["jp_name"]
        allowed_names.append(name)
        snippet = (r["text"] or "")[:420]
        snippet_ja = translate_english_to_japanese(snippet) if not has_japanese(snippet) else snippet
        blocks.append(
            f"Segment {i}: {name}\n"
            f"Text_JA: {snippet_ja}\n"
        )

    allowed_block = json.dumps(allowed_names, ensure_ascii=False)

    sys_msg = (
        "You are an Amazon Ads strategist. Respond entirely in Japanese.\n"
        "Return ONLY valid JSON.\n"
        "CRITICAL REQUIREMENTS:\n"
        f"- You MUST create exactly {len(allowed_names)} complete segments using ALL the names provided.\n"
        "- Use ONLY segment names from Allowed Segment Names (verbatim).\n"
        "- Do NOT invent or rephrase segment names.\n"
        "- For keywords, use ONLY terms that appear in the provided Text_JA for that segment.\n"
        "- Each segment must have EXACTLY 10 keywords.\n"
        "- DO NOT stop early - complete ALL segments fully.\n"
    )

    user_msg = (
        f"=== Campaign Brief ===\n{campaign_brief}\n\n"
        f"=== Retrieved Segments (from your data) ===\n{'\n\n'.join(blocks)}\n\n"
        f"=== Allowed Segment Names (use ALL {len(allowed_names)} names) ===\n{allowed_block}\n\n"
        f"Return JSON array with exactly {len(allowed_names)} objects in this shape:\n"
        "[\n"
        "  {\n"
        "    \"segment_name\": \"<one allowed name>\",\n"
        "    \"why_it_fits\": \"1-2 concise lines in Japanese\",\n"
        "    \"keywords\": [\"kw1\",\"kw2\",...\"kw10\"]\n"
        "  },\n"
        "  ...\n"
        "]\n"
        f"IMPORTANT: Your response must contain exactly {len(allowed_names)} complete segment objects.\n"
    )
    return sys_msg + "\n\n" + user_msg

def _generate_segments_json(prompt: str) -> str:
    client = _openai_client()
    resp = _chat_create(
        client,
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_completion_tokens=2500  # Increased for 10 complete segments
    )
    return resp.choices[0].message.content.strip()

def _validate_and_render(campaign_brief: str, rows: List[dict], json_text: str) -> Tuple[str, List[dict]]:
    allowed = [r["jp_name"] for r in rows]
    allowed_set = set(allowed)

    try:
        data = json.loads(json_text)
    except Exception:
        raise ValueError("Model did not return valid JSON.")

    if not isinstance(data, list):
        raise ValueError("JSON root must be a list.")

    # Validate objects
    seen = set()
    cleaned: List[dict] = []
    for obj in data:
        if not isinstance(obj, dict):
            continue
        name = obj.get("segment_name")
        why = obj.get("why_it_fits")
        kws = obj.get("keywords")

        if not isinstance(name, str) or name not in allowed_set:
            continue
        if name in seen:
            continue
        if not isinstance(why, str) or len(why.strip()) < 3:
            continue
        if not (isinstance(kws, list) and len(kws) == 10 and all(isinstance(x, str) and x.strip() for x in kws)):
            continue

        seen.add(name)
        cleaned.append({
            "segment_name": name,
            "why_it_fits": why.strip(),
            "keywords": [k.strip() for k in kws]
        })

    # Ensure we have all segments. If not, hard fail so you notice instead of silently lying.
    if len(cleaned) != len(allowed):
        missing = [n for n in allowed if n not in seen]
        raise ValueError(f"Incomplete generation. Missing segments: {missing}")

    # Render markdown in stable order
    cleaned_by_name = {c["segment_name"]: c for c in cleaned}

    md_lines = []
    md_lines.append("💡 Proposed Target Segments\n")
    for i, name in enumerate(allowed, 1):
        c = cleaned_by_name[name]
        md_lines.append(f"**Segment {i}: {name}**")
        md_lines.append(f"**Why it fits:** {c['why_it_fits']}")
        md_lines.append("**Keywords:** " + "、".join(c["keywords"]))
        md_lines.append("")  # blank line

    return "\n".join(md_lines).strip(), cleaned

# ---------------------------
# Output 1: Pretty printing
# ---------------------------
def print_matches(rows: List[dict]):
    print("\n🔎 Matched segments (from YOUR data):\n")
    for i, r in enumerate(rows, 1):
        name = r["jp_name"]
        mp = r["match_pct"]
        print(f"{i}) {name} ({mp:.1f}%)")

# ---------------------------
# Save output
# ---------------------------
def save_generation(
    brief: str,
    ai_kws: List[str],
    rows: List[dict],
    md_output: str,
    generation_json: Optional[List[dict]] = None,
    path: str = "generated_segments.jsonl"
):
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "brief": brief,
            "ai_keywords": ai_kws,
            "retrieved_segments": [r["keyword"] for r in rows],
            "scores": [
                {"keyword": r["keyword"], "cosine": r["cosine"], "match_pct": r["match_pct"], "est_ctr_pct": r["est_ctr_pct"]}
                for r in rows
            ],
            "generation_json": generation_json,
            "output_markdown": md_output
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
        debug_print(f"💾 Saved generation → {path}")
    except Exception as e:
        debug_print(f"⚠️ Could not save to {path}: {e}")

# ---------------------------
# CLI
# ---------------------------
def main():
    global DEBUG
    ap = argparse.ArgumentParser(description="Amazon Ads Automation - Segment Generator (fixed)")

    ap.add_argument("--brief", type=str, default=None, help="Inline campaign brief.")
    ap.add_argument("--no-extract", action="store_true", help="Disable AI keyword extraction.")
    ap.add_argument("--kw-weight", type=float, default=0.4, help="Blend weight for keyword embedding (0..1).")
    ap.add_argument("--top-k", type=int, default=10, help="How many segments to retrieve.")
    ap.add_argument("--min-cos", type=float, default=0.20, help="Minimum cosine threshold for primary pass.")
    ap.add_argument("--retrieval-only", action="store_true", help="Only print matches, skip generation.")
    ap.add_argument("--base-ctr", type=float, default=1.0, help="Base CTR %% prior (default 1.0).")
    ap.add_argument("--debug", action="store_true", help="Show debug output.")
    args = ap.parse_args()

    DEBUG = args.debug

    brief = args.brief or input("Enter campaign brief: ").strip()
    if not brief:
        print("❌ Campaign brief is required")
        return

    rows, ai_kws, error = retrieve_segments_detailed(
        brief=brief,
        top_k=args.top_k,
        use_extract=not args.no_extract,
        kw_weight=max(0.0, min(1.0, args.kw_weight)),
        min_cos=args.min_cos,
        base_ctr_pct=args.base_ctr
    )

    if error:
        print(f"\n❌ {error}")
        return

    print_matches(rows)

    if args.retrieval_only:
        return

    try:
        prompt = _build_generation_prompt_json(brief, rows)
        raw_json = _generate_segments_json(prompt)
        md, cleaned = _validate_and_render(brief, rows, raw_json)

        print("\n" + md + "\n")
        save_generation(brief, ai_kws, rows, md_output=md, generation_json=cleaned)

    except Exception as e:
        print(f"\n❌ Generation failed: {str(e)}")
        if DEBUG:
            traceback.print_exc()

if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
