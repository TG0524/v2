#!/usr/bin/env python3
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import json
import subprocess
import re
import httpx
from packaging import version
from openai import OpenAI

# ---------------------------------------
# HTTPX client + OpenAI client (proxy-safe)
# ---------------------------------------
def _make_httpx_client(proxy: str | None, timeout: float = 30.0) -> httpx.Client:
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
        # Fallback if current httpx doesn't support the chosen kw
        kwargs.pop(kw, None)
        alt_kw = "proxies" if kw == "proxy" else "proxy"
        if proxy:
            kwargs[alt_kw] = proxy
        return httpx.Client(**kwargs)

_openai_singleton = None
def get_openai_client() -> OpenAI:
    global _openai_singleton
    if _openai_singleton is not None:
        return _openai_singleton

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")

    proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or os.getenv("ALL_PROXY")
    http_client = _make_httpx_client(proxy, timeout=float(os.getenv("OPENAI_HTTP_TIMEOUT", "90")))

    # Optional: prevent SDK from re-reading proxy env vars internally
    for k in ("HTTP_PROXY","http_proxy","HTTPS_PROXY","https_proxy","ALL_PROXY","all_proxy"):
        os.environ.pop(k, None)

    _openai_singleton = OpenAI(
        api_key=api_key,
        http_client=http_client,
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        project=os.getenv("OPENAI_PROJECT") or None,
    )
    return _openai_singleton

# ---------------------------------------
# Data paths & Japanese mapping
# ---------------------------------------
def _path(*parts):
    p1 = os.path.join("data", *parts)
    p2 = os.path.join("Data", *parts)
    return p1 if os.path.exists(p1) else p2

JAPAN_MAP_PATH = _path("japan.json")
japanese_names = {}
if os.path.exists(JAPAN_MAP_PATH):
    with open(JAPAN_MAP_PATH, "r", encoding="utf-8") as f:
        japanese_names = json.load(f)

def get_japanese_name(english_name: str) -> str:
    """Get Japanese name for English segment name, fallback to English if not found."""
    return japanese_names.get(english_name, english_name)

# ---------------------------------------
# Translation helper (keywords → Japanese)
# ---------------------------------------
def translate_keywords_to_japanese(keywords: list[str]) -> list[str]:
    """
    Translate a list of English keywords to Japanese using OpenAI.
    Designed to be safe for gpt-5-nano (no temperature/top_p).
    """
    if not keywords:
        print("No keywords to translate")  # Debug
        return keywords

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("No OpenAI API key found")  # Debug
        return keywords  # Fallback to original if no API key

    try:
        client = get_openai_client()
    except Exception as e:
        print(f"OpenAI client init error: {e}")
        return keywords

    model = os.getenv("OPENAI_GEN_MODEL", "gpt-4o-mini")
    keywords_text = ", ".join(keywords)
    print(f"Translating keywords: {keywords_text}")  # Debug

    sys_msg = (
        "Translate the following English keywords to Japanese. "
        "Return ONLY a JSON array of Japanese translations in the same order. "
        "Keep marketing and product terms natural for Japanese Amazon users."
    )
    messages = [{"role": "system", "content": sys_msg},
                {"role": "user", "content": keywords_text}]

    try:
        # Use max_completion_tokens for newer models
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=400
        )
        content = resp.choices[0].message.content.strip()
        print(f"Translation response: {content}")  # Debug
        translated = json.loads(content)
        if isinstance(translated, list) and len(translated) == len(keywords):
            print(f"Successfully translated to: {translated}")  # Debug
            return translated
        else:
            print(f"Translation format error: expected {len(keywords)} items, got {len(translated) if isinstance(translated, list) else 'not a list'}")  # Debug
    except Exception as e:
        print(f"Translation error: {e}")

    # Fallback to original keywords if translation fails
    print("Falling back to original keywords")  # Debug
    return keywords

# ---------------------------------------
# Flask app
# ---------------------------------------
app = Flask(__name__)
CORS(app)

# Ensure required environment variables
required_env = ['OPENAI_API_KEY', 'EMBEDDING_BACKEND', 'EMBEDDING_MODEL']
missing_env = [var for var in required_env if not os.getenv(var)]
if missing_env:
    print(f"⚠️  Missing environment variables: {', '.join(missing_env)}")
    print("Set them like:")
    print("export OPENAI_API_KEY=sk-...")
    print("export EMBEDDING_BACKEND=openai")
    print("export EMBEDDING_MODEL=text-embedding-3-small")

@app.route('/')
def index():
    return send_from_directory('public', 'index.html')

# ---------------------------------------
# API: retrieval only
# ---------------------------------------
@app.route('/api/retrieve', methods=['POST'])
def retrieve_segments():
    try:
        data = request.json
        print(f"Received data: {data}")

        campaign_brief = data.get('campaign_brief', '').strip()
        top_k = int(data.get('top_k', 10))
        keyword_weight = float(data.get('keyword_weight', 0.4))
        enable_keywords = data.get('enable_keywords', True)

        if not campaign_brief:
            return jsonify({'error': 'Campaign brief is required'}), 400
        # Minimum characters for a useful marketing brief
        MIN_BRIEF_CHARS = 10

        if len(campaign_brief) < MIN_BRIEF_CHARS:
            return jsonify({
        "error": f"キャンペーン説明が短すぎます。最低 {MIN_BRIEF_CHARS} 文字必要です。"
        }), 400


        # Build command for 12.py
        cmd = [
            sys.executable, '12.py',
            '--brief', campaign_brief,
            '--kw-weight', str(keyword_weight),
            '--retrieval-only'  # Only get matches, no LLM generation
        ]
        if not enable_keywords:
            cmd.append('--no-extract')

        print(f"Running command: {' '.join(cmd)}")

        # Run the script
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')

        print(f"Return code: {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")

        if result.returncode != 0:
            return jsonify({'error': f'Script error: {result.stderr}'}), 500

        # Parse the output from 12.py
        segments = parse_retrieval_output(result.stdout)

        return jsonify({
            'segments': segments,
            'total_found': len(segments)
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

# ---------------------------------------
# API: full generate (retrieval + LLM output)
# ---------------------------------------
@app.route('/api/generate', methods=['POST'])
def generate_segments():
    try:
        data = request.json
        print(f"Generate received data: {data}")

        campaign_brief = data.get('campaign_brief', '').strip()
        top_k = int(data.get('top_k', 10))
        keyword_weight = float(data.get('keyword_weight', 0.4))
        enable_keywords = data.get('enable_keywords', True)

        if not campaign_brief:
            return jsonify({'error': 'Campaign brief is required'}), 400

        # Build command for 12.py (without --retrieval-only for full generation)
        cmd = [
            sys.executable, '12.py',
            '--brief', campaign_brief,
            '--kw-weight', str(keyword_weight)
        ]
        if not enable_keywords:
            cmd.append('--no-extract')

        print(f"Running generate command: {' '.join(cmd)}")

        # Run the script
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')

        print(f"Generate return code: {result.returncode}")
        print(f"Generate STDOUT: {result.stdout}")
        print(f"Generate STDERR: {result.stderr}")

        if result.returncode != 0:
            return jsonify({'error': f'Script error: {result.stderr}'}), 500

        # Parse both retrieval results and generated segments
        segments, generated_segments = parse_full_output(result.stdout, campaign_brief)

        return jsonify({
            'segments': segments,
            'generated_segments': generated_segments,
            'total_found': len(segments)
        })

    except Exception as e:
        print(f"Generate error: {e}")
        return jsonify({'error': str(e)}), 500

# ---------------------------------------
# Parsing helpers
# ---------------------------------------
def parse_retrieval_output(output):
    """Parse the retrieval-only output from 12.py"""
    segments = []
    lines = output.split('\n')

    for i, line in enumerate(lines):
        # Look for lines like: "1) Tech Enthusiasts 25-34 (75.2%)" or "1) Tech Enthusiasts 25-34"
        match = re.match(r'(\d+)\)\s+(.+?)(?:\s*\(([\d.]+)%\))?\s*$', line.strip())
        if match:
            segment_name = match.group(2)
            match_percent = float(match.group(3)) if match.group(3) else None
            
            # If no percentage in the line, look for it in the next line
            if match_percent is None and i + 1 < len(lines):
                score_line = lines[i + 1]
                # Parse various formats: "match: 89.1%" or "• match: 89.1%"
                score_match = re.search(r'match:\s*([\d.]+)%', score_line)
                if score_match:
                    match_percent = float(score_match.group(1))

            # Add segment even if no percentage found (for compatibility)
            japanese_segment_name = get_japanese_name(segment_name)
            segments.append({
                'name': japanese_segment_name,
                'match_percent': match_percent or 0.0
            })
    return segments

def parse_full_output(output, brief=""):
    """Parse both retrieval results and generated segments from full 12.py output"""
    # Split output into retrieval and generation parts
    parts = output.split('💡 Proposed Target Segments')

    # Parse retrieval part
    segments = parse_retrieval_output(parts[0])

    # Parse generated segments
    generated_segments = []
    if len(parts) > 1:
        generated_segments = parse_generated_segments(parts[1], brief)

    return segments, generated_segments

def parse_generated_segments(output_text, brief=""):
    """Parse the generated segments from JSON output"""
    segments = []

    print(f"Parsing output text: {output_text[:500]}...")  # Debug

    # Try to parse as JSON first (new format)
    try:
        # Look for JSON array in the output
        json_match = re.search(r'\[[\s\S]*\]', output_text)
        if json_match:
            json_text = json_match.group(0)
            data = json.loads(json_text)
            
            print(f"Found JSON with {len(data)} segments")  # Debug
            
            for item in data:
                if isinstance(item, dict):
                    segment = {
                        'name': item.get('segment_name', ''),
                        'why_fits': item.get('why_it_fits', ''),
                        'keywords': item.get('keywords', [])
                    }
                    segments.append(segment)
                    print(f"Parsed JSON segment: {segment['name']}")  # Debug
            
            return segments
    except Exception as e:
        print(f"JSON parsing failed: {e}, trying markdown fallback")  # Debug

    # Fallback to markdown parsing (old format)
    segment_blocks = re.split(r'\*\*Segment \d+:', output_text)
    print(f"Found {len(segment_blocks)} markdown blocks")  # Debug

    for i, block in enumerate(segment_blocks[1:], 1):  # Skip first empty part
        print(f"Processing markdown block {i}: {block[:200]}...")  # Debug

        segment = {}

        # Extract segment name (first line after the segment marker)
        lines = block.strip().split('\n')
        if lines:
            name_line = lines[0].strip().replace('**', '').strip()
            if name_line:
                segment['name'] = name_line

        # Extract "Why it fits"
        why_match = re.search(r'\*\*Why it fits:\*\*\s*([^\*]+?)(?=\*\*|$)', block, re.DOTALL)
        if why_match:
            segment['why_fits'] = why_match.group(1).strip()

        # Extract keywords - handle both comma-separated and bullet format
        keywords_match = re.search(r'\*\*Keywords:\*\*\s*([^\*]+?)(?=\*\*|$)', block, re.DOTALL)
        if keywords_match:
            keywords_text = keywords_match.group(1).strip()
            # Try Japanese comma first, then regular comma
            keywords = [kw.strip() for kw in keywords_text.split('、') if kw.strip()]
            if len(keywords) <= 1:
                keywords = [kw.strip() for kw in keywords_text.split(',') if kw.strip()]
            # If no commas, try bullet points
            if len(keywords) <= 1:
                keywords = re.findall(r'[•·-]\s*([^\n]+)', keywords_text)
                keywords = [kw.strip() for kw in keywords if kw.strip()]
            
            print(f"Original markdown keywords: {keywords}")  # Debug

            # Only translate if keywords are in English
            if keywords and any(re.search(r'[a-zA-Z]', kw) for kw in keywords):
                try:
                    japanese_keywords = translate_keywords_to_japanese(keywords)
                    print(f"Translated keywords: {japanese_keywords}")  # Debug
                    segment['keywords'] = japanese_keywords
                except Exception as e:
                    print(f"Keyword translation failed: {e}")
                    segment['keywords'] = keywords
            else:
                segment['keywords'] = keywords

        print(f"Parsed markdown segment: {segment}")  # Debug

        # Only add segment if it has essential fields
        if segment.get('name') and (segment.get('why_fits') or segment.get('keywords')):
            segments.append(segment)

    print(f"Final parsed segments: {segments}")  # Debug
    
    # If no segments were parsed, try alternative parsing
    if not segments:
        print("No segments found with standard parsing, trying alternative method...")
        segments = parse_segments_alternative(output_text)
    
    return segments

def parse_segments_alternative(markdown_text):
    """Alternative parsing method for generated segments"""
    segments = []
    
    # Look for any text that looks like segment names
    lines = markdown_text.split('\n')
    current_segment = {}
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Look for segment names (usually after "Segment N:" or as headers)
        if re.match(r'.*セグメント.*:|.*Segment.*:', line) or line.startswith('**') and line.endswith('**'):
            if current_segment.get('name'):
                segments.append(current_segment)
            current_segment = {'name': re.sub(r'\*\*|Segment \d+:\s*', '', line).strip()}
        elif 'なぜ適合' in line or 'Why it fits' in line:
            # Extract reason
            reason = re.sub(r'\*\*.*?\*\*\s*', '', line).strip()
            if reason:
                current_segment['why_fits'] = reason
        elif 'キーワード' in line or 'Keywords' in line:
            # Extract keywords from next few lines
            keywords = re.findall(r'[^\s,]+', line.replace('キーワード', '').replace('Keywords', '').replace('**', '').replace(':', ''))
            if keywords:
                current_segment['keywords'] = keywords[:10]
    
    # Add the last segment
    if current_segment.get('name'):
        segments.append(current_segment)
    
    return segments

# ---------------------------------------
# Healthcheck
# ---------------------------------------
@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'message': 'Flask app is running'})

@app.route('/test')
def test():
    return jsonify({'message': 'Test endpoint working', 'files_exist': {
        'docs.jsonl': os.path.exists('Data/docs.jsonl'),
        'faiss.index': os.path.exists('Data/faiss.index'),
        'japan.json': os.path.exists('Data/japan.json')
    }})

# ---------------------------------------
# Entrypoint
# ---------------------------------------
if __name__ == '__main__':
    # Check if 12.py exists
    if not os.path.exists('12.py'):
        print("❌ 12.py not found in current directory")
        print("Make sure 12.py is in the same folder as app.py")


    print("🚀 Starting Flask server...")
    print("📁 Serving from current directory")
    print("🌐 Open http://localhost:5000 in your browser")

    app.run(debug=True, host='0.0.0.0', port=5000)
