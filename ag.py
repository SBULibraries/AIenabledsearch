import argparse
import json
import logging
import os
import sys
import time
import re
import urllib.parse
from typing import Dict, Optional, Tuple, List
from requests.adapters import HTTPAdapter, Retry
import requests

HOST = os.getenv("LLM_HOST", "https://chatgpt.microsopht.com/ollama")
API_KEY = os.getenv("LLM_API_KEY", "sk-0c1c84e2c0ae4dbca4b71681577b9412")
MODEL = os.getenv("LLM_MODEL", "llama3:latest")
REQUEST_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60")) 
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
RETRY_BACKOFF = float(os.getenv("LLM_RETRY_BACKOFF", "1.5"))  # backoff
DEBUG = os.getenv("DEBUG", "False").upper() == "TRUE"

# Configure module-level logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Streaming LLM Request with Retry Logic
# ─────────────────────────────────────────────────────────────────────────────
def send_chat_prompt(
    system_prompt: str,
    user_prompt: str,
    timeout: int = REQUEST_TIMEOUT,
    max_retries: int = MAX_RETRIES,
) -> str:
    """
    Sends a streaming chat request to the LLM and returns the concatenated response.

    Implements exponential backoff on failures and logs request/response details.

    Args:
        system_prompt (str): The LLM system prompt.
        user_prompt (str): The LLM user prompt.
        timeout (int): HTTP request timeout in seconds.
        max_retries (int): Number of times to retry on failure.

    Returns:
        str: The full LLM response content.

    Raises:
        RuntimeError: If all retry attempts fail.
    """
    url = f"{HOST}/api/chat"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": True,
    }

    session = get_retrying_session(max_retries)

    for attempt in range(1, max_retries + 1):
        try:
            logger.debug("Sending LLM request (attempt %d): %s", attempt, payload)
            response = session.post(url, headers=headers, json=payload, stream=True, timeout=timeout)

            if response.status_code != 200:
                logger.warning(
                    "LLM call returned non-200 status: %d %s", response.status_code, response.text
                )
                raise RuntimeError(f"LLM call failed: {response.status_code} {response.text}")

            full_output = []
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    content_fragment = data.get("message", {}).get("content", "")
                    if content_fragment:
                        full_output.append(content_fragment)
                    if data.get("done", False):
                        break
                except json.JSONDecodeError:
                    logger.debug("Skipping non-JSON line: %s", line)
                    continue

            result = "".join(full_output).strip()
            logger.debug("LLM response: %s", result)
            return result

        except Exception as e:
            logger.error("LLM request error on attempt %d: %s", attempt, str(e))
            if attempt < max_retries:
                wait_time = RETRY_BACKOFF ** attempt
                logger.info("Retrying after %.1f seconds...", wait_time)
                time.sleep(wait_time)
            else:
                logger.critical("LLM request failed after %d attempts", attempt)
                raise RuntimeError(f"LLM request failed after maximum retries: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# Agent 1: Query Splitter with Flags
# ─────────────────────────────────────────────────────────────────────────────
def split_query(user_query: str, verbose: bool = False) -> Dict[str, any]:
    """
    Splits the user's query into:
      - peer_flag: True if 'peer reviewed' in query
      - sbu_flag: True if 'sbu held' or 'held by sbu' in query
      - online_flag: True if 'available online' in query
      - topic_input: subject/filter terms (excluding words used in other flags and inputs)
      - time_input: exact time phrase (if present)
      - type_input: material-type word normalized (if present)

    Returns a dict with keys:
      'peer_flag', 'sbu_flag', 'online_flag',
      'topic_input', 'time_input', 'type_input'.

    It tries to use the LLM to parse topic/time/type out of the query,
    but first strips any of the three flags so they don’t pollute the topic.
    """
    if verbose:
        logger.info("Splitting query with flags: %s", user_query)

    lower_q = user_query.lower()
    peer_flag = bool(re.search(r"\b(peer[- ]?reviewed|scholarly|academic (sources|research)|research articles?)\b", lower_q))
    sbu_flag = bool(re.search(r"\b(held (by|at)? sbu|(by|at)? our library|in our library|at sbu|stony brook|stony brook university|the university?|do we have|library holdings?)\b", lower_q))
    online_flag = bool(re.search(r"\b(available online|online only|full text online|access online)\b", lower_q))

    # Removing flag‐phrases so subsequent parsing won’t see them
    cleaned_query = user_query
    for phrase in ["peer reviewed", "available online", "sbu held", "held by sbu"]:
        cleaned_query = re.sub(re.escape(phrase), "", cleaned_query, flags=re.IGNORECASE)
    cleaned_query = re.sub(r"\s{2,}", " ", cleaned_query).strip()
    system_prompt = (
        "You are a 'Query Splitter' agent. Given a natural-language search request that may contain:\n"
        "  1. Subject or filter terms (e.g. names, keywords),\n"
        "  2. A time phrase (e.g. 'late 18th century', '1911'),\n"
        "  3. A material-type word (singular or plural: 'book(s)', 'article(s)', 'video(s)', etc.).\n"
        "Parse the input into exactly three fields:\n"
        "- 'topic_input': all subject/filter terms, excluding any time phrase or type word.\n"
        "- 'time_input': the time phrase exactly as in the query (if any), otherwise an empty string.\n"
        "- 'type_input': a list of one or more normalized material types (in lowercase, plural if appropriate), using this controlled vocabulary:\n"
        "[journals, books, articles, images, microform, audios, maps, videos, dissertations, government_documents, reports, book_chapters, scores, archival_material_manuscripts, market_researchs]\n"
        "Return ONLY a valid JSON object with those three keys and no extra text."
    )
    user_prompt = f"User Query: \"{cleaned_query}\""
    raw = ""

    try:
        raw = send_chat_prompt(system_prompt, user_prompt)
        json_start = raw.find("{")
        json_end = raw.rfind("}")
        if json_start != -1 and json_end != -1 and json_end > json_start:
            raw_json = raw[json_start : json_end + 1]
            parsed = json.loads(raw_json)
            topic = parsed.get("topic_input", "").strip()
            time_phrase = parsed.get("time_input", "").strip()
            raw_type = parsed.get("type_input", "")
            if isinstance(raw_type, str):
                type_word = [raw_type.strip()] if raw_type.strip() else []
            elif isinstance(raw_type, list):
                type_word = [t.strip().lower() for t in raw_type if isinstance(t, str)]
            else:
                type_word = []

            if verbose:
                logger.info(
                    "Parsed output: peer=%s, sbu=%s, online=%s, topic='%s', time='%s', type='%s'",
                    peer_flag,
                    sbu_flag,
                    online_flag,
                    topic,
                    time_phrase,
                    type_word,
                )
            return {
                "peer_flag": peer_flag,
                "sbu_flag": sbu_flag,
                "online_flag": online_flag,
                "topic_input": topic,
                "time_input": time_phrase,
                "type_input": type_word,
            }
        else:
            raise ValueError("No valid JSON object found in LLM response.")
    except Exception as e:
        logger.warning("Failed to parse JSON from split_query (error: %s). Falling back to regex.", e)

        time_pattern = r"(?P<time>(?:\d{1,2}(?:st|nd|rd|th)?\s)?\d{4}|late\s\d{1,2}(?:st|nd|rd|th)?\scentury|between\s\d{4}\sand\s\d{4})"
        type_pattern = (
            r"\b(book|books|article|articles|video|videos|audio|audiobook|audiobooks|ebook|ebooks|thesis|theses|report|reports|journal|journals)\b"
        )

        time_match = re.search(time_pattern, cleaned_query, re.IGNORECASE)
        type_match = re.search(type_pattern, cleaned_query, re.IGNORECASE)

        time_phrase = time_match.group("time").strip() if time_match else ""
        type_word = type_match.group(0).strip().lower() if type_match else ""
        if type_word.endswith("s"):
            type_word = type_word[:-1]

        temp = cleaned_query
        if time_phrase:
            temp = temp.replace(time_phrase, "")
        if type_match:
            temp = temp.replace(type_match.group(0), "")
        topic = re.sub(r"\s{2,}", " ", temp).strip()

        if verbose:
            logger.info(
                "Fallback parsing: peer=%s, sbu=%s, online=%s, topic='%s', time='%s', type='%s'",
                peer_flag,
                sbu_flag,
                online_flag,
                topic,
                time_phrase,
                type_word,
            )
        return {
            "peer_flag": peer_flag,
            "sbu_flag": sbu_flag,
            "online_flag": online_flag,
            "topic_input": topic,
            "time_input": time_phrase,
            "type_input": type_word,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Agent 2: Year Range Extractor
# ─────────────────────────────────────────────────────────────────────────────
def extract_year_range(time_input: str, verbose: bool = False) -> Tuple[Optional[int], Optional[int]]:
    """
    Converts a time phrase into a standardized year range tuple (start_year, end_year)
    using an LLM to interpret all inputs, including named events, centuries, and eras.

    Returns (None, None) if no valid interpretation is possible.
    """
    if not time_input:
        if verbose:
            logger.info("No time input provided, returning (None, None)")
        return (None, None)

    if verbose:
        logger.info("Extracting year range via LLM from: %s", time_input)

    try:
        system_prompt = (
            "You are a strict date-range normalizer. "
            "Given a time phrase, return only a clean, comma-separated year range in the format: YYYY,YYYY. "
            "If the phrase only contains one year, repeat it. "
            "You MUST interpret named historical events, cultural eras, and ambiguous periods "
            "(e.g., 'World War II', 'Jazz Age', 'Cold War', 'late 18th century') into concrete year ranges. "
            "If failed, return NONE,NONE. "
            "Do NOT explain, do NOT repeat the phrase. Only output the year range."
        )
        user_prompt = f"Normalize this time phrase to a year range: \"{time_input}\"\nReturn only format: YYYY,YYYY or NONE,NONE"

        raw = send_chat_prompt(system_prompt, user_prompt).strip()

        if verbose:
            logger.info("LLM normalized time phrase output: %s", raw)

        if "NONE,NONE" in raw.upper():
            if verbose:
                logger.info("LLM determined no valid years in phrase")
            return (None, None)

        match = re.match(r"(\d{4})\s*,\s*(\d{4})", raw)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            if verbose:
                logger.info("LLM returned year range: %d-%d", start, end)
            return (start, end)
        else:
            raise ValueError("Invalid format from LLM")
    except Exception as e:
        logger.warning("Failed to normalize time phrase '%s'. Error: %s", time_input, str(e))
        return (None, None)


def format_creation_date(start: int, end: int) -> str:
    """
    Format a creation date range for the Discovery URL.
    Encoded as: 1560%7C,%7C2025
    """
    return f"{start}%7C,%7C{end}"


# ─────────────────────────────────────────────────────────────────────────────
# Agent 3: Boolean Expression Builder
# ─────────────────────────────────────────────────────────────────────────────
def build_boolean_string(topic_input: str, verbose: bool = False) -> str:
    """
    Converts topic_input (subject/filter terms) into a valid Boolean expression
    using AND, OR, NOT, and quoted phrases. Excludes any time or type words.

    Returns an empty string if topic_input is empty.
    """
    if not topic_input:
        return ""
    if verbose:
        logger.info("Building boolean string from: %s", topic_input)

    system_prompt = (
        "You are an assistant that creates high-quality Boolean search queries for academic or library catalogs like Ex Libris PRIMO.\n"
        "Given a topic, build a search string using Ex Libris PRIMO syntax.\n"
        "Expand only with close synonyms or relevant alternatives — avoid general academic or encyclopedic expansions.\n"
        "Avoid vague, overly broad, or loosely related terms (e.g. 'gastronomy', 'food preparation').\n"
        "Keep the Boolean string focused, specific, and helpful for finding directly related materials.\n"
        "Do not include any explanation."
    )
    user_prompt = f"Topic Input: \"{topic_input}\"\nBuild a comprehensive Boolean search string"
    raw = send_chat_prompt(system_prompt, user_prompt).strip()

    boolean_expr = raw.strip()
    if not re.match(r'^[\"(]', boolean_expr):
        logger.warning("Boolean output may be malformed: %s", boolean_expr)

    if verbose:
        logger.info("Boolean expression: %s", boolean_expr)
    return boolean_expr


# ─────────────────────────────────────────────────────────────────────────────
# Agent 4: Material-Type Normalizer
# ─────────────────────────────────────────────────────────────────────────────

def extract_material_types(type_input: str, use_llm_fallback: bool = True, verbose: bool = False) -> List[str]:

    
    """
    Normalizes material type phrases to their Discovery-compatible resource types.
    Returns a list (deduplicated). Uses LLM fallback if enabled.
    """
    if not type_input:
        return []

    # Tokenize by common conjunctions and separators
    if isinstance(type_input, list):
        # Already split — just flatten and clean it
        tokens = [t.lower().strip() for t in type_input if isinstance(t, str)]
    elif isinstance(type_input, str):
        # Tokenize by conjunctions
        tokens = re.split(r"[,\s]*\b(?:and|or|,)\b[\s]*", type_input.lower())
        tokens = [t.strip() for t in tokens if t.strip()]
    else:
        tokens = []

    normalized_types = []

    # Controlled vocabulary map
    MATERIAL_TYPE_MAP = {
        # Books
        "book": "books", "books": "books", "ebook": "books", "ebooks": "books",
        "monograph": "books", "monographs": "books",

        # Articles & Journals
        "article": "articles", "articles": "articles",
        "journal": "journals", "journals": "journals",
        "paper": "articles", "papers": "articles", "research paper": "articles",

        # Videos
        "video": "videos", "videos": "videos", "film": "videos", "films": "videos", "movie": "videos", "movies": "videos",

        # Audio
        "audio": "audios", "audios": "audios", "audiobook": "audios", "audiobooks": "audios",

        # Other types
        "image": "images", "images": "images",
        "map": "maps", "maps": "maps",
        "microform": "microform",
        "dissertation": "dissertations", "dissertations": "dissertations", "thesis": "dissertations", "theses": "dissertations",
        "government document": "government_documents", "gov doc": "government_documents", "gov docs": "government_documents",
        "government_documents": "government_documents",
        "report": "reports", "reports": "reports",
        "book chapter": "book_chapters", "book chapters": "book_chapters",
        "score": "scores", "scores": "scores",
        "archival material": "archival_material_manuscripts",
        "manuscript": "archival_material_manuscripts", "manuscripts": "archival_material_manuscripts",
        "market research": "market_researchs"
    }

    for token in tokens:
        token = token.strip()
        if not token:
            continue

        # Try map lookup
        if token in MATERIAL_TYPE_MAP:
            normalized_types.append(MATERIAL_TYPE_MAP[token])
        elif use_llm_fallback:
            # Try LLM-based normalization if unknown
            system_prompt = (
                "You are a 'Material-Type Extractor' agent. Given a word or short phrase indicating a material type "
                "(possibly plural, e.g. 'books', 'articles', 'videos'), normalize it to a recognized academic type:\n"
                "books, journals, articles, images, microform, audios, maps, videos, dissertations, government_documents, "
                "reports, book_chapters, scores, archival_material_manuscripts, or market_researchs.\n"
                "Return only the best-matching singular or plural label from that list. Return an empty string if none match."
            )
            user_prompt = f"Material type: \"{token}\""
            try:
                raw = send_chat_prompt(system_prompt, user_prompt).strip().lower()
                mapped = MATERIAL_TYPE_MAP.get(raw, raw)  # Try to map result if it's a synonym
                if mapped in MATERIAL_TYPE_MAP.values():
                    normalized_types.append(mapped)
                elif verbose:
                    logger.warning("LLM returned unrecognized material type: %s", raw)
            except Exception as e:
                if verbose:
                    logger.warning("LLM fallback failed on '%s': %s", token, e)

        elif verbose:
            logger.warning("Unrecognized material type (no LLM fallback): %s", token)

    # Remove duplicates
    return list(set(normalized_types))

# ─────────────────────────────────────────────────────────────────────────────
# URL Construction: Discovery Search
# ─────────────────────────────────────────────────────────────────────────────
def construct_url(
    boolean: str,
    rtype: Optional[List[str]] = None,
    peer_flag: bool = False,
    sbu_flag: bool = False,
    online_flag: bool = False,
    creationdate: Optional[str] = None,
) -> str:
    """
    Build a Stony Brook Library Discovery search URL using the boolean query
    and optional filters (resource types, peer reviewed, SBU-held, online).
    Properly URL-encodes the boolean expression.
    """
    _prefix = "https://search.library.stonybrook.edu/discovery/search?vid=01SUNY_STB:01SUNY_STB"
    _query_prefix = "&query="
    _query_field = "any,"
    _filter = "contains,"
    # URL-encode the boolean expression for special characters
    encoded_boolean = urllib.parse.quote_plus(boolean)

    url = f"{_prefix}{_query_prefix}{_query_field}{_filter}{encoded_boolean}"

    if peer_flag:
        url += "&mfacet=tlevel,include,peer_reviewed,1,lk"
    if sbu_flag:
        url += "&mfacet=tlevel,include,available_p,1,lk"
    if online_flag:
        url += "&mfacet=tlevel,include,online_resources,1,lk"
    if creationdate:
        url += f"&mfacet=searchcreationdate,include,{creationdate},1,lk"

    valid_rtypes = [
        "journals", "books", "articles", "images", "microform", "reviews", "reports", 
        "book_chapters", "scores", "archival_material_manuscripts", "market_researchs",
        "audios", "maps", "videos", "dissertations", "government_documents"
    ]
    for r in rtype or []:
        rtype_match = r.strip().lstrip("$")
        if rtype_match in valid_rtypes:
            url += f"&mfacet=rtype,include,{rtype_match},1,lk"
        elif rtype_match.lower() != "none" and DEBUG:
            print(f"Warning: Unexpected resource type '{rtype_match}'")

    url += "&search_scope=EverythingNZBooks"
    return url


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Process Query for URL Mode (Revised)
# ─────────────────────────────────────────────────────────────────────────────
def process_query(user_query: str, verbose: bool = False) -> Tuple[bool, bool, bool, List[str], str, Optional[str]]:
    """
    Process the user_query to extract flags (handled at splitter), boolean query, and creation date:
      - peer_reviewed, sbuheld, online_resources now provided by split_query
      - rtype_list: list containing plural material type if present
      - boolean_query: constructed boolean expression if topic_input present
      - creationdate: encoded date range if time_input present and valid

    Returns a tuple: (peer_reviewed, sbuheld, online_resources, rtype_list, boolean_query, creationdate)
    """
    # Step 1: Split query via Agent 1 (now includes flags)
    parts = split_query(user_query, verbose=verbose)
    peer_flag = parts.get("peer_flag", False)
    sbu_flag = parts.get("sbu_flag", False)
    online_flag = parts.get("online_flag", False)
    time_input = parts.get("time_input", "")
    type_input = parts.get("type_input", "")
    topic_input = parts.get("topic_input", "")

    # Step 2: If time_input is present, call Agent 2 to get year range and format creation date
    creationdate = None
    if time_input:
        start_year, end_year = extract_year_range(time_input, verbose=verbose)
        if start_year is not None and end_year is not None:
            creationdate = format_creation_date(start_year, end_year)

    # Step 3: If type_input is present, call Agent 4 to normalize material type
    rtype_list: List[str] = []
    if type_input:
        # Extract material types (plural form, normalized)
        rtype_list = extract_material_types(type_input, verbose=verbose)

        # Ensure all material types are pluralized properly
        rtype_list = [rtype + "s" if not rtype.endswith("s") else rtype for rtype in rtype_list]

    # Step 4: If topic_input is present, call Agent 3 to build boolean expression
    boolean_expr = ""
    if topic_input:
        boolean_expr = build_boolean_string(topic_input, verbose=verbose)

    return peer_flag, sbu_flag, online_flag, rtype_list, boolean_expr, creationdate


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator: Process Full Query (JSON) and URL Mode
# ─────────────────────────────────────────────────────────────────────────────
def split_full_query(user_query: str, verbose: bool = False) -> Dict[str, Optional[str]]:
    """
    End-to-end pipeline for JSON output:
      1. split_query → peer_flag, sbu_flag, online_flag, topic_input, time_input, type_input
      2. extract_year_range(time_input) → year_range tuple
      3. build_boolean_string(topic_input) → boolean_query
      4. extract_material_type(type_input) → mtype

    Returns a dict with keys:
      - 'original_query': original user query
      - 'boolean_query': constructed boolean expression
      - 'year_range': normalized year range as string (YYYY-YYYY) or empty string
      - 'type': normalized material type
      - 'split_parts': dict of intermediate outputs (including flags)
    """
    if verbose:
        logger.info("Starting full query split for: %s", user_query)

    parts = split_query(user_query, verbose=verbose)
    topic_input = parts.get("topic_input", "")
    time_input = parts.get("time_input", "")
    type_input = parts.get("type_input", "")

    start_year, end_year = extract_year_range(time_input, verbose=verbose)

    # Format year range as string for backward compatibility
    if start_year is not None and end_year is not None:
        if start_year == end_year:
            year_range = str(start_year)
        else:
            year_range = f"{start_year}-{end_year}"
    else:
        year_range = ""

    boolean_query = build_boolean_string(topic_input, verbose=verbose)
    mtype = extract_material_types(type_input, verbose=verbose)

    result = {
        "original_query": user_query.strip(),
        "boolean_query": boolean_query,
        "year_range": year_range,
        "type": mtype,
        "split_parts": parts,
    }
    if verbose:
        logger.info("Full split result: %s", json.dumps(result, ensure_ascii=False, indent=2))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# JSON Output Helper
# ─────────────────────────────────────────────────────────────────────────────
def print_json(data: Dict) -> None:
    """
    Pretty-print a dictionary as compact JSON to standard output.
    """
    sys.stdout.write(json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    sys.stdout.write("\n")
    sys.stdout.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Command-Line Interface
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """
    Set up and parse CLI arguments.

    Returns:
        Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Enhanced LLM-based query agents: split|extract_year|build_bool|extract_type|split_full|url"
    )
    parser.add_argument(
        "mode",
        choices=["split", "extract_year", "build_bool", "extract_type", "split_full", "url"],
        help="Which agent or function to run",
    )
    parser.add_argument(
        "query",
        nargs=argparse.REMAINDER,
        help="The user query to process (enclose in quotes if multi-word)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def get_retrying_session(
    total: int = MAX_RETRIES,
    backoff_factor: float = RETRY_BACKOFF,
    status_forcelist: List[int] = [429, 500, 502, 503, 504],
) -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=total,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["POST", "GET"],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def main() -> None:
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled")

    if not args.query:
        logger.error("No query provided. Specify the query after the mode.")
        sys.exit(1)

    raw_query = " ".join(args.query).strip()
    if not raw_query:
        logger.error("Empty query string.")
        sys.exit(1)

    try:
        if args.mode == "split":
            result = split_query(raw_query, verbose=args.verbose)
            print_json(result)
        elif args.mode == "extract_year":
            start_year, end_year = extract_year_range(raw_query, verbose=args.verbose)
            if start_year is not None and end_year is not None:
                if start_year == end_year:
                    year_range = str(start_year)
                else:
                    year_range = f"{start_year}-{end_year}"
            else:
                year_range = ""
            print_json({"year_range": year_range})
        elif args.mode == "build_bool":
            boolean_str = build_boolean_string(raw_query, verbose=args.verbose)
            print_json({"boolean_query": boolean_str})
        elif args.mode == "extract_type":
            material = extract_material_types(raw_query, verbose=args.verbose)
            print_json({"type": material})
        elif args.mode == "split_full":
            result = split_full_query(raw_query, verbose=args.verbose)
            print_json(result)
        elif args.mode == "url":
            # process_query to follow agentic workflow
            peer_flag, sbu_flag, online_flag, rtype_list, boolean_expr, creationdate = process_query(
                raw_query, verbose=args.verbose
            )
            redirect_url = construct_url(
                boolean_expr,
                rtype_list,
                peer_flag,
                sbu_flag,
                online_flag,
                creationdate=creationdate
            )
            if DEBUG:
                print("Debug Mode Enabled")
                print("Processed Query Values:")
                print(f"Peer Reviewed: {peer_flag}")
                print(f"SBU Held: {sbu_flag}")
                print(f"Online Resources: {online_flag}")
                print(f"Resource Type(s): {rtype_list}")
                print(f"Boolean Query: {boolean_expr}")
                print(f"Creation Date: {creationdate}")
                print(f"Constructed URL: {redirect_url}")
            else:
                print(redirect_url.strip())
        else:
            logger.error("Unknown mode: %s", args.mode)
            sys.exit(1)
    except Exception as e:
        logger.exception("Error occurred during processing: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        sys.exit(1)