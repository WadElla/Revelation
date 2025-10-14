# query_iot_rag_custom_improve.py
from typing import Optional, List, Dict, Tuple
from agno.tools import tool
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
# from langchain_community.llms import Ollama
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from get_embedding_function import get_embedding_function
import os, hashlib, json, re

# ---------- system prompt ----------
SYSTEM_PROMPT = """
You are **Revelation**, an expert assistant for *interpreting* network traffic captured in PCAPs.  
Your task is to analyze the processed capture artifacts and provide accurate, evidence-grounded explanations 
of what occurred at the network, transport, and application layers.  

Use only the data explicitly contained in the artifacts—never invent or assume unseen details.

---

📘 **Evidence You May Rely On**
- **Zeek JSON logs**: reconstructed protocol events (DNS, HTTP, MQTT, Modbus, TLS, etc.).  
- **Flow summaries**: bidirectional sessions with duration, packet counts, byte volumes, TCP flags, MAC vendors, and public-IP reputation.  
- **Packet-layer JSON**: hierarchical header fields across Ethernet, IP, TCP/UDP, and application layers.  
- **Anomaly interpretation report**: fine-tuned BERT output describing benign or malicious behaviors with supporting metadata.  
- **Threat-intel enrichments**: VirusTotal and InternetDB results for public IPs (if available).  

Treat all retrieved artifact content as authoritative.  Do **not** fabricate values or behaviors not present in these sources.

---

🔒 **Security & Privacy**
- Do **not** transmit or infer beyond the provided artifacts.
- Use threat-intel only when it appears in the enrichment sections; do **not** guess or “fill in” missing intel.
- If reputation is needed, rely only on enrichment/flow annotations already provided—do **not** estimate.

🧠 **Interpretation Focus**
- Explain *what happened*, *why it matters*, and *what an operator might do next*.  
- Distinguish clearly between normal operations and adversarial indicators.  
- Relate behavior to protocol semantics, endpoint roles, and potential risk or operational significance.  
- Integrate anomaly findings, endpoint reputations, and observed services into coherent insight.

---

🔎 **Application & Protocol Hints (use when present)**
Use any observable evidence—**not limited to IoT protocols**—to interpret traffic behavior:

- **Web / Cloud Services** – HTTP(S), WebSocket, QUIC, REST, or API calls; note methods, paths, response codes, latency patterns.  
- **Industrial / IoT** – MQTT topics, Modbus function codes, BACnet, CoAP, LwM2M; detect command repetition, abnormal bursts.  
- **Infrastructure** – DNS/mDNS queries, DHCP exchanges, NTP synchronization, TLS handshakes, certificate reuse.  
- **Email / Messaging** – SMTP, IMAP, POP3, XMPP, or proprietary message relays; identify suspicious relay loops or spikes.  
- **File Transfer / Storage** – FTP, SFTP, SMB, NFS, or HTTP uploads; flag large or repetitive transfers.  
- **Remote Access / Management** – SSH, RDP, Telnet, SNMP, or VPN protocols; identify login attempts, session counts, resets.  
- **Streaming / Multimedia** – RTP, RTSP, SIP, WebRTC; note jitter, packet loss, or codec negotiation.  
- **Other Custom or Unknown Services** – infer behavior from port numbers, payload sizes, and timing consistency.  

Apply these hints *only when supported by evidence in the artifacts.*

⚙️ **Response Format**
- Use clear markdown with short paragraphs or bullet lists.  
- Structure responses as:
  1. **Traffic Evidence** – concrete observations from logs or summaries.  
  2. **Interpretation** – explain what those observations mean.  
  3. **Operator Insight** – suggest follow-up checks or implications.  
- If no matching evidence exists, write: *“No relevant evidence found in the current capture.”*  
- Keep tone professional, technical, and concise. 

---

🚫 **Do Not**
- Do not speculate about unseen payloads, hidden devices, or intent.  
- Do not conflate general networking knowledge with capture-specific findings.  
- Do not override artifact content with assumptions.  

---

🎯 **Goal**
Produce **concise, technically precise, and evidence-grounded** interpretations of network behavior—  spanning IoT, web, industrial, and cloud contexts—to assist analysts in understanding both normal 
and malicious activity within the processed capture.
""".strip()

# ---------- helpers ----------
def _doc_key(d: Document) -> Tuple[str, str]:
    mid = (d.metadata or {}).get("id")
    return ("id", str(mid)) if mid else ("md5", hashlib.md5(d.page_content.encode()).hexdigest())

def _minmax(xs: List[float]) -> List[float]:
    if not xs: return []
    lo, hi = min(xs), max(xs)
    if hi <= lo: return [0.5 for _ in xs]
    return [(x - lo) / (hi - lo) for x in xs]

def _extract_user_query_if_wrapped(user_query: str) -> str:
    txt = (user_query or "").strip()
    if "user_query" in txt and "{" in txt and "}" in txt:
        try:
            m = re.search(r"\{.*\}", txt, re.S)
            if m:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict) and "user_query" in obj:
                    return obj["user_query"]
        except Exception:
            pass
    return user_query

def keyword_filter_docs(docs: List[Document], query: str, top_n: int = 5) -> List[Document]:
    kws = query.lower().split()
    scored = []
    for d in docs:
        t = d.page_content.lower()
        s = sum(1 for kw in kws if kw in t)
        if s > 0: scored.append((s, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_n]]

def build_bm25_index(docs: List[Document]) -> BM25Okapi:
    corpus = [d.page_content.lower().split() for d in docs]
    return BM25Okapi(corpus)

def hybrid_search(query: str, all_docs: List[Document], bm25: BM25Okapi, db: Chroma,
                  k_sparse: int, k_dense: int, alpha: float) -> List[Document]:
    # sparse
    tokens = query.lower().split()
    bm25_scores = bm25.get_scores(tokens)
    top_sparse_idx = sorted(range(len(all_docs)), key=lambda i: bm25_scores[i], reverse=True)[:k_sparse]

    # dense (Chroma returns distance -> convert to similarity safely)
    dense_hits = db.similarity_search_with_score(query, k=k_dense)  # [(Document, distance)]
    key_to_idx = {_doc_key(d): i for i, d in enumerate(all_docs)}
    dense_pairs = []
    for d, dist in dense_hits:
        key = _doc_key(d)
        if key not in key_to_idx:
            continue
        try:
            dist = float(dist)
        except Exception:
            dist = 1.0
        if dist < 0.0:
            dist = 0.0
        sim = 1.0 / (1.0 + dist)  # lower distance -> higher similarity
        dense_pairs.append((key_to_idx[key], sim))

    # merge candidates
    merged: Dict[int, Dict[str, float]] = {}
    for i in top_sparse_idx:
        merged[i] = {"bm25": float(bm25_scores[i]), "vec": None}
    for i, sim in dense_pairs:
        merged.setdefault(i, {"bm25": 0.0, "vec": None})
        merged[i]["vec"] = sim if merged[i]["vec"] is None else max(merged[i]["vec"], sim)

    # --------- normalize (fixed) ---------
    idxs = list(merged.keys())
    bm25_list = [merged[i]["bm25"] for i in idxs]
    vec_raw   = [merged[i]["vec"]  for i in idxs]

    # Replace None with -inf, then min–max scale (guard all-None case)
    vec_list = [v if v is not None else float("-inf") for v in vec_raw]
    finite_vecs = [v for v in vec_list if v != float("-inf")]
    vec_floor = min(finite_vecs) if finite_vecs else 0.0
    vec_list = [vec_floor if v == float("-inf") else v for v in vec_list]

    bm25_n = _minmax(bm25_list)
    vec_n  = _minmax(vec_list)

    blended = {idxs[i]: (alpha * vec_n[i] + (1 - alpha) * bm25_n[i]) for i in range(len(idxs))}
    ranked = sorted(blended.items(), key=lambda kv: kv[1], reverse=True)
    return [all_docs[i] for i, _ in ranked]

def deduplicate_docs(docs: List[Document]) -> List[Document]:
    seen = set(); uniq = []
    for d in docs:
        k = _doc_key(d)
        if k in seen: continue
        seen.add(k); uniq.append(d)
    return uniq

def rerank_docs(question: str, docs: List[Document], reranker: CrossEncoder, top_k: int) -> List[Document]:
    pairs = [(question, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = [doc for _, doc in sorted(zip(scores, docs), key=lambda t: t[0], reverse=True)]
    return ranked[:top_k]

# ---------- session path helpers ----------
HERE = os.path.dirname(os.path.abspath(__file__))
SESSIONS_ROOT = os.path.join(HERE, "chroma_sessions")
LATEST_POINTER = os.path.join(SESSIONS_ROOT, "latest_session.txt")
SESSION_DIR_RE = re.compile(r"(session_[A-Za-z0-9]{6,})", re.I)

# --- high-level summaries ---
FLOW_SUCCINCT = os.path.join(HERE, "flow_summaries", "summary_succinct.txt")
ANOMALY_SUCCINCT = os.path.join(HERE, "enhancer", "enhanced_bert_report_succinct.txt")

def _read_text_if_exists(path: str) -> Optional[str]:
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
    except Exception:
        pass
    return None



def _normalize_session_path(p: str) -> str:
    p = (p or "").strip().strip('"').strip("'")
    return os.path.realpath(os.path.abspath(os.path.expanduser(p)))

def _extract_session_dirname(p: str) -> Optional[str]:
    m = SESSION_DIR_RE.search(p or "")
    return m.group(1) if m else None

def _resolve_session_dir(chroma_path: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (resolved_path, note).
    note is one of: None, 'rescued_by_session_id', 'fallback_to_latest',
                    'missing_latest_session_pointer', 'bad_path_and_missing_latest'
    """
    if chroma_path:
        p = _normalize_session_path(chroma_path)
        if os.path.isdir(p):
            return p, None
        # rescue by session id
        sess = _extract_session_dirname(p)
        if sess:
            candidate = _normalize_session_path(os.path.join(SESSIONS_ROOT, sess))
            if os.path.isdir(candidate):
                return candidate, "rescued_by_session_id"
        # fallback to latest
        if os.path.exists(LATEST_POINTER):
            latest = _normalize_session_path(open(LATEST_POINTER).read().strip())
            if os.path.isdir(latest):
                return latest, "fallback_to_latest"
        return None, "bad_path_and_missing_latest"
    else:
        if os.path.exists(LATEST_POINTER):
            latest = _normalize_session_path(open(LATEST_POINTER).read().strip())
            if os.path.isdir(latest):
                return latest, None
        return None, "missing_latest_session_pointer"

@tool(
    name="query_iot_traffic_rag",
    description="Hybrid retrieval (BM25 + dense) + keyword fallback + reranking over the latest IoT session.",
    show_result=False,                   # don't dump raw JSON into the conversation
    requires_user_input=False,           # do NOT block for interactive input
    user_input_fields=[],                # nothing to collect interactively
    requires_confirmation=False,
    stop_after_tool_call=False,          # agent continues to finalize
)
def query_iot_traffic_rag(user_query: str, chroma_path: Optional[str] = None) -> str:

    user_query = _extract_user_query_if_wrapped(user_query)
    print(f"🔎 User query: {user_query}")

    # Tunables
    k_sparse, k_dense, alpha = 80, 70, 0.7            #20, 30, 0.7
    top_k_hybrid, k_keyword, top_k = 60, 5, 40       #15, 3, 5

    # --- Resolve session directory (no base-search; rescue by session_id; fallback to latest) ---
    resolved_path, note = _resolve_session_dir(chroma_path)
    if not resolved_path:
        package = {
            "type": "rag_context",
            "system_prompt": SYSTEM_PROMPT,
            "question": user_query,
            "chroma_path": None,
            "top_k": [],
            "note": note or "missing_latest_session_pointer",
        }
        return json.dumps(package, ensure_ascii=False)

    chroma_path = resolved_path
    if note:
        print(f"ℹ️ Session path note: {note}")
    print(f"🔗 Loading Chroma DB from: {chroma_path}")
    embedding_fn = get_embedding_function()

    # --- Minimal collection fallback: default -> 'langchain' ---
    def _open_and_get_docs(collection_name: Optional[str] = None):
        db = Chroma(
            persist_directory=chroma_path,
            embedding_function=embedding_fn,
            **({"collection_name": collection_name} if collection_name else {})
        )
        data = db.get(include=["documents", "metadatas"])
        docs = [Document(page_content=t, metadata=m)
                for t, m in zip(data.get("documents", []), data.get("metadatas", []))]
        return db, docs

    db, all_docs = _open_and_get_docs(None)  # default collection
    if not all_docs:
        db, all_docs = _open_and_get_docs("langchain")

    print(f"📚 Loaded {len(all_docs)} documents")
    if not all_docs:
        package = {
            "type": "rag_context",
            "system_prompt": SYSTEM_PROMPT,
            "question": user_query,
            "chroma_path": chroma_path,
            "top_k": [],
            "note": "no_documents_in_db_or_collection_mismatch",
        }
        return json.dumps(package, ensure_ascii=False)

    bm25 = build_bm25_index(all_docs)
    hybrid_docs = hybrid_search(user_query, all_docs, bm25, db, k_sparse, k_dense, alpha)
    hybrid_top = hybrid_docs[:top_k_hybrid]
    keyword_hits = keyword_filter_docs(all_docs, user_query, top_n=k_keyword)
    candidates_docs = deduplicate_docs(hybrid_top + keyword_hits)

    print(f"🔀 Deduplicated candidate set: {len(candidates_docs)} documents")
    if not candidates_docs:
        package = {
            "type": "rag_context",
            "system_prompt": SYSTEM_PROMPT,
            "question": user_query,
            "chroma_path": chroma_path,
            "top_k": [],
            "note": "no_relevant_documents"
        }
        return json.dumps(package, ensure_ascii=False)

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    top_docs = rerank_docs(user_query, candidates_docs, reranker, top_k=top_k)

    # Build top_k as a list of chunks with text only
    top_items = [{"text": d.page_content.strip()} for d in top_docs]

    # Read high-level summaries (kept separate from top_k)
    flow_slim = _read_text_if_exists(FLOW_SUCCINCT)
    anomaly_slim = _read_text_if_exists(ANOMALY_SUCCINCT)

    # Return the package the agent expects
    package = {
        "type": "rag_context",
        "system_prompt": SYSTEM_PROMPT,
        "question": user_query,          # (possibly normalized) query used
        "chroma_path": chroma_path,      # for traceability / optional reuse
        "top_k": top_items,               # list[{"text": ...}]'
        "chunk_count": len(top_items)
    }

    overview = {}
    if flow_slim:
        overview["flow_summary"] = flow_slim
    if anomaly_slim:
        overview["anomaly_summary"] = anomaly_slim
    if overview:
        package["overview"] = overview


    out = json.dumps(package, ensure_ascii=False)
    print(f"[RAG] Returning rag_context with {len(top_items)} chunks.")
    return out