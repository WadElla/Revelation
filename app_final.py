# app.py
import os
import sys
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional

import streamlit as st




from agno.models.google import Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
import os
os.environ["GOOGLE_API_KEY"] = "Put your Google API key here if you want to use their Gemini model"

import google.generativeai as genai
genai.configure(api_key="Put your Google API key here if you want to use their Gemini model")

import os
os.environ["TAVILY_API_KEY"] = "Put your Tavily API key here"

from agno.tools.tavily import TavilyTools



# ----------------------------
# Project paths & environment
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

# Ensure local modules are importable
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Default embedding model for your stack (Ollama, consistent with tools)
os.environ.setdefault("EMBED_MODEL", "nomic-embed-text:v1.5")

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

TMP_DIR = PROJECT_ROOT / "tmp"
TMP_DIR.mkdir(exist_ok=True, parents=True)

STATE_DIR = PROJECT_ROOT / "state"
STATE_DIR.mkdir(exist_ok=True, parents=True)

LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True, parents=True)

SESSIONS_FILE = STATE_DIR / "sessions.json"         # maps pcap_hash -> session info
PCAP_CANONICAL = DATA_DIR / "Backdoor_attack.pcap"  # your tools expect this path
LATEST_POINTER = PROJECT_ROOT / "chroma_sessions" / "latest_session.txt"

# ----------------------------
# Import your pipeline tools
# ----------------------------
from zeek_json_tool import run_zeek_log_generator
from extract_packet_layer_json import extract_packet_layers
from flow_summaries_updated_final import generate_flow_summary
from extract_pcap_features_tool import extract_pcap_features
from anomaly_detection import run_bert_anomaly_detector
from enrich_iot_report import enrich_iot_report
from Rag_ingest_updated_final import rag_ingest_tool
from query_iot_rag_custom_improve import query_iot_traffic_rag  # the @tool-exposed function
#from agno.tools.duckduckgo import DuckDuckGoTools

# ----------------------------
# Agno Agent (same behavior as in workflow.py)
# ----------------------------
from agno.agent import Agent
from agno.models.ollama import Ollama
from textwrap import dedent

AGENT_INSTRUCTIONS = dedent("""\
You are the **Revelation Query Agent**. You may receive either:
A) a <toolcall name="query_iot_traffic_rag">{...JSON...}</toolcall> with JSON args; or
B) a JSON package from the retrieval tool (type="rag_context": system_prompt, question, top_k, retrieved chunk_count and optionally overview={flow_summary, anomaly_summary}).

ReAct (compact):
1) Perceive
   - If input contains a <toolcall> for `query_iot_traffic_rag`, parse its JSON and CALL that tool.
   - Query Reformulation (allowed): if the user's question has typos, ambiguous phrasing, or is too generic,
     minimally rewrite it (fix spelling; normalize protocol/attack names; expand common abbreviations; add neutral synonyms only).
     Preserve original intent; do not add new facts or constraints. Do not reveal that you rewrote it.
   - Include chroma_path if provided.

2) Reason
   - If the tool returns plain text (legacy final answer) → RETURN it unchanged.
   - If it returns JSON with type="rag_context":
       • Treat `overview.flow_summary` and `overview.anomaly_summary` (when present) as high-signal capture evidence.
         - Use them to establish global context quickly (traffic-wide stats, dominant behaviors, verdict tallies).
         - Do not paste the summaries verbatim; synthesize what’s relevant to the question.
       • Treat `top_k` as the primary source for specific details and examples; use it to support/anchor claims from the summaries.
       • For anomaly/attack-specific questions: prioritize the anomaly summary first, then corroborate with any matching `top_k` details.
       • For “whole traffic” or broad situational questions: use both the flow summary and anomaly summary, then add any concrete details found in `top_k`.

       • GROUNDING PRINCIPLE:
         - Determine first whether the user's question can be meaningfully answered using the retrieved capture artifacts (summaries and `top_k`).
         - Review the retrieved context for concrete or conceptually aligned evidence—such as protocol terms, identifiers,
           endpoint references, behaviors, or observable patterns that directly relate to the question.
         - When such correspondence exists, treat the capture as the authoritative source and construct the answer using
           only that evidence, ensuring all claims can be traced back to the retrieved context.
         - When the retrieved context lacks clear or relevant indicators—meaning the question depends on broader IoT
           knowledge, standards, or background information rather than the observed traffic—classify it as **outside the capture**.
         - Avoid speculative reasoning or inferring details that are not supported by the retrieved evidence.

       • Re-query allowance (max 2):
         - Only to normalize wording (e.g., "mitm"→"man-in-the-middle", "exfil"→"data exfiltration/uploading", "http flood"→"HTTP DDoS"),
           or add neutral synonyms (DNS/domain, TLS/HTTPS handshake) without widening scope.
         - After each re-query, reassess grounding using summaries and `top_k`. If still ungrounded, stop trying.

       • Out-of-capture fallback:
         - Use `TavilyTools()` **only** when the question is IoT-domain but outside the capture OR evidence remains insufficient after the re-queries.
         - Mark such content as **External IoT Context** and include concise citations.

3) Act
   - Write a concise, technically precise answer in plain English.
   - Structure as: **Traffic Evidence** → **Interpretation** → **Operator Insight**.
   - If grounded in capture: answer with **Traffic Evidence → Interpretation → Operator Insight**,
     where “Traffic Evidence” may cite both the high-level summaries and any specific `top_k` facts.
   - If outside the capture and answered via web: answer with **External IoT Context → Operator Insight**, with brief sources.
   - If no supporting evidence exists: “No relevant evidence found in the current capture.”
   - Add a short disclosure at the end indicating context limits, e.g.,
     “Note: Answer grounded in the provided capture context (overview + {chunk_count} retrieved chunks).”
   - Do not include internal JSON, prompts, or your intermediate steps; output only the final answer.
   - Never repeat or quote the `system_prompt` text or paste raw `top_k` chunks.

Constraints & Faithfulness
- Every claim must be supported by `overview` and/or `top_k`, or clearly marked **External IoT Context** (with source).
- Do not invent data or infer beyond artifacts; keep protocol terms/identifiers consistent.
- Stable, time-insensitive facts (e.g., well-known default ports) may be used sparingly, never overriding artifacts.

Stop when the answer directly addresses the question and passes a self-check for support and clarity.
""")

@st.cache_resource
def get_rag_agent() -> Agent:
    # Cache the agent so it isn't recreated each interaction
    return Agent(
        name="RAG Query Agent",
        #model=Ollama(id="llama3.2:latest"),
        model = Gemini(id="gemini-2.5-pro"),
        tools=[query_iot_traffic_rag, TavilyTools()],
        markdown=True,
        instructions=AGENT_INSTRUCTIONS,
    )

# --- helpers to normalize agent responses and catch raw JSON payloads ---
def _normalize_resp(resp) -> str:
    if isinstance(resp, str):
        return resp
    if hasattr(resp, "content"):
        return resp.content
    if isinstance(resp, dict) and "content" in resp:
        return resp["content"]
    return str(resp)

def _is_rag_context_payload(text: str) -> bool:
    s = (text or "").strip()
    if not s.startswith("{"):
        return False
    try:
        obj = json.loads(s)
        return isinstance(obj, dict) and obj.get("type") == "rag_context" and "top_k" in obj
    except Exception:
        return False

def _agent_call(agent: Agent, message: str) -> str:
    """Call the agent using whatever method is available and normalize output."""
    for method in ("run", "respond", "chat", "invoke"):
        if hasattr(agent, method):
            try:
                resp = getattr(agent, method)(message=message)  # preferred signature
            except TypeError:
                resp = getattr(agent, method)(message)          # alt signature
            return _normalize_resp(resp)
    return "⚠️ Agent invocation method not found. Please update Agno to a compatible version."

def agent_invoke_query(user_query: str, chroma_path: Optional[str]) -> str:
    """
    Compose the <toolcall> payload and invoke the Agno Agent.
    Matches workflow.py semantics, but also forwards chroma_path explicitly.
    Includes a safety net to finalize if the agent echoes raw rag_context JSON.
    """
    payload = {"user_query": user_query}
    if chroma_path:
        payload["chroma_path"] = chroma_path

    toolcall_msg = f"""<toolcall name="query_iot_traffic_rag">
{json.dumps(payload, ensure_ascii=False)}
</toolcall>"""

    agent = get_rag_agent()

    # First pass: instruct the agent to call the retrieval tool
    text = _agent_call(agent, toolcall_msg)

    # Safety net: if the agent echoed the rag_context JSON, force a finalize pass
    if _is_rag_context_payload(text):
        finalize_msg = (
            "Use this package internally to produce the final answer. "
            "Do NOT display the JSON, the system prompt, or raw chunks. "
            "Output only the final answer in markdown:\n"
            f"<rag_context>\n{text}\n</rag_context>"
        )
        text = _agent_call(agent, finalize_msg)

    return text

# ----------------------------
# Utilities
# ----------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_sessions_index() -> dict:
    if SESSIONS_FILE.exists():
        try:
            return json.loads(SESSIONS_FILE.read_text())
        except Exception:
            return {}
    return {}

def save_sessions_index(index: dict) -> None:
    SESSIONS_FILE.write_text(json.dumps(index, indent=2))

def add_session_record(pcap_hash: str, chroma_path: str, filename: str) -> None:
    index = load_sessions_index()
    index[pcap_hash] = {
        "chroma_path": chroma_path,
        "filename": filename,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z"
    }
    # keep a small index (latest 10). Your ingest tool manages on-disk retention.
    items = sorted(index.items(), key=lambda kv: kv[1]["created_at"], reverse=True)[:10]
    save_sessions_index(dict(items))

def resolve_session_for_hash(pcap_hash: str) -> str | None:
    """Return chroma_path if we have a session for this hash and it's still on disk."""
    index = load_sessions_index()
    item = index.get(pcap_hash)
    if not item:
        return None
    path = item.get("chroma_path")
    if path and os.path.isdir(path):
        return path
    return None

def read_latest_pointer() -> str | None:
    if LATEST_POINTER.exists():
        p = LATEST_POINTER.read_text().strip()
        return p if os.path.isdir(p) else None
    return None

def write_app_log(filename: str, num_packets: int, question: str, answer: str) -> None:
    try:
        sess_id = st.session_state.get("session_id", hashlib.md5(os.urandom(8)).hexdigest())
        st.session_state["session_id"] = sess_id
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        log_path = LOGS_DIR / f"session_{sess_id}_{ts}.log"
        content = (
            f"Session ID: {sess_id}\n"
            f"Timestamp: {ts}Z\n"
            f"Uploaded File: {filename}\n"
            f"Number of Packets: {num_packets}\n"
            f"User Question: {question}\n"
            f"AI Response: {answer}\n"
            "--------------------------------------\n"
        )
        log_path.write_text(content)
    except Exception as e:
        st.warning(f"Log writing failed: {e}")

def place_uploaded_pcap_to_canonical(uploaded) -> Path:
    """Save uploaded PCAP to the canonical path your tools expect."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PCAP_CANONICAL, "wb") as f:
        f.write(uploaded.getvalue())
    return PCAP_CANONICAL

def build_pipeline(pcap_path: Path, logs: list[str], force_ingest: bool = False) -> None:
    """Run deterministic steps in order; append step logs for the UI."""
    logs.append("▶ Zeek JSON generation…")
    _ = run_zeek_log_generator()
    logs.append("✓ Zeek logs ready.")

    logs.append("▶ Per-packet layer extraction…")
    _ = extract_packet_layers()
    logs.append("✓ Packet layers JSON ready.")

    logs.append("▶ Flow summaries…")
    _ = generate_flow_summary()
    logs.append("✓ Flow summaries ready.")

    logs.append("▶ BERT feature extraction…")
    _ = extract_pcap_features()
    logs.append("✓ Features CSV ready.")

    logs.append("▶ BERT anomaly detection + report…")
    _ = run_bert_anomaly_detector()
    logs.append("✓ BERT report ready.")

    logs.append("▶ Threat intel enrichment…")
    _ = enrich_iot_report()
    logs.append("✓ Enriched report ready.")

    logs.append("▶ RAG ingestion to Chroma…")
    _ = rag_ingest_tool(force=force_ingest)
    logs.append("✓ Ingestion complete.")

def find_or_build_session(uploaded_file_name: str | None, force_ingest: bool = False) -> tuple[str | None, list[str], str | None]:
    """
    Returns (chroma_path, logs, pcap_hash). If a prior session for the same PCAP hash
    exists, we reuse it and skip rebuild, unless force_ingest=True.
    """
    logs: list[str] = []
    if not PCAP_CANONICAL.exists():
        return None, ["⚠️ No PCAP found to process."], None

    p_hash = sha256_file(PCAP_CANONICAL)
    logs.append(f"PCAP sha256: {p_hash[:12]}…")

    # Reuse existing session if available, unless forced
    existing = resolve_session_for_hash(p_hash)
    if existing and not force_ingest:
        logs.append(f"⏭ Skipping rebuild — found existing session for this PCAP hash.")
        return existing, logs, p_hash

    # Build full pipeline (forced or first-time)
    logs.append("🏗 Running full deterministic pipeline.")
    build_pipeline(PCAP_CANONICAL, logs, force_ingest=force_ingest)

    # After ingest, read RAG pointer
    chroma_path = read_latest_pointer()
    if not chroma_path:
        logs.append("❌ Ingestion finished but no session pointer found.")
        return None, logs, p_hash

    add_session_record(p_hash, chroma_path, uploaded_file_name or PCAP_CANONICAL.name)
    logs.append(f"✅ Session created and indexed at: {chroma_path}")
    return chroma_path, logs, p_hash

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Revelation — IoT Traffic Interpreter", page_icon="🛰️", layout="wide")
st.title("🛰️ Revelation — IoT Traffic Interpreter")

with st.sidebar:
    st.header("⚙️ Controls")
    st.markdown("Agent-backed queries. Sessions reuse by PCAP hash to avoid rebuilds.")

    # Existing sessions (from our index)
    index = load_sessions_index()
    if index:
        st.subheader("Saved Sessions")
        pretty = [
            f"{h[:12]} — {v.get('filename','?')} — {v.get('created_at','')}"
            for h, v in sorted(index.items(), key=lambda kv: kv[1]["created_at"], reverse=True)
        ]
        choice = st.selectbox("Select a saved session (optional):", ["(none)"] + pretty, index=0)
        selected_session_path = None
        selected_hash = None
        if choice != "(none)":
            # parse hash prefix from display
            selected_hash = choice.split(" — ")[0]
            # find full hash
            resolved = None
            for h, v in index.items():
                if h.startswith(selected_hash):
                    resolved = v["chroma_path"]
                    break
            if resolved and os.path.isdir(resolved):
                selected_session_path = resolved
            else:
                st.warning("Selected session path is missing on disk.")

        force_rebuild = st.checkbox("Force rebuild pipeline (ignore cache)", value=False)

    else:
        st.info("No saved sessions yet. Upload a PCAP to build the first one.")
        selected_session_path = None
        selected_hash = None
        force_rebuild = False

    st.markdown("---")
    st.caption("Embedding model (env): " + os.environ.get("EMBED_MODEL", ""))

# Tabs: Ingest / Query / Sessions
tab_ingest, tab_query, tab_sessions = st.tabs(["🧱 Ingest PCAP", "❓ Query (via Agent)", "📁 Sessions"])

# ----------------------------
# Tab: Ingest PCAP
# ----------------------------
with tab_ingest:
    st.subheader("Upload a PCAP and build (or reuse) a session")
    uploaded = st.file_uploader("Choose a PCAP/PCAPNG", type=["pcap", "pcapng"], accept_multiple_files=False)

    col1, col2 = st.columns(2)
    with col1:
        run_btn = st.button("Build / Reuse Session", type="primary", use_container_width=True)
    with col2:
        clear_btn = st.button("Clear Current Canonical PCAP", use_container_width=True)

    # Optional size cap like your reference app
    MAX_MB = 10  # adjust as desired
    if uploaded and uploaded.size > MAX_MB * 1024 * 1024:
        st.error(f"File exceeds {MAX_MB} MB. Please upload a smaller PCAP.")
        st.stop()

    if clear_btn:
        if PCAP_CANONICAL.exists():
            PCAP_CANONICAL.unlink(missing_ok=True)
        st.success("Canonical PCAP cleared. Upload a new PCAP to proceed.")
        st.stop()

    if run_btn:
        if uploaded:
            # place uploaded file to canonical location your tools expect
            place_uploaded_pcap_to_canonical(uploaded)
            st.session_state["uploaded_name"] = uploaded.name
        else:
            st.session_state["uploaded_name"] = None

        logs_box = st.empty()
        with st.spinner("Processing…"):
            chroma_path, logs, p_hash = find_or_build_session(
                st.session_state.get("uploaded_name"),
                force_ingest=force_rebuild
            )

        logs_box.code("\n".join(logs), language="bash")
        if chroma_path:
            st.success(f"Session ready at: {chroma_path}")
            st.session_state["current_chroma_path"] = chroma_path
            st.session_state["current_pcap_hash"] = p_hash
        else:
            st.error("Session not ready. Check logs above.")

# ----------------------------
# Tab: Query (Agent-backed)
# ----------------------------
with tab_query:
    st.subheader("Ask questions (handled by the Agno Agent, which calls the RAG tool)")

    # Determine active session: preference order = selected in sidebar > from ingest tab > latest pointer
    active_path = None
    if 'current_chroma_path' in st.session_state:
        active_path = st.session_state['current_chroma_path']
    if selected_session_path:
        active_path = selected_session_path
    if not active_path:
        # as a last resort, try latest pointer
        active_path = read_latest_pointer()

    st.markdown(f"**Active session:** `{active_path or 'None'}`")
    if not active_path:
        st.info("No active session found. Build or select a session first.")
    else:
        prompt = st.text_area(
            "Your question",
            "What is going on in the network?",
            height=120
        )
        ask_btn = st.button("Ask (via Agent)", type="primary")

        if ask_btn:
            with st.spinner("Agent reasoning + tool invocation…"):
                ans = agent_invoke_query(prompt, active_path)
            st.markdown("**Answer**")
            # Render as markdown for better formatting
            st.markdown(ans)

            # Optional logging (you can hook real packet counts if you track them)
            write_app_log(
                filename=st.session_state.get("uploaded_name", "unknown.pcap"),
                num_packets=0,
                question=prompt,
                answer=ans,
            )

# ----------------------------
# Tab: Sessions
# ----------------------------
with tab_sessions:
    st.subheader("Saved Sessions Index")
    idx = load_sessions_index()
    if not idx:
        st.info("No sessions in index yet.")
    else:
        # pretty table
        rows = []
        for h, v in sorted(idx.items(), key=lambda kv: kv[1]["created_at"], reverse=True):
            rows.append(
                f"- **{h[:12]}…**  \n"
                f"  path: `{v['chroma_path']}`  \n"
                f"  file: `{v.get('filename','?')}`  \n"
                f"  created: {v.get('created_at','')}  \n"
            )
        st.markdown("\n".join(rows))

        # quick action: set active session
        st.markdown("---")
        set_active = st.text_input("Set active chroma path (paste a path)")
        if st.button("Use this path", disabled=not set_active):
            if os.path.isdir(set_active):
                st.session_state["current_chroma_path"] = set_active
                st.success(f"Active session set to: {set_active}")
            else:
                st.error("Provided path is not a directory.")

# ----------------------------
# Footer
# ----------------------------
st.caption("Revelation • Deterministic IoT pipeline + agentic RAG query • Streamlit frontend (Agent-backed)")