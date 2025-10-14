#Workflow.py

import os
import hashlib
from pathlib import Path
from textwrap import dedent
import json

from agno.workflow.v2.types import StepInput, StepOutput
from agno.workflow.v2.workflow import Workflow
from agno.storage.sqlite import SqliteStorage
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools.tavily import TavilyTools

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
os.environ.setdefault("PYTHONPATH", f"{PROJECT_ROOT}:{os.environ.get('PYTHONPATH','')}")
#os.environ.setdefault("EMBED_MODEL", "nomic-embed-text:v1.5")

DATA_DIR = PROJECT_ROOT / "data"
PCAP_PATH = DATA_DIR / "Backdoor_attack.pcap"
STATE_DIR = PROJECT_ROOT / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
PCAP_HASH_FILE = STATE_DIR / "pcap_hash.txt"

# Deterministic steps
from zeek_json_tool import run_zeek_log_generator
from extract_packet_layer_json import extract_packet_layers
from flow_summaries_updated_final import generate_flow_summary
from extract_pcap_features_tool import extract_pcap_features
from anomaly_detection import run_bert_anomaly_detector
from enrich_iot_report import enrich_iot_report
from Rag_ingest_updated_final import rag_ingest_tool

# Agent-exposed tool
from query_iot_rag_custom_improve import query_iot_traffic_rag


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def should_skip_pipeline() -> tuple[bool, str]:
    if not PCAP_PATH.exists():
        return False, ""
    current = sha256_file(PCAP_PATH)
    previous = PCAP_HASH_FILE.read_text().strip() if PCAP_HASH_FILE.exists() else ""
    return (current == previous and previous != ""), current

def step_skip_guard(_: StepInput) -> StepOutput:
    skip, current_hash = should_skip_pipeline()
    msg = (
        f"[Skip Guard] PCAP unchanged ({current_hash[:12]}) — skipping rebuild."
        if skip else
        f"[Skip Guard] PCAP changed ({current_hash[:12] if current_hash else 'none'}) — running full pipeline."
    )
    return StepOutput(content={"skip": skip, "pcap_hash": current_hash, "log": [msg]})

def step_zeek(step_input: StepInput) -> StepOutput:
    state = step_input.previous_step_content or {}
    if state.get("skip"):
        state["log"].append("[Zeek] skipped.")
        return StepOutput(content=state)
    out = run_zeek_log_generator()
    state["log"].append(f"[Zeek]\n{out}")
    return StepOutput(content=state)

def step_packet_layers(step_input: StepInput) -> StepOutput:
    state = step_input.previous_step_content or {}
    if state.get("skip"):
        state["log"].append("[Packet Layers] skipped.")
        return StepOutput(content=state)
    out = extract_packet_layers()
    state["log"].append(f"[Packet Layers]\n{out}")
    return StepOutput(content=state)

def step_flow_summary(step_input: StepInput) -> StepOutput:
    state = step_input.previous_step_content or {}
    if state.get("skip"):
        state["log"].append("[Flow Summary] skipped.")
        return StepOutput(content=state)
    out = generate_flow_summary()
    state["log"].append(f"[Flow Summary]\n{out}")
    return StepOutput(content=state)

def step_features(step_input: StepInput) -> StepOutput:
    state = step_input.previous_step_content or {}
    if state.get("skip"):
        state["log"].append("[BERT Features] skipped.")
        return StepOutput(content=state)
    out = extract_pcap_features()
    state["log"].append(f"[BERT Features]\n{out}")
    return StepOutput(content=state)

def step_bert(step_input: StepInput) -> StepOutput:
    state = step_input.previous_step_content or {}
    if state.get("skip"):
        state["log"].append("[BERT Anomaly Detection] skipped.")
        return StepOutput(content=state)
    out = run_bert_anomaly_detector()
    state["log"].append(f"[BERT Anomaly Detection]\n{out}")
    return StepOutput(content=state)

def step_enrich(step_input: StepInput) -> StepOutput:
    state = step_input.previous_step_content or {}
    if state.get("skip"):
        state["log"].append("[Threat Intel Enrichment] skipped.")
        return StepOutput(content=state)
    out = enrich_iot_report()
    state["log"].append(f"[Threat Intel Enrichment]\n{out}")
    return StepOutput(content=state)

def step_rag_ingest(step_input: StepInput) -> StepOutput:
    state = step_input.previous_step_content or {}
    if state.get("skip"):
        state["log"].append("[RAG Ingestion] skipped.")
        return StepOutput(content=state)
    out = rag_ingest_tool()
    state["log"].append(f"[RAG Ingestion]\n{out}")
    if state.get("pcap_hash"):
        PCAP_HASH_FILE.write_text(state["pcap_hash"])
    return StepOutput(content=state)

def step_prepare_query(step_input: StepInput) -> StepOutput:
    user_q = step_input.message or "Summarize notable threats and benign behaviors."
    payload = {"user_query": user_q}
    msg = dedent(f"""\
        <toolcall name="query_iot_traffic_rag">
        {json.dumps(payload, ensure_ascii=False)}
        </toolcall>
    """).strip()
    return StepOutput(content=msg)

rag_agent = Agent(
    name="RAG Query Agent",
    model=Ollama(id="llama3.2:latest"),
    tools=[query_iot_traffic_rag, TavilyTools()],
    markdown=True,
    instructions=AGENT_INSTRUCTIONS
)

def step_capture_answer(step_input: StepInput) -> StepOutput:
    return StepOutput(content=step_input.previous_step_content or "")

iot_interpretation_workflow = Workflow(
    name="Revelation — Deterministic IoT Pipeline + Agentic Query",
    description="Skip heavy steps when PCAP unchanged. Otherwise: Zeek→Features→BERT→Enrich→Flows→Packets→Ingest, then query via agent tool.",
    storage=SqliteStorage(
        table_name="workflow_v2",
        db_file=str(PROJECT_ROOT / "tmp" / "workflow_v2.db"),
        mode="workflow_v2",
    ),
    steps=[
        step_skip_guard,
        step_zeek,
        step_packet_layers,
        step_flow_summary,
        step_features,
        step_bert,
        step_enrich,
        step_rag_ingest,
        step_prepare_query,
        rag_agent,
        step_capture_answer,
    ],
)

if __name__ == "__main__":
    iot_interpretation_workflow.print_response(
        message="What is the dominant attack in the traffic?",
        markdown=True,
    )