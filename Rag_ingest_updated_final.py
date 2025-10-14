import os
import shutil
import json
import re
import hashlib
from typing import List
from uuid import uuid4
from collections import defaultdict
from langchain.schema.document import Document
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, JSONLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import OllamaEmbeddings
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import time

# -------------------------------------------------------------------------
# Configuration & Embeddings
# -------------------------------------------------------------------------

# Safety buffer below ChromaDB max batch size
BATCH_SIZE = 1000

# === Embedding Function ===
def get_embedding_function():
    return OllamaEmbeddings(model="nomic-embed-text:v1.5")

"""
# Alternate embedding function (kept as in the original file, but unused)
def get_embedding_function():
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
"""

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZEEK_PATH = os.path.join(BASE_DIR, "zeek_json_outputs")
REPORT_PATH = os.path.join(BASE_DIR, "enhancer")
FLOW_PATH = os.path.join(BASE_DIR, "flow_summaries")
PACKET_LAYER_PATH = os.path.join(BASE_DIR, "packet_layer_json")

ENHANCED_REPORT_FILE = os.path.join(REPORT_PATH, "enhanced_bert_report.txt")
FLOW_SUMMARY_FILE = os.path.join(FLOW_PATH, "summary.txt")

SESSIONS_DIR = os.path.join(BASE_DIR, "chroma_sessions")
LATEST_SESSION_FILE_LEGACY_DOC = "latest_session.txt"  # doc note only; see function-scoped var below
HASH_TRACK_FILE = os.path.join(SESSIONS_DIR, "ingested_hashes.json")


# -------------------------------------------------------------------------
# Public tool entry
# -------------------------------------------------------------------------

def rag_ingest_tool(force: bool = False, **kwargs) -> str:
    """
    Tool entrypoint. Set force=True to bypass the dataset-hash skip guard
    and always build a brand-new session.
    """
    if kwargs:
        print(f"Ignoring unexpected tool inputs: {kwargs}")
    return ingest_all_chunks(force=force)


# -------------------------------------------------------------------------
# Utility: hashing & session rotation
# -------------------------------------------------------------------------

def hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def compute_dataset_hash() -> str:
    all_files = []
    for folder in [ZEEK_PATH, REPORT_PATH, FLOW_PATH, PACKET_LAYER_PATH]:
        if not os.path.isdir(folder):
            # folder might legitimately not exist yet (e.g., missing optional source)
            continue
        for fname in sorted(os.listdir(folder)):
            full_path = os.path.join(folder, fname)
            if os.path.isfile(full_path):
                all_files.append(hash_file(full_path))
    combined = ''.join(sorted(all_files))
    return hashlib.sha256(combined.encode()).hexdigest()

def has_been_ingested(new_hash: str) -> bool:
    if not os.path.isfile(HASH_TRACK_FILE):
        return False
    try:
        with open(HASH_TRACK_FILE, "r") as f:
            past_hashes = json.load(f)
        return new_hash in past_hashes
    except Exception:
        return False

def update_ingested_hashes(new_hash: str):
    past_hashes = []
    if os.path.isfile(HASH_TRACK_FILE):
        try:
            with open(HASH_TRACK_FILE, "r") as f:
                past_hashes = json.load(f)
        except Exception:
            past_hashes = []
    past_hashes.insert(0, new_hash)
    past_hashes = past_hashes[:3]  # keep only latest 3
    os.makedirs(os.path.dirname(HASH_TRACK_FILE), exist_ok=True)
    with open(HASH_TRACK_FILE, "w") as f:
        json.dump(past_hashes, f, indent=2)

def cleanup_old_sessions():
    if not os.path.exists(SESSIONS_DIR):
        return
    sessions = [d for d in os.listdir(SESSIONS_DIR) if d.startswith("session_")]
    sessions = sorted(sessions, key=lambda x: os.path.getmtime(os.path.join(SESSIONS_DIR, x)), reverse=True)
    for old in sessions[3:]:
        shutil.rmtree(os.path.join(SESSIONS_DIR, old), ignore_errors=True)


# -------------------------------------------------------------------------
# Ingestion pipeline (builds all chunks and writes a new session)
# -------------------------------------------------------------------------

def ingest_all_chunks(force: bool = False) -> str:
    """
    Build chunks from Zeek logs, enhanced BERT report, flow summaries, and packet-layer JSONs,
    then write them into a brand-new Chroma session directory (unless force=False and hash matches).
    """
    # Create a brand-new session path per ingestion call
    sessions_dir = SESSIONS_DIR
    chroma_session_dir = os.path.join(sessions_dir, f"session_{uuid4().hex[:8]}")
    latest_session_file = os.path.join(sessions_dir, "latest_session.txt")

    os.makedirs(chroma_session_dir, exist_ok=True)
    os.makedirs(os.path.dirname(latest_session_file), exist_ok=True)

    dataset_hash = compute_dataset_hash()
    if (not force) and has_been_ingested(dataset_hash):
        print("⚠️  Duplicate dataset detected. Skipping ingestion.")
        return "⚠️  This dataset has already been ingested in one of the last 3 sessions."

    all_chunks: List[Document] = []

    print("🔍  Loading Zeek documents...")
    zeek_chunks = chunk_zeek_by_uid()
    print(f"  → Generated {len(zeek_chunks)} UID-based Zeek chunks")
    all_chunks.extend(zeek_chunks)

    print("🔍  Loading BERT reports...")
    bert_docs = load_bert_report_documents()
    print(f"  → Loaded {len(bert_docs)} BERT reports")

    print("✂️   Splitting BERT reports...")
    bert_chunks = split_bert_reports(bert_docs)
    print(f"  → Generated {len(bert_chunks)} BERT chunks")
    all_chunks.extend(bert_chunks)

    print("🔍  Loading Flow Summary...")
    flow_chunks = load_and_split_flow_summary()
    print(f"  → Generated {len(flow_chunks)} Flow Summary chunks")
    all_chunks.extend(flow_chunks)

    print("🔍  Loading Packet Layer Chunks with Semantic Chunking...")
    packet_chunks = load_and_chunk_packet_layers()
    print(f"  → Generated {len(packet_chunks)} Packet Layer chunks")
    all_chunks.extend(packet_chunks)

    print(f"\n📦  Total chunks to ingest: {len(all_chunks)}")

    # Write chunks into THIS session
    add_to_chroma(all_chunks, chroma_session_dir)

    # Update session pointer AFTER successful ingest
    with open(latest_session_file, "w") as f:
        f.write(chroma_session_dir)

    # Track hash and rotate old sessions
    update_ingested_hashes(dataset_hash)
    cleanup_old_sessions()

    return f"✅ Ingested {len(all_chunks)} chunks into Chroma session: {chroma_session_dir}"


# -------------------------------------------------------------------------
# Chunk builders (unchanged logic)
# -------------------------------------------------------------------------

def chunk_zeek_by_uid() -> List[Document]:
    records = []
    if not os.path.isdir(ZEEK_PATH):
        return []

    for fname in os.listdir(ZEEK_PATH):
        if fname.endswith(".json"):
            path = os.path.join(ZEEK_PATH, fname)
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for rec in data:
                        rec['_log_source'] = fname
                        records.append(rec)
            except Exception as e:
                print(f"❌ Failed to load {fname}: {e}")

    uid_map = defaultdict(list)
    for rec in records:
        uid = rec.get('uid', f"NO_UID_{uuid4().hex[:6]}")
        uid_map[uid].append(rec)

    chunks = []
    for uid, group in uid_map.items():
        chunk_text = json.dumps(group, indent=2)
        meta = {
            'chunk_type': 'zeek_uid',
            'uid': uid,
            'source': group[0]['_log_source'] if group and '_log_source' in group[0] else 'unknown'
        }
        chunks.append(Document(page_content=chunk_text, metadata=meta))
    return chunks

def load_bert_report_documents() -> List[Document]:
    docs = []
    if os.path.isfile(ENHANCED_REPORT_FILE):
        loader = TextLoader(ENHANCED_REPORT_FILE, encoding="utf-8")
        for doc in loader.load():
            doc.metadata["source"] = "enhanced_bert_report.txt"
            docs.append(doc)
    else:
        print(f"❌ Enhanced report not found at: {ENHANCED_REPORT_FILE}")
    return docs

def split_bert_reports(docs: List[Document]) -> List[Document]:
    chunks = []
    for doc in docs:
        text = doc.page_content
        src = doc.metadata.get("source", "unknown.txt")
        chunks.extend(split_bert_report_by_section(text, src))
    return chunks

def split_bert_report_by_section(text: str, source: str) -> List[Document]:
    docs = []

    stats_match = re.search(
        r"Total traffic samples analyzed: .+?\nNormal traffic: .+?\nAnomalous traffic: .+?",
        text, re.DOTALL
    )
    if stats_match:
        docs.append(Document(
            page_content=stats_match.group(0).strip(),
            metadata={"chunk_type": "traffic_stats", "source": source}
        ))

    summary = re.search(
        r"=== Interpretation Summary ===\n(.+?)(?=\n=== Unique Endpoints|\Z)",
        text, re.DOTALL
    )
    if summary:
        docs.append(Document(
            page_content=summary.group(0).strip(),
            metadata={"chunk_type": "summary", "source": source}
        ))

    normal_match = re.search(
        r"\[Normal\]\n- Count:.*?\n- No action needed.*?(?=\n\[|\n===|\Z)",
        text, re.DOTALL
    )
    if normal_match:
        docs.append(Document(
            page_content=normal_match.group(0).strip(),
            metadata={"chunk_type": "behavior", "attack_type": "Normal", "source": source}
        ))

    for match in re.finditer(
        r"\[(?P<attack>[A-Za-z0-9_ ]+)]\n.*?Severity: (?P<severity>\w+).*?(?=\n\[|\n=== Interpretation Summary|\n=== Unique Endpoints|\Z)",
        text, re.DOTALL
    ):
        attack = match.group("attack").strip()
        if attack == "Normal":
            continue
        docs.append(Document(
            page_content=match.group(0).strip(),
            metadata={"chunk_type": "behavior", "attack_type": attack, "source": source}
        ))

    meta = re.search(
        r"=== Unique Endpoints and Metadata per Attack ===\n(.+)",
        text, re.DOTALL
    )
    if meta:
        blocks = re.split(r"\n(?=\[[A-Za-z0-9_ ]+])", meta.group(1))
        for block in blocks:
            m2 = re.match(r"\[(?P<attack>[A-Za-z0-9_ ]+)]", block)
            if m2:
                attack = m2.group("attack").strip()
                docs.append(Document(
                    page_content=block.strip(),
                    metadata={"chunk_type": "metadata", "attack_type": attack, "source": source}
                ))
    return docs

def load_and_split_flow_summary() -> List[Document]:
    if not os.path.isfile(FLOW_SUMMARY_FILE):
        print(f"❌ Flow summary not found at: {FLOW_SUMMARY_FILE}")
        return []

    with open(FLOW_SUMMARY_FILE, "r") as f:
        text = f.read()

    chunks = []
    source = "summary.txt"

    overview_match = re.search(r"=== IoT Traffic Flow Summary ===.*?={5,}", text, re.DOTALL)
    if overview_match:
        chunks.append(Document(
            page_content=overview_match.group(0).strip(),
            metadata={"chunk_type": "flow_overview", "source": source}
        ))

    flow_blocks = re.findall(r"(=== Flow ID: .+?)(?=(?:=== Flow ID: |\Z))", text, re.DOTALL)
    for block in flow_blocks:
        match = re.match(r"=== Flow ID: (?P<flow_id>.+?) ===", block)
        if match:
            flow_id = match.group("flow_id").strip()
            chunks.append(Document(
                page_content=block.strip(),
                metadata={
                    "chunk_type": "flow_summary",
                    "flow_id": flow_id,
                    "source": source
                }
            ))
    return chunks

def load_and_chunk_packet_layers() -> List[Document]:
    if not os.path.isdir(PACKET_LAYER_PATH):
        return []
    embedding_fn = get_embedding_function()
    text_splitter = SemanticChunker(embedding_fn)
    chunks: List[Document] = []

    for fname in os.listdir(PACKET_LAYER_PATH):
        if fname.endswith(".json"):
            file_path = os.path.join(PACKET_LAYER_PATH, fname)
            try:
                loader = JSONLoader(
                    file_path=file_path,
                    jq_schema=""".[] | ._source.layers | del(.data)""",
                    text_content=False
                )
                pages = loader.load_and_split()
                semantic_chunks = text_splitter.split_documents(pages)
                for i, chunk in enumerate(semantic_chunks):
                    chunk.metadata = {
                        "chunk_type": "packet_layer",
                        "source": fname,
                        "packet_index": i
                    }
                    chunks.append(chunk)
            except Exception as e:
                print(f"❌ Error processing packet layer JSON {fname}: {e}")
    return chunks


# -------------------------------------------------------------------------
# Add-to-Chroma with deterministic IDs (prevents silent duplication)
# -------------------------------------------------------------------------

def _chunk_id(doc: Document, idx: int) -> str:
    m = doc.metadata or {}
    src   = m.get("source", "unknown")
    ctype = m.get("chunk_type", "unknown")
    tail  = m.get("attack_type") or m.get("flow_id") or m.get("uid") or m.get("packet_index", idx)
    h     = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()[:12]
    return f"{src}:{ctype}:{tail}:{h}"

def add_to_chroma(chunks: List[Document], session_dir: str):
    db = Chroma(
        persist_directory=session_dir,
        embedding_function=get_embedding_function()
    )

    print(f"📥 Ingesting {len(chunks)} chunks into {session_dir} (batched)…")
    for start in range(0, len(chunks), BATCH_SIZE):
        end   = min(start + BATCH_SIZE, len(chunks))
        batch = chunks[start:end]
        ids   = [_chunk_id(d, i) for i, d in enumerate(batch, start)]
        db.add_documents(batch, ids=ids)
        print(f"   → Batch {start}–{end} ingested")
        # Optional pause for API rate limits (kept disabled):
        # time.sleep(8)

    db.persist()
    print(f"✅ Fully ingested {len(chunks)} chunks into {session_dir}")


# -------------------------------------------------------------------------
# Debug helper (unchanged)
# -------------------------------------------------------------------------

def nuke_all_sessions_debug_only():
    base = os.path.join(BASE_DIR, "chroma_sessions")
    if os.path.exists(base):
        shutil.rmtree(base)
        print("✅ All session DBs removed.")
"""
WARNING: Debug-only function. Removes all Chroma session databases, including the latest ones.
Not used in production. Do not call unless explicitly resetting the system.
"""