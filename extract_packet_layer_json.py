import os
import json
import subprocess


# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PCAP_PATH = os.path.join(BASE_DIR, "data", "Backdoor_attack.pcap")
PACKET_LAYER_DIR = os.path.join(BASE_DIR, "packet_layer_json")
os.makedirs(PACKET_LAYER_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(PACKET_LAYER_DIR, "packet_layers.json")


def extract_packet_layers() -> str:
    if not os.path.isfile(PCAP_PATH):
        raise FileNotFoundError(f"PCAP file not found: {PCAP_PATH}")

    cmd = f'tshark -nlr "{PCAP_PATH}" -T json > "{OUTPUT_FILE}"'
    subprocess.run(cmd, shell=True, check=True)

    try:
        with open(OUTPUT_FILE, "r") as file:
            data = json.load(file)

        for packet in data:
            layers = packet.get("_source", {}).get("layers", {})

            # Strip verbose or encrypted hex payloads
            if "udp" in layers:
                layers["udp"].pop("udp.payload", None)
            if "tcp" in layers:
                layers["tcp"].pop("tcp.payload", None)
                layers["tcp"].pop("tcp.segment_data", None)
                layers["tcp"].pop("tcp.reassembled.data", None)
            if "tls" in layers and isinstance(layers["tls"], dict):
                layers["tls"].pop("tls.segment.data", None)
                tls_record = layers["tls"].get("tls.record", {})
                if isinstance(tls_record, list):
                    for rec in tls_record:
                        hs = rec.get("tls.handshake", {})
                        tree = hs.get("tls.handshake.random_tree", {})
                        if isinstance(tree, dict):
                            tree.pop("tls.handshake.random_bytes", None)
                elif isinstance(tls_record, dict):
                    hs = tls_record.get("tls.handshake", {})
                    tree = hs.get("tls.handshake.random_tree", {})
                    if isinstance(tree, dict):
                        tree.pop("tls.handshake.random_bytes", None)

        with open(OUTPUT_FILE, "w") as f:
            json.dump(data, f, indent=2)

        return f"✅ Packet-layer JSON extracted and cleaned. Total packets: {len(data)}. Output: {OUTPUT_FILE}"

    except json.JSONDecodeError as e:
        raise ValueError(f"JSON decoding failed: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during packet cleaning: {e}")
