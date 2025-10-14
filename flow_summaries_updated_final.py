import os
import pyshark
import statistics
import requests
import ipaddress
from collections import defaultdict
from manuf import manuf

# === CONFIGURATION ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PCAP_PATH = os.path.join(BASE_DIR, "data", "Backdoor_attack.pcap")
FLOW_SUMMARY_DIR = os.path.join(BASE_DIR, "flow_summaries")
FLOW_SUMMARY_FILE = os.path.join(FLOW_SUMMARY_DIR, "summary.txt")
ABUSEIPDB_API_KEY = "Put your AbuseIPDB API key here"
ABUSEIPDB_ENDPOINT = "https://api.abuseipdb.com/api/v2/check"
FLOW_SUMMARY_FILE_SUCCINCT = os.path.join(FLOW_SUMMARY_DIR, "summary_succinct.txt")


# === UTILITIES ===
def decode_tcp_flags(flag_hex):
    flags = []
    try:
        value = int(flag_hex, 16)
        if value & 0x01: flags.append("FIN")
        if value & 0x02: flags.append("SYN")
        if value & 0x04: flags.append("RST")
        if value & 0x08: flags.append("PSH")
        if value & 0x10: flags.append("ACK")
        if value & 0x20: flags.append("URG")
        if value & 0x40: flags.append("ECE")
        if value & 0x80: flags.append("CWR")
    except:
        pass
    return ", ".join(flags) if flags else "None"

def annotate_reputation(ip_str):
    try:
        ip = ipaddress.ip_address(ip_str)
        if ip.is_loopback:
            return "Loopback IP (Localhost)"
        if ip.is_multicast:
            return "Multicast IP"
        if ip.is_link_local:
            return "Link-Local IP"
        if ip.is_private:
            return "Private IP"
        return query_abuseipdb(ip_str)
    except:
        return "Invalid IP"

def query_abuseipdb(ip):
    try:
        response = requests.get(
            ABUSEIPDB_ENDPOINT,
            params={"ipAddress": ip, "maxAgeInDays": 90},
            headers={"Key": ABUSEIPDB_API_KEY, "Accept": "application/json"},
            timeout=5
        )
        if response.status_code == 200:
            score = response.json().get("data", {}).get("abuseConfidenceScore", 0)
            return f"Malicious Activity Detected (score: {score})" if score >= 25 else f"No Threat Detected (score: {score})"
    except Exception as e:
        print(f"[WARN] AbuseIPDB query failed for {ip}: {e}")
    return "Reputation Unknown"

def format_summary_block(flow_id, meta):
    lines = [
        f"=== Flow ID: {flow_id} ===",
        f"- Duration: {meta['flow_time']:.2f}s",
        f"- Packets: {meta['total_packets']}",
        f"- Source IP: {meta['src_ip']} ({meta['src_reputation']}) → Dest IP: {meta['dst_ip']} ({meta['dst_reputation']})",
        f"- MAC: {meta['src_mac']} ({meta['mac_src_vendor']}) → {meta['dst_mac']} ({meta['mac_dst_vendor']})",
        f"- IP Version: {meta['ip_version']}",
        f"- TTL Range: {meta['ttl_range']}",
        f"- Protocol: {meta['protocol']}",
        f"- Ports: {meta['src_port']} → {meta['dst_port']}",
        f"- Total bytes exchanged: {meta['total_packet_length']}",
        f"- App Payload Size: {meta['app_payload_size']} bytes"
    ]
    if meta.get('application_summary'):
        lines.append(f"- Application Summary: {meta['application_summary']}")
    if meta.get('extra_info'):
        lines.append(f"- Extra Info: {meta['extra_info']}")
    lines.append(f"- TCP Flag Pattern: {' → '.join(meta['flag_sequence']) if meta['flag_sequence'] else 'N/A'}\n")
    return "\n".join(lines)


def generate_flow_summary() -> str:
    if not os.path.isfile(PCAP_PATH):
        raise FileNotFoundError(f"PCAP not found: {PCAP_PATH}")
    os.makedirs(FLOW_SUMMARY_DIR, exist_ok=True)

    cap = pyshark.FileCapture(PCAP_PATH, keep_packets=True)
    parser = manuf.MacParser()
    flows = defaultdict(lambda: {
        "packets": [], "times": [], "total_packet_length": 0,
        "flag_sequence": [], "ttl_values": [], "app_payload_size": 0,
        "src_ip": "N/A", "dst_ip": "N/A", "eth_src": "N/A", "eth_dst": "N/A",
        "mac_src_vendor": "Unknown", "mac_dst_vendor": "Unknown", "ip_version": "N/A",
        "app_summary": "", "extra_info": ""
    })
    reputation_cache = {}
    seen_ips = set()          # NEW: for succinct stats
    public_ips = set()        # NEW
    private_ips = set()       # NEW

    for pkt in cap:
        try:
            ts = pkt.sniff_time.timestamp()
            ip_src = getattr(pkt.ip, 'src', 'N/A') if hasattr(pkt, 'ip') else 'N/A'
            ip_dst = getattr(pkt.ip, 'dst', 'N/A') if hasattr(pkt, 'ip') else 'N/A'
            ip_version = getattr(pkt.ip, 'version', 'N/A') if hasattr(pkt, 'ip') else 'N/A'
            ttl = int(getattr(pkt.ip, 'ttl', 0)) if hasattr(pkt, 'ip') else 0
            eth_src = getattr(pkt.eth, 'src', 'N/A') if hasattr(pkt, 'eth') else 'N/A'
            eth_dst = getattr(pkt.eth, 'dst', 'N/A') if hasattr(pkt, 'eth') else 'N/A'
            src_vendor = parser.get_manuf(eth_src) or "Unknown"
            dst_vendor = parser.get_manuf(eth_dst) or "Unknown"

            # track IP categories (for succinct)
            for ip_ in (ip_src, ip_dst):
                if ip_ != "N/A":
                    seen_ips.add(ip_)
                    try:
                        ipobj = ipaddress.ip_address(ip_)
                        if ipobj.is_private or ipobj.is_loopback or ipobj.is_link_local:
                            private_ips.add(ip_)
                        else:
                            public_ips.add(ip_)
                    except Exception:
                        pass

            proto, stream_id = "OTHER", None
            if 'TCP' in pkt:
                proto = 'TCP'
                stream_id = f"TCP_{pkt.tcp.stream}"
                flags = getattr(pkt.tcp, 'flags', '0x00')
                flows[stream_id]['flag_sequence'].append(decode_tcp_flags(flags))
            elif 'UDP' in pkt:
                proto = 'UDP'
                stream_id = f"UDP_{pkt.udp.stream}"
            else:
                continue

            # Application-layer summary
            app_summary = ""
            extra_info = ""
            if hasattr(pkt, 'http'):
                app_summary = f"HTTP {getattr(pkt.http, 'request_method', '')} {getattr(pkt.http, 'request_uri', '')}"
            elif hasattr(pkt, 'mqtt'):
                app_summary = f"MQTT {getattr(pkt.mqtt, 'msgtype', '')} on topic {getattr(pkt.mqtt, 'topic', '')}"
            elif hasattr(pkt, 'dns'):
                app_summary = f"DNS Query {getattr(pkt.dns, 'qry_name', '')}"
                if hasattr(pkt.dns, 'a'):
                    extra_info = f"Answers: {getattr(pkt.dns, 'a')}"
            elif hasattr(pkt, 'mbtcp'):
                app_summary = f"Modbus Unit {getattr(pkt.mbtcp, 'unit_id', '')}"
                if hasattr(pkt.mbtcp, 'func_code'):
                    extra_info = f"Function Code: {getattr(pkt.mbtcp, 'func_code')}"
            elif hasattr(pkt, 'tls'):
                app_summary = "TLS handshake detected"
            elif hasattr(pkt, 'coap'):
                app_summary = f"CoAP {getattr(pkt.coap, 'code', '')} to {getattr(pkt.coap, 'uri_path', '')}"
            elif hasattr(pkt, 'bacnet'):
                app_summary = f"BACnet {getattr(pkt.bacnet, 'bacnet_service', '')}"
            elif hasattr(pkt, 'oma_lwm2m'):
                app_summary = "LwM2M device management operation"
            elif hasattr(pkt, 'mqttsn'):
                app_summary = f"MQTT-SN {getattr(pkt.mqttsn, 'msgtype', '')}"

            # Accumulate metadata
            flows[stream_id]['packets'].append(pkt)
            flows[stream_id]['times'].append(ts)
            flows[stream_id]['ttl_values'].append(ttl)
            flows[stream_id]['total_packet_length'] += int(pkt.length)
            total_len = int(getattr(pkt, 'length', 0))
            ip_len = int(getattr(pkt.ip, 'len', 0)) if hasattr(pkt, 'ip') else 0
            flows[stream_id]['app_payload_size'] += max(total_len - ip_len, 0)

            flows[stream_id].update({
                'src_ip': ip_src, 'dst_ip': ip_dst,
                'eth_src': eth_src, 'eth_dst': eth_dst,
                'mac_src_vendor': src_vendor, 'mac_dst_vendor': dst_vendor,
                'ip_version': ip_version,
                'app_summary': app_summary or flows[stream_id]['app_summary'],
                'extra_info': extra_info or flows[stream_id]['extra_info']
            })
        except Exception as e:
            print(f"Error processing packet: {e}")
            continue

    cap.close()

    all_blocks, flow_counts, talkers = [], defaultdict(int), defaultdict(int)
    # lightweight app counters for succinct summary
    app_tally = defaultdict(int)   # e.g., HTTP, DNS, MQTT, Modbus, TLS, CoAP, BACnet, LwM2M, MQTT-SN, Other

    for flow_id, data in flows.items():
        if not data['packets']:
            continue
        proto = flow_id.split("_")[0]
        flow_counts[proto] += 1
        talkers[f"{data['src_ip']} → {data['dst_ip']}"] += 1
        pkt_layer = data['packets'][0].tcp if proto == 'TCP' else data['packets'][0].udp
        src_port = getattr(pkt_layer, 'srcport', 'N/A')
        dst_port = getattr(pkt_layer, 'dstport', 'N/A')
        flow_time = max(data['times']) - min(data['times']) if len(data['times']) > 1 else 0
        ttl_vals = data['ttl_values']
        ttl_range = f"{min(ttl_vals)}–{max(ttl_vals)} (avg: {statistics.mean(ttl_vals):.1f})" if ttl_vals else "N/A"

        # reputation (cached)
        src_rep = reputation_cache.get(data['src_ip'])
        dst_rep = reputation_cache.get(data['dst_ip'])
        if src_rep is None:
            src_rep = annotate_reputation(data['src_ip'])
            reputation_cache[data['src_ip']] = src_rep
        if dst_rep is None:
            dst_rep = annotate_reputation(data['dst_ip'])
            reputation_cache[data['dst_ip']] = dst_rep

        # tally app type for succinct
        app = data['app_summary'].lower()
        def bump(name): app_tally[name] += 1
        if app.startswith("http"):
            bump("HTTP")
        elif app.startswith("dns"):
            bump("DNS")
        elif app.startswith("mqttsn"):
            bump("MQTT-SN")
        elif app.startswith("mqtt"):
            bump("MQTT")
        elif app.startswith("modbus"):
            bump("Modbus")
        elif app.startswith("tls"):
            bump("TLS")
        elif app.startswith("coap"):
            bump("CoAP")
        elif app.startswith("bacnet"):
            bump("BACnet")
        elif app.startswith("lwm2m"):
            bump("LwM2M")
        elif app:
            bump("Other")

        block = format_summary_block(flow_id, {
            'flow_time': flow_time,
            'total_packets': len(data['packets']),
            'src_ip': data['src_ip'],
            'dst_ip': data['dst_ip'],
            'src_mac': data['eth_src'],
            'dst_mac': data['eth_dst'],
            'mac_src_vendor': data['mac_src_vendor'],
            'mac_dst_vendor': data['mac_dst_vendor'],
            'ip_version': data['ip_version'],
            'ttl_range': ttl_range,
            'protocol': proto,
            'src_port': src_port,
            'dst_port': dst_port,
            'total_packet_length': data['total_packet_length'],
            'app_payload_size': data['app_payload_size'],
            'application_summary': data['app_summary'],
            'extra_info': data['extra_info'],
            'flag_sequence': data['flag_sequence'][:6],
            'src_reputation': src_rep,
            'dst_reputation': dst_rep
        })
        all_blocks.append(block)

    # ------- write full holistic summary (existing behavior) -------
    with open(FLOW_SUMMARY_FILE, "w") as f:
        f.write("=== IoT Traffic Flow Summary ===\n")
        f.write(f"Total Flows Detected: {len(flows)}\n")
        for proto, count in flow_counts.items():
            f.write(f"- {proto}: {count} flows\n")
        top_talkers = sorted(talkers.items(), key=lambda x: x[1], reverse=True)[:5]
        f.write("Top Talkers:\n")
        for pair, count in top_talkers:
            f.write(f"- {pair} ({count} flows)\n")
        f.write("================================\n\n")
        for block in all_blocks:
            f.write(block + "\n")

    # ------- write succinct summary (NEW) -------
    # reputation counts (public IPs only)
    rep_counts = {"malicious": 0, "benign": 0, "unknown": 0}
    for ip_, rep in reputation_cache.items():
        try:
            if not ipaddress.ip_address(ip_).is_private:
                if "Malicious Activity Detected" in rep:
                    rep_counts["malicious"] += 1
                elif "No Threat Detected" in rep:
                    rep_counts["benign"] += 1
                else:
                    rep_counts["unknown"] += 1
        except Exception:
            pass

    with open(FLOW_SUMMARY_FILE_SUCCINCT, "w") as g:
        g.write("=== Succinct Flow Summary ===\n")
        g.write(f"Total flows: {sum(flow_counts.values())}\n")
        # protocol distribution (compact)
        if flow_counts:
            proto_parts = [f"{p}:{c}" for p, c in sorted(flow_counts.items())]
            g.write("Protocol distribution: " + ", ".join(proto_parts) + "\n")
        # IP overview
        g.write(f"Unique IPs: {len(seen_ips)}  •  Private: {len(private_ips)}  •  Public: {len(public_ips)}\n")
        # reputation overview
        g.write(f"Public IP reputation — Malicious: {rep_counts['malicious']}, "
                f"Benign: {rep_counts['benign']}, Unknown: {rep_counts['unknown']}\n")
        # application overview (only non-zero)
        if app_tally:
            apps = ", ".join(f"{k}:{v}" for k, v in sorted(app_tally.items()) if v > 0)
            g.write(f"Application hints: {apps if apps else 'None'}\n")
        # top talkers (compact)
        top_talkers = sorted(talkers.items(), key=lambda x: x[1], reverse=True)[:5]
        if top_talkers:
            g.write("Top talkers: " + " | ".join(f"{pair} ({cnt})" for pair, cnt in top_talkers) + "\n")
        g.write("================================\n")

    return (f"✅ Flow summary written:\n"
            f" - Full: {FLOW_SUMMARY_FILE}\n"
            f" - Succinct: {FLOW_SUMMARY_FILE_SUCCINCT}")