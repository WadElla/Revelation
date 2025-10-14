import os
import json
import re
import csv
import ipaddress
import requests
from time import sleep
from typing import Dict
from datetime import datetime, timezone


VT_API_KEY = "Put your VirusTotal API Key here"
# This sets the base path relative to *this file's* location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_INPUT_REPORT = os.path.join(BASE_DIR, "outputs", "bert_traffic_report.txt")
ENHANCER_DIR = os.path.join(BASE_DIR, "enhancer")


os.makedirs(ENHANCER_DIR, exist_ok=True)

CSV_OUTPUT = os.path.join(ENHANCER_DIR, "malicious_public_ips.csv")
TXT_IP_LIST = os.path.join(ENHANCER_DIR, "enrichment_targets.txt")
JSON_CONTEXT_MAP = os.path.join(ENHANCER_DIR, "ip_context_map.json")
ENRICHED_JSON = os.path.join(ENHANCER_DIR, "ip_enrichment.json")
ENHANCED_REPORT = os.path.join(ENHANCER_DIR, "enhanced_bert_report.txt")
ENHANCED_REPORT_SUCCINCT = os.path.join(ENHANCER_DIR, "enhanced_bert_report_succinct.txt")



def is_public_ip(ip_str):
    try:
        ip_obj = ipaddress.ip_address(ip_str)
        return (
            ip_obj.version == 4 and
            ip_obj.is_global and
            not ip_obj.is_multicast and
            not ip_obj.is_loopback and
            not ip_obj.is_reserved and
            not ip_obj.is_link_local
        )
    except ValueError:
        return False


def extract_public_ips_from_report(report_path, csv_output_path, ip_list_output_path, context_map_output_path):
    public_ip_records = []
    unique_public_ips = set()
    ip_context_map = {}

    current_attack_type = None
    capture_ips = False

    with open(report_path, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("[") and line.endswith("]") and line not in ("[Normal]", "[Interpretation Summary]"):
                current_attack_type = line.strip("[]")
                capture_ips = False

            if "Unique IP Pairs:" in line and current_attack_type:
                capture_ips = True
                continue

            if any(proto in line for proto in ("MQTT Topics:", "DNS Queries:", "Modbus Unit IDs:")):
                capture_ips = False

            if capture_ips and "→" in line:
                parts = [ip.strip() for ip in line.split("→")]
                if len(parts) == 2:
                    src_ip, dst_ip = parts
                    src_is_public = is_public_ip(src_ip)
                    dst_is_public = is_public_ip(dst_ip)

                    if src_is_public or dst_is_public:
                        public_ip_records.append((current_attack_type, src_ip, dst_ip))
                        if src_is_public:
                            unique_public_ips.add(src_ip)
                            ip_context_map.setdefault(src_ip, []).append({"attack": current_attack_type, "role": "source"})
                        if dst_is_public:
                            unique_public_ips.add(dst_ip)
                            ip_context_map.setdefault(dst_ip, []).append({"attack": current_attack_type, "role": "destination"})

    with open(csv_output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Attack_Type", "Source_IP", "Destination_IP"])
        writer.writerows(public_ip_records)

    with open(ip_list_output_path, "w") as ipfile:
        for ip in sorted(unique_public_ips):
            ipfile.write(ip + "\n")

    with open(context_map_output_path, "w") as jsonfile:
        json.dump(ip_context_map, jsonfile, indent=2)


def query_virustotal(ip, api_key):
    url = f"https://www.virustotal.com/api/v3/ip_addresses/{ip}"
    headers = {"x-apikey": api_key}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        stats = data.get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
        last_seen = data.get("data", {}).get("attributes", {}).get("last_analysis_date", None)
        return {
            "was_marked_malicious_by_any_engine": stats.get("malicious", 0) > 0,
            "number_of_engines_flagging_as_malicious": stats.get("malicious", 0),
            "total_number_of_engines_checked": sum(stats.values()),
            "last_analysis_unix_timestamp": last_seen
        }
    return None


def query_internetdb(ip):
    url = f"https://internetdb.shodan.io/{ip}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                "open_tcp_udp_ports_observed": data.get("ports", []),
                "inferred_host_tags": data.get("tags", []),
                "known_cve_vulnerabilities": data.get("vulns", []),
                "associated_software_cpes": data.get("cpes", []),
                "dns_hostnames_observed": data.get("hostnames", [])
            }
        return {"error": "No data found"}
    except Exception as e:
        return {"error": str(e)}


def enrich_public_ips(ip_list_path, context_map_path, output_path, vt_api_key, delay=2):
    with open(ip_list_path, "r") as f:
        ip_list = [line.strip() for line in f if line.strip()]

    with open(context_map_path, "r") as f:
        ip_context_map = json.load(f)

    enriched_data = {}

    for ip in ip_list:
        ip_data = {"attack_context_categories": ip_context_map.get(ip, [])}
        vt_info = query_virustotal(ip, vt_api_key)
        if vt_info:
            ip_data["virustotal_threat_intel"] = vt_info
        shodan_info = query_internetdb(ip)
        if shodan_info:
            ip_data["shodan_internetdb_observations"] = shodan_info
        enriched_data[ip] = ip_data
        sleep(delay)

    with open(output_path, "w") as f:
        json.dump(enriched_data, f, indent=2)


def summarize_enrichment(ip: str, data: Dict) -> str:
    vt = data.get("virustotal_threat_intel", {})
    shodan = data.get("shodan_internetdb_observations", {})
    parts = []

    if vt:
        malicious = vt.get("was_marked_malicious_by_any_engine", False)
        positives = vt.get("number_of_engines_flagging_as_malicious", 0)
        total = vt.get("total_number_of_engines_checked", 0)
        last_seen_raw = vt.get("last_analysis_unix_timestamp", "N/A")
        try:
            last_seen_int = int(last_seen_raw)
            last_seen = datetime.fromtimestamp(last_seen_int, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        except (TypeError, ValueError):
            last_seen = "N/A"
        verdict = "[malicious]" if malicious else "[clean]"
        vt_summary = (
            f"IP {ip} is associated with {verdict} activity — VirusTotal identified this IP as {'suspicious' if malicious else 'benign'} based on consensus across multiple antivirus and threat intelligence engines. "
            f"{positives} out of {total} engines flagged it as potentially harmful. Last known analysis occurred on {last_seen}."
        )
        parts.append(vt_summary)

    if shodan:
        tags = ", ".join(shodan.get("inferred_host_tags", [])) or "none"
        ports = shodan.get("open_tcp_udp_ports_observed", [])
        cves = shodan.get("known_cve_vulnerabilities", [])
        cpes = shodan.get("associated_software_cpes", [])
        dns = ", ".join(shodan.get("dns_hostnames_observed", [])) or "N/A"
        if tags != "none":
            parts.append(f"  • Tags: {tags} — inferred descriptors for the host based on observed behavior or services.")
        if ports:
            parts.append(f"  • Open Ports: {sorted(ports)} — network services currently reachable on this host, which may expose it to attacks.")
        if cves:
            parts.append(f"  • Known CVEs: {', '.join(cves)} — publicly disclosed vulnerabilities possibly affecting this system.")
        if cpes:
            parts.append(f"  • Software CPEs: {', '.join(cpes)} — detected software components and versions, useful for vulnerability assessment.")
        if dns != "N/A":
            parts.append(f"  • DNS Hostnames: {dns} — domain names linked to this IP address, which may help in attribution or lookup.")

    return f"🧠 Threat Intelligence Summary (from VirusTotal and Shodan/InternetDB):\n- " + "\n".join(parts) if parts else ""


def build_succinct_enrichment_summary(enriched_json_path: str, original_report_path: str, output_path: str) -> None:
    """
    Build a succinct, human-readable anomaly summary derived from the BERT report and enrichment data.
    Sections:
      1) Overall Classification Results
      2) Attack Distribution
      3) Public IP Insights (with verdict -> IP lists)
      4) Attacks Involving Public IPs
    Values are computed to match the main report's calculations (no hardcoding).
    """
    # ---------------------------
    # Load entire BERT report text
    # ---------------------------
    report_text = ""
    if os.path.isfile(original_report_path):
        with open(original_report_path, "r", encoding="utf-8") as f:
            report_text = f.read()

    # ---------------------------
    # Robust overall counts
    # ---------------------------
    def _parse_overall_counts(txt: str):
        total = 0
        normal = 0
        anomalous = 0

        m_total = re.search(r"Total (?:traffic )?samples analyzed:\s*([\d,]+)", txt, flags=re.I)
        if m_total:
            try:
                total = int(m_total.group(1).replace(",", ""))
            except Exception:
                total = 0

        m_norm = re.search(r"Normal traffic:\s*([\d,]+)", txt, flags=re.I)
        if m_norm:
            try:
                normal = int(m_norm.group(1).replace(",", ""))
            except Exception:
                normal = 0
        else:
            # fallback from [Normal] block if present
            m_norm_count = re.search(r"\[Normal\][\s\S]*?-+\s*Count:\s*([\d,]+)", txt, flags=re.I)
            if m_norm_count:
                try:
                    normal = int(m_norm_count.group(1).replace(",", ""))
                except Exception:
                    normal = 0

        m_anom = re.search(r"Anomalous traffic:\s*([\d,]+)", txt, flags=re.I)
        if m_anom:
            try:
                anomalous = int(m_anom.group(1).replace(",", ""))
            except Exception:
                anomalous = 0
        else:
            if total and normal <= total:
                anomalous = total - normal

        if total == 0 and (normal or anomalous):
            total = normal + anomalous

        return total, normal, anomalous

    total_samples, normal_samples, anomalous_samples = _parse_overall_counts(report_text)

    # ---------------------------
    # Attack counts from '- Count: N' inside each [Attack] block
    # ---------------------------
    def _parse_attack_counts(txt: str):
        counts = {}
        pattern = re.compile(
            r"\[(?P<attack>[A-Za-z0-9_ /-]+)\]\s*"
            r"(?:(?!\[[^\]]+\]).)*?"        # within this block
            r"-\s*Count:\s*(?P<count>[\d,]+)",
            flags=re.I | re.S
        )
        for m in pattern.finditer(txt):
            attack = m.group("attack").strip()
            if attack.lower() in {"normal", "interpretation summary"}:
                continue
            try:
                c = int(m.group("count").replace(",", ""))
            except Exception:
                continue
            counts[attack] = counts.get(attack, 0) + c
        return counts

    attack_counts = _parse_attack_counts(report_text)

    # ---------------------------
    # Load enrichment (public IPs, ports, tags, CVEs, attack→IP map)
    # ---------------------------
    public_ip_verdict_counts = {"malicious": 0, "benign": 0, "unknown": 0}
    verdict_ip_map = {"malicious": set(), "benign": set(), "unknown": set()}
    attacks_to_ips = {}  # attack -> set(ip)
    ports, tags, cves = set(), set(), set()

    if os.path.isfile(enriched_json_path):
        with open(enriched_json_path, "r", encoding="utf-8") as f:
            enrichment = json.load(f)
    else:
        enrichment = {}

    for ip, details in enrichment.items():
        vt = details.get("virustotal_threat_intel", {}) or {}
        total_checked = int(vt.get("total_number_of_engines_checked", 0) or 0)
        malicious = bool(vt.get("was_marked_malicious_by_any_engine", False))

        if malicious:
            verdict = "malicious"
        elif total_checked > 0:
            verdict = "benign"
        else:
            verdict = "unknown"

        public_ip_verdict_counts[verdict] += 1
        verdict_ip_map[verdict].add(ip)

        sh = details.get("shodan_internetdb_observations", {}) or {}
        for p in sh.get("open_tcp_udp_ports_observed", []) or []:
            ports.add(str(p))
        for t in sh.get("inferred_host_tags", []) or []:
            tags.add(t)
        for cve in sh.get("known_cve_vulnerabilities", []) or []:
            cves.add(cve)

        for ctx in details.get("attack_context_categories", []) or []:
            atk = (ctx.get("attack") or "Unlabeled").strip()
            attacks_to_ips.setdefault(atk, set()).add(ip)

    total_public_ips = sum(public_ip_verdict_counts.values())

    # ---------------------------
    # Helpers (local)
    # ---------------------------
    def pct(part: int, whole: int) -> float:
        try:
            return round((part / whole) * 100, 2) if whole else 0.0
        except ZeroDivisionError:
            return 0.0

    def join_list(iterable, limit=None):
        items = sorted(set(iterable))
        if limit and len(items) > limit:
            return ", ".join(items[:limit]) + f", +{len(items) - limit} more"
        return ", ".join(items) if items else "N/A"

    # ---------------------------
    # Write succinct summary file (same format, computed)
    # ---------------------------
    with open(output_path, "w", encoding="utf-8") as g:
        g.write("==== Succinct Anomaly & Threat Intelligence Summary ====\n\n")

        # 1) Overall Classification Results
        g.write("📊 Overall Classification Results\n")
        g.write(f"- Total traffic samples analyzed: {total_samples if total_samples else 'N/A'}\n")
        g.write(f"- Normal traffic: {normal_samples} ({pct(normal_samples, total_samples)}%)\n")
        g.write(f"- Anomalous traffic: {anomalous_samples} ({pct(anomalous_samples, total_samples)}%)\n\n")

        # 2) Attack Distribution
        if attack_counts:
            g.write("🧠 Attack Distribution\n")
            total_attacks = sum(attack_counts.values())
            for attack, count in sorted(attack_counts.items(), key=lambda x: x[1], reverse=True):
                g.write(f"- {attack}: {count} samples ({pct(count, total_attacks)}%)\n")
            g.write("\n")

        # 3) Public IP Insights (with verdict -> IP lists)
        g.write("🌐 Public IP Insights\n")
        g.write(f"- Total public IPs observed: {total_public_ips}\n")
        g.write("- VirusTotal verdicts:\n")
        g.write(f"    • Malicious ({public_ip_verdict_counts['malicious']}): {join_list(verdict_ip_map['malicious'])}\n")
        g.write(f"    • Benign ({public_ip_verdict_counts['benign']}): {join_list(verdict_ip_map['benign'])}\n")
        g.write(f"    • Unknown ({public_ip_verdict_counts['unknown']}): {join_list(verdict_ip_map['unknown'])}\n")
        if ports:
            g.write(f"- Frequent open ports: {join_list(ports)}\n")
        if tags:
            g.write(f"- Common host tags: {join_list(tags)}\n")
        if cves:
            g.write(f"- Associated CVEs: {join_list(cves)}\n")
        g.write("\n")

        # 4) Attacks Involving Public IPs
        if attacks_to_ips:
            g.write("🧩 Attacks Involving Public IPs\n")
            for atk, ips in sorted(attacks_to_ips.items(), key=lambda x: len(x[1]), reverse=True):
                g.write(f"- {atk}: {join_list(ips)}\n")

        g.write("\n========================================================\n")


def enhance_bert_report(original_path: str, enrichment_path: str, output_path: str):
    with open(original_path, 'r', encoding='utf-8') as f:
        text = f.read()
    with open(enrichment_path, 'r', encoding='utf-8') as f:
        enrichment_data = json.load(f)

    attack_intel_map = {}
    for ip, data in enrichment_data.items():
        attack_entries = data.get("attack_context_categories", [])
        for entry in attack_entries:
            attack = entry.get("attack")
            if attack:
                summary = summarize_enrichment(ip, data)
                if summary:
                    attack_intel_map.setdefault(attack, []).append(summary)

    def patch_metadata_block(match):
        attack = match.group("attack").strip()
        block = match.group(0).strip()
        if attack in attack_intel_map:
            enriched_block = [block, "- Threat Intelligence:"]
            for s in attack_intel_map[attack]:
                enriched_block.extend(line.strip() for line in s.strip().splitlines())
            return "\n".join(enriched_block) + "\n\n"
        return block + "\n\n"

    updated_text = re.sub(r"\[(?P<attack>[A-Za-z0-9_ ]+)]\n.*?metadata observed.*?(?=(\n\[|\Z))", patch_metadata_block, text, flags=re.DOTALL)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(updated_text)


def enrich_iot_report() -> str:
    extract_public_ips_from_report(DEFAULT_INPUT_REPORT, CSV_OUTPUT, TXT_IP_LIST, JSON_CONTEXT_MAP)
    enrich_public_ips(TXT_IP_LIST, JSON_CONTEXT_MAP, ENRICHED_JSON, VT_API_KEY)
    enhance_bert_report(DEFAULT_INPUT_REPORT, ENRICHED_JSON, ENHANCED_REPORT)
    # NEW: succinct summary for agent context
    build_succinct_enrichment_summary(ENRICHED_JSON, DEFAULT_INPUT_REPORT, ENHANCED_REPORT_SUCCINCT)
    return (f"✅ Report enhanced with threat intel.\n"
            f" - Full text: {ENHANCED_REPORT}\n"
            f" - Succinct:  {ENHANCED_REPORT_SUCCINCT}")