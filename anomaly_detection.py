import os
import pandas as pd
import torch
from collections import Counter
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

def run_bert_anomaly_detector() -> str:
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)
    model_path = os.path.join(BASE_DIR, "saved_model")
    csv_path = os.path.join(BASE_DIR, "features", "test.csv")
    output_csv = os.path.join(BASE_DIR, "outputs", "unlabeled_results.csv")
    report_path = os.path.join(BASE_DIR, "outputs", "bert_traffic_report.txt")

    # Load model and tokenizer
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # Load dataset
    test_df = pd.read_csv(csv_path, low_memory=False)

    # Drop irrelevant or high-cardinality fields
    columns_to_remove = ['tcp.payload', 'http.file_data', 'arp.src.proto_ipv4']
    test_df.drop(columns=[col for col in columns_to_remove if col in test_df.columns], inplace=True)

    # Rename label column if necessary
    if 'Attack_label' in test_df.columns:
        test_df.rename(columns={'Attack_label': 'Label'}, inplace=True)

    # Build input text for BERT
    exclude_for_text = ['Label', 'Attack_type', 'ip.src_host', 'ip.dst_host']
    text_cols = [col for col in test_df.columns if col not in exclude_for_text]
    test_df['text'] = test_df[text_cols].apply(lambda row: ' '.join([f"{col}: {row[col]}" for col in text_cols]), axis=1)

    # Load labels
    has_labels = 'Label' in test_df.columns
    true_labels = test_df['Label'].tolist() if has_labels else None

    # Class mappings
    attack_label_map = {
        0: "Normal", 1: "MITM", 2: "Fingerprinting", 3: "Ransomware", 4: "Uploading",
        5: "SQL Injection", 6: "DDoS_HTTP", 7: "DDoS_TCP", 8: "Password", 9: "Port Scanning",
        10: "Vul Scanner", 11: "Backdoor", 12: "XSS", 13: "DDoS_UDP", 14: "DDoS_ICMP"
    }

    attack_categories = {
        1: ("MITM", "Interception / Impersonation", "High", "Man-in-the-Middle (MITM) attacks attempt to intercept or alter communications between devices, often through ARP spoofing or rogue gateways.", "Sensitive data, including credentials or configuration commands, can be silently stolen or manipulated.", "Isolate the affected subnet, inspect ARP tables and DHCP leases, and enforce certificate validation for encrypted traffic."),
        2: ("Fingerprinting", "Reconnaissance", "Low", "The attacker actively profiles connected devices or services to determine software versions, open ports, and vulnerabilities.", "This is often the prelude to a targeted exploit or vulnerability scan.", "Restrict unnecessary service exposure, use device fingerprint obfuscation where possible, and log abnormal scan behavior."),
        3: ("Ransomware", "Malware Deployment", "Critical", "Malicious code designed to encrypt device data or lock access until a ransom is paid, possibly spreading laterally across the network.", "Can disrupt industrial systems or critical home infrastructure, causing major data loss or service outages.", "Immediately isolate affected hosts, restore from clean backups, and assess lateral movement. Report incidents if needed."),
        4: ("Uploading", "Unauthorized File Transfer", "Medium", "Traffic patterns consistent with unexpected file uploads, possibly exploiting unsecured endpoints (e.g., FTP, HTTP PUT).", "Could result in remote code execution or storage of malicious tools on IoT gateways.", "Audit upload endpoints, enforce authentication, inspect logs for unusual payloads, and restrict executable file transfers."),
        5: ("SQL Injection", "Injection Exploits", "High", "Malicious SQL statements are injected into application inputs to exfiltrate data or manipulate backend databases.", "Can compromise user records, leak credentials, or allow remote access to control systems.", "Sanitize all input fields, apply least privilege DB access, and use parameterized queries or ORMs."),
        6: ("DDoS_HTTP", "Volumetric Denial-of-Service", "High", "Large volumes of HTTP requests target web servers or REST APIs to exhaust resources.", "May lead to downtime of dashboards, control panels, or external APIs used by IoT systems.", "Deploy rate limiting, reverse proxies, or cloud-based DDoS protection. Identify attacker IPs for blacklisting."),
        7: ("DDoS_TCP", "Connection Flooding", "High", "TCP SYN/ACK floods or spoofed session establishment to overwhelm TCP stacks.", "Often affects routers, embedded Linux devices, or gateway firewalls, causing memory exhaustion.", "Enable SYN cookies, tune OS TCP parameters, and monitor for high-rate IPs with excessive half-open connections."),
        8: ("Password", "Credential Brute Force", "Medium", "Numerous login attempts with different passwords, targeting SSH, Telnet, HTTP, or MQTT services.", "Can lead to device hijacking or access to administrative interfaces if default or weak passwords are used.", "Enforce strong credentials, lock accounts after failures, and monitor authentication logs for abuse patterns."),
        9: ("Port Scanning", "Service Discovery", "Low", "Sequential or randomized probes across TCP/UDP ports to discover active services.", "Indicates an attacker is mapping the network to find exploitable targets.", "Block unsolicited scanning at the perimeter, use port knockers or segmentation to protect sensitive devices."),
        10: ("Vul Scanner", "Exploit Scanning", "Medium", "Automated tools (e.g., Nmap NSE, Nikto, Nessus) scan for known vulnerabilities or misconfigurations.", "These tools often identify weak firmware, default credentials, or exposed admin panels.", "Patch exposed services, disable unused ports, and fingerprint the scanning tool for possible attribution."),
        11: ("Backdoor", "Persistent Access", "Critical", "Hidden channels or implants enabling an attacker to regain access even after reboots or remediation.", "Can allow full device control and long-term compromise of the IoT environment.", "Hunt for unrecognized binaries, reverse shells, or custom beaconing behavior. Reflash firmware if needed."),
        12: ("XSS", "Web Exploits", "Medium", "Malicious JavaScript is injected into web-based IoT dashboards or user-facing interfaces.", "Can hijack sessions, modify settings, or exfiltrate sensitive values from browser sessions.", "Sanitize all inputs and outputs on web UIs. Apply Content Security Policies (CSP) and browser hardening."),
        13: ("DDoS_UDP", "Amplification / Flooding", "High", "High-speed UDP floods, often spoofed, target vulnerable services like NTP, DNS, or SSDP.", "Can saturate bandwidth, disrupt video streams, or bring down IoT controllers.", "Disable unused UDP services, rate-limit critical ports, and monitor for abnormal packet size bursts."),
        14: ("DDoS_ICMP", "Ping Flood / Tunnel Abuse", "High", "High-frequency ICMP traffic including echo requests or ICMP tunneling for covert channels.", "Impacts CPU usage, routing tables, or can be used to bypass firewalls.", "Rate-limit ICMP, block unneeded types (e.g., redirect, router discovery), and alert on anomalous RTTs.")
    }

    # Prediction loop
    predictions, results = [], []
    for _, row in test_df.iterrows():
        inputs = tokenizer(row['text'], return_tensors='pt', truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        pred = logits.argmax().item()
        predictions.append(pred)
        result = {
            'text': row['text'],
            'predicted_label': pred,
            'predicted_attack_type': attack_label_map[pred],
            'ip.src_host': row.get('ip.src_host', 'N/A'),
            'ip.dst_host': row.get('ip.dst_host', 'N/A'),
            'mqtt.topic': row.get('mqtt.topic', 'N/A'),
            'dns.qry.name': row.get('dns.qry.name', 'N/A'),
            'mbtcp.unit_id': row.get('mbtcp.unit_id', 'N/A'),
            'frame.time': row.get('frame.time', 'N/A')
        }
        if has_labels:
            result['true_label'] = row['Label']
        results.append(result)

    result_df = pd.DataFrame(results)
    result_df.sort_values(by="predicted_attack_type", inplace=True)
    result_df.to_csv(output_csv, index=False)

    if has_labels:
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
        _ = classification_report(
            true_labels,
            predictions,
            labels=list(range(15)),
            target_names=[attack_label_map[i] for i in range(15)],
            zero_division=0
        )

    pred_counts = Counter(predictions)
    total = len(predictions)
    normal_count = pred_counts.get(0, 0)
    anomaly_count = total - normal_count
    anomaly_percent = (anomaly_count / total) * 100

    report_lines = []
    report_lines.append("==== BERT-Based IoT Traffic Interpretation Report ====\n")
    report_lines.append(f"Total traffic samples analyzed: {total}")
    report_lines.append(f"Normal traffic: {normal_count} ({normal_count / total:.2%})")
    report_lines.append(f"Anomalous traffic: {anomaly_count} ({anomaly_percent:.2f}%)\n")

    report_lines.append("=== Category-wise Breakdown and Analysis ===\n")
    for class_id in range(15):
        count = pred_counts.get(class_id, 0)
        if count == 0:
            continue
        pct = (count / anomaly_count) * 100
        if class_id == 0:
            report_lines.append(f"[Normal]\n- Count: {count} ({pct:.2f}%)\n- No action needed. Traffic appears regular.\n")
        else:
            name, category, severity, behavior, implication, recommendation = attack_categories[class_id]
            report_lines.append(f"[{name}]\nThis section provides a summary of how {name} activity manifested in the analyzed traffic. A total of {count} samples ({pct:.2f}% of traffic) were classified under this attack type. The summary includes its behavioral signature, severity level, potential impact on IoT infrastructure, and recommended mitigation strategies.\n- Count: {count} ({pct:.2f}%)\n- Category: {category}\n- Severity: {severity}\n- Behavior: {behavior}\n- Risk: {implication}\n- Recommendation: {recommendation}\n")

    report_lines.append("=== Interpretation Summary ===")
    if anomaly_count == 0:
        report_lines.append("✅ No anomalies detected. The traffic appears clean.")
    elif anomaly_percent < 5:
        report_lines.append("🟡 Low anomaly rate detected. Possibly benign or low-risk activities.")
    elif anomaly_percent < 20:
        report_lines.append("🟠 Moderate anomaly level. Suggests potential early-stage attacks.")
    else:
        report_lines.append("🔴 High anomaly volume detected! Immediate investigation recommended.")

    dominant = max([(cid, cnt) for cid, cnt in pred_counts.items() if cid != 0], key=lambda x: x[1], default=None)
    if dominant:
        dom_id, dom_count = dominant
        dom_name = attack_label_map[dom_id]
        dom_pct = (dom_count / anomaly_count) * 100
        report_lines.append(f"\n🛑 Dominant Detected Attack: {dom_name} ({dom_count} samples, {dom_pct:.2f}%)")

    # Endpoint metadata per attack
    report_lines.append("\n=== Unique Endpoints and Metadata per Attack ===\n")
    for class_id in sorted(pred_counts.keys()):
        if class_id == 0:
            continue
        name = attack_label_map[class_id]
        attack_df = result_df[result_df['predicted_label'] == class_id]
        unique_conns = set()
        unique_mqtt = set()
        unique_dns = set()
        unique_modbus = set()

        for _, row in attack_df.iterrows():
            src = row.get('ip.src_host', 'N/A')
            dst = row.get('ip.dst_host', 'N/A')
            if pd.notna(src) and pd.notna(dst):
                unique_conns.add(f"{src} → {dst}")
            if pd.notna(row.get('mqtt.topic')) and row.get('mqtt.topic') != 'N/A':
                unique_mqtt.add(row['mqtt.topic'])
            if pd.notna(row.get('dns.qry.name')) and row.get('dns.qry.name') != 'N/A':
                unique_dns.add(row['dns.qry.name'])
            if pd.notna(row.get('mbtcp.unit_id')) and row.get('mbtcp.unit_id') != 'N/A':
                unique_modbus.add(str(row['mbtcp.unit_id']))

        report_lines.append(f"[{name}]")
        report_lines.append(f"This section summarizes all unique network endpoints and protocol-level metadata observed during {name} activity. It includes IP communication pairs, MQTT topics, DNS queries, and Modbus Unit IDs involved in this category of traffic.")
        report_lines.append(f"- Unique IP Pairs: {len(unique_conns)}")
        for pair in sorted(unique_conns):
            report_lines.append(f"  - {pair}")
        if unique_mqtt:
            report_lines.append(f"- MQTT Topics: {len(unique_mqtt)}")
            for topic in sorted(unique_mqtt):
                report_lines.append(f"  - {topic}")
        if unique_dns:
            report_lines.append(f"- DNS Queries: {len(unique_dns)}")
            for query in sorted(unique_dns):
                report_lines.append(f"  - {query}")
        if unique_modbus:
            report_lines.append(f"- Modbus Unit IDs: {len(unique_modbus)}")
            for unit_id in sorted(unique_modbus):
                report_lines.append(f"  - {unit_id}")
        report_lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    return f"✅ Reports generated.\nCSV: {output_csv}\nTXT: {report_path}"
