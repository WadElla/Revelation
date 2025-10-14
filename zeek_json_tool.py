import subprocess
import tempfile
import glob
import os
from typing import List


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PCAP_PATH = os.path.join(BASE_DIR, "data", "Backdoor_attack.pcap")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "zeek_json_outputs")

def run_zeek_log_generator() -> str:
    if not os.path.isfile(DEFAULT_PCAP_PATH):
        raise FileNotFoundError(f"PCAP file not found: {DEFAULT_PCAP_PATH}")

    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    generated_files: List[str] = []

    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = ["zeek", "-C", "-r", DEFAULT_PCAP_PATH, "LogAscii::use_json=T"]
        result = subprocess.run(cmd, cwd=temp_dir, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Zeek error: {result.stderr}")

        log_files = sorted(glob.glob(os.path.join(temp_dir, "*.log")))
        for log_file in log_files:
            base = os.path.splitext(os.path.basename(log_file))[0]
            json_path = os.path.join(DEFAULT_OUTPUT_DIR, f"{base}.json")

            with open(log_file, "r") as lf, open(json_path, "w") as jf:
                jf.write('[\n')
                first = True
                for line in lf:
                    line = line.strip()
                    if not line:
                        continue
                    if not first:
                        jf.write(',\n')
                    jf.write(line)
                    first = False
                jf.write('\n]\n')

            generated_files.append(json_path)

    return f"✅ Zeek logs generated and saved to: {DEFAULT_OUTPUT_DIR}\n" + "\n".join(generated_files)
