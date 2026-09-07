#!/usr/bin/env bash
set -euo pipefail

# From the package root:
python scripts/evaluate_frontier_pcap_baseline.py   --reference-csv canonical_data/reference_answers.csv   --prediction-csv canonical_data/chatgpt_pcap_only_predictions.csv   --output-dir results/chatgpt_pcap_only   --model-name ChatGPT_PCAP_only

# Optional, slower, and may require model download/cache:
# python scripts/evaluate_frontier_pcap_baseline.py #   --reference-csv canonical_data/reference_answers.csv #   --prediction-csv canonical_data/chatgpt_pcap_only_predictions.csv #   --output-dir results/chatgpt_pcap_only_bertscore #   --model-name ChatGPT_PCAP_only #   --use-bertscore
