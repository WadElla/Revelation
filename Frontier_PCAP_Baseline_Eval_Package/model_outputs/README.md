# Model Outputs Folder

This folder contains the included ChatGPT PCAP-only outputs generated earlier from the strict PCAP-only prompt. The cleaned DOCX files remove the `Evidence used:` paragraphs and keep only the question text and answer.

For a new ChatGPT run, either:

1. Replace `canonical_data/chatgpt_pcap_only_predictions.csv` with a new prediction CSV using the same columns, or
2. Put new DOCX outputs in a separate folder and run:

```bash
python scripts/evaluate_frontier_pcap_baseline.py   --reference-csv canonical_data/reference_answers.csv   --prediction-csv /path/to/new_predictions.csv   --output-dir results/new_chatgpt_run   --model-name ChatGPT_New_Run
```

Keep reference answers separate from model outputs. Do not put gold QA files in this folder.
