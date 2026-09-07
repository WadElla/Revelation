# Frontier PCAP-Only Baseline Evaluation Package

This package is organized for the ChatGPT/frontier-model PCAP-only baseline experiment. It keeps benchmark questions, gold/reference answers, packet captures, and model outputs separate so the evaluation setup does not conflate them.

## Decisive experimental setup

- **Gold/reference answers:** `reference_answers/*_reference_answers.docx`, converted into `canonical_data/reference_answers.csv` for scoring.
- **Benchmark questions:** `questions/*_questions.docx`, converted into `canonical_data/questions.csv`.
- **Packet captures:** `pcaps/*.pcap`, renamed by scenario.
- **Included model output:** `model_outputs/chatgpt_pcap_only_docx/ChatGPT_PCAPOnly_*_answers.docx`, also converted into `canonical_data/chatgpt_pcap_only_predictions.csv`.
- **Evaluation code:** `scripts/evaluate_frontier_pcap_baseline.py`.

The current uploaded benchmark contains **161 QA pairs**:

- Backdoor: 41 questions
- DDoS: 40 questions
- Normal: 40 questions
- Uploading: 40 questions

This package does not silently drop or merge questions.

## Scenario mapping

| Scenario name | Original question file | Original reference file | Original PCAP | Included ChatGPT PCAP-only output |
|---|---|---|---|---|
| Backdoor | Scenario_A_questions.docx | Backdoor_test3.docx | Scenario_A.pcap | Scenario_A_PCAP_Evidence_Answers(1).docx |
| DDoS | Scenario_B_questions.docx | DDoS_test3.docx | Scenario_B.pcap | Scenario_B_PCAP_Evidence_Answers(1).docx |
| Normal | Scenario_C_questions.docx | Normal_test3.docx | Scenario_C.pcap | Scenario_C_PCAP_Evidence_Answers(1).docx |
| Uploading | Scenario_D_questions.docx | uploading_test3.docx | Scenario_D.pcap | Scenario_D_PCAP_Evidence_Answers(1).docx |

## What the evaluation does

The script compares already-generated model answers against the gold/reference QA answers. It does **not** run Revelation, Chroma, RAG, Zeek, BERT, or any LLM. This is intentional because this experiment is a PCAP-only frontier baseline.

The outputs include:

- `per_question_results.csv`
- `overall_summary.csv`
- `scenario_summary.csv`
- `category_summary.csv`
- `answerability_analysis.csv`
- `paper_summary.txt`

## Run

From the package root:

```bash
pip install -r requirements.txt
bash run_default_evaluation.sh
```

For optional BERTScore, run:

```bash
python scripts/evaluate_frontier_pcap_baseline.py   --reference-csv canonical_data/reference_answers.csv   --prediction-csv canonical_data/chatgpt_pcap_only_predictions.csv   --output-dir results/chatgpt_pcap_only_bertscore   --model-name ChatGPT_PCAP_only   --use-bertscore
```

## Important interpretation for the paper

This experiment should be described as a **frontier PCAP-only baseline**. The reference answers are based on the full benchmark QA files, which include answers from Revelation artifacts such as anomaly reports, flow summaries, Zeek-style logs, threat-intelligence summaries, and protocol records. The included ChatGPT PCAP-only outputs were produced under a stricter setting that asks the model to answer only from the packet capture. Therefore, many `cannot be determined` answers are not merely language mismatches; they reveal which benchmark questions require structured artifacts beyond raw PCAP inspection.

Use `answerability_analysis.csv` and `category_summary.csv` together with BLEU/ROUGE/METEOR/BERTScore to write a fair analysis.
