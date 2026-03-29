# 🛡️ Grok-1 + Hancock — AI Cybersecurity Agent

This repository combines the **Grok-1 314B Mixture-of-Experts** language model with
**Hancock**, CyberViser's AI-powered cybersecurity agent for pentesting, SOC analysis,
incident response, and more.

---

## 🚀 Quick Start

### 1. Install dependencies

```shell
pip install -r requirements.txt
```

### 2. Run the original Grok-1 demo

Make sure to download the checkpoint and place the `ckpt-0` directory in `checkpoints` — see [Downloading the weights](#downloading-the-weights).

```shell
python run.py
```

### 3. Run the Hancock cybersecurity agent

**Interactive CLI (with Ollama backend):**

```shell
# Install and start Ollama first: https://ollama.com
export HANCOCK_LLM_BACKEND=ollama
python hancock_agent.py
```

**REST API server:**

```shell
python hancock_agent.py --server --port 5000
```

**With Grok-1 native backend (requires multi-GPU):**

```shell
python hancock_agent.py --backend grok
```

---

## 🤖 Hancock — Cybersecurity Modes

| Mode | Endpoint | Description |
|------|----------|-------------|
| 🔴 **Pentest** | `/v1/chat` (mode: pentest) | Recon, exploitation, CVE analysis, PTES reporting |
| 🔵 **SOC** | `/v1/triage`, `/v1/hunt`, `/v1/respond` | Alert triage, SIEM queries, PICERL incident response |
| ⚡ **Auto** | `/v1/chat` (mode: auto) | Context-aware pentest + SOC combined |
| 💻 **Code** | `/v1/code` | Security code: YARA, KQL, SPL, Sigma, Python |
| 👔 **CISO** | `/v1/ciso` | Compliance, risk reporting, board summaries |
| 🔍 **Sigma** | `/v1/sigma` | Sigma detection rule authoring |
| 🦠 **YARA** | `/v1/yara` | YARA malware detection rule authoring |
| 🔎 **IOC** | `/v1/ioc` | Threat intelligence enrichment |

### API Examples

**Alert Triage:**
```bash
curl -X POST http://localhost:5000/v1/triage \
  -H "Content-Type: application/json" \
  -d '{"alert": "Mimikatz detected on DC01 at 03:14 UTC"}'
```

**Threat Hunting (Splunk):**
```bash
curl -X POST http://localhost:5000/v1/hunt \
  -H "Content-Type: application/json" \
  -d '{"target": "lateral movement via PsExec", "siem": "splunk"}'
```

**Sigma Rule Generation:**
```bash
curl -X POST http://localhost:5000/v1/sigma \
  -H "Content-Type: application/json" \
  -d '{"description": "Detect LSASS memory dump", "logsource": "windows sysmon", "technique": "T1003.001"}'
```

**YARA Rule Generation:**
```bash
curl -X POST http://localhost:5000/v1/yara \
  -H "Content-Type: application/json" \
  -d '{"description": "Cobalt Strike beacon default HTTP profile", "file_type": "PE"}'
```

**IOC Enrichment:**
```bash
curl -X POST http://localhost:5000/v1/ioc \
  -H "Content-Type: application/json" \
  -d '{"indicator": "185.220.101.35", "type": "ip"}'
```

### CLI Commands

```
/mode pentest | soc | auto | code | ciso | sigma | yara | ioc
/clear          — clear conversation history
/history        — show history
/model <id>     — switch model
/exit           — quit
```

### Backends

| Backend | Env Var | Description |
|---------|---------|-------------|
| **grok** | `HANCOCK_LLM_BACKEND=grok` | Grok-1 native 314B MoE (requires GPU cluster) |
| **ollama** | `HANCOCK_LLM_BACKEND=ollama` | Local Ollama server (default) |
| **nvidia** | `HANCOCK_LLM_BACKEND=nvidia` + `NVIDIA_API_KEY` | NVIDIA NIM cloud API |
| **openai** | `HANCOCK_LLM_BACKEND=openai` + `OPENAI_API_KEY` | OpenAI API |

---

## 📋 File Structure

| File | Purpose |
|------|---------|
| `model.py` | Grok-1 model architecture (314B MoE, JAX) |
| `checkpoint.py` | Checkpoint loading and restoration |
| `runners.py` | Grok-1 inference runner and sampling |
| `run.py` | Original Grok-1 demo script |
| `hancock_agent.py` | Hancock cybersecurity agent (CLI + REST API) |
| `hancock_runner.py` | Bridge between Hancock and Grok-1 inference |
| `hancock_constants.py` | Shared constants and utilities |
| `tests/test_hancock.py` | Unit tests for Hancock modules |

---

# Model Specifications

Grok-1 is currently designed with the following specifications:

- **Parameters:** 314B
- **Architecture:** Mixture of 8 Experts (MoE)
- **Experts Utilization:** 2 experts used per token
- **Layers:** 64
- **Attention Heads:** 48 for queries, 8 for keys/values
- **Embedding Size:** 6,144
- **Tokenization:** SentencePiece tokenizer with 131,072 tokens
- **Additional Features:**
  - Rotary embeddings (RoPE)
  - Supports activation sharding and 8-bit quantization
- **Maximum Sequence Length (context):** 8,192 tokens

# Downloading the weights

You can download the weights using a torrent client and this magnet link:

```
magnet:?xt=urn:btih:5f96d43576e3d386c9ba65b883210a393b68210e&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce
```

or directly using [HuggingFace 🤗 Hub](https://huggingface.co/xai-org/grok-1):
```
git clone https://github.com/xai-org/grok-1.git && cd grok-1
pip install huggingface_hub[hf_transfer]
huggingface-cli download xai-org/grok-1 --repo-type model --include ckpt-0/* --local-dir checkpoints --local-dir-use-symlinks False
```

# Running Tests

```shell
pip install flask openai
python -m pytest tests/ -v
```

# License

The code and associated Grok-1 weights in this release are licensed under the
Apache 2.0 license. The license only applies to the source files in this
repository and the model weights of Grok-1.
