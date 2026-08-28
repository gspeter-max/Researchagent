# Research Review Agent

An autonomous research agent built with **Loop Engineering**, continuous context accumulation, verification threshold gating, and deterministic Human-in-the-Loop (HITL) steering.

## Architecture & Loop Engineering Flow

```
1. Initial Query & Alignment Checkpoint (Proceed / Cancel / Custom Focus)
   ↓
2. Query Formulation (user_query + state.findings + state.evaluation + state.feedback)
   ↓
3. Web Research & Information Retrieval
   ↓
4. Quality Verification & Scoring (0.0 to 1.0)
   ↓
5. Threshold Routing:
   ├── Score >= Threshold ──→ LLM Synthesizer ──→ Final Report
   └── Score < Threshold  ──→ Human-in-the-Loop (HITL):
                               ├── [1] Search More (+ Text Feedback) ──→ (Loops to Step 2)
                               ├── [2] Proceed Anyway (Override)     ──→ LLM Synthesizer
                               └── [3] Cancel                        ──→ Abort ($0 Cost)
```

## Features
- **Loop Engineering**: Continuous Reason-Act-Observe cycle with context accumulation across iterations.
- **Threshold Gating**: Automated scoring against sufficiency criteria with deterministic routing.
- **Cost-Optimized HITL**: Deterministic branching ($0$ token overhead for control actions) paired with targeted query reformulation.
- **Multi-Provider**: Works with Anthropic Claude, OpenAI, Gemini, or local heuristic engine with zero mandatory external dependencies.

## Usage

```bash
# Interactive run
python3 cli.py --topic "Post-Quantum Cryptography standards in 2026"

# Run with custom verification threshold
python3 cli.py --topic "Autonomous AI Agents" --threshold 0.85

# Offline / Mock search test
python3 cli.py --topic "Autonomous AI Agents" --mock-search

# Save report to markdown
python3 cli.py --topic "Fusion Energy Progress" --output fusion_report.md
```

## Running Tests
```bash
python3 -m unittest discover tests
```
