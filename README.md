# Dify DSL Generator

A self-healing Dify workflow/chat-flow YAML generator with strict validation.

## Features

- **Validator Engine** — 25+ structural & semantic checks (node/edge connectivity, variable references, value selectors, node-type-specific rules, graph reachability)
- **Dual Mode** — supports both `workflow` and `chat` (advanced-chat) Dify app modes
- **Self-healing Loop** — calls Anthropic API to generate YAML, validates it, feeds errors back to the LLM for auto-correction (max 50 iterations)
- **Comprehensive Test Suite** — 6 positive + 19 negative test cases covering all supported node types

## Quick Start

```bash
pip install pyyaml anthropic
python dify_dsl_generator.py          # runs test suite (no API key needed)
python dify_dsl_generator.py --test   # runs test suite explicitly
```

For full generation mode:

```bash
export ANTHROPIC_API_KEY='sk-...'
python dify_dsl_generator.py
```

## Supported Node Types

`start`, `end`, `llm`, `code`, `if-else`, `iteration`, `knowledge-retrieval`, `http-request`, `answer`, `template-transform`, `variable-aggregator`, `parameter-extractor`, `question-classifier`, `tool`, `variable-assigner`

## Demo YAMLs

- `demo_workflow_clean_json.yml` — start → code → iteration(llm) → end
- `demo_chat_flow.yml` — chat mode: start → llm → answer
- `demo_workflow_ifelse.yml` — branching with if-else
- `demo_workflow_knowledge.yml` — knowledge retrieval pipeline
- `demo_workflow_http.yml` — HTTP API integration workflow
