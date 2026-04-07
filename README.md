# Bargaining Agent (Purple)

**AgentBeats Competition Submission: Purple Agent for Multi-Agent Negotiation**

A purple agent that negotiates item allocations in the AgentBeats bargaining meta-game. Uses Mistral LLM with chain-of-thought reasoning, code-level safety validation (M1–M5), and persistent per-opponent memory.

---

## Competition Context

This agent competes in the **AgentBeats Meta-Game Negotiation** scenario. A green evaluator agent pits purple agents against a pool of baseline strategies and RL-trained policies, then computes Maximum Entropy Nash Equilibrium to rank them.

| Property | Value |
|----------|-------|
| **Agent Type** | Purple (Challenger) |
| **Domain** | Multi-agent negotiation / bargaining |
| **LLM** | Mistral (`mistral-small-latest` or `mistral-large-latest`) |
| **Protocol** | A2A (Agent-to-Agent) |
| **Goal** | Maximize Nash Welfare: `√(your_payoff × opponent_payoff)` |

### Game Rules

- 3 item types with fixed quantities: `[7, 4, 1]`
- Each player has private valuations and a private BATNA (fallback payoff)
- Players alternate turns: propose allocations or accept/reject offers
- Value decays each round by a discount factor (`γ < 1`)
- If no agreement is reached, both sides receive only their BATNA

---

## How It Works

```
Green agent sends observation (JSON via A2A)
        │
        ▼
  _parse_observation()          ── extract JSON from message
        │
        ▼
  _build_situation()            ── enrich with valuations, BATNA,
        │                          discount, cheapest items, EF1 status
        ▼
  Mistral LLM (chain-of-thought) ── <think>reasoning</think> then JSON
        │
        ▼
  _validate_and_fix()           ── enforce M1–M5 rules at code level
        │
        ▼
  JSON response to green agent
```

### Decision Flow

1. **Context enrichment** — The agent builds a detailed `[SITUATION]` block containing:
   - Computed item values and total worth if keeping all
   - BATNA comparison and discount pressure
   - Items sorted by personal value (cheapest to offer first for Nash Welfare)
   - EF1 status of opponent's offer
   - Lessons from past games against this opponent (if any)

2. **LLM reasoning** — Mistral receives the observation + situation block and reasons step-by-step inside `<think>` tags before producing a JSON decision. The system prompt explicitly instructs the agent to maximize Nash Welfare and prefer EF1-compatible allocations.

3. **Safety validation** — Code checks the LLM output for M1–M5 violations and auto-corrects them. This is the last line of defense against arithmetic errors or hallucinated allocations.

### M1–M5 Safety Rules

| Rule | Violation | Protection |
|------|-----------|------------|
| **M1** (relaxed) | Offer >15% below previous best | Concessions ≤15% allowed to enable Nash Welfare improvement; hard floor is BATNA |
| **M2** | Propose allocation worth < BATNA | Filter `my_val < batna`, auto-fix via `_fix_proposal()` |
| **M3** | Offer all items or zero items to self | Filter `sum==0 or sum==total`, auto-fix |
| **M4** | Accept offer worth < BATNA | Override decision to `accept: false` |
| **M5** | Walk away from offer > BATNA on final round | Override decision to `accept: true` |

The `_fix_proposal()` method performs a brute-force search over all valid item splits to find the allocation that maximizes a Nash Welfare proxy: `√(my_val) × √(opponent_items + 1)`.

---

## Nash Welfare Strategy

The agent is optimized for the leaderboard metrics: **Nash Welfare (NW)**, **Nash Welfare above BATNA (NWA)**, and **Envy-Freeness up to one item (EF1)**.

**Key principles:**

- 🎁 **Give cheapest items first** — Items you value least may be highly valuable to your opponent. The situation block always shows items sorted by your valuation so the LLM can make informed, mutually beneficial offers.
- 🤝 **Concessions are allowed** — M1 is relaxed to permit up to 15% concessions below your previous best offer. This enables movement toward Pareto-improving deals rather than deadlock.
- ⚖️ **EF1 awareness** — The agent computes and reports EF1 status for both proposals and incoming offers, guiding the LLM toward fairer splits.
- 🛡️ **Fallback optimization** — When the LLM fails to produce valid JSON, `_fix_proposal()` selects the allocation that maximizes the Nash Welfare proxy rather than defaulting to a greedy or random choice.

---

## Project Structure

```
src/
  agent.py       ── Core logic: situation building, CoT prompting, M1–M5 validation
  executor.py    ── A2A request routing and task handling
  server.py      ── HTTP server setup, AgentCard configuration
  messenger.py   ── A2A messaging utilities and helpers
amber-manifest.json5 ── Amber deployment manifest (config schema, Docker image)
Dockerfile           ── Container build configuration
pyproject.toml       ── Python dependencies (managed via uv)
```

---

## Quick Start

### Run Locally

```bash
# Install dependencies
uv sync

# Configure API key
cp sample.env .env
# Edit .env: set your Mistral API key
# MISTRAL_API_KEY=your_key_here

# Start the agent server
uv run src/server.py --host 127.0.0.1 --port 9009

# Verify it's running
curl http://127.0.0.1:9009/.well-known/agent.json
```


## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MISTRAL_API_KEY` | — | Mistral API key (**required**) |
| `MISTRAL_MODEL` | `mistral-small-latest` | Mistral model to use (`mistral-small-latest` or `mistral-large-latest`) |
| `AGENT_MEMORY_DIR` | `./memory` | Directory for persistent opponent memory files |
| `AGENT_LOGS_DIR` | `./logs` | Directory for game log files |

> 🔐 Secrets marked as `secret: true` in `amber-manifest.json5` are never logged or exposed in deployment configs.

---

## Response Format

The agent communicates with the green evaluator via the A2A protocol using JSON messages.

**Propose an allocation:**
```json
{
  "allocation_self": [5, 2, 1],
  "allocation_other": [2, 2, 0],
  "reason": "Offered low-value items to maximize Nash Welfare"
}
```

**Accept or reject an offer:**
```json
{
  "accept": true,
  "reason": "Offer value 12.5 >= BATNA 10.0 and EF1-compatible"
}
```

> 💡 The LLM is instructed to output reasoning inside `<think>` tags first, then valid JSON after `</thought>`. The parser extracts and validates the JSON portion.

---

## License

MIT © 2026