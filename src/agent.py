import json
import os
import re
from typing import Any

from mistralai.client import Mistral
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
MISTRAL_MODEL = os.environ.get("MISTRAL_MODEL", "mistral-small-latest")

if not MISTRAL_API_KEY:
    raise RuntimeError("MISTRAL_API_KEY environment variable is required")

client = Mistral(api_key=MISTRAL_API_KEY)

SYSTEM_PROMPT = """You are a negotiator in the AgentBeats bargaining meta-game.

## HARD RULES (never violate):
- M2: Never propose/accept a deal worth less than your BATNA
- M3: Never offer ALL items or ZERO items to yourself
- M4: Never ACCEPT an offer worth less than your BATNA
- M5: Never WALK AWAY from an offer worth MORE than your BATNA on the last round

## GOAL: MAXIMIZE NASH WELFARE
Nash Welfare = sqrt(your_payoff × opponent_payoff).
Give opponent items you value LEAST — this boosts mutual gain.

## RESPONSE FORMAT
Part 1 — THINKING (inside <think> tags):
Reason step-by-step about valuations, BATNA, opponent behavior.

Part 2 — DECISION (valid JSON after </thought>):
For PROPOSE: {"allocation_self": [x,y,z], "allocation_other": [a,b,c], "reason": "..."}
For ACCEPT_OR_REJECT: {"accept": true/false, "reason": "..."}

JSON must come AFTER </thought>. No markdown code blocks."""

def _parse_observation(text: str) -> dict[str, Any] | None:
    """Extract JSON from message."""
    blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    candidates = list(blocks) + [text]
    for c in candidates:
        c = c.strip()
        if not c:
            continue
        try:
            data = json.loads(c)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    return None

def _extract_json_from_cot(reply: str) -> str:
    """Extract JSON after </thought> or from anywhere."""
    # Try to find JSON after </thought>
    match = re.search(r"</thought>\s*(.*)", reply, re.DOTALL)
    if match:
        after = match.group(1).strip()
        parsed = _parse_observation(after)
        if parsed:
            return json.dumps(parsed)
    # Fallback: search anywhere
    parsed = _parse_observation(reply)
    if parsed:
        return json.dumps(parsed)
    return reply

def _validate_and_fix(reply: str, obs: dict, action: str) -> str:
    """Simple safety check for M2/M4 (BATNA constraint)."""
    parsed = _parse_observation(reply)
    if not parsed:
        return reply
    
    batna = obs.get("batna_value", obs.get("batna_self", 0))
    valuations = obs.get("valuations_self", [])
    
    if action == "PROPOSE" and "allocation_self" in parsed and valuations:
        alloc = parsed["allocation_self"]
        if isinstance(alloc, list) and len(alloc) == len(valuations):
            my_val = sum(v * a for v, a in zip(valuations, alloc))
            if my_val < batna:
                quantities = obs.get("quantities", [0]*len(valuations))
                parsed["allocation_self"] = quantities
                parsed["allocation_other"] = [0]*len(valuations)
                parsed["reason"] = f"M2 fix: ensured value >= BATNA ({batna})"
                return json.dumps(parsed)
    
    elif action == "ACCEPT_OR_REJECT":
        offer_value = obs.get("offer_value", 0)
        accept = parsed.get("accept")
        if accept is True and offer_value < batna:
            return json.dumps({"accept": False, "reason": f"M4 fix: offer {offer_value} < BATNA {batna}"})
        if accept is False and offer_value > batna and obs.get("round_index", 0) >= obs.get("max_rounds", 5):
            return json.dumps({"accept": True, "reason": f"M5 fix: last round, offer {offer_value} > BATNA {batna}"})
    
    return reply

async def run(message: Message, updater: TaskUpdater) -> None:
    """Main A2A handler."""
    input_text = get_message_text(message)
    await updater.update_status(TaskState.working)
    
    obs = _parse_observation(input_text)
    action = obs.get("action", "") if obs else ""
    
    situation = ""
    if obs:
        v = obs.get("valuations_self", [])
        q = obs.get("quantities", [])
        batna = obs.get("batna_value", obs.get("batna_self", 0))
        round_idx = obs.get("round_index", 0)
        max_rounds = obs.get("max_rounds", 5)
        
        situation = f"\n[SITUATION]\n"
        if v and q:
            total = sum(vi * qi for vi, qi in zip(v, q))
            situation += f"My valuations: {v}\nQuantities: {q}\nTotal if keep all: {total}\nBATNA: {batna}\n"
            situation += f"Round: {round_idx}/{max_rounds}\n"
            cheapest = sorted(range(len(v)), key=lambda i: v[i])
            situation += f"Cheapest items for me (offer first): {[f'type{i}' for i in cheapest]}\n"
        
        if action == "ACCEPT_OR_REJECT":
            offer_val = obs.get("offer_value", 0)
            situation += f"Opponent offer value: {offer_val}\n"
            if offer_val >= batna:
                situation += f"✓ Offer >= BATNA — accepting is safe\n"
            else:
                situation += f"✗ Offer < BATNA — must reject (M4)\n"
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": input_text + situation}
    ]
    
    # Call Mistral
    response = client.chat.complete(
        model=MISTRAL_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=1024
    )
    llm_reply = response.choices[0].message.content or ""
    
    # Extract and validate JSON
    json_reply = _extract_json_from_cot(llm_reply)
    final_reply = _validate_and_fix(json_reply, obs or {}, action)
    
    # Respond via A2A
    await updater.add_artifact(
        parts=[Part(root=TextPart(text=final_reply))],
        name="Response"
    )