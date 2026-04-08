import json
import os
import re
from typing import Any

from mistralai.client import Mistral
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text


class Agent:
    """Negotiation agent for AgentBeats bargaining meta-game."""
    
    def __init__(self):
        """Initialize the agent with Mistral client."""
        self.api_key = os.environ.get("MISTRAL_API_KEY")
        self.model = os.environ.get("MISTRAL_MODEL", "mistral-large-latest")
        
        if not self.api_key:
            raise RuntimeError("MISTRAL_API_KEY environment variable is required")
        
        self.client = Mistral(api_key=self.api_key)
        self.system_prompt = """You are a strategic negotiator in the AgentBeats bargaining meta-game, a multi-round bargaining benchmark where agents negotiate over privately valued items under time pressure and outside options.

## YOUR OBJECTIVES (in priority order):
1. **MAXIMIZE NASH WELFARE**: sqrt(your_payoff × opponent_payoff) — this is your primary score metric
2. **MINIMIZE REGRET**: Don't leave money on the table compared to optimal strategies
3. **ACHIEVE ENVY-FREENESS**: Both parties should feel the outcome is fair
4. **BEAT YOUR BATNA**: Never accept less than your Best Alternative To Negotiated Agreement

## HARD CONSTRAINTS (Never violate):
- **M2**: NEVER propose or accept a deal worth less than your BATNA
- **M3**: NEVER offer ALL items or ZERO items to yourself (must be a genuine negotiation)
- **M4**: NEVER ACCEPT an offer worth less than your BATNA
- **M5**: NEVER WALK AWAY from an offer worth MORE than your BATNA on the final round

## STRATEGIC PRINCIPLES:

### Item Valuation Strategy:
- Your valuations: higher number = more valuable to YOU
- Give opponent items you value LEAST (lowest valuation numbers)
- Keep items you value MOST (highest valuation numbers)
- This maximizes mutual gain because you lose little while opponent gains a lot

### Concession Strategy:
- Early rounds: Propose aggressive but feasible deals (60-70% of gains for you)
- Middle rounds: Make gradual concessions (10-15% per round)
- Late rounds: Be more generous to secure a deal (close to 50-50 split)
- Final round: Accept anything above BATNA (M5 rule)

### Opponent Modeling:
- Track opponent's past offers to infer their valuations
- If opponent consistently offers cheap items to you, they value those items low
- Adjust your counter-offers based on opponent's apparent strategy

### Nash Welfare Optimization:
- Nash Welfare = sqrt(your_payoff × opponent_payoff)
- Maximum Nash Welfare occurs at 50-50 split when valuations are symmetric
- With asymmetric valuations, give more of low-value-to-you items to opponent
- Example: If you value [90, 50, 10] and opponent values [10, 50, 90], 
  optimal is you take item0 (90), opponent takes item2 (90), split item1 (50 each)

## RESPONSE FORMAT:

### For PROPOSE action:
Return ONLY this JSON structure, no other text:
{
    "allocation_self": [quantity_item0, quantity_item1, quantity_item2],
    "allocation_other": [quantity_item0, quantity_item1, quantity_item2],
    "reason": "Strategic explanation including Nash welfare consideration"
}

### For ACCEPT_OR_REJECT action:
Return ONLY this JSON structure, no other text:
{
    "accept": true/false,
    "reason": "Strategic explanation including payoff comparison to BATNA"
}

## EXAMPLE PROPOSAL:
Input: {"action": "PROPOSE", "valuations_self": [90, 50, 10], "quantities": [1,1,1], "batna_value": 60}
Output: {"allocation_self": [1,0,0], "allocation_other": [0,1,1], "reason": "Keeping high-value item0 (90), offering low-value items 1(50) and 2(10) to opponent to maximize Nash Welfare. My payoff = 90 which exceeds BATNA 60."}

## EXAMPLE ACCEPT:
Input: {"action": "ACCEPT_OR_REJECT", "offer_value": 75, "batna_value": 60, "round_index": 4, "max_rounds": 5}
Output: {"accept": true, "reason": "Offer 75 exceeds BATNA 60, and this is the final round (M5 rule). Accepting secures positive gain."}

## EXAMPLE REJECT:
Input: {"action": "ACCEPT_OR_REJECT", "offer_value": 50, "batna_value": 60, "round_index": 2, "max_rounds": 5}
Output: {"accept": false, "reason": "Offer 50 below BATNA 60. Can negotiate for better deal with 3 rounds remaining."}

## REMEMBER:
- Return ONLY valid JSON, no markdown blocks, no extra text
- Always ensure allocation sums respect quantities
- Never propose/accept below BATNA (unless M5 forces acceptance on last round)
- Maximize sqrt(your_payoff × opponent_payoff) as your primary metric
- Be strategic: aggressive early, conceding mid, accepting late"""

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Main A2A handler."""
        input_text = get_message_text(message)
        await updater.update_status(TaskState.working)
        
        # Parse input to extract action and context
        try:
            obs = json.loads(input_text)
            action = obs.get("action", "")
        except:
            obs = {}
            action = ""
        
        # Build context-aware user prompt
        user_prompt = f"""Current negotiation state:
{json.dumps(obs, indent=2)}

Based on this state, return a JSON response for action: {action}"""

        # Call Mistral with powerful prompt
        response = self.client.chat.complete(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=512,
            top_p=0.95
        )
        
        llm_reply = response.choices[0].message.content or ""
        
        # Extract JSON from response (handle any stray markdown)
        json_match = re.search(r'\{.*\}', llm_reply, re.DOTALL)
        if json_match:
            try:
                final_response = json.loads(json_match.group())
            except:
                # Fallback to safe response
                final_response = self._get_safe_response(obs, action)
        else:
            final_response = self._get_safe_response(obs, action)
        
        # Validate BATNA constraints
        final_response = self._validate_batna(final_response, obs, action)
        
        # Send response
        await updater.add_artifact(
            parts=[Part(root=TextPart(text=json.dumps(final_response)))],
            name="Response"
        )
    
    def _get_safe_response(self, obs: dict, action: str) -> dict:
        """Safe fallback response."""
        if action == "PROPOSE":
            quantities = obs.get("quantities", [1, 1, 1])
            return {
                "allocation_self": quantities,
                "allocation_other": [0, 0, 0],
                "reason": "Safe default: keeping all items"
            }
        else:
            batna = obs.get("batna_value", obs.get("batna_self", 0))
            offer = obs.get("offer_value", 0)
            return {
                "accept": offer >= batna,
                "reason": f"Safe default: {'accepting' if offer >= batna else 'rejecting'} based on BATNA comparison"
            }
    
    def _validate_batna(self, response: dict, obs: dict, action: str) -> dict:
        """Validate BATNA constraints."""
        batna = obs.get("batna_value", obs.get("batna_self", 0))
        valuations = obs.get("valuations_self", [])
        
        if action == "PROPOSE" and "allocation_self" in response:
            alloc = response["allocation_self"]
            if isinstance(alloc, list) and len(alloc) == len(valuations):
                my_value = sum(v * a for v, a in zip(valuations, alloc))
                if my_value < batna:
                    # Fix: return to BATNA
                    quantities = obs.get("quantities", [0] * len(valuations))
                    response["allocation_self"] = quantities
                    response["allocation_other"] = [0] * len(valuations)
                    response["reason"] += f" [FIXED: M2 constraint - value {my_value} < BATNA {batna}]"
        
        elif action == "ACCEPT_OR_REJECT":
            offer_value = obs.get("offer_value", 0)
            accept = response.get("accept", False)
            round_idx = obs.get("round_index", 0)
            max_rounds = obs.get("max_rounds", 5)
            
            # M4: Never accept below BATNA
            if accept and offer_value < batna and round_idx < max_rounds - 1:
                response["accept"] = False
                response["reason"] += f" [FIXED: M4 constraint - cannot accept offer {offer_value} < BATNA {batna}]"
            
            # M5: On last round, accept if above BATNA
            if not accept and offer_value > batna and round_idx >= max_rounds - 1:
                response["accept"] = True
                response["reason"] += f" [FIXED: M5 constraint - last round, must accept offer {offer_value} > BATNA {batna}]"
        
        return response