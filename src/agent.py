import json, re, os, logging
from typing import Any
from mistralai.client import Mistral
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text

# Системный промпт с правилами M1-M5
SYSTEM_PROMPT = """Вы — переговорщик в мета-игре торга AgentBeats.

## ЖЁСТКИЕ ПРАВИЛА (нарушение = проигрыш):
- M2: Не предлагайте сделку дешевле вашего BATNA
- M3: Не предлагайте ВСЕ или НОЛЬ предметов себе
- M4: Не ПРИНИМАЙТЕ предложение дешевле вашего BATNA
- M5: Не ОТКАЗЫВАЙТЕСЬ от предложения дороже вашего BATNA на последнем раунде

## ЦЕЛЬ: Максимизировать Nash Welfare = sqrt(ваш_выигрыш × выигрыш_оппонента)
Ключ: предметы, которые вы цените МЕНЬШЕ, могут быть ОЧЕНЬ ценны для оппонента.

## Формат ответа:
1. Думайте шаг за шагом внутри <think>...</think>
2. Затем дайте валидный JSON:
   - Для предложения: {"allocation_self": [x,y,z], "allocation_other": [a,b,c], "reason": "..."}
   - Для принятия: {"accept": true/false, "reason": "..."}
"""

class GameMemory:
    """Память об одной игре: история предложений, статистика оппонента."""
    def __init__(self, opp_key: str = ""):
        self.opp_key = opp_key
        self.my_offers = []      # [(allocation, value), ...]
        self.opp_offers = []     # [(allocation_to_me, value_for_me), ...]
        self.round_index = 0
        self.valuations = []     # [v1, v2, v3]
        self.batna = 0           # fallback payoff
        self.quantities = []     # [q1, q2, q3]
        self.discount = 0.98
        self.best_offer_value = 0

class Agent:
    def __init__(self):
        # Инициализация LLM клиента (Mistral / OpenRouter)
        api_key = os.environ.get("MISTRAL_API_KEY")
        self.client = Mistral(api_key=api_key)
        self.model = os.environ.get("MISTRAL_MODEL", "mistral-large-latest")
        self.conversation_history = []
        self.game_memory = GameMemory()
    
    def _parse_observation(self, text: str) -> dict | None:
        """Извлекает JSON-наблюдение из входящего сообщения."""
        blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.I)
        for c in blocks + [text]:
            try:
                data = json.loads(c.strip())
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                continue
        return None
    
    def _build_situation(self, obs: dict, action: str) -> str:
        """Строит контекстный блок [SITUATION] для LLM с числами и стратегией."""
        mem = self.game_memory
        lines = ["\n[SITUATION — используйте эти факты для рассуждений]"]
        
        # Позиция агента
        if mem.valuations and mem.quantities:
            total = sum(v*q for v, q in zip(mem.valuations, mem.quantities))
            lines.append(f"\n## Моя позиция")
            lines.append(f" Оценки предметов: {mem.valuations}")
            lines.append(f" Количество: {mem.quantities}")
            lines.append(f" BATNA: {mem.batna}, Всего возможно: {total}")
            lines.append(f" Раунд: {mem.round_index}, Дисконт: {mem.discount}")
            
            # Стратегическая подсказка: какие предметы дешевле отдать
            sorted_items = sorted(range(len(mem.valuations)), 
                               key=lambda i: mem.valuations[i])
            cheapest = [f"type{i}(val={mem.valuations[i]})" for i in sorted_items]
            lines.append(f" Предметы, которые мне дешевле отдать: {', '.join(cheapest)}")
        
        # История предложений оппонента
        if mem.opp_offers:
            lines.append(f"\n## Предложения оппонента:")
            for i, (alloc, val) in enumerate(mem.opp_offers):
                lines.append(f"  Раунд {i+1}: дал мне {alloc} (ценность для меня = {val})")
        
        # Контекст действия
        if action == "PROPOSE":
            lines.append(f"\n## Задача: ПРЕДЛОЖИТЬ распределение")
            lines.append(f" Ограничение: ценность >= BATNA ({mem.batna})")
            lines.append(f" Стратегия: отдавайте оппоненту предметы, которые вы цените меньше!")
        elif action == "ACCEPT_OR_REJECT":
            offer_val = obs.get("offer_value", 0)
            lines.append(f"\n## Задача: ПРИНЯТЬ или ОТКЛОНИТЬ")
            lines.append(f" Предложение: {offer_val}, BATNA: {mem.batna}")
            if offer_val >= mem.batna:
                lines.append(f" ✓ Предложение ВЫШЕ BATNA — принятие безопасно")
            else:
                lines.append(f" ✗ Предложение НИЖЕ BATNA — M4 требует отклонить")
        
        return "\n".join(lines)
    
    def _validate_and_fix(self, reply: str, obs: dict, action: str) -> str:
        """Проверка правил M1-M5 и авто-исправление нарушений."""
        try:
            parsed = json.loads(reply.strip())
        except:
            return reply
        
        mem = self.game_memory
        v, q = mem.valuations, mem.quantities
        
        if action == "PROPOSE" and v and q:
            alloc = parsed.get("allocation_self")
            if alloc and isinstance(alloc, list):
                my_val = sum(vi*ai for vi, ai in zip(v, alloc))
                
                # M2: ценность < BATNA
                if my_val < mem.batna:
                    return self._fix_proposal(parsed, v, q, mem.batna)
                
                # M3: все или ничего
                if sum(alloc) == 0 or sum(alloc) == sum(q):
                    return self._fix_proposal(parsed, v, q, mem.batna)
        
        elif action == "ACCEPT_OR_REJECT":
            offer_val = obs.get("offer_value", 0)
            batna_val = obs.get("batna_value", mem.batna)
            
            # M4: принять предложение < BATNA
            if parsed.get("accept") is True and offer_val < batna_val:
                return json.dumps({"accept": False, "reason": "M4 fix: offer < BATNA"})
            
            # M5: отклонить предложение > BATNA на последнем раунде
            if parsed.get("accept") is False and offer_val > batna_val:
                if mem.round_index >= mem.max_rounds:
                    return json.dumps({"accept": True, "reason": "M5 fix: last round"})
        
        return reply
    
    def _fix_proposal(self, parsed: dict, v: list, q: list, batna: int) -> str:
        """Находит валидное распределение, максимизирующее Nash Welfare."""
        import itertools
        target = batna
        best_alloc, best_nw = None, -1
        total_items = sum(q)
        
        for combo in itertools.product(*[range(qi+1) for qi in q]):
            a = list(combo)
            if sum(a) in (0, total_items):  # M3
                continue
            my_val = sum(vi*ai for vi, ai in zip(v, a))
            if my_val < target:  # M2
                continue
            # Nash Welfare proxy: баланс между моей ценностью и предметами оппонента
            opp_items = total_items - sum(a)
            nw_proxy = (my_val**0.5) * (opp_items + 1)**0.5
            if nw_proxy > best_nw:
                best_nw, best_alloc = nw_proxy, a
        
        if best_alloc:
            parsed["allocation_self"] = best_alloc
            parsed["allocation_other"] = [q[i]-best_alloc[i] for i in range(len(q))]
        return json.dumps(parsed)
    
    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Основной цикл обработки A2A-сообщения."""
        input_text = get_message_text(message)
        await updater.update_status(TaskState.working)
        
        # Парсинг наблюдения
        obs = self._parse_observation(input_text)
        action = obs.get("action", "") if obs else ""
        
        if obs:
            # Обновление памяти игры
            self._update_memory(obs)
            # Построение контекста для LLM
            situation = self._build_situation(obs, action)
            enriched_input = input_text + "\n" + situation
            self.conversation_history.append({"role": "user", "content": enriched_input})
        
        # Вызов LLM
        llm_reply = self._call_llm_with_retry([
            {"role": "system", "content": SYSTEM_PROMPT},
            *self.conversation_history
        ])
        
        # Извлечение JSON из chain-of-thought ответа
        json_reply = self._extract_json_from_cot(llm_reply)
        
        # Валидация и исправление
        final_reply = self._validate_and_fix(json_reply, obs or {}, action)
        
        # Ответ через A2A
        self.conversation_history.append({"role": "assistant", "content": final_reply})
        await updater.add_artifact(
            parts=[Part(root=TextPart(text=final_reply))],
            name="Response"
        )