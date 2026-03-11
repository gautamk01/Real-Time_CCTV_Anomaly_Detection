"""Cloud AI Model - Groq (Llama) Wrapper with Enhanced Prompting."""

import json
import re
import concurrent.futures
from openai import OpenAI
from typing import List, Dict, Optional
from .examples_database import get_diverse_examples, format_example_for_prompt

# Timeout for Groq API calls (seconds)
_CLOUD_TIMEOUT = 5

# Keywords for edge-only fallback when cloud times out
_ALERT_KEYWORDS = re.compile(
    r"\b(fight|fighting|punch|punching|kick|kicking|struggle|struggling|"
    r"attack|attacking|aggression|aggressive|hit|hitting|slap|slapping|"
    r"choke|choking|wrestle|wrestling|shove|shoving|drag|dragging|"
    r"weapon|knife|gun|blood|unconscious|injured|fallen|down)\b",
    re.IGNORECASE,
)
_CLEAR_KEYWORDS = re.compile(
    r"\b(standing|walking|talking|sitting|normal|calm|casual|friendly|"
    r"relaxed|peaceful|quiet|strolling|chatting)\b",
    re.IGNORECASE,
)


class CloudAI:
    """
    Cloud AI for threat assessment and investigation orchestration.
    Uses Groq (Llama 3.3 70B) via OpenAI-compatible API for sub-second inference.

    Performance: Static prompt (decision framework + few-shot examples) is passed
    as system message. Only the dynamic observation log is sent per call.
    """

    def __init__(self, api_key: str, model_id: str = "llama-3.3-70b-versatile"):
        """
        Initialize the cloud AI model.

        Args:
            api_key: Groq API key
            model_id: Model identifier (default: llama-3.3-70b-versatile)
        """
        self.model_id = model_id

        print(f"🔧 [CLOUD] Configuring {model_id} via Groq...")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )

        # Pre-build static system instruction
        self.system_prompt = self._build_system_instruction()

        print(f"✅ [CLOUD] Model configured successfully")
        print(f"✨ [CLOUD] Using Groq LPU for sub-second inference")

    def _build_system_instruction(self) -> str:
        """Build the static system instruction once at init time.

        This contains the decision framework, few-shot examples, rules,
        and output format — everything that doesn't change between calls.
        """
        examples = get_diverse_examples(limit=3)
        examples_text = "\n".join([
            format_example_for_prompt(ex, i) for i, ex in enumerate(examples)
        ])

        return f"""You are a Violence Detection AI analyzing REAL-TIME CCTV footage.

CRITICAL RULE: When in doubt, ALWAYS choose ALERT. A false alarm is acceptable — a missed attack is not.

EXAMPLES:
{examples_text}

ALERT immediately (do NOT investigate first):
- Any punching, kicking, slapping, choking, wrestling, shoving, dragging
- Person on ground, unconscious, injured, or fallen
- Weapons visible (knife, gun, bat, bottle used as weapon)
- Forceful restraint or aggressive physical control
- Person in distress, agitation, or fear
- Scattered/broken objects suggesting a fight occurred
- Someone approaching a distressed/downed person aggressively

INVESTIGATE (ask ONE short question) — ONLY when ALL of these are true:
- No physical contact observed
- No one is on the ground or in distress
- Scene is ambiguous (e.g. tense argument, unclear gathering)
- This is round 1 (if round 2+, you MUST decide ALERT or CLEAR)

CLEAR — ONLY when the scene is clearly safe:
- Casual walking, talking, standing, sitting
- Friendly social interaction with no tension
- Sports/exercise in appropriate venue
- Empty scene or no people

ESCALATION RULE: If this is round 2 or later and you are unsure, choose ALERT. Do not return INVESTIGATE after round 1 unless you are highly confident the scene is merely ambiguous.

NOT violence: sports in gym/field, choreographed performance, playful roughhousing with smiling, yoga.

Confidence: 100=active striking/weapons, 95=person down/injured, 90=restraint/distress, 85=agitation/fear/scattered objects, 75=heated argument, 50=normal.

JSON output: {{"status":"CLEAR|INVESTIGATE|ALERT","confidence":<0-100>,"question":"<if INVESTIGATE>","reason":"<cite timestamps, max 2 sentences>"}}"""

    def _edge_fallback(self, history: List[str]) -> Dict:
        """Fast keyword-based fallback when the cloud call times out."""
        text = " ".join(history).lower()

        alert_hits = len(_ALERT_KEYWORDS.findall(text))
        clear_hits = len(_CLEAR_KEYWORDS.findall(text))

        if alert_hits > 0:
            return {
                "status": "ALERT",
                "confidence": 85,
                "question": "",
                "reason": f"Cloud timeout — edge fallback triggered on {alert_hits} alert keyword(s)",
            }
        if clear_hits > 0:
            return {
                "status": "CLEAR",
                "confidence": 70,
                "question": "",
                "reason": f"Cloud timeout — edge fallback: {clear_hits} benign keyword(s)",
            }
        return {
            "status": "INVESTIGATE",
            "confidence": 60,
            "question": "Describe any aggressive actions visible in the scene.",
            "reason": "Cloud timeout — edge fallback: ambiguous scene, needs follow-up",
        }

    def _format_rag_cases(self, rag_context: List[dict]) -> str:
        """Format RAG-retrieved cases as a prompt section.

        Args:
            rag_context: List of case dicts from RAG retrieval, each with
                         'history' (str), 'metadata' (dict), 'similarity' (float)

        Returns:
            Formatted string to append to the system prompt
        """
        lines = ["\nSIMILAR PAST CASES (from system memory):"]
        for i, case in enumerate(rag_context):
            meta = case.get("metadata", {})
            similarity = case.get("similarity", 0)
            history_text = case.get("history", "")
            # Truncate long histories to keep prompt compact
            if len(history_text) > 250:
                history_text = history_text[:247] + "..."

            lines.append(f"Past Case {i + 1} (similarity: {similarity:.0%}):")
            lines.append(f"  Observations: {history_text}")
            lines.append(f"  Verdict: {meta.get('status', '?')} ({meta.get('confidence', '?')}%)")
            lines.append(f"  Reason: {meta.get('reason', 'N/A')}")
            lines.append("")

        return "\n".join(lines)

    def assess_threat(self, history: List[str],
                      rag_context: Optional[List[dict]] = None) -> Dict:
        """
        Analyze the situation and decide next action.

        Sends the system prompt + dynamic observation log to Groq.
        Uses a 5-second timeout with keyword fallback to prevent stalling.

        Args:
            history: List of observation strings with timestamps
            rag_context: Optional list of similar past cases from RAG

        Returns:
            Dict with keys: status, confidence, question (optional), reason
        """
        # Build system message: static prompt + optional RAG context
        system_msg = self.system_prompt
        if rag_context:
            system_msg += self._format_rag_cases(rag_context)

        history_text = "\n".join([f"- {h}" for h in history])

        user_prompt = f"""OBSERVATION LOG:
{history_text}

Analyze and provide JSON decision."""

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self.client.chat.completions.create,
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=200,
                    temperature=0,
                )
                response = future.result(timeout=_CLOUD_TIMEOUT)

            decision = json.loads(response.choices[0].message.content)

            # Ensure all required fields exist
            decision.setdefault("status", "CLEAR")
            decision.setdefault("confidence", 0)
            decision.setdefault("question", "Describe any aggressive actions.")
            decision.setdefault("reason", "No reason provided")

            return decision

        except concurrent.futures.TimeoutError:
            print(
                f"   ⏱️ [CLOUD] Timeout ({_CLOUD_TIMEOUT}s) — using edge fallback")
            return self._edge_fallback(history)

        except Exception as e:
            print(f"   ❌ [CLOUD ERROR] {e}")
            return {
                "status": "CLEAR",
                "confidence": 0,
                "question": "",
                "reason": f"Error: {str(e)}"
            }
