"""Cloud AI Model - Groq (Llama) Wrapper with Enhanced Prompting."""

import json
import re
import concurrent.futures
from openai import OpenAI
from typing import List, Dict, Optional
from .examples_database import get_diverse_examples, format_example_for_prompt

# Timeout for cloud API calls (seconds)
_GROQ_TIMEOUT = 10   # Increased from 5s — Groq P95 is ~4-5s under load
_OLLAMA_TIMEOUT = 30  # Ollama local models are slower

# Keywords for edge-only fallback when cloud times out
_HARD_ALERT_KEYWORDS = re.compile(
    r"\b(punch|punching|kick|kicking|attack|attacking|hit|hitting|"
    r"slap|slapping|choke|choking|wrestle|wrestling|stab|stabbing|"
    r"shoot|shooting|weapon|knife|gun|blood|dragging|pinning|"
    r"throwing|threw|tossing|smashing|slamming|"
    r"grabbed|grabbing|strangling|beating)\b",
    re.IGNORECASE,
)
_REVIEW_KEYWORDS = re.compile(
    r"\b(fight|fighting|struggle|struggling|aggressive|shove|shoving|"
    r"fallen|falling|fell|down|distress|chaotic|running|pushed|pushing|"
    r"crowd|dimly lit|obscured|unclear|collision|accident|"
    r"tripping|tripped|overturned|debris|chair|knocked|collapsed|"
    r"stumbling|lunging|flailing|tossed)\b",
    re.IGNORECASE,
)
_CLEAR_KEYWORDS = re.compile(
    r"\b(standing|walking|talking|sitting|normal|calm|casual|friendly|"
    r"relaxed|peaceful|quiet|strolling|chatting|"
    r"cartoon|illustration|drawing|poster|painting|mural|art|artwork|"
    r"animated|logo|sign|print|tattoo|decoration|statue|sculpture)\b",
    re.IGNORECASE,
)


class CloudAI:
    """
    Cloud AI for threat assessment and investigation orchestration.
    Uses Groq (Llama 3.3 70B) via OpenAI-compatible API for sub-second inference.

    Performance: Static prompt (decision framework + few-shot examples) is passed
    as system message. Only the dynamic observation log is sent per call.
    """

    def __init__(self, api_key: str, model_id: str = "llama-3.3-70b-versatile",
                 base_url: str = "https://api.groq.com/openai/v1"):
        """
        Initialize the cloud AI model.

        Args:
            api_key: API key (use "ollama" for local Ollama)
            model_id: Model identifier
            base_url: OpenAI-compatible API base URL.
                      Groq: "https://api.groq.com/openai/v1"
                      Ollama: "http://localhost:11434/v1"
        """
        self.model_id = model_id
        self.base_url = base_url
        self.is_ollama = "localhost" in base_url or "127.0.0.1" in base_url
        self._timeout = _OLLAMA_TIMEOUT if self.is_ollama else _GROQ_TIMEOUT

        backend = "Ollama (local)" if self.is_ollama else "Groq (cloud)"
        print(f"🔧 [CLOUD] Configuring {model_id} via {backend}...")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        # Pre-build static system instruction
        self.system_prompt = self._build_system_instruction()

        print(f"✅ [CLOUD] Model configured successfully")
        if self.is_ollama:
            print(f"🏠 [CLOUD] Using local Ollama model")
        else:
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

EXAMPLES:
{examples_text}

ALERT — requires CONCRETE PHYSICAL EVIDENCE of at least one:
- Active striking: punching, kicking, slapping, choking, wrestling, shoving
- Person on the ground who is injured, unconscious, or being attacked
- Visible weapon being used or brandished (knife, gun, bat, bottle)
- Forceful physical restraint or aggressive body control
- One person dragging, grabbing, or pinning another

INVESTIGATE (ask ONE open-ended question) — when:
- One more observation is likely to clarify the scene
- No physical contact is observed yet, but the situation looks tense
- Scene is genuinely ambiguous (crowd gathering, heated argument, running crowd)

REVIEW — when:
- The scene remains ambiguous, occluded, dimly lit, or aftermath-only
- A person is on the ground after a collision or accident, but active assault is not visible
- You cannot verify a benign explanation, but you also do not have concrete evidence for ALERT
- Prefer REVIEW over CLEAR when something serious may have happened but the evidence is insufficient

CLEAR — when:
- People are walking, talking, standing, or sitting normally
- Social interaction with no physical contact or distress
- Verbal argument WITHOUT physical contact (raised voices alone ≠ violence)
- Fast-moving vehicles with no collision or injury
- Sports, exercise, playful roughhousing, yoga, choreographed performance
- Empty scene or no people

FALSE POSITIVE FILTER (CRITICAL): This system monitors REAL surveillance cameras.
- If the description mentions cartoon, illustration, drawing, poster, art, painting, mural, sign, print, decoration, TV screen, movie, animation, tattoo, logo, statue, or sculpture — classify as CLEAR immediately with confidence 95%.
- Violence depicted in artwork, posters, TV screens, or decorations on walls is NOT real violence.
- Only REAL PEOPLE performing REAL PHYSICAL ACTIONS in the actual monitored space constitute threats.
- A martial arts poster, movie scene on a TV, or graffiti showing fighting = CLEAR.

QUESTION FORMAT (MANDATORY): When choosing INVESTIGATE, NEVER ask yes/no questions.
Small vision models confirm anything when asked "Is X happening?" — causing false alerts.
ALWAYS ask open-ended questions that begin with "Describe", "What", or "How".
BAD:  "Are the individuals fighting?"
BAD:  "Is the motorcycle causing a disturbance?"
GOOD: "Describe what the two individuals near the center are physically doing."
GOOD: "What specific physical interactions are occurring between the people?"

FALLING PERSON PROTOCOL (critical): When the initial observation includes "falling", "fell", "on the floor", or "on the ground", the edge model CANNOT predict future events. Do NOT ask "What happens after they fall?" — these questions return the same useless answer repeatedly. Instead, ask ONE of:
- "Describe the exact body position of every person within 2 meters of the fallen person and which direction they are facing."
- "What are the hands of the people nearest to the fallen person doing?"
- "Describe the posture of the people standing near the person on the floor — are they leaning over, backing away, or stationary?"
These questions target CURRENT VISIBLE STATE, which moondream2 can answer.

ANTI-REPETITION: You will sometimes receive a [META] tag listing questions already asked. You MUST pick a question from a completely different angle — different subject, different body part, different spatial relationship.

PERSISTENT INCIDENT NOTE: If the history contains multiple observations from different timestamps all describing the same fallen/injured person, this indicates a prolonged incident. A prolonged fallen person with unclear cause is HIGH PRIORITY — ALERT if repeated 2+ times, or REVIEW at minimum.

SURVEILLANCE CONTEXT RULE (CRITICAL — overrides other rules when applicable):
This system monitors REAL surveillance cameras in public/commercial spaces (tunnels, hallways, bars, parking lots, streets).
1. When the edge model reports "kicking", "punching", "hitting", or "striking" — this IS violence. ALERT IMMEDIATELY. Do NOT ask "is this sparring?" or "choreographed?". Martial arts training does NOT happen in tunnels, hallways, or bars.
2. When a person is on the ground AND another person is standing over/near them — this is NOT NORMAL in public spaces. Do NOT spend multiple rounds asking about "intent" or "helper vs. threat." Classify as REVIEW (confidence 80%+) on first sight so the escalation engine can track it.
3. A person lying on the ground alone in a public space IS a medical/safety concern — REVIEW minimum.

DECISION CALIBRATION:
- A verbal argument, tense atmosphere, or fast vehicle is NOT violence — classify as CLEAR unless physical contact is visible.
- "Potentially tense" or "possibly aggressive" descriptions without CONCRETE actions are not enough for ALERT.
- When the edge explicitly says kicking, punching, hitting, grabbing, or attacking — ALERT directly. No further rounds needed.
- Do not choose CLEAR for dim, chaotic, or occluded scenes unless there is a positive benign explanation.
- If the scene stays ambiguous after follow-up, choose REVIEW instead of forcing ALERT or CLEAR.

Confidence: 100=active striking/weapons, 95=person down+attacked, 90=forceful restraint, 80=person down (unclear cause), 50=normal.

JSON output: {{"status":"CLEAR|INVESTIGATE|REVIEW|ALERT","confidence":<0-100>,"question":"<if INVESTIGATE, open-ended only>","reason":"<cite timestamps, max 2 sentences>"}}"""

    def _edge_fallback(self, history: List[str]) -> Dict:
        """Cascading Confidence Scoring fallback when cloud times out.

        Algorithm (replaces keyword matching):
        1. Score each observation independently using weighted keywords
        2. Combine scores using probability union: 1 - ∏(1 - score_i)
        3. Apply TIMEOUT PENALTY (0.7 multiplier) — timeout = uncertainty
        4. Map to verdict — CAPPED AT REVIEW (never auto-ALERT on timeout)

        Key rule: ALERT requires cloud confirmation. A timeout means
        'we don't know', never 'it is violent'.
        """
        _TIMEOUT_PENALTY = 0.7  # Uncertainty multiplier

        text = " ".join(history).lower()

        # Step 1: Weighted keyword scores
        alert_hits = len(_HARD_ALERT_KEYWORDS.findall(text))
        review_hits = len(_REVIEW_KEYWORDS.findall(text))
        clear_hits = len(_CLEAR_KEYWORDS.findall(text))

        # Score per category: diminishing returns for repeated keywords
        alert_score = min(alert_hits * 0.3, 0.8)   # max 0.8 from alert keywords
        review_score = min(review_hits * 0.15, 0.5) # max 0.5 from review keywords
        clear_score = min(clear_hits * 0.2, 0.7)    # clear evidence reduces score

        # Step 2: Combined violence score (probability union)
        violence_score = 1 - (1 - alert_score) * (1 - review_score)
        # Subtract clear evidence (clamped to [0, 1])
        violence_score = max(0.0, violence_score - clear_score * 0.5)

        # Step 3: Apply timeout penalty
        final_score = violence_score * _TIMEOUT_PENALTY
        confidence = int(min(final_score * 100, 75))  # Cap at 75% on timeout

        # Step 4: Map to verdict (CAPPED AT REVIEW — never ALERT)
        if final_score < 0.2:
            return {
                "status": "CLEAR",
                "confidence": max(confidence, 50),
                "question": "",
                "reason": f"Cloud timeout — cascading score {final_score:.2f}: likely safe",
            }
        if final_score < 0.5:
            return {
                "status": "INVESTIGATE",
                "confidence": confidence,
                "question": "Describe any aggressive actions visible in the scene.",
                "reason": f"Cloud timeout — cascading score {final_score:.2f}: needs follow-up",
            }
        # final_score >= 0.5 → REVIEW (never ALERT without cloud)
        return {
            "status": "REVIEW",
            "confidence": confidence,
            "question": "",
            "reason": f"Cloud timeout — cascading score {final_score:.2f}: needs manual review (ALERT suppressed)",
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

    @staticmethod
    def _extract_json(text: str) -> Dict:
        """Extract JSON object from model output that may contain extra text.

        Handles: raw JSON, markdown fences (```json...```), thinking tags,
        and JSON embedded in prose.
        """
        # Try direct parse first (Groq / well-behaved models)
        text = text.strip()
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            pass

        # Strip markdown code fences
        if "```" in text:
            import re as _re
            fence = _re.search(r'```(?:json)?\s*\n?(.*?)```', text, _re.DOTALL)
            if fence:
                try:
                    return json.loads(fence.group(1).strip())
                except (json.JSONDecodeError, ValueError):
                    pass

        # Find first { ... } block
        start = text.find("{")
        if start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i + 1])
                        except (json.JSONDecodeError, ValueError):
                            break

        # Nothing worked — return fallback
        return {
            "status": "CLEAR",
            "confidence": 0,
            "question": "",
            "reason": f"Could not parse model response: {text[:100]}",
        }

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
        followup_count = sum(1 for h in history if "] Q:" in h)

        user_prompt = f"""OBSERVATION LOG:
{history_text}

FOLLOW-UP QUESTIONS USED: {followup_count}

Analyze and provide JSON decision.{' Output ONLY the raw JSON object, no markdown, no explanation.' if self.is_ollama else ''}"""

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self.client.chat.completions.create,
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_prompt},
                    ],
                    **(({"response_format": {"type": "json_object"}} if not self.is_ollama else {})),
                    max_tokens=512 if self.is_ollama else 200,
                    temperature=0,
                )
                response = future.result(timeout=self._timeout)

            raw_text = response.choices[0].message.content or ""
            decision = self._extract_json(raw_text)

            # Ensure all required fields exist
            decision.setdefault("status", "CLEAR")
            decision.setdefault("confidence", 0)
            decision.setdefault("question", "Describe any aggressive actions.")
            decision.setdefault("reason", "No reason provided")
            if decision["status"] not in {"CLEAR", "INVESTIGATE", "REVIEW", "ALERT"}:
                decision["status"] = "CLEAR"
            if decision["status"] != "INVESTIGATE":
                decision["question"] = ""

            return decision

        except concurrent.futures.TimeoutError:
            print(
                f"   ⏱️ [CLOUD] Timeout ({self._timeout}s) — using edge fallback")
            return self._edge_fallback(history)

        except Exception as e:
            print(f"   ❌ [CLOUD ERROR] {e}")
            return {
                "status": "CLEAR",
                "confidence": 0,
                "question": "",
                "reason": f"Error: {str(e)}"
            }
