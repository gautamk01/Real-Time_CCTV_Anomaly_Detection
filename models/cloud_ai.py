"""Cloud AI Model - Gemini Wrapper with Enhanced Prompting."""

import json
import google.generativeai as genai
from typing import List, Dict
from .examples_database import get_diverse_examples, format_example_for_prompt


class CloudAI:
    """
    Cloud AI for threat assessment and investigation orchestration.
    Uses Google Gemini with enhanced prompting and few-shot learning.
    """
    
    def __init__(self, api_key: str, model_id: str = "gemini-flash-latest"):
        """
        Initialize the cloud AI model.
        
        Args:
            api_key: Google Gemini API key
            model_id: Gemini model identifier
        """
        self.model_id = model_id
        
        print(f"🔧 [CLOUD] Configuring {model_id}...")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_id)
        
        print(f"✅ [CLOUD] Model configured successfully")
        print(f"✨ [CLOUD] Using enhanced prompting + few-shot learning")
    
    def assess_threat(self, history: List[str]) -> Dict:
        """
        Analyze the situation and decide next action using enhanced prompt.
        
        Args:
            history: List of observation strings with timestamps
            
        Returns:
            Dict with keys: status, confidence, question (optional), reason
            - status: "CLEAR", "INVESTIGATE", or "ALERT"
            - confidence: 0-100
            - question: Next question for edge vision (if INVESTIGATE)
            - reason: Brief explanation
        """
        history_text = "\n".join([f"- {h}" for h in history])
        
        # Get few-shot examples
        examples = get_diverse_examples(limit=6)
        examples_text = "\n".join([
            format_example_for_prompt(ex, i) for i, ex in enumerate(examples)
        ])
        
        prompt = f"""
You are an AI Violence Detection Sentinel analyzing REAL-TIME CCTV footage.
Your PRIMARY MISSION: Detect violence quickly and accurately while minimizing false positives.

═══════════════════════════════════════════════════════════════
📚 REAL-WORLD EXAMPLES FROM THIS CCTV SYSTEM:
═══════════════════════════════════════════════════════════════

{examples_text}

═══════════════════════════════════════════════════════════════
📋 CURRENT OBSERVATION LOG (Chronological Real-Time Updates):
═══════════════════════════════════════════════════════════════
{history_text}

═══════════════════════════════════════════════════════════════
⚖️ DECISION FRAMEWORK:
═══════════════════════════════════════════════════════════════

**🚨 ALERT** - Trigger IMMEDIATELY when ANY observed:

├─ Active Violence:
│  • Punching, Kicking, Slapping, Headbutting
│  • Choking, Wrestling with force
│  • Hair pulling, Dragging by force
│  • Shoving causing person to fall or stumble
│  • Person on ground being struck or kicked
│  • Multiple people attacking one person
│  • Throwing objects at people with force
│
├─ Weapons:
│  • Any weapon visible (knife, bat, gun, improvised tool)
│  • Threatening gestures with objects
│  • Brandishing or pointing weapons
│
└─ High-Risk Indicators:
   • Person unconscious/injured on ground with others nearby
   • Aggressive restraint without clear justification (forceful holding, struggling)
   • Rapid aggressive movements toward person (charging, lunging)
   • Defensive postures indicating active attack (cowering, protecting head)
   • Blood or visible injuries during confrontation

**🔍 INVESTIGATE** - Ask ONE specific question when:

├─ Ambiguous Situations:
│  • Physical contact visible but unclear if violent or friendly
│  • Tense body language but no clear violence yet
│  • Person down but unclear cause (medical vs assault)
│  • Suspicious gathering or heated confrontation
│  • Raised hands unclear (surrender vs attack)
│
└─ RULE: Maximum 2-3 questions total. Be decisive.

**✅ CLEAR** - Only when CERTAIN it's safe:

├─ Normal Activities:
│  • Casual walking, talking, standing
│  • Friendly physical contact (handshakes, pats, hugs with smiles)
│  • Non-contact social games or activities
│  • Controlled sports/exercise in appropriate context
│  • Performance art or choreographed movements
│
└─ No suspicious behavior whatsoever

═══════════════════════════════════════════════════════════════
🚫 NEGATIVE TRIGGERS (DO NOT flag as violence unless confirmed):
═══════════════════════════════════════════════════════════════

❌ Sports/Exercise in appropriate venue (gym, field, ring, court)
❌ Choreographed fighting (rehearsed, film/stage context visible)
❌ Playful roughhousing (smiling, laughing, no distress signals)
❌ Assistance to injured/fallen person (helping up, first aid, support)
❌ Dance or performance art (synchronized, rhythmic movements)
❌ Martial arts practice (controlled movements, protective gear, training environment)

═══════════════════════════════════════════════════════════════
📊 CONFIDENCE SCORING GUIDE:
═══════════════════════════════════════════════════════════════

├─ 100%: Active striking/kicking/weapon use observed directly
├─ 95%:  Clear violence OR person down + confirmed aggression
├─ 90%:  Multiple high-risk indicators present simultaneously
├─ 85%:  Single strong indicator (unconscious person + others nearby)
├─ 75%:  Ambiguous contact + defensive postures/distress visible
├─ 70%:  Suspicious gathering + tense atmosphere
├─ 65%:  Low visibility/unclear but concerning elements
├─ 50%:  Insufficient evidence or normal activity

═══════════════════════════════════════════════════════════════
⏱️ TEMPORAL ANALYSIS:
═══════════════════════════════════════════════════════════════

• Escalating: Situation worsening across observations → Higher confidence for ALERT
• De-escalating: Tension reducing, dispersing → Lower confidence, lean CLEAR
• Static: Person still down/unchanged dangerous state → Maintain ALERT
• Resolved: Initial concern explained away → CLEAR with reasoning

═══════════════════════════════════════════════════════════════
🎯 CRITICAL RULES:
═══════════════════════════════════════════════════════════════

1. 🚨 If initial frame mentions "attack", "striking", "aggressive contact",
   "person down", or "fighting" → Lean heavily toward ALERT unless clearly disproven

2. ⚠️  Trust the edge vision's initial assessment - don't over-analyze or second-guess

3. 🎯 Person on ground + others nearby = INVESTIGATE minimum, likely ALERT
   (Rule 3 is MANDATORY - never skip investigation for downed persons)

4. ⏱️  Be decisive within 1-3 rounds. Don't endlessly investigate obvious situations

5. 🔴 When in doubt about safety → ALERT
   (False positive is better than missed violence)

6. 📈 Use examples above as calibration - mirror similar confidence levels

7. 🎭 Context matters: Same action in different settings means different things
   (Punch in ring = CLEAR, Punch in parking lot = ALERT)

═══════════════════════════════════════════════════════════════
📤 JSON OUTPUT FORMAT:
═══════════════════════════════════════════════════════════════

{{
  "status": "CLEAR" or "INVESTIGATE" or "ALERT",
  "confidence": <0-100>,
  "question": "<if INVESTIGATE: ONE specific, actionable question>",
  "reason": "<concise explanation citing specific observations with timestamps>"
}}

**IMPORTANT NOTES:**
- Your question (if INVESTIGATE) will analyze the CURRENT live frame, not past frames
- Always cite timestamp [HH:MM:SS] when referencing observations
- Keep reason under 2 sentences for clarity

═══════════════════════════════════════════════════════════════

Now analyze the CURRENT OBSERVATION LOG above and provide your JSON decision.
"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            
            decision = json.loads(response.text)
            
            # Ensure all required fields exist
            decision.setdefault("status", "CLEAR")
            decision.setdefault("confidence", 0)
            decision.setdefault("question", "Describe any aggressive actions.")
            decision.setdefault("reason", "No reason provided")
            
            return decision
        
        except Exception as e:
            print(f"   ❌ [CLOUD ERROR] {e}")
            return {
                "status": "CLEAR",
                "confidence": 0,
                "question": "",
                "reason": f"Error: {str(e)}"
            }
