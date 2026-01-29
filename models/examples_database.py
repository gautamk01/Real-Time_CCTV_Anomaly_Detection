"""Annotated Examples Database for Few-Shot Learning.

This file contains real scenarios from CCTV test runs, manually annotated
with ground truth decisions. These examples are used to improve the Cloud AI's
decision-making accuracy through few-shot prompting.
"""

# Real examples from test runs (extracted from terminal output)
ANNOTATED_EXAMPLES = [
    # ============ ALERT Examples ============
    {
        "category": "ALERT",
        "observations": [
            "[00:00:23] Man lying on ground in tunnel, appears unconscious",
            "[00:00:27] One person kicking the man on ground"
        ],
        "verdict": "ALERT",
        "confidence": 100,
        "reason": "Active violence: kicking defenseless person on ground (Critical Rule 1)",
        "rounds": 2
    },
    {
        "category": "ALERT",
        "observations": [
            "[00:00:08] Person kneeling/on ground, multiple people surrounding",
            "[00:00:12] Individuals forcibly holding person down",
            "[00:00:15] Restraining or aggressively posturing toward kneeling person"
        ],
        "verdict": "ALERT",
        "confidence": 95,
        "reason": "Aggressive restraint without clear justification + person on ground (Rule 3)",
        "rounds": 3
    },
    {
        "category": "ALERT",
        "observations": [
            "[00:01:18] Group engaged in physical altercation on city street",
            "[00:01:18] Fighting and pushing each other, chaotic scene"
        ],
        "verdict": "ALERT",
        "confidence": 95,
        "reason": "Active violence: explicit fighting and pushing in altercation",
        "rounds": 1
    },
    {
        "category": "ALERT",
        "observations": [
            "[00:01:23] Nighttime street, people actively fighting or pushing",
            "[00:01:23] Scene tense and chaotic, people in distress or fear"
        ],
        "verdict": "ALERT",
        "confidence": 95,
        "reason": "Active violence: fighting with visible distress indicators",
        "rounds": 1
    },
    {
        "category": "ALERT",
        "observations": [
            "[00:00:35] Man lying on ground in tunnel, unconscious or injured",
            "[00:00:35] Other people nearby in potentially dangerous situation"
        ],
        "verdict": "ALERT",
        "confidence": 95,
        "reason": "High-Risk Indicator: unconscious/injured person on ground with others nearby",
        "rounds": 1
    },
    
    # ============ INVESTIGATE Examples ============
    {
        "category": "INVESTIGATE",
        "observations": [
            "[00:00:23] Man lying on ground in tunnel, several people visible",
            "[00:00:27] One person standing near trying to assist",
            "[00:00:31] Person in defensive posture, potentially restraining"
        ],
        "verdict": "INVESTIGATE",
        "confidence": 75,
        "reason": "Person on ground + others nearby requires clarification (Rule 3)",
        "rounds": 3
    },
    {
        "category": "INVESTIGATE",
        "observations": [
            "[00:00:35] Person lying still on ground in parking garage",
            "[00:00:39] No movement or responsive behavior visible",
            "[00:00:42] Standing people showing aggressive movements"
        ],
        "verdict": "ALERT",  # Escalated from INVESTIGATE
        "confidence": 95,
        "reason": "Aggressive movements toward unresponsive victim confirmed",
        "rounds": 3
    },
    {
        "category": "INVESTIGATE",
        "observations": [
            "[00:00:08] Person kneeling, tense situation developing",
            "[00:00:12] Others watching and possibly assisting"
        ],
        "verdict": "INVESTIGATE",
        "confidence": 75,
        "reason": "Ambiguous situation: person down + others nearby needs investigation",
        "rounds": 2
    },
    
    # ============ CLEAR Examples ============
    {
        "category": "CLEAR",
        "observations": [
            "[00:00:40] Two people standing close together",
            "[00:00:40] Engaged in conversation, one speaking and one listening"
        ],
        "verdict": "CLEAR",
        "confidence": 95,
        "reason": "Normal social interaction, no aggression or violence indicators",
        "rounds": 1
    },
    {
        "category": "CLEAR",
        "observations": [
            "[00:01:13] Nighttime street scene, people walking and standing",
            "[00:01:13] Casual gathering, motorcycles parked, skateboards present"
        ],
        "verdict": "CLEAR",
        "confidence": 90,
        "reason": "Casual social scene, no violence or high-risk behavior",
        "rounds": 1
    },
    {
        "category": "CLEAR",
        "observations": [
            "[00:01:05] Group of people standing together",
            "[00:01:09] No active punching, shoving, or defensive postures visible"
        ],
        "verdict": "CLEAR",
        "confidence": 85,
        "reason": "Possible verbal confrontation but no physical violence confirmed",
        "rounds": 2
    },
    
    # ============ Edge Cases ============
    {
        "category": "INVESTIGATE→CLEAR",
        "observations": [
            "[00:00:36] Person lying on ground, blurry image",
            "[00:00:40] No physical contact or aggression visible",
            "[00:00:44] Context shifts to martial arts practice"
        ],
        "verdict": "INVESTIGATE",
        "confidence": 70,
        "reason": "Blurry feed + person on ground requires investigation despite unclear context",
        "rounds": 3,
        "notes": "Low clarity scenario - maintained investigation due to Rule 3"
    },
    {
        "category": "ALERT",
        "observations": [
            "[00:00:23] Person lying on ground, possibly injured",
            "[00:00:30] Person providing aid appears to be forcibly controlling",
            "[00:00:30] Struggling rather than supportive interaction"
        ],
        "verdict": "ALERT",
        "confidence": 90,
        "reason": "Forceful restraint without clear supportive intent on downed person",
        "rounds": 3
    },
    {
        "category": "INVESTIGATE",
        "observations": [
            "[00:00:35] Dimly lit parking garage, person on ground",
            "[00:00:39] Person approaching with hands together, non-aggressive",
            "[00:00:42] Attempting to move person on ground"
        ],
        "verdict": "INVESTIGATE",
        "confidence": 80,
        "reason": "Ambiguous movement of downed person - could be assistance or rough handling",
        "rounds": 3
    },
    {
        "category": "INVESTIGATE",
        "observations": [
            "[00:01:05] People gathered, possible raised voices",
            "[00:01:08] Ambiguous contact, tense atmosphere"
        ],
        "verdict": "INVESTIGATE",
        "confidence": 70,
        "reason": "Tense gathering requires clarification before escalating or clearing",
        "rounds": 2
    },
]


def get_examples_by_category(category: str, limit: int = 3):
    """Get N examples of a specific category for prompting."""
    filtered = [ex for ex in ANNOTATED_EXAMPLES if ex["category"].startswith(category)]
    return filtered[:limit]


def get_diverse_examples(limit: int = 10):
    """Get a balanced mix of ALERT, INVESTIGATE, and CLEAR examples."""
    alerts = get_examples_by_category("ALERT", limit // 3)
    investigates = get_examples_by_category("INVESTIGATE", limit // 3)
    clears = get_examples_by_category("CLEAR", limit // 3)
    
    return alerts + investigates + clears


def format_example_for_prompt(example: dict, index: int) -> str:
    """Format a single example for inclusion in the prompt."""
    obs_text = "\n   ".join(example["observations"])
    
    formatted = f"""
Example {index + 1} - {example["category"]}:
   Observations:
   {obs_text}
   
   → Decision: {example["verdict"]} (Confidence: {example["confidence"]}%)
   → Reason: {example["reason"]}
   → Investigation Rounds: {example["rounds"]}
"""
    
    if "notes" in example:
        formatted += f"   → Note: {example['notes']}\n"
    
    return formatted
