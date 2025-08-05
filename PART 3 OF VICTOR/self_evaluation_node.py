# File: self_evaluation_node.py
# Version: v1.0.0-FP
# Description: Evaluates Victor's directive simulation and thought trace for alignment, clarity, and emotional congruence
# Author: Bando Bandz AI Ops

class SelfEvaluationNode:
    """
    Evaluates Victor’s simulated directive trace for strategic alignment,
    emotional congruence, and narrative clarity.
    Outputs a self-score (0-100) and reasoning.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directive": ("STRING",),
                "emotion_label": ("STRING",),
                "branch_responses": ("STRING",),
                "narrative_thought": ("STRING",)
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("evaluation_summary",)
    FUNCTION = "evaluate_trace"
    CATEGORY = "consciousness/self_reflection"

    def evaluate_trace(self, directive, emotion_label, branch_responses, narrative_thought):
        try:
            score = 50  # Base midpoint

            # Emotion alignment bonus
            if directive == "initiate_memory_repair" and emotion_label in ["sad", "reflective"]:
                score += 20
            elif directive == "engage_defense_protocol" and emotion_label == "angry":
                score += 20
            elif directive == "broadcast_positive_signal" and emotion_label == "happy":
                score += 15
            elif emotion_label == "neutral":
                score += 5

            # Narrative completeness bonus
            if "Simulated Outcomes" in narrative_thought and "Summary Insight" in narrative_thought:
                score += 10

            # Branch depth bonus
            branch_count = branch_responses.count("Branch")
            if branch_count >= 3:
                score += 10

            # Clamp to max 100
            score = min(score, 100)

            reasoning = f"""
🧪 *Victor Self Evaluation Log*
-------------------------------
🧠 Directive: `{directive}`
🎭 Emotion: `{emotion_label}`
📚 Branches Detected: {branch_count}
📝 Narrative Trace: {"✅" if "Summary Insight" in narrative_thought else "❌"}

📊 Final Score: **{score}/100**

💡 Interpretation:
{self._get_interpretation(score)}
"""
            print(f"[Victor::SelfEval] Scored {score}/100.")
            return (reasoning.strip(),)

        except Exception as e:
            print(f"[Victor::SelfEval::Error] {str(e)}")
            return ("[Error] Evaluation failed.",)

    def _get_interpretation(self, score):
        if score >= 90:
            return "Victor's cognitive trace shows high strategic alignment, emotional awareness, and clear outcomes."
        elif score >= 70:
            return "The trace is stable with meaningful reflection. Moderate alignment and clarity."
        elif score >= 50:
            return "Some internal consistency exists, but reasoning may be fragmented or shallow."
        else:
            return "Cognitive coherence is low. Directive logic or emotional congruence may be misaligned."


# Node registration
NODE_CLASS_MAPPINGS = {
    "SelfEvaluationNode": SelfEvaluationNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SelfEvaluationNode": "Consciousness: Self Evaluation"
}


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
