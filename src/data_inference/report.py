def build_report(text, label, explanation, retrieved_docs=None):

    return {
        "input_text": text,
        "prediction": label,
        "explanation": explanation,

        "risk_level": (
            "high" if "False" in explanation else
            "medium" if "Unverified" in explanation else
            "low"
        ),

        "evidence_count": len(retrieved_docs) if retrieved_docs else 0
    }