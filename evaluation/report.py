def markdown_report(summary: dict) -> str:
    lines = ["# AIC 2026 Evaluation", ""]
    lines.extend(f"- {key}: {value:.4f}" if isinstance(value, float) else f"- {key}: {value}" for key, value in summary.items())
    return "\n".join(lines) + "\n"
