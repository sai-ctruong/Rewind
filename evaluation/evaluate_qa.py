from .official_eval import evaluate_labels


def evaluate(labels, predictor, **kwargs):
    return evaluate_labels([row for row in labels if row["task"].lower() in {"qa", "q&a", "vqa"}], predictor, **kwargs)
