"""Deterministic explanations of recorded comparison metrics, not model rankings."""

MEASURES = {
    "baseline": (
        "Accuracy is the percentage of predictions that match the expected label. The chart "
        "compares overall accuracy and accuracy on supporting and refuting cases separately. "
        "Higher values indicate better performance on the evaluated cases."
    ),
    "consistency": (
        "The chart groups claims by whether the model was correct every run, wrong every run, "
        "unstable, or incomplete. The headline score measures identical verdicts across all "
        "five runs. Repeating a verdict does not necessarily mean it is correct."
    ),
    "robustness": (
        "The charts compare accuracy before and after evidence is reordered or a distractor "
        "is added, using the same eligible cases within each test. A smaller accuracy drop "
        "indicates less degradation. Verdict stability is reported separately from accuracy."
    ),
    "sensitivity": (
        "The chart shows the percentage of claim pairs answered correctly with both supporting "
        "and refuting evidence. Higher values indicate better evidence sensitivity on these "
        "pairs. Changing the verdict alone is not enough: both answers must be correct."
    ),
}


def compare_scores(names, values, measure):
    """Compare unrounded scores, without calling rounded equality an exact tie."""
    a, b = values
    if a is None or b is None:
        missing = ", ".join(name for name, value in zip(names, values) if value is None)
        return f"{measure.capitalize()} cannot be compared because a score is unavailable for {missing}."
    if a == b:
        return f"{names[0]} and {names[1]} have equal {measure}, at {a:.1%}."
    if f"{a:.1%}" == f"{b:.1%}":
        return (
            f"Both models display {a:.1%} for {measure}; their unrounded scores differ "
            "by less than 0.1 percentage point."
        )
    lead, other = (0, 1) if a > b else (1, 0)
    return (
        f"{names[lead]} scores higher on {measure}: {values[lead]:.1%}, "
        f"compared with {values[other]:.1%} for {names[other]}."
    )


def interpretation(kind, metrics, names, *, demo=False):
    if kind == "baseline":
        sentences = [
            compare_scores(names, [m["overall_accuracy"] for m in metrics], "overall accuracy")
        ]
        for label, description in (
            ("SUPPORTED", "supporting cases"),
            ("REFUTED", "refuting cases"),
        ):
            sentences.append(
                compare_scores(
                    names, [m["class_accuracy"].get(label) for m in metrics], description
                )
            )
    elif kind == "consistency":
        sentences = [
            compare_scores(
                names, [m["exact_consistency_rate"] for m in metrics], "verdict repeatability"
            )
        ]
        for name, m in zip(names, metrics):
            states = m["stability_classification"]
            sentences.append(
                f"Across {m['total_cases']:,} claims, {name} was correct every run on "
                f"{states['STABLE_CORRECT']['count']:,} and wrong every run on "
                f"{states['STABLE_WRONG']['count']:,}."
            )
    elif kind == "robustness":
        sentences = []
        for variant, label in (
            ("reorder", "reordered evidence"),
            ("distractor", "added distractors"),
        ):
            sentences.append(
                compare_scores(
                    names,
                    [m[variant]["perturbed_accuracy"] for m in metrics],
                    f"accuracy with {label}",
                )
            )
            for name, m in zip(names, metrics):
                block = m[variant]
                before, after = block["control_accuracy"], block["perturbed_accuracy"]
                if before is not None and after is not None:
                    sentences.append(
                        f"With {label}, {name}'s accuracy changed from {before:.1%} to {after:.1%}."
                    )
        sentences.append(
            "Higher changed accuracy does not necessarily mean a smaller drop from the original score."
        )
    else:
        sentences = [
            compare_scores(
                names,
                [m["correct_sensitivity_rate"] for m in metrics],
                "correct evidence sensitivity",
            )
        ]
        for name, m in zip(names, metrics):
            sentences.append(
                f"{name} answered both evidence versions correctly for {m['correct_sensitivity_count']:,} of {m['total_pairs']:,} pairs."
            )
    sentences.append(
        "This quick demo is too small to establish a reliable model ranking."
        if demo
        else "These comparisons describe the evaluated cases, not overall model quality."
    )
    return " ".join(sentences)
