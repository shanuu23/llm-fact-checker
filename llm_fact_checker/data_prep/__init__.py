"""Dataset loading, sampling, and preparation."""

# FEVER and VitaminC use the same three source labels.
LABEL_MAP = {
    "SUPPORTS": "SUPPORTED",
    "REFUTES": "REFUTED",
    "NOT ENOUGH INFO": "NOT ENOUGH INFORMATION",
}


def map_label(label: str, dataset: str) -> str:
    """Map a dataset label to the canonical label the model is asked to produce."""
    try:
        return LABEL_MAP[label]
    except KeyError as error:
        raise ValueError(f"Unknown {dataset} label: {label!r}") from error
