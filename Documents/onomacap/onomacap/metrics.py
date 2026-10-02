import unicodedata


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def cer(prediction, reference):
    return edit_distance(prediction, reference) / max(len(reference), 1)


def score_prediction(prediction, references):
    prediction = unicodedata.normalize("NFC", prediction).strip()
    references = [unicodedata.normalize("NFC", ref).strip() for ref in references]
    compact = lambda s: "".join(s.split())
    p = compact(prediction)
    refs = [compact(ref) for ref in references]
    return {
        "min_cer": min(cer(p, ref) for ref in refs),
        "min_cer_with_spaces": min(cer(prediction, ref) for ref in references),
        "min_jamo_cer": min(cer(unicodedata.normalize("NFD", p),
                                 unicodedata.normalize("NFD", ref)) for ref in refs),
        "exact_match": float(prediction in references),
        "exact_match_no_spaces": float(p in refs),
    }


def aggregate_scores(rows):
    if not rows:
        raise ValueError("No predictions to evaluate")
    keys = score_prediction("", [""]).keys()
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}
