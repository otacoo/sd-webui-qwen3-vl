import re


def clean_output(text):
    if text is None:
        return ""
    text = str(text).strip()

    if text.startswith("```") or text.startswith("~~~"):
        lines = text.splitlines()
        if lines and (lines[0].startswith("```") or lines[0].startswith("~~~")):
            lines = lines[1:]
        if lines and (lines[-1].startswith("```") or lines[-1].startswith("~~~")):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    for prefix in ("Output:", "Result:", "Answer:", "Caption:", "Tags:"):
        lowered = text.lower()
        if lowered.startswith(prefix.lower()):
            text = text[len(prefix):].strip()

    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)

    if len(text) >= 2:
        quote = text[0]
        if quote in ("\"", "'", "“", "‘") and text[-1] in ("\"", "'", "”", "’"):
            if (quote, text[-1]) in (("\"", "\""), ("'", "'"), ("“", "”"), ("‘", "’")):
                text = text[1:-1].strip()

    return text


def normalize_tags(text):
    parts = []
    seen = set()
    for part in re.split(r"[,\n;]", text):
        tag = part.strip()
        if not tag:
            continue
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        parts.append(tag)
    return ", ".join(parts)


def strip_punctuation_for_filename(name):
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name)