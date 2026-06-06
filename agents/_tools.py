import re

_TYPE_TO_JSON_SCHEMA: dict[type, str] = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def python_type_to_json_schema(tp) -> str:
    origin = getattr(tp, "__origin__", None)
    if origin in (list, tuple, set, frozenset):
        return "array"
    if origin is dict:
        return "object"
    return _TYPE_TO_JSON_SCHEMA.get(tp, "string")


def parse_param_descriptions(docstring: str | None) -> dict[str, str]:
    if not docstring:
        return {}
    match = re.search(
        r"(?:^|\n)\s*(?:Args?|Arguments?|Parameters?)\s*:\s*\n(.*?)(?=\n\s*\n|\Z)",
        docstring,
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return {}
    descriptions: dict[str, str] = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)", line)
        if m:
            descriptions[m.group(1)] = m.group(2).strip()
    return descriptions
