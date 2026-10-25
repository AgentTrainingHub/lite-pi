"""Semantic accents; important explanations retain the terminal's foreground."""

from prompt_toolkit.styles import Style

STYLE = Style.from_dict(
    {
        "brand": "ansicyan bold",
        "assistant": "ansicyan bold",
        "user": "ansimagenta bold",
        "tool": "ansiblue bold",
        "success": "ansigreen bold",
        "failure": "ansired bold",
        "attention": "bold",
        "meta": "nobold",
        "code": "underline",
        "border": "ansicyan",
    }
)
