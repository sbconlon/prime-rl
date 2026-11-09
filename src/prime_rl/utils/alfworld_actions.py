"""ALFWorld action-string helpers shared by the probe's gate metrics and the
branched-dataset collect (plan branched-dataset, Phase 4 §2).

The probe set is not drawn from the train split, so the gate splits the
oracle-agreement rate by the action's *verb class* rather than by per-game
labelled sets (master doc §8, G-C).
"""

from __future__ import annotations

# The ALFWorld command verbs. "move" is ALFWorld's second put-syntax
# ("move <obj> to <recep>") and is frequent enough to keep apart from "other".
VERB_CLASSES: tuple[str, ...] = (
    "go",
    "take",
    "put",
    "move",
    "open",
    "close",
    "clean",
    "heat",
    "cool",
    "use",
    "examine",
    "inventory",
    "look",
)


def verb_class(action: str | None) -> str:
    """First token of the action string when it is a known verb, else "other".

    Two-word receptacle names ("go to sinkbasin 1", "put soapbar 1 in/on
    sinkbasin 1") do not matter: only the first token is read.
    """
    if not action:
        return "other"
    first = str(action).strip().split(" ", 1)[0].lower()
    return first if first in VERB_CLASSES else "other"
