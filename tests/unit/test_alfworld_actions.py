"""branched-dataset Phase 4 -- verb_class for the gate's per-class split."""

import pytest

from prime_rl.utils.alfworld_actions import VERB_CLASSES, verb_class


@pytest.mark.parametrize(
    "action, cls",
    [
        ("go to sinkbasin 1", "go"),
        ("go to desk 1", "go"),
        ("take soapbar 1 from sinkbasin 1", "take"),
        ("put soapbar 1 in/on sinkbasin 1", "put"),
        ("move soapbar 1 to sinkbasin 1", "move"),
        ("open drawer 1", "open"),
        ("close drawer 1", "close"),
        ("clean soapbar 1 with sinkbasin 1", "clean"),
        ("heat egg 1 with microwave 1", "heat"),
        ("cool tomato 1 with fridge 1", "cool"),
        ("use desklamp 1", "use"),
        ("examine shelf 1", "examine"),
        ("inventory", "inventory"),
        ("look", "look"),
        ("  Look  ", "look"),
        ("GO TO garbagecan 1", "go"),
    ],
)
def test_known_verbs(action, cls):
    assert verb_class(action) == cls
    assert cls in VERB_CLASSES


@pytest.mark.parametrize("action", ["", None, "help", "dance wildly", "<action>go</action>"])
def test_other(action):
    assert verb_class(action) == "other"
