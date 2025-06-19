# This script sets up a debate scenario between an angel and a demon
from pyscrai.components.debate_template import run_debate

if __name__ == "__main__":
    entity1_params = {
        'name': 'An Angel',
        'goal': 'Defend the beauty and purpose of human existence',
        'force_time_horizon': False,
        'description':"A benevolent and wise angel with an optimistic perspective on human life."
    }
    entity2_params = {
        'name': 'A Demon',
        'goal': 'Argue for nihilism and the meaninglessness of existence',
        'force_time_horizon': False,
        'description':"A cunning and mischievous demon with a cynical view of human existence."
    }
    topic = "Free Will vs. Determinism"
    premise = """In a timeless void between Heaven and Hell, fundamental questions of existence echo through eternity.
A demon and an angel meet to debate the meaning and purpose of human life.
The demon, with their cynical perspective, argues for nihilism and the meaninglessness of existence.
The angel, drawing from their divine wisdom, defends the inherent beauty and purpose in human life."""

    run_debate(entity1_params, entity2_params, topic, premise, turns=2)
