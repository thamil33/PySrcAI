from pyscrai.components.debate_template import run_debate

if __name__ == "__main__":
    entity1_params = {
        'name': 'Libertarian',
        'goal': 'Argue for the existence of free will',
        'force_time_horizon': False,
        'description': 'A philosopher who believes in human freedom and moral responsibility.'
    }
    entity2_params = {
        'name': 'Determinist',
        'goal': 'Argue that all events are determined by prior causes',
        'force_time_horizon': False,
        'description': 'A scientist who believes every event is the result of prior states and natural laws.'
    }
    topic = "Free Will vs. Determinism"
    premise = (
        "A philosopher and a scientist meet to debate whether humans truly have free will, "
        "or if every action is determined by prior causes."
    )
    run_debate(entity1_params, entity2_params, topic, premise, turns=2)
