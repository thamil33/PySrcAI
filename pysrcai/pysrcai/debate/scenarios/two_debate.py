"""Simple two participant debate scenario."""

import datetime
from concordia.typing import prefab as prefab_lib
from concordia.typing import scene as scene_lib
from concordia.typing import entity as entity_lib

# Central configuration
WORD_LIMITS = {
    "opening": {"min": 30, "max": 60},
    "rebuttal": {"min": 30, "max": 60},
    "closing": {"min": 50, "max": 80},
    "judgment": {"min": 50, "max": 80},
}


PARTICIPANT_A = "Alice"
PARTICIPANT_B = "Bob"
MODERATOR = "Moderator"

PREMISE = (
    "Two participants engage in a friendly debate on a generic topic. "
    "A moderator ensures fair play and decides the winner."
)


def _limit_text(phase: str) -> str:
    limits = WORD_LIMITS[phase]
    return f"LIMIT: Keep your response to {limits['min']}-{limits['max']} words."


OPENING_SCENE = scene_lib.SceneTypeSpec(
    name="opening",
    game_master_name="orchestrator",
    possible_participants=[PARTICIPANT_A, PARTICIPANT_B, MODERATOR],
    action_spec=entity_lib.free_action_spec(
        call_to_action=f'Present your opening statement. {_limit_text("opening")}'
    ),
    default_premise={
        PARTICIPANT_A: ["State your position clearly."],
        PARTICIPANT_B: ["State your position clearly."],
        MODERATOR: ["Introduce the debate and give each participant the floor."],
    },
)

REBUTTAL_SCENE = scene_lib.SceneTypeSpec(
    name="rebuttal",
    game_master_name="orchestrator",
    possible_participants=[PARTICIPANT_A, PARTICIPANT_B, MODERATOR],
    action_spec=entity_lib.free_action_spec(
        call_to_action=f'Respond to the previous statement. {_limit_text("rebuttal")}'
    ),
    default_premise={
        PARTICIPANT_A: ["Respond to your opponent."],
        PARTICIPANT_B: ["Respond to your opponent."],
        MODERATOR: ["Guide the rebuttal phase."],
    },
)

CLOSING_SCENE = scene_lib.SceneTypeSpec(
    name="closing",
    game_master_name="orchestrator",
    possible_participants=[PARTICIPANT_A, PARTICIPANT_B, MODERATOR],
    action_spec=entity_lib.free_action_spec(
        call_to_action=f'Provide your closing remarks. {_limit_text("closing")}'
    ),
    default_premise={
        PARTICIPANT_A: ["Summarize your stance."],
        PARTICIPANT_B: ["Summarize your stance."],
        MODERATOR: ["Invite closing remarks from each participant."],
    },
)

JUDGMENT_SCENE = scene_lib.SceneTypeSpec(
    name="judgment",
    game_master_name="orchestrator",
    possible_participants=[MODERATOR],
    action_spec=entity_lib.free_action_spec(
        call_to_action=f'Who won the debate and why? {_limit_text("judgment")}'
    ),
    default_premise={
        MODERATOR: ["Declare the winner with reasoning."],
    },
)

SCENES = [
    scene_lib.SceneSpec(
        scene_type=OPENING_SCENE,
        participants=[MODERATOR, PARTICIPANT_A, PARTICIPANT_B],
        num_rounds=3,
        start_time=datetime.datetime(2025, 1, 1, 9, 0),
    ),
    scene_lib.SceneSpec(
        scene_type=REBUTTAL_SCENE,
        participants=[MODERATOR, PARTICIPANT_A, PARTICIPANT_B],
        num_rounds=3,
        start_time=datetime.datetime(2025, 1, 1, 9, 15),
    ),
    scene_lib.SceneSpec(
        scene_type=CLOSING_SCENE,
        participants=[MODERATOR, PARTICIPANT_A, PARTICIPANT_B],
        num_rounds=3,
        start_time=datetime.datetime(2025, 1, 1, 9, 30),
    ),
    scene_lib.SceneSpec(
        scene_type=JUDGMENT_SCENE,
        participants=[MODERATOR],
        num_rounds=1,
        start_time=datetime.datetime(2025, 1, 1, 9, 45),
    ),
]

INSTANCES = [
    prefab_lib.InstanceConfig(
        prefab="participant",
        role=prefab_lib.Role.ENTITY,
        params={
            "name": PARTICIPANT_A,
            "goal": "Argue in favour of position A.",
            "context": "",
            "word_limits": WORD_LIMITS,
        },
    ),
    prefab_lib.InstanceConfig(
        prefab="participant",
        role=prefab_lib.Role.ENTITY,
        params={
            "name": PARTICIPANT_B,
            "goal": "Argue in favour of position B.",
            "context": "",
            "word_limits": WORD_LIMITS,
        },
    ),
    prefab_lib.InstanceConfig(
        prefab="moderator",
        role=prefab_lib.Role.ENTITY,
        params={
            "name": MODERATOR,
            "goal": "Ensure fair play and judge the debate.",
            "context": "",
        },
    ),
    prefab_lib.InstanceConfig(
        prefab="dialogic_and_dramaturgic_gm",
        role=prefab_lib.Role.GAME_MASTER,
        params={
            "name": "orchestrator",
            "scenes": SCENES,
        },
    ),
]
