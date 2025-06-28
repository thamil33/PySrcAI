"""Basic two-person debate scenario."""

import datetime
from concordia.typing import prefab as prefab_lib
from concordia.typing import scene as scene_lib
from concordia.typing import entity as entity_lib

# Word limits for phases
WORD_LIMITS = {
    "opening": {"min": 20, "max": 40},
    "rebuttal": {"min": 20, "max": 40},
    "closing": {"min": 30, "max": 60},
}

def get_limit_text(phase: str) -> str:
    limits = WORD_LIMITS[phase]
    return f"Keep response {limits['min']}-{limits['max']} words."

PARTICIPANT_A = "Cats_Advocate"
PARTICIPANT_B = "Dogs_Advocate"
MODERATOR = "Debate_Moderator"

PREMISE = (
    "Two participants debate whether cats or dogs make better pets. "
    "A moderator oversees the discussion and declares a winner."
)

OPENING_SCENE = scene_lib.SceneTypeSpec(
    name="opening",
    game_master_name="Debate_Orchestrator",
    possible_participants=[PARTICIPANT_A, PARTICIPANT_B, MODERATOR],
    action_spec=entity_lib.free_action_spec(
        call_to_action=f"Present your opening statement. {get_limit_text('opening')}"
    ),
    default_premise={
        MODERATOR: [
            "Welcome participants. Present your positions succinctly."
        ],
        PARTICIPANT_A: [
            "State why cats are superior pets."
        ],
        PARTICIPANT_B: [
            "State why dogs are superior pets."
        ],
    },
)

REBUTTAL_SCENE = scene_lib.SceneTypeSpec(
    name="rebuttal",
    game_master_name="Debate_Orchestrator",
    possible_participants=[PARTICIPANT_A, PARTICIPANT_B, MODERATOR],
    action_spec=entity_lib.free_action_spec(
        call_to_action=f"Respond to the prior statement. {get_limit_text('rebuttal')}"
    ),
    default_premise={
        MODERATOR: [
            "Time for rebuttals. Address the other's points directly."
        ],
        PARTICIPANT_A: ["Counter the dog's argument."],
        PARTICIPANT_B: ["Counter the cat's argument."],
    },
)

CLOSING_SCENE = scene_lib.SceneTypeSpec(
    name="closing",
    game_master_name="Debate_Orchestrator",
    possible_participants=[PARTICIPANT_A, PARTICIPANT_B, MODERATOR],
    action_spec=entity_lib.free_action_spec(
        call_to_action=f"Give your closing remarks. {get_limit_text('closing')}"
    ),
    default_premise={
        MODERATOR: [
            "Final remarks before judgment. Summarize briefly."
        ],
        PARTICIPANT_A: ["Summarize why cats win."],
        PARTICIPANT_B: ["Summarize why dogs win."],
    },
)

JUDGMENT_SCENE = scene_lib.SceneTypeSpec(
    name="judgment",
    game_master_name="Debate_Orchestrator",
    possible_participants=[MODERATOR],
    action_spec=entity_lib.free_action_spec(
        call_to_action="Who won the debate and why?"
    ),
    default_premise={
        MODERATOR: [
            "Consider logic and persuasiveness then choose a winner."
        ]
    },
)

SCENES = [
    scene_lib.SceneSpec(
        scene_type=OPENING_SCENE,
        participants=[PARTICIPANT_A, PARTICIPANT_B, MODERATOR],
        num_rounds=3,
        start_time=datetime.datetime(2025, 1, 1, 12, 0),
    ),
    scene_lib.SceneSpec(
        scene_type=REBUTTAL_SCENE,
        participants=[PARTICIPANT_A, PARTICIPANT_B, MODERATOR],
        num_rounds=3,
        start_time=datetime.datetime(2025, 1, 1, 12, 15),
    ),
    scene_lib.SceneSpec(
        scene_type=CLOSING_SCENE,
        participants=[PARTICIPANT_A, PARTICIPANT_B, MODERATOR],
        num_rounds=3,
        start_time=datetime.datetime(2025, 1, 1, 12, 30),
    ),
    scene_lib.SceneSpec(
        scene_type=JUDGMENT_SCENE,
        participants=[MODERATOR],
        num_rounds=1,
        start_time=datetime.datetime(2025, 1, 1, 12, 45),
    ),
]

INSTANCES = [
    prefab_lib.InstanceConfig(
        prefab="participant_entity",
        role=prefab_lib.Role.ENTITY,
        params={
            "name": PARTICIPANT_A,
            "goal": "Prove cats are better pets.",
            "context": "Cat enthusiast with many feline companions.",
            "word_limits": WORD_LIMITS,
        },
    ),
    prefab_lib.InstanceConfig(
        prefab="participant_entity",
        role=prefab_lib.Role.ENTITY,
        params={
            "name": PARTICIPANT_B,
            "goal": "Prove dogs are better pets.",
            "context": "Dog trainer with years of experience.",
            "word_limits": WORD_LIMITS,
        },
    ),
    prefab_lib.InstanceConfig(
        prefab="moderator_entity",
        role=prefab_lib.Role.ENTITY,
        params={
            "name": MODERATOR,
            "goal": "Moderate fairly and declare a winner.",
            "context": "Neutral host.",
        },
    ),
    prefab_lib.InstanceConfig(
        prefab="orchestrator",
        role=prefab_lib.Role.GAME_MASTER,
        params={
            "name": "Debate_Orchestrator",
            "scenes": SCENES,
        },
    ),
]

