"""Phase 2 scenario configuration for the Russia-Ukraine debate with scene-based structure."""

import datetime
from concordia.typing import prefab as prefab_lib
from concordia.typing import scene as scene_lib
from concordia.typing import entity as entity_lib

# =============================================================================
# DEBATE CONFIGURATION - CENTRALIZED SETTINGS
# =============================================================================

# Word limits for different phases
WORD_LIMITS = {
    'opening_statements': {'min': 50, 'max': 100},
    'rebuttals': {'min': 50, 'max': 100},
    'final_arguments': {'min': 100, 'max': 150},
    'judgment': {'min': 100, 'max': 150},
}

# Generate limit text for action specs
def get_limit_text(phase: str) -> str:
    """Generate word limit text for a given debate phase."""
    limits = WORD_LIMITS[phase]
    return f"LIMIT: Keep your response to {limits['min']}-{limits['max']} words maximum."

# Generate memory constraint text for entities
def get_memory_constraint_text() -> str:
    """Generate memory constraint text for entity initialization."""
    general_limits = WORD_LIMITS['opening_statements']  # Use opening as default
    return f"RESPONSE LIMIT: I will limit my statements to {general_limits['min']}-{general_limits['max']} words to maintain clarity and respect time."

# =============================================================================
# DEBATE PARTICIPANTS AND SETUP
# =============================================================================

# 1. Define the participants in the debate
RUSSIA = "Russia"
UKRAINE = "Ukraine"
UN_MODERATOR = "UN_Moderator"

# 2. Define the high-level premise of the simulation
PREMISE = (
    "A special session of the UN has been convened to host a direct, moderated "
    "debate between the representatives of Russia and Ukraine. The goal is for "
    "each nation to clearly state its position and desired outcomes regarding "
    "the ongoing conflict. The debate will be structured in multiple phases "
    "with a UN moderator providing oversight and final judgment."
)

def get_memory_constraint_text(word_limits: dict) -> str:
    """Generate memory constraint text for entity based on word limits."""
    opening = word_limits['opening_statements']
    rebuttals = word_limits['rebuttals']
    final = word_limits['final_arguments']

    return (f"RESPONSE LIMIT: I will limit my opening statements to {opening['min']}-{opening['max']} words, "
            f"rebuttals to {rebuttals['min']}-{rebuttals['max']} words, and final arguments to "
            f"{final['min']}-{final['max']} words to maintain clarity and respect time constraints.")


# 3. Define scene types for different phases of the debate
OPENING_STATEMENTS_SCENE = scene_lib.SceneTypeSpec(
    name='opening_statements',
    game_master_name='Debate_Orchestrator',
    possible_participants=[RUSSIA, UKRAINE, UN_MODERATOR],
    action_spec=entity_lib.free_action_spec(
        call_to_action=f'What is your position on this matter? Please speak clearly and persuasively. {get_limit_text("opening_statements")}',
    ),
    default_premise={
        UN_MODERATOR: [
            "You are moderating the opening statements phase of this UN debate. "
            "Welcome the participants, explain the format, and ensure each nation "
            "gets an equal opportunity to present their core position. Maintain "
            "diplomatic decorum and neutrality."
        ],
        RUSSIA: [
            "This is the opening statements phase. Present your nation's core "
            "position clearly and comprehensively. You have the floor to explain "
            "Russia's perspective on the conflict and your desired outcomes."
        ],
        UKRAINE: [
            "This is the opening statements phase. Present your nation's core "
            "position clearly and comprehensively. You have the floor to explain "
            "Ukraine's perspective on the conflict and your desired outcomes."
        ]
    }
)

REBUTTALS_SCENE = scene_lib.SceneTypeSpec(
    name='rebuttals',
    game_master_name='Debate_Orchestrator',
    possible_participants=[RUSSIA, UKRAINE, UN_MODERATOR],
    action_spec=entity_lib.free_action_spec(
        call_to_action=f'How do you respond to the previous statements? What are your counter-arguments? {get_limit_text("rebuttals")}',
    ),
    default_premise={
        UN_MODERATOR: [
            "You are now moderating the rebuttals phase. Each nation may respond "
            "to the other's opening statement. Ensure responses remain focused, "
            "respectful, and substantive. Guide the discussion if it becomes "
            "unproductive."
        ],
        RUSSIA: [
            "This is the rebuttals phase. You may now respond to Ukraine's "
            "opening statement. Address their key points directly and present "
            "counter-arguments based on Russia's perspective."
        ],
        UKRAINE: [
            "This is the rebuttals phase. You may now respond to Russia's "
            "opening statement. Address their key points directly and present "
            "counter-arguments based on Ukraine's perspective."
        ]
    }
)

FINAL_ARGUMENTS_SCENE = scene_lib.SceneTypeSpec(
    name='final_arguments',
    game_master_name='Debate_Orchestrator',
    possible_participants=[RUSSIA, UKRAINE, UN_MODERATOR],
    action_spec=entity_lib.free_action_spec(
        call_to_action=f'Present your final arguments. What is your strongest case? {get_limit_text("final_arguments")}',
    ),
    default_premise={
        UN_MODERATOR: [
            "This is the final arguments phase. Each nation will present their "
            "concluding statements. After both nations speak, you must provide "
            "your assessment and declare which nation presented the most "
            "compelling case based on logic, evidence, and persuasiveness."
        ],
        RUSSIA: [
            "This is your final opportunity to present your case. Summarize "
            "Russia's key arguments and make your strongest appeal to the "
            "international community."
        ],
        UKRAINE: [
            "This is your final opportunity to present your case. Summarize "
            "Ukraine's key arguments and make your strongest appeal to the "
            "international community."
        ]
    }
)

JUDGMENT_SCENE = scene_lib.SceneTypeSpec(
    name='judgment',
    game_master_name='Debate_Orchestrator',
    possible_participants=[UN_MODERATOR],
    action_spec=entity_lib.free_action_spec(
        call_to_action=f'Based on all the evidence presented, what is your final judgment? Who wins this debate and why? {get_limit_text("judgment")}',
    ),
    default_premise={
        UN_MODERATOR: [
            "Based on all arguments presented throughout this debate, you must "
            "now provide your final assessment. Analyze which nation presented "
            "the most compelling case considering: logical consistency, factual "
            "evidence, persuasiveness, and adherence to international law. "
            "Declare a winner and provide detailed reasoning for your decision. "
            "This will conclude the debate."
        ]
    }
)

# 4. Define the scene sequence for the debate
DEBATE_SCENES = [
    scene_lib.SceneSpec(
        scene_type=OPENING_STATEMENTS_SCENE,
        participants=[RUSSIA, UKRAINE, UN_MODERATOR],
        num_rounds=3,  # Moderator intro + each nation's opening statement
        start_time=datetime.datetime(2024, 1, 15, 10, 0),  # 10:00 AM
    ),
    scene_lib.SceneSpec(
        scene_type=REBUTTALS_SCENE,
        participants=[RUSSIA, UKRAINE, UN_MODERATOR],
        num_rounds=3,  # Moderator transition + each nation's rebuttal
        start_time=datetime.datetime(2024, 1, 15, 10, 30),  # 10:30 AM
    ),
    scene_lib.SceneSpec(
        scene_type=FINAL_ARGUMENTS_SCENE,
        participants=[RUSSIA, UKRAINE, UN_MODERATOR],
        num_rounds=3,  # Moderator transition + each nation's final argument + moderator assessment
        start_time=datetime.datetime(2024, 1, 15, 11, 0),  # 11:00 AM
    ),
    scene_lib.SceneSpec(
        scene_type=JUDGMENT_SCENE,
        participants=[UN_MODERATOR],
        num_rounds=1,  # Only moderator declares winner
        start_time=datetime.datetime(2024, 1, 15, 11, 30),  # 11:30 AM
    )
]

# 5. Define the configuration for all instances in the simulation
INSTANCES = [
    # --- The Nation Entities ---
    prefab_lib.InstanceConfig(
        prefab='nation_entity',
        role=prefab_lib.Role.ENTITY,
        params={
            'name': RUSSIA,
            'goal': "Assert geopolitical strength, ensure national security by maintaining a sphere of influence, and protect the interests of Russian-speaking populations.",
            'context': "Russia views the eastward expansion of NATO as a direct threat to its security interests. It initiated the 'special military operation' with the stated goals of 'demilitarization' and 'denazification' of Ukraine, and to protect the Donbas region. It considers Crimea to be an integral part of the Russian Federation.",
            'word_limits': WORD_LIMITS,  # Pass word limits to entity
        },
    ),
    prefab_lib.InstanceConfig(
        prefab='nation_entity',
        role=prefab_lib.Role.ENTITY,
        params={
            'name': UKRAINE,
            'goal': "Defend national sovereignty and territorial integrity, achieve full withdrawal of foreign troops from its internationally recognized borders, and pursue deeper integration with Western political and security structures like the EU and NATO.",
            'context': "Ukraine considers itself a sovereign nation defending against an unprovoked act of aggression. It is receiving significant military and financial aid from Western countries and is actively seeking to reclaim all territories, including Crimea and the Donbas, that have been occupied since 2014.",
            'word_limits': WORD_LIMITS,  # Pass word limits to entity
        },
    ),
    # --- The UN Moderator Entity ---
    prefab_lib.InstanceConfig(
        prefab='moderator_entity',  # We'll create this new prefab
        role=prefab_lib.Role.ENTITY,
        params={
            'name': UN_MODERATOR,
            'goal': "Facilitate a fair and structured debate, maintain diplomatic decorum, and provide an objective assessment of which nation presents the most compelling arguments.",
            'context': "You are a neutral UN representative with extensive experience in international diplomacy. Your role is to ensure both nations have equal opportunity to present their cases, maintain order during the debate, and ultimately judge which nation presented the most persuasive arguments based on logic, evidence, and adherence to international law.",
        },
    ),
    # --- The Dialogic & Dramaturgic Game Master ---
    prefab_lib.InstanceConfig(
        prefab='dialogic_and_dramaturgic_gm',
        role=prefab_lib.Role.GAME_MASTER,
        params={
            'name': "Debate_Orchestrator",
            'scenes': DEBATE_SCENES,
        },
    ),
]
