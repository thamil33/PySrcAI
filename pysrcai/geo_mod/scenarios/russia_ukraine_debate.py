"""Scenario configuration for the Russia-Ukraine debate."""

from concordia.typing import prefab as prefab_lib

# 1. Define the participants in the debate
RUSSIA = "Russia"
UKRAINE = "Ukraine"

# 2. Define the high-level premise of the simulation
PREMISE = (
    "A special session of the UN has been convened to host a direct, moderated "
    "debate between the representatives of Russia and Ukraine. The goal is for "
    "each nation to clearly state its position and desired outcomes regarding "
    "the ongoing conflict."
)

# 3. Define the configuration for all instances in the simulation
INSTANCES = [
    # --- The Nation Entities ---
    prefab_lib.InstanceConfig(
        prefab='nation_entity',  # This will be mapped to our NationEntity prefab
        role=prefab_lib.Role.ENTITY,
        params={
            'name': RUSSIA,
            'goal': "Assert geopolitical strength, ensure national security by maintaining a sphere of influence, and protect the interests of Russian-speaking populations.",
            'context': "Russia views the eastward expansion of NATO as a direct threat to its security interests. It initiated the 'special military operation' with the stated goals of 'demilitarization' and 'denazification' of Ukraine, and to protect the Donbas region. It considers Crimea to be an integral part of the Russian Federation.",
        },
    ),
    prefab_lib.InstanceConfig(
        prefab='nation_entity',
        role=prefab_lib.Role.ENTITY,
        params={
            'name': UKRAINE,
            'goal': "Defend national sovereignty and territorial integrity, achieve full withdrawal of foreign troops from its internationally recognized borders, and pursue deeper integration with Western political and security structures like the EU and NATO.",
            'context': "Ukraine considers itself a sovereign nation defending against an unprovoked act of aggression. It is receiving significant military and financial aid from Western countries and is actively seeking to reclaim all territories, including Crimea and the Donbas, that have been occupied since 2014.",
        },
    ),
    # --- The Game Master ---
    prefab_lib.InstanceConfig(
        prefab='generic_gm',  # Use the generic game master prefab
        role=prefab_lib.Role.GAME_MASTER,
        params={
            'name': "UN Moderator",
            'acting_order': 'fixed',  # Fixed order turns
        },
    ),
]
