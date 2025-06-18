"""
Modular Debate Scenario Template

This script allows you to quickly set up a two-actor philosophical debate using Concordia's prefab system.
Just provide entity parameters and a debate topic/context.

Usage Example:
    python debate_template.py

Or import and call run_debate() from another script.
"""

import os
from dotenv import load_dotenv
from concordia.prefabs.entity.basic_with_plan import Entity as BasicEntity
from concordia.prefabs.game_master.dialogic import GameMaster as DebateGameMaster
from concordia.associative_memory.basic_associative_memory import AssociativeMemoryBank
from pyscrai.embedding.sentence_transformers import embedder
from concordia.language_model.openrouter_model import OpenRouterLanguageModel

# Load environment variables
load_dotenv()

# --- Utility: Setup memory and model ---
def setup_memory_and_model():
    memory_bank = AssociativeMemoryBank()
    memory_bank.set_embedder(embedder)
    lm = OpenRouterLanguageModel(
        model_name="mistralai/mistral-small-3.1-24b-instruct:free",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )
    return memory_bank, lm

# --- Main Debate Runner ---
def run_debate(entity1_params, entity2_params, topic, premise, turns=6):
    memory_bank, lm = setup_memory_and_model()

    # Build entities
    entity1 = BasicEntity(
        params=entity1_params,
        description=entity1_params.get('description', '')
    ).build(model=lm, memory_bank=memory_bank)

    entity2 = BasicEntity(
        params=entity2_params,
        description=entity2_params.get('description', '')
    ).build(model=lm, memory_bank=memory_bank)

    # Build game master
    game_master = DebateGameMaster(entities=[entity1, entity2]).build(
        model=lm, memory_bank=memory_bank
    )

    # Run the debate
    print(f"\nDebate Topic: {topic}\n{'='*60}")
    print(f"Premise: {premise}\n{'='*60}")
    for entity in [entity1, entity2]:
        entity.observe(premise)

    for turn in range(turns):
        current_entity = entity1 if turn % 2 == 0 else entity2
        print(f"\nTurn {turn+1}: {current_entity.name}'s perspective...")
        response = current_entity.act()
        print(f"{current_entity.name}: {response}\n{'-'*50}")
        entity1.observe(response)
        entity2.observe(response)

# --- Example Usage ---
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
    context = (
        "A philosopher and a scientist meet to debate whether humans truly have free will, "
        "or if every action is determined by prior causes."
    )
    run_debate(entity1_params, entity2_params, topic, context, turns=6)
