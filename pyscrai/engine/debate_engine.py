"""Debate engine implementation for philosophical debates."""

from collections.abc import Mapping, Sequence
from typing import Any

from concordia.typing import entity as entity_lib
from concordia.environment import engine
from concordia.document import interactive_document

class DebateEngine(engine.Engine):
    """Engine for managing philosophical debates between entities."""

    def __init__(self):
        self.current_turn = 0
        self.history = []

    def make_observation(
        self,
        game_master: entity_lib.Entity,
        entity: entity_lib.Entity,
    ) -> str:
        """Create an observation for the entity based on the current state."""
        # Return the last event if there is one, otherwise return empty string
        return self.history[-1] if self.history else ""

    def next_acting(
        self,
        game_master: entity_lib.Entity,
        entities: Sequence[entity_lib.Entity],
    ) -> tuple[entity_lib.Entity, entity_lib.ActionSpec]:
        """Determine which entity should act next and what their action spec should be."""
        # Alternate between entities
        current_entity = entities[self.current_turn % len(entities)]

        # Create a debate-specific action spec
        action_spec = entity_lib.free_action_spec(
            call_to_action=(
                f"As {current_entity.name}, continue the philosophical debate about human existence. "
                "Consider the previous statements and respond with your perspective."
            )
        )

        return current_entity, action_spec

    def resolve(
        self,
        game_master: entity_lib.Entity,
        event: str,
    ) -> None:
        """Process and record the debate event."""
        self.history.append(event)
        self.current_turn += 1

    def terminate(
        self,
        game_master: entity_lib.Entity,
    ) -> bool:
        """Determine if the debate should end."""
        # We could add more sophisticated termination conditions here
        return self.current_turn >= 6  # 3 turns each for a total of 6 exchanges

    def next_game_master(
        self,
        game_master: entity_lib.Entity,
        game_masters: Sequence[entity_lib.Entity],
    ) -> entity_lib.Entity:
        """Return the game master for the next phase."""
        # For our simple debate, we'll keep the same game master
        return game_master

    def run_loop(
        self,
        game_masters: Sequence[entity_lib.Entity],
        entities: Sequence[entity_lib.Entity],
        premise: str,
        max_steps: int,
        verbose: bool = True,
        log: list[Mapping[str, Any]] | None = None,
    ):
        """Run the debate simulation loop."""
        if log is None:
            log = []

        print("\n Debate between Entities:")
        print("=" * 50)
        print(f"\nPremise: {premise}\n")
        print("=" * 50)

        # Initialize with the premise
        for entity in entities:
            entity.observe(premise)

        game_master = game_masters[0]
        step = 0

        while step < max_steps and not self.terminate(game_master):
            try:
                # Determine next actor and their action
                actor, action_spec = self.next_acting(game_master, entities)

                if verbose:
                    print(f"\nTurn {self.current_turn + 1}: {actor.name}'s perspective...")

                # Get the actor's response
                response = actor.act(action_spec)

                if verbose:
                    print(f"{actor.name}: {response}\n")
                    print("-" * 50)

                # Record the event
                self.resolve(game_master, response)

                # Let all entities observe the response
                for entity in entities:
                    entity.observe(response)

                # Log the turn if requested
                if log is not None:
                    log.append({
                        'turn': self.current_turn,
                        'actor': actor.name,
                        'response': response
                    })

                step += 1

            except Exception as e:
                print(f"Error during turn {self.current_turn}: {str(e)}")
                break

        return log
