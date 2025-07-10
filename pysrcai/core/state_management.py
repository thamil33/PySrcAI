"""State management for PySrcAI.

This module provides classes and utilities for managing simulation state
in a way that works across different simulation types.
"""

from typing import Dict, Any, List, Optional, TypeVar, Generic, Callable
import json
import copy
import time
import os


StateT = TypeVar('StateT')


class StateSnapshot(Generic[StateT]):
    """A snapshot of a simulation state at a point in time."""
    
    def __init__(self, state: StateT, timestamp: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None):
        """Initialize a state snapshot.
        
        Args:
            state: The state to snapshot.
            timestamp: Optional timestamp for the snapshot.
            metadata: Optional metadata about the snapshot.
        """
        self.state = copy.deepcopy(state)
        self.timestamp = timestamp or time.time()
        self.metadata = metadata or {}
    
    def with_metadata(self, key: str, value: Any) -> 'StateSnapshot[StateT]':
        """Create a new snapshot with additional metadata.
        
        Args:
            key: Metadata key.
            value: Metadata value.
            
        Returns:
            New snapshot with updated metadata.
        """
        new_metadata = self.metadata.copy()
        new_metadata[key] = value
        return StateSnapshot(self.state, self.timestamp, new_metadata)


class StateContainer(Generic[StateT]):
    """Container for managing simulation state with history."""
    
    def __init__(self, initial_state: StateT, max_history: int = 100):
        """Initialize the state container.
        
        Args:
            initial_state: Initial simulation state.
            max_history: Maximum number of historical states to keep.
        """
        self._current_state = copy.deepcopy(initial_state)
        self._history: List[StateSnapshot[StateT]] = []
        self._max_history = max_history
        self._change_listeners: List[Callable[[StateT, StateT], None]] = []
        
        # Take initial snapshot
        self._add_snapshot(initial_state)
    
    def get_current(self) -> StateT:
        """Get the current state.
        
        Returns:
            Current state.
        """
        return copy.deepcopy(self._current_state)
    
    def set_current(self, state: StateT, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Set the current state.
        
        Args:
            state: New state.
            metadata: Optional metadata about the state change.
        """
        old_state = self._current_state
        self._current_state = copy.deepcopy(state)
        self._add_snapshot(state, metadata)
        
        # Notify listeners
        for listener in self._change_listeners:
            try:
                listener(old_state, self._current_state)
            except Exception as e:
                print(f"[StateContainer] Error in state change listener: {e}")
    
    def update(self, updater: Callable[[StateT], StateT], metadata: Optional[Dict[str, Any]] = None) -> StateT:
        """Update the current state using a function.
        
        Args:
            updater: Function that takes the current state and returns a new state.
            metadata: Optional metadata about the state change.
            
        Returns:
            The new state after the update.
        """
        new_state = updater(copy.deepcopy(self._current_state))
        self.set_current(new_state, metadata)
        return copy.deepcopy(new_state)
    
    def _add_snapshot(self, state: StateT, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a snapshot to the history.
        
        Args:
            state: State to snapshot.
            metadata: Optional metadata about the snapshot.
        """
        snapshot = StateSnapshot(state, metadata=metadata)
        self._history.append(snapshot)
        
        # Trim history if needed
        while len(self._history) > self._max_history:
            self._history.pop(0)
    
    def get_history(self) -> List[StateSnapshot[StateT]]:
        """Get the state history.
        
        Returns:
            List of state snapshots.
        """
        return list(self._history)
    
    def get_snapshot(self, index: int) -> Optional[StateSnapshot[StateT]]:
        """Get a specific snapshot from the history.
        
        Args:
            index: Index of the snapshot to get.
            
        Returns:
            The snapshot or None if the index is out of range.
        """
        if 0 <= index < len(self._history):
            return self._history[index]
        return None
    
    def revert_to(self, index: int) -> bool:
        """Revert to a specific snapshot.
        
        Args:
            index: Index of the snapshot to revert to.
            
        Returns:
            True if the revert was successful, False otherwise.
        """
        if 0 <= index < len(self._history):
            snapshot = self._history[index]
            old_state = self._current_state
            self._current_state = copy.deepcopy(snapshot.state)
            
            # Trim history to the snapshot
            self._history = self._history[:index + 1]
            
            # Notify listeners
            for listener in self._change_listeners:
                try:
                    listener(old_state, self._current_state)
                except Exception as e:
                    print(f"[StateContainer] Error in state change listener: {e}")
            
            return True
        
        return False
    
    def add_change_listener(self, listener: Callable[[StateT, StateT], None]) -> None:
        """Add a listener for state changes.
        
        Args:
            listener: Function that takes the old and new state.
        """
        self._change_listeners.append(listener)
    
    def remove_change_listener(self, listener: Callable[[StateT, StateT], None]) -> bool:
        """Remove a state change listener.
        
        Args:
            listener: The listener to remove.
            
        Returns:
            True if the listener was found and removed, False otherwise.
        """
        try:
            self._change_listeners.remove(listener)
            return True
        except ValueError:
            return False


class JsonSerializableStateContainer(StateContainer[Dict[str, Any]]):
    """State container that can be serialized to and from JSON."""
    
    def save_to_file(self, file_path: str) -> None:
        """Save the state container to a file.
        
        Args:
            file_path: Path to the file to save to.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Serialize state and history
        data = {
            "current_state": self._current_state,
            "history": [
                {
                    "state": snapshot.state,
                    "timestamp": snapshot.timestamp,
                    "metadata": snapshot.metadata
                }
                for snapshot in self._history
            ],
            "max_history": self._max_history
        }
        
        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'JsonSerializableStateContainer':
        """Load a state container from a file.
        
        Args:
            file_path: Path to the file to load from.
            
        Returns:
            Loaded state container.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create state container
        container = cls(data["current_state"], data["max_history"])
        
        # Clear default history
        container._history.clear()
        
        # Load history
        for snapshot_data in data["history"]:
            snapshot = StateSnapshot(
                snapshot_data["state"],
                snapshot_data["timestamp"],
                snapshot_data["metadata"]
            )
            container._history.append(snapshot)
        
        # Set current state
        container._current_state = copy.deepcopy(data["current_state"])
        
        return container
