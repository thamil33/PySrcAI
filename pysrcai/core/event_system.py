"""Event system for PySrcAI.

This module provides an event bus implementation for PySrcAI that enables
communication between different modules in the simulation.
"""

from typing import Dict, List, Any, Callable, Optional
import time
import uuid
import threading
import queue


class Event:
    """Represents an event in the PySrcAI system."""
    
    def __init__(self, event_type: str, data: Any = None, source: Optional[str] = None):
        """Initialize a new event.
        
        Args:
            event_type: Type of the event.
            data: Data associated with the event.
            source: Source of the event (e.g., module name).
        """
        self.id = str(uuid.uuid4())
        self.event_type = event_type
        self.data = data
        self.source = source
        self.timestamp = time.time()


EventHandler = Callable[[Event], None]


class EventBus:
    """Event bus for communication between PySrcAI modules."""
    
    def __init__(self):
        """Initialize the event bus."""
        self._handlers: Dict[str, List[EventHandler]] = {}
        self._global_handlers: List[EventHandler] = []
        self._event_history: List[Event] = []
        self._max_history = 1000
        self._lock = threading.RLock()
        
        # For async handling
        self._async_enabled = False
        self._event_queue: queue.Queue = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
    
    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Subscribe a handler to a specific event type.
        
        Args:
            event_type: Type of event to subscribe to.
            handler: Function to call when the event occurs.
        """
        with self._lock:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)
    
    def subscribe_all(self, handler: EventHandler) -> None:
        """Subscribe a handler to all events.
        
        Args:
            handler: Function to call when any event occurs.
        """
        with self._lock:
            self._global_handlers.append(handler)
    
    def unsubscribe(self, event_type: str, handler: EventHandler) -> bool:
        """Unsubscribe a handler from a specific event type.
        
        Args:
            event_type: Type of event to unsubscribe from.
            handler: Handler to remove.
            
        Returns:
            True if the handler was found and removed, False otherwise.
        """
        with self._lock:
            if event_type in self._handlers:
                try:
                    self._handlers[event_type].remove(handler)
                    return True
                except ValueError:
                    return False
        return False
    
    def unsubscribe_all(self, handler: EventHandler) -> bool:
        """Unsubscribe a handler from all events.
        
        Args:
            handler: Handler to remove.
            
        Returns:
            True if the handler was found and removed, False otherwise.
        """
        with self._lock:
            try:
                self._global_handlers.remove(handler)
                return True
            except ValueError:
                return False
        return False
    
    def publish(self, event: Event) -> None:
        """Publish an event to all subscribed handlers.
        
        Args:
            event: The event to publish.
        """
        with self._lock:
            # Add to history
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)
            
            if self._async_enabled:
                # Add to queue for async processing
                self._event_queue.put(event)
            else:
                # Process synchronously
                self._dispatch_event(event)
    
    def publish_event(self, event_type: str, data: Any = None, source: Optional[str] = None) -> Event:
        """Create and publish an event.
        
        Args:
            event_type: Type of the event.
            data: Data associated with the event.
            source: Source of the event.
            
        Returns:
            The published event.
        """
        event = Event(event_type, data, source)
        self.publish(event)
        return event
    
    def _dispatch_event(self, event: Event) -> None:
        """Dispatch an event to all relevant handlers.
        
        Args:
            event: The event to dispatch.
        """
        # Call specific handlers
        if event.event_type in self._handlers:
            for handler in self._handlers[event.event_type]:
                try:
                    handler(event)
                except Exception as e:
                    print(f"[EventBus] Error in handler for event {event.event_type}: {e}")
        
        # Call global handlers
        for handler in self._global_handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"[EventBus] Error in global handler for event {event.event_type}: {e}")
    
    def enable_async(self) -> None:
        """Enable asynchronous event processing."""
        with self._lock:
            if not self._async_enabled:
                self._async_enabled = True
                self._running = True
                self._worker_thread = threading.Thread(target=self._event_worker, daemon=True)
                self._worker_thread.start()
    
    def disable_async(self) -> None:
        """Disable asynchronous event processing and wait for all events to process."""
        with self._lock:
            if self._async_enabled:
                self._running = False
                if self._worker_thread:
                    self._worker_thread.join()
                self._async_enabled = False
    
    def _event_worker(self) -> None:
        """Worker thread for processing events asynchronously."""
        while self._running:
            try:
                # Get an event with a timeout to allow checking _running
                try:
                    event = self._event_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Process the event
                self._dispatch_event(event)
                self._event_queue.task_done()
                
            except Exception as e:
                print(f"[EventBus] Error in event worker: {e}")
    
    def get_history(self, event_type: Optional[str] = None, limit: int = 100) -> List[Event]:
        """Get event history, optionally filtered by type.
        
        Args:
            event_type: Optional type to filter by.
            limit: Maximum number of events to return.
            
        Returns:
            List of events.
        """
        with self._lock:
            if event_type is None:
                return list(self._event_history[-limit:])
            else:
                return [e for e in self._event_history if e.event_type == event_type][-limit:]
    
    def clear_history(self) -> None:
        """Clear the event history."""
        with self._lock:
            self._event_history.clear()


# Global event bus instance
event_bus = EventBus()
