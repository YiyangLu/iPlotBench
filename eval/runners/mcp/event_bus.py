"""Event bus for decoupled event distribution."""

from typing import Any, Callable

EventHandler = Callable[[dict[str, Any]], None]


class EventBus:
    """Simple synchronous event bus for tool coordination."""

    def __init__(self):
        self._subscribers: dict[str, list[EventHandler]] = {}

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Register handler for event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Remove handler for event type."""
        if event_type in self._subscribers:
            self._subscribers[event_type] = [
                h for h in self._subscribers[event_type] if h != handler
            ]

    def publish(self, event_type: str, data: dict[str, Any]) -> None:
        """Publish event to all subscribers."""
        if event_type in self._subscribers:
            for handler in self._subscribers[event_type]:
                try:
                    handler(data)
                except Exception as e:
                    print(f"Event handler error for {event_type}: {e}")

    def clear_subscribers(self, event_type: str | None = None) -> None:
        """Clear subscribers for event type or all."""
        if event_type:
            self._subscribers.pop(event_type, None)
        else:
            self._subscribers.clear()


# Global singleton
_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Get global event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus
