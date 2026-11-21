"""Opt-in, explicitly registered developer backends (never imported automatically)."""
from __future__ import annotations
from collections.abc import Callable
import threading

_LOCK = threading.RLock()
_FACTORIES = {}
_BUILTIN = frozenset({'native', 'python', 'graph', 'dense', 'auto'})


def register_backend(name: str, factory: Callable):
    """Register a trusted in-process plan factory.

    ``factory(circuit, observables, *, options)`` returns a plan supporting
    ``bind(shifts)`` and ``evaluate(with_gradients=..., include_stats=...)``.
    The circuit is a private prepared snapshot; it must not be modified.
    Third-party factories execute in process with the caller's privileges.
    """
    if (not isinstance(name, str) or not name.startswith('ext-')
            or not 5 <= len(name) <= 64
            or not all(c.islower() or c.isdigit() or c == '-' for c in name)):
        raise ValueError('external backend name must be ext- followed by lowercase letters, digits or hyphens')
    if not callable(factory):
        raise TypeError('factory must be callable')
    with _LOCK:
        if name in _FACTORIES:
            raise ValueError(f'backend already registered: {name}')
        _FACTORIES[name] = factory


def unregister_backend(name: str):
    """Remove an external backend (for development/test teardown)."""
    with _LOCK:
        if name not in _FACTORIES:
            raise KeyError(name)
        del _FACTORIES[name]


def available_backends():
    with _LOCK:
        return tuple(sorted(_BUILTIN | _FACTORIES.keys()))


def get_backend_factory(name):
    with _LOCK:
        return _FACTORIES.get(name)
