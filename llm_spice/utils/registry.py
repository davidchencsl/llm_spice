from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")

MODEL_REGISTRY = {}


def register_model(name: str, hf_id: str, **kwargs):
    def deco(cls: type[T]) -> type[T]:
        MODEL_REGISTRY[name] = (cls, hf_id, kwargs)
        return cls

    return deco


PROCESSOR_REGISTRY = {}


def register_processor(name: str, **kwargs):
    def deco(cls: type[T]) -> type[T]:
        kwargs.update({"name": name})
        PROCESSOR_REGISTRY[name] = (cls, kwargs)
        return cls

    return deco


MEMORY_REGISTRY = {}


def register_memory(name: str, **kwargs):
    def deco(cls: type[T]) -> type[T]:
        kwargs.update({"name": name})
        MEMORY_REGISTRY[name] = (cls, kwargs)
        return cls

    return deco


TECHNODE_REGISTRY = {}


def register_technode(name: str, **kwargs):
    def deco(cls: type[T]) -> type[T]:
        kwargs.update({"name": name})
        TECHNODE_REGISTRY[name] = (cls, kwargs)
        return cls

    return deco


TRACE_REGISTRY = {}


def register_trace(name: str, **kwargs):
    def deco(cls: type[T]) -> type[T]:
        kwargs.update({"name": name})
        TRACE_REGISTRY[name] = (cls, kwargs)
        return cls

    return deco


NETWORK_REGISTRY = {}


def register_network(name: str, **kwargs):
    def deco(cls: type[T]) -> type[T]:
        kwargs.update({"name": name})
        NETWORK_REGISTRY[name] = (cls, kwargs)
        return cls

    return deco
