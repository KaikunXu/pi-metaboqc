"""Describe processing methods with a shared, immutable registry model.

``MethodSpec`` centralizes canonical identifiers, display labels, aliases, and
handler metadata. Stage packages retain their own registries, allowing their
parameters and implementations to differ without duplicating name resolution.
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping


def _method_token(value: object) -> str:
    """Normalize a method identifier for alias lookup."""
    return "".join(char for char in str(value).upper() if char.isalnum())


@dataclass(frozen=True)
class MethodSpec:
    """Describe one processing method independently of its presentation."""

    key: str
    display_name: str
    aliases: tuple[str, ...] = ()
    handler_name: str | None = None
    supports_auto: bool = True
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "options", MappingProxyType(dict(self.options))
        )

    @property
    def tokens(self) -> set[str]:
        """Return normalized lookup tokens accepted by this specification."""
        return {
            _method_token(self.key),
            _method_token(self.display_name),
            *(_method_token(alias) for alias in self.aliases),
        }


class MethodRegistry:
    """Resolve canonical method specifications for one processing stage."""

    def __init__(self, specs: Iterable[MethodSpec]) -> None:
        """Build an alias lookup for a stage's supported methods.

        Args:
            specs: Canonical method specifications for one processing stage.

        Raises:
            ValueError: If two specifications claim the same alias.
        """
        self._specs = tuple(specs)
        self._lookup: dict[str, MethodSpec] = {}
        # Index canonical names, display labels, and aliases through the same
        # normalized token while rejecting ambiguous registrations eagerly.
        for spec in self._specs:
            for token in spec.tokens:
                existing = self._lookup.get(token)
                if existing is not None and existing != spec:
                    raise ValueError(f"Duplicate method alias: {token}")
                self._lookup[token] = spec

    def resolve(self, value: object) -> MethodSpec:
        """Return the specification matching a canonical name or alias."""
        token = _method_token(value)
        try:
            return self._lookup[token]
        except KeyError as exc:
            raise ValueError(f"Unsupported processing method: {value}") from exc

    def get(self, value: object) -> MethodSpec | None:
        """Return a matching specification, or ``None`` when unsupported."""
        return self._lookup.get(_method_token(value))

    def canonicalize(self, value: object, strict: bool = True) -> str:
        """Return a canonical key while optionally preserving unknown values."""
        spec = self.get(value)
        if spec is not None:
            return spec.key
        if strict:
            raise ValueError(f"Unsupported processing method: {value}")
        return str(value).strip()

    def display_name(self, value: object, strict: bool = False) -> str:
        """Return the configured display name for a method."""
        spec = self.get(value)
        if spec is not None:
            return spec.display_name
        if strict:
            raise ValueError(f"Unsupported processing method: {value}")
        return str(value).strip()

    def __iter__(self):
        return iter(self._specs)
