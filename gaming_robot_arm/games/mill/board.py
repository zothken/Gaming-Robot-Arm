"""Brettdefinitionen fuer Muehle mit A1..C8-Labels."""

from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple

BOARD_LABELS: List[str] = [
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "A6",
    "A7",
    "A8",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
    "C8",
]

RINGS: Dict[str, Sequence[str]] = {
    "A": ("A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"),
    "B": ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"),
    "C": ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"),
}


def _ring_edges(labels: Sequence[str]) -> List[Tuple[str, str]]:
    edges: List[Tuple[str, str]] = []
    count = len(labels)
    for idx, src in enumerate(labels):
        dst = labels[(idx + 1) % count]
        edges.append((src, dst))
    return edges


_edges: List[Tuple[str, str]] = []
for ring in RINGS.values():
    _edges.extend(_ring_edges(ring))

_edges.extend(
    [
        ("A2", "B2"),
        ("B2", "C2"),
        ("A4", "B4"),
        ("B4", "C4"),
        ("A6", "B6"),
        ("B6", "C6"),
        ("A8", "B8"),
        ("B8", "C8"),
    ]
)

ADJACENT: Dict[str, Set[str]] = {label: set() for label in BOARD_LABELS}
for src, dst in _edges:
    ADJACENT[src].add(dst)
    ADJACENT[dst].add(src)

MILLS: List[Tuple[str, str, str]] = [
    ("A1", "A2", "A3"),
    ("A3", "A4", "A5"),
    ("A5", "A6", "A7"),
    ("A7", "A8", "A1"),
    ("B1", "B2", "B3"),
    ("B3", "B4", "B5"),
    ("B5", "B6", "B7"),
    ("B7", "B8", "B1"),
    ("C1", "C2", "C3"),
    ("C3", "C4", "C5"),
    ("C5", "C6", "C7"),
    ("C7", "C8", "C1"),
    ("A2", "B2", "C2"),
    ("A4", "B4", "C4"),
    ("A6", "B6", "C6"),
    ("A8", "B8", "C8"),
]

MILLS_BY_POSITION: Dict[str, List[Tuple[str, str, str]]] = {label: [] for label in BOARD_LABELS}
for mill in MILLS:
    for pos in mill:
        MILLS_BY_POSITION[pos].append(mill)

__all__ = ["ADJACENT", "BOARD_LABELS", "MILLS", "MILLS_BY_POSITION", "RINGS"]
