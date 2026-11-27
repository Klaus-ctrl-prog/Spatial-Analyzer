"""Belegte Geraetegrenzen Mosconi Pico 8/10 (Quelle: AGENTS.md 4.3, docs/SMART453_BASS_SPEC.md).

Nur Grenzen, die in den Projektdokumenten belegt sind. Nicht belegte
Eigenschaften (Kanalzahl, PEQ-Typen ausser PEQ, Frequenz-/Gain-/Q-Bereiche,
max. Delay) werden bewusst NICHT vorausgesetzt oder geprueft.
"""

from __future__ import annotations

DELAY_GRID_MS = 0.02  # Hardware-Granularitaet (nicht 0,01 ms)
CROSSOVER_SLOPES = ("12dB-LR", "24dB-LR")
MAX_PEQ_PER_CHANNEL = 6
PHASE_OPTIONS_DEG = (0, 180)
TWEETER_MIN_HPF_HZ = 2500.0
CHANNEL_ROLES = ("tweeter", "midrange", "woofer", "subwoofer")
