"""Geraete-Adapter (Paket C): uebersetzen und validieren nur.

Akustik- und EQ-Berechnung bleiben ausschliesslich im Kern (eq_solver.py,
pipeline.py). Adapter lesen `filter_export.json` (siehe filter_export.py),
fuehren keine Messung/Optimierung durch und sprechen niemals Hardware oder
REW an.
"""
