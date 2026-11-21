# Spatial Analyzer

Mehrpositions-Analyzer fuer REW-Messungen (Room EQ Wizard) der Wohnzimmer-Anlage
KEF LS50 Wireless Gen1 + Saxx deepSOUND DS 120 DSP. Getrennte L/R-Auswertung
ueber 11 Messpositionen (Sofamitte + 10 Wanderpositionen), energetische
Mittelung, Bassresonanz- und Subwooferanalyse.

Lizenz: MIT. Eigenstaendiges Projekt, getrennt von `Klaus-ctrl-prog/KEFplayer`.

Vollstaendige fachliche Spezifikation: [`docs/SPECIFICATION.md`](docs/SPECIFICATION.md).

## Status

Phase 1 (Datenimport, Frequenzraster, energetische Mittelung) — siehe
`docs/SPECIFICATION.md`, Abschnitt 4.

## Lizenz-Hinweis

Keine Code-Uebernahme aus AGPL-/GPL-Projekten (SECS 2R, pierreaubert/autoeq
bzw. sotf). Alle Kernmodule sind eigene Implementierungen nach Spezifikation.
Der `.mdat`-Parser folgt der `secs-mdat`-Datenmodell-Spezifikation
(Feldnamen/-bedeutungen), enthaelt aber keinen aus SECS 2R uebernommenen Code.

## Entwicklung

```bash
pip install -e ".[dev]"
pytest
```
