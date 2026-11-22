# Spatial Analyzer

Mehrpositions-Analyzer fuer REW-Messungen (Room EQ Wizard) der Wohnzimmer-Anlage
KEF LS50 Wireless Gen1 + Saxx deepSOUND DS 120 DSP. Getrennte L/R-Auswertung
ueber 11 Messpositionen (Sofamitte + 10 Wanderpositionen), energetische
Mittelung, Bassresonanz- und Subwooferanalyse.

Lizenz: MIT. Eigenstaendiges Projekt, getrennt von `Klaus-ctrl-prog/KEFplayer`.

Vollstaendige fachliche Spezifikation: [`docs/SPECIFICATION.md`](docs/SPECIFICATION.md).

## Status

Phasen 1–3 abgeschlossen (siehe `docs/SPECIFICATION.md` Abschnitt 4):
Datenimport & Validierung, Frequenzraster & energetische Mittelung,
Gesamtspektrum-/Bassresonanzanalyse, Subwoofer-Optimierer,
Zielkurve/PEQ-Vorschlagslogik, Exporte/Berichte/CLI. Dazu ergaenzt:
Windows-Starterdatei (`start_analyzer.bat`), REW-Notes-Durchreichung mit
Copy-Paste-Vorlage, Schema-Validierung (`--dry-run`) und
A/B-Vergleichsmodus (`compare`).

Seit 2026-11-22 laeuft eine reine Haertungsphase ohne neue Funktionen bis
reale Messdaten ab Fruehjahr 2027 vorliegen (siehe
`docs/ROADMAP_2027.md` fuer den vollstaendigen Verlauf). Abgeschlossen:
Prioritaet 1 (Zeit-/Pegel-/Phasentreue), Prioritaet 2 (komplexe
Mehrwege-Summierung), Prioritaet 3 (Fazenda-Schwellenwert-Kalibrierung,
schliesst GitHub-Issue #2), Prioritaet 4 (`mdat.py`-Parser-Haertung,
bezieht sich auf Issue #7, schliesst es aber nicht), Prioritaet 5
(Import-/Validierungspfad-Fuzzing), Prioritaet 6 Teil 1
(Sensitivitaetsanalyse der Optimierer-Gewichtung). Prioritaet 6 Teil 2
(`validate_no_harm()`-Grenzfall, `eq_solver.py`-Sensitivitaet) und
Prioritaet 7 (Dokumentations-/Test-Konsistenz) folgen.

Bewusst zurueckgestellt bis Fruehjahr 2027 (reale Messdaten
erforderlich, siehe die offenen GitHub-Issues): `.mdat`-Binaerparser-
Validierung gegen eine echte REW-`.mdat`-Datei (`parse_bytes()` bleibt
bewusst nicht implementiert, Issue #7), ETC-Reflexions-Schwellenwert-
Kalibrierung anhand realer Wohnzimmer-Messungen (Issue #5),
FDW-/Window-Kennzeichnung in REW-API-Metadaten (Issue #4), optionaler
Car-Audio-Export (Issue #6). `mdat.py` ist dabei bewusst noch nicht mit
`pipeline.py` verdrahtet (die CLI/Pipeline liest weiterhin ihre eigene
JSON-Zwischenform, siehe Docstring von `pipeline.py`).

Das separate, bereits abgeschlossene Smart-453-Bass-Dokumentationspaket
(`docs/SMART453_BASS_SPEC.md`) ruht ebenfalls bis zu realen
Fahrzeugmessungen ab Fruehjahr 2027 und wird von dieser Haertungsphase
nicht beruehrt.

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
