# REW-`Notes`-Vorlage: Wohnzimmer & Auto

REW kann die Regler- und DSP-Einstellungen **externer** Geräte (Saxx deepSOUND
DS 120 DSP im Wohnzimmer, Mosconi-DSP im Auto) nicht automatisch auslesen.
Diese Einstellungen müssen deshalb bei jeder Messung von Hand ins
REW-`Notes`-Feld eingetragen werden – sonst lässt sich die Messung Monate
später nicht mehr exakt reproduzieren oder einordnen.

Beide Blöcke unten sind zum direkten Kopieren gedacht: Platzhalter ausfüllen,
nicht benötigte Zeilen weglassen, **Pflichtfelder (⚠) nie leer lassen**.

Hinweis zu `Spatial-Analyzer`: Dieser Text landet, sofern er beim Erzeugen
der Eingabe-JSON ins optionale `notes`-Feld der jeweiligen Position bzw.
des Kanals (`l`/`r`/`sub`) übernommen wird, unverändert im Berichtsabschnitt
"Mess- & DSP-Notizen (aus REW)" – siehe `pipeline.py`/`reporting.py`.

## Wohnzimmer (KEF LS50 Wireless Gen1 + Saxx DS 120 DSP)

```
Position: sofa_mitte (bzw. wander_<n>)
KEF-Lautstärke: ___ ⚠ Pflicht - wurde im Archiv wiederholt vergessen
KEF Highpass: ___ Hz, Flankensteilheit ___ dB/Oct
KEF Bass Extension: Standard / Mehr / Weniger
KEF Phasenkorrektur: ein / aus
KEF Desk-EQ: ___ dB
KEF Wall-EQ: ___ dB
KEF Höhen-Anhebung: ___ dB

Saxx Lowpass: ___ Hz, Filtertyp (z. B. Bessel 24 dB/Oct): ___
Saxx Phase: ___ Grad
Saxx Subpegel: ___ dB
Saxx EQ 1: Fc ___ Hz, Gain ___ dB, Q ___ (oder "Bypass")
Saxx EQ 2: Fc ___ Hz, Gain ___ dB, Q ___ (oder "Bypass")
Saxx EQ 3: Fc ___ Hz, Gain ___ dB, Q ___ (oder "Bypass")
(weitere EQ-Bänder nach Bedarf ergänzen)

Mess-Kontext:
Mikrofonposition/-ID: ___ ⚠ Pflicht (z. B. sofa_mitte, wander_3)
UMIK-1-Ausrichtung: 90° (Standard laut Messregeln)
Kalibrierdatei: ___
Sweep-Typ/-Länge: ___ (z. B. 256k Log Sweep)
Ausgabepegel: ___ dBFS
```

## Auto (Mosconi-DSP) – Anhang für spätere Car-Audio-Messungen

```
Preset-Name: ___ ⚠ Pflicht
Kanal Front L: Trennfrequenz(en) ___ Hz, Flankensteilheit ___ dB/Oct,
               Laufzeit ___ ms (___ cm), Phase 0°/180°
Kanal Front R: Trennfrequenz(en) ___ Hz, Flankensteilheit ___ dB/Oct,
               Laufzeit ___ ms (___ cm), Phase 0°/180°
Kanal Sub:     Trennfrequenz(en) ___ Hz, Flankensteilheit ___ dB/Oct,
               Laufzeit ___ ms (___ cm), Phase 0°/180°
EQ je Kanal: Fc ___ Hz, Gain ___ dB, Q ___ (oder "Bypass")
Subpegel: ___ dB
Messposition: ___ (z. B. Fahrersitz Ohrhöhe)
```

**Hinweis:** Ein möglicher Car-Audio-Export aus `Spatial-Analyzer` (Mosconi-
kompatibles Format) ist ausdrücklich **nicht** Teil des Projekt-Kerns (siehe
`SPECIFICATION.md` Abschnitt 1, "Abgrenzung", und GitHub-Issue zur Klärung
dieses Punkts). Dieser Block dient vorerst nur der sauberen Dokumentation
von Auto-Messungen in REW selbst.
