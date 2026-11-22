# Arbeitsplan bis Fruehjahr 2027 (Haertungsphase ohne neue Messdaten)

Dieses Dokument haelt den verabschiedeten Arbeitsplan fest, solange keine
neuen Fahrzeug- oder Wohnzimmer-Messungen moeglich sind. Grundsatz: nur
mathematische und testseitige Haertung des bestehenden `Spatial-Analyzer`
(`Klaus-ctrl-prog/Spatial-Analyzer`, Referenzstand `main`), keine neuen
Fahrzeugoptimierer, keine Erweiterung oder Aenderung von
`docs/SMART453_BASS_SPEC.md` (dieses Paket ist eigenstaendig abgeschlossen
und ruht bis zu realen Smart-453-Messungen, siehe dort).

## Prioritaet 1 - Zeit-, Pegel- und Phasentreue (Pruefung, Haertung & Tests)

Umgesetzt per PR, siehe Commit-Historie ab 2026-11-22:

1. **REW-Zeitbezug (`start_time_s`):** Gepruefter und behobener Fund: das
   Referenzmodell (`secs-mdat`, siehe `claude/secs_mdat_pruefbericht.md`
   Abschnitt 2) fuehrt `start_time_s` als eigenes Feld der Impulsantwort;
   der eigene Nachbau (`mdat.py` `MdatImpulseResponse`, `models.py`
   `ImpulseResponseData`) hatte dieses Feld bislang nicht nachgebildet - es
   waere beim Parsen stillschweigend verworfen worden (bei `mdat.py` sogar
   ohne in `unknown_fields` zu landen, da das auf der verschachtelten
   `impulse_response`-Ebene nicht erfasst wird). Behoben: Feld ergaenzt,
   Default `None` (nicht `0.0`, um "fehlt" von "ist Null" zu unterscheiden),
   Tests fuer Erhalt/Abgrenzung von `measurement_delay_s` ergaenzt.
   Architektur-Hinweis dokumentiert: `mdat.py` ist noch nicht mit
   `pipeline.py` verdrahtet (die Pipeline liest eine eigene JSON-
   Zwischenform direkt) - `start_time_s` bleibt deshalb vorerst ohne
   nachgelagerte Verwendung, das ist ein bestehender, hier nur benannter
   Architekturzustand, kein neuer Mangel.
2. **Absolute Pegelbasis:** Gepruefter Befund: `complex_response.py`
   (`to_complex`/`from_complex`) ist eine reine, punktweise Formel ohne
   jede Normalisierung - verifiziert durch neue Roundtrip-Tests ueber einen
   weiten Pegelbereich (-40 bis 120 dB) mit dokumentierter Fehlergrenze
   (1e-9 dB, reine Gleitkomma-Rundung). Keine Code-Aenderung notwendig,
   nur Docstring-Praezisierung und Testabsicherung.
3. **Fraktionale Sample-Delays:** Gepruefter Befund: `sub_optimizer.py`
   wendet Delays ausschliesslich ueber den linearen Phasenterm
   `exp(-j*2*pi*f*tau)` im Frequenzbereich an - `tau` ist ein
   Gleitkommawert, es findet keine Rundung auf ganze Samples statt. Neuer
   deterministischer Regressionstest belegt das mit einem Delay deutlich
   unterhalb einer Sample-Periode bei 48 kHz (0,0037 ms vs. ~0,0208 ms
   Sample-Periode), das exakt gefunden wird, plus Gegenprobe mit falscher
   Polaritaet. Docstring in `_complex_sum` entsprechend ergaenzt.
4. **Synthetische Referenztests & Fehlergrenzen:** siehe 1-3 - alle drei
   Punkte wurden mit deterministischen, synthetischen Tests mit bekannten
   Soll-Werten (Pegel, Delay, Polaritaet) abgesichert, Fehlergrenzen direkt
   in den Tests bzw. Docstrings dokumentiert statt impliziter Annahmen.

## Prioritaet 2 - Robustheit des Import-/Validierungspfads (Vorschlag, noch nicht begonnen)

- Fuzz-/Grenzwerttests fuer `validator.py` (z. B. NaN/Inf in `freq_hz`/
  `spl_db`, leere Arrays, extrem dichte oder extrem duenne Frequenzraster)
  ueber die bereits in `mdat.py` etablierte `AllocationBudget`-Haertung
  hinaus.
- Pruefen, ob `grid.py`s R160-Interpolation an den Bandraendern (sehr
  niedrige/hohe Frequenzen) numerisch stabil bleibt, inkl. Tests mit
  bewusst knappen Ueberlappungsbereichen (`FrequencyGrid.overlap_with`).

## Prioritaet 3 - Nachvollziehbarkeit der Optimierer-Zielfunktionen (Vorschlag, noch nicht begonnen)

- `sub_optimizer.py` und `eq_solver.py`: dokumentierte Sensitivitaetstests
  (wie stark aendert sich das Optimum bei kleinen Aenderungen der
  Modell-Gewichtung `ObjectiveWeights`?), um die bereits als Modell
  gekennzeichnete 70/30-Gewichtung nicht unbeabsichtigt als robuster
  erscheinen zu lassen, als sie ist.
- Pruefen, ob `validate_no_harm()` (Do-No-Harm-Check) auch fuer den
  Mehrpositions-Fall mit sehr wenigen Positionen (z. B. nur 2) noch
  aussagekraeftige Streuungskennzahlen liefert.

## Prioritaet 4 - Dokumentations- und Test-Konsistenz (Vorschlag, noch nicht begonnen)

- Abgleich aller Modul-Docstrings gegen `docs/SPECIFICATION.md`, ob
  zwischenzeitliche Praezisierungen (wie die in Prioritaet 1 ergaenzten
  Haertungshinweise) konsistent uebernommen sind.
- Pruefen, ob `docs/REW_NOTES_TEMPLATE.md` und die Notes-Durchreichung
  (`pipeline.py`/`reporting.py`) nach den seither ergaenzten Funktionen
  (Compare-Modus, `--dry-run`) noch vollstaendig und aktuell sind.

## Zurueckgestellt bis reale Messdaten vorliegen (ab Fruehjahr 2027)

- Jede Validierung gegen eine echte, unveraenderte REW-`.mdat`-Datei
  (`mdat.py` Abschnitt G.4 der Spezifikation; `parse_bytes()` bleibt bis
  dahin bewusst `NotImplementedError`).
- Unabhaengige Plausibilisierung von `mdat.py`/`models.py` gegen den
  REW-Text-Export bzw. die REW-API (siehe
  `claude/secs_mdat_pruefbericht.md` Abschnitt 8).
- Jede reale Kontrollmessung fuer `sub_optimizer.py`-Ergebnisse (Messregel 1
  aus den Projekt-Vorgaben: Simulation vor Messung klar als
  "simulationsbasiert, noch nicht real verifiziert" kennzeichnen).
- Das gesamte Smart-453-Themenpaket (`docs/SMART453_BASS_SPEC.md`) - ruht
  unveraendert, siehe dort.
