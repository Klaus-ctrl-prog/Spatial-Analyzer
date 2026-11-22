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

## Prioritaet 2 - Komplexe Mehrwege-Summierung (Pruefung, Haertung & Tests)

Umgesetzt per PR, siehe Commit-Historie ab 2026-11-22. Alle Tests sind
ausdruecklich algorithmische Verifikation an synthetischen Daten, keine
Fahrzeug- oder Raumvalidierung.

1. **Synthetische Mehrwege-Referenztests:** Exakte analytische Referenzwerte
   fuer konstruktive (0 Grad, +6,0206 dB bei gleich starken, phasengleichen
   Quellen) und destruktive (180 Grad, exakte Ausloeschung) Addition waren
   fuer `complex_response.py` und `lr_sum.py` bereits vorhanden
   (`test_in_phase_equal_sources_double_amplitude`,
   `test_out_of_phase_equal_sources_cancel`); fuer `sub_optimizer.py`
   ergaenzt (`TestExactAnalyticConstructiveSum`, Score ≈ 0 bei exakt
   analytischem Zielwert, Fehlergrenze 1e-9).
2. **Frequenzraster- & NaN-Robustheit - echter Fund:** `optimize()` nutzte
   `np.mean()`/`np.std()` statt `np.nanmean()`/`np.nanstd()` - ein einziger
   NaN/Inf-Wert im Uebergangsband (z. B. aus einer Interpolation ausserhalb
   des Quellbereichs, siehe `grid.py`) machte `score_grid` an JEDEM
   Gitterpunkt zu NaN. Da `NaN < best_score` immer `False` ist, wurde
   `best` dann nie aktualisiert und die Funktion lieferte unbemerkt den
   bedeutungslosen Default (Gain 0 dB, Delay 0 ms, erste Polaritaet,
   Score `inf`) zurueck, als sei das ein echtes Ergebnis - in `pipeline.py`
   haette das (ohne den dort vorhandenen `except ValueError`, der bisher
   nichts zu fangen hatte) unbemerkt als scheinbar echte Empfehlung im
   Report gelandet. Behoben: expliziter `np.isfinite()`-Guard ueber das
   Uebergangsband, der jetzt einen klaren `ValueError` wirft (von
   `pipeline.py`s bestehendem Fehlerpfad abgefangen und als "Subwoofer-
   Optimierung uebersprungen: ..." im Report vermerkt, statt eine
   bedeutungslose Empfehlung zu zeigen).
3. **Mehrdeutige Optimierungsergebnisse & Fehlergrenzen:** Dokumentierter
   und testgestuetzter Befund zur 360-Grad-Periodizitaet bei schmalen
   Baendern (z. B. dem Modenband 60-62 Hz): Delays, die sich um ein
   ganzzahliges Vielfaches der Periode der Bandmittenfrequenz unterscheiden
   (~16,39 ms bei 61 Hz), sind bei schmaler Bandbreite NAHEZU, aber nicht
   exakt, entartet (Score-Unterschied < 1e-3 Score-Einheiten in einem
   konkreten Testfall), waehrend eine halbe Periode (destruktiv statt
   konstruktiv) einen um mehr als drei Groessenordnungen schlechteren Score
   ergibt. In `optimize()`s Docstring als bekannte Eigenschaft der
   Zielfunktion dokumentiert, nicht als Implementierungsfehler.

## Prioritaet 3 - Bassresonanz-Bewertung / Fazenda-Kalibrierung (GitHub-Issue #2)

Umgesetzt per PR, siehe Commit-Historie ab 2026-11-22. Schliesst Issue #2.

1. **Abgleich gegen Fazenda, Stephenson & Goldberg (2015):** Die bisherigen
   `PERCEPTUAL_THRESHOLD_MODEL`-Stuetzpunkte in `bass_resonance.py` waren
   laut eigenem Docstring eine selbst gewaehlte, nicht verifizierte
   Naeherung. Jetzt direkt aus dem Fliesstext der Originalpublikation
   (JASA 137(3), pp. 1088-1098) uebernommen - Abschnitt "Conclusion",
   "artificial stimuli"-Mittelwerte: 0.90s bei 32Hz, 0.30s bei 63Hz,
   0.27s bei 100Hz, 0.18s bei 150Hz, 0.17s bei 200Hz. Bewusst NICHT die
   "music stimuli"-Kurve verwendet (methodisch weiter von der eigenen
   toenerzeugten Abklingmessung entfernt). Dokumentierte, nicht selbst
   aufgeloeste Einschraenkung: die Publikation widerspricht sich selbst
   beim Knickpunkt (Abstract: 63Hz: Ergebnisteil: ~100Hz) - beide
   umliegenden Stuetzpunkte wurden deshalb unveraendert aus der Konklusion
   uebernommen, statt den Widerspruch selbst zu entscheiden.
2. **Interpolation & Randbereiche:** Dokumentiert, dass die Interpolation
   zwischen den Stuetzpunkten (linear in log10(Frequenz) UND log10(Zeit))
   eine EIGENE Modellentscheidung ist, nicht Teil der Publikation (diese
   gibt keine gefittete Kurve an). Ausserhalb des Stuetzpunktbereichs
   (< 32 Hz oder > 200 Hz) wird bewusst NICHT extrapoliert, sondern der
   Randwert konstant fortgesetzt (`np.clip`) - entspricht der expliziten
   Extrapolationswarnung der Originalstudie.
3. **Grenzfaelle mit Tests abgesichert:** exakte Stuetzpunkt-Werte,
   unabhaengig nachgerechneter Interpolationspunkt (log-log-Formel direkt
   gegenpruefen), Klemmverhalten unter 32 Hz und ueber 200 Hz, sowie der
   hoerbar/unhoerbar-Uebergang exakt an einem kalibrierten Stuetzpunkt
   (100 Hz / 0.27s: knapp darunter/exakt/knapp darueber).

## Prioritaet 4 - REW-MDAT-Parser Haertung (GitHub-Issue #7)

Umgesetzt per PR, siehe Commit-Historie ab 2026-11-22. Haertet den
bestehenden `parse_dict()`-Pfad in `mdat.py`; schliesst NICHT Issue #7
selbst (siehe Leitplanke unten).

1. **`parse_dict()` gegen fehlerhafte/unvollstaendige Strukturen
   abgesichert:** jeder verschachtelte Record (`frequency_spacing`,
   `frequency_response`, `impulse_response`, `room_dimensions`) wird jetzt
   explizit auf `dict`-Typ geprueft; nicht-numerische Werte (`ppo`,
   `peak_index`, `measurement_delay_s`, `spl_offset_db`, `start_time_s`,
   Raummasse) werden mit klarer `MdatValidationError` abgelehnt statt mit
   rohem `TypeError`/`ValueError` durchzuschlagen; leere Mess-Arrays
   (`frequency_hz`/`spl_db`/`samples`) werden abgelehnt; optionale
   Nebenarrays (`phase_deg`, `unwrapped_phase_deg`, `group_delay_s`) werden
   jetzt auf Laengengleichheit mit `frequency_hz` geprueft (vorher
   stillschweigend akzeptierter Laengenunterschied, der nachgelagerte
   punktweise Berechnungen falsch zugeordnet haette).
2. **Speicherbudget, Versions-Fallback, Sweep/RTA-Erkennung geprueft:**
   `AllocationBudget` deckt jetzt auch die optionalen
   frequency_response-Nebenarrays einzeln ab. Versions-Fallback-Verhalten
   fuer leere, unbekannte/zukuenftige und sehr alte Versionsstrings explizit
   dokumentiert und getestet (Fallback greift nur bei fehlendem/leerem
   Feld, nicht bei einer dem Projekt unbekannten Versionsnummer). Die
   Sweep-vs-RTA-Heuristik (`is_sweep`) um Grenzfalltests (ein Sample,
   durchgaengig stumme Samples, Peak exakt/knapp unter dem dokumentierten
   Faktor-3-Schwellenwert) ergaenzt; dabei einen kleinen Typ-Inkonsistenz-
   Fund behoben (ein Rueckgabepfad lieferte `np.bool_` statt `bool`).
3. **Strikte Leitplanke zu `parse_bytes()` (Issue #7, unveraendert):**
   `parse_bytes()` bleibt weiterhin bewusst NICHT implementiert. Diese
   Haertungsrunde betrifft ausschliesslich den bereits nutzbaren
   `parse_dict()`-Pfad und ist KEINE Byte-Level-Verifikation gegen eine
   echte REW-.mdat-Datei. Issue #7 selbst bleibt offen und blockiert bis
   fruehestens Maerz 2027.

## Prioritaet 5 - Robustheit des Import-/Validierungspfads (Vorschlag, noch nicht begonnen)

- Fuzz-/Grenzwerttests fuer `validator.py` (z. B. NaN/Inf in `freq_hz`/
  `spl_db`, leere Arrays, extrem dichte oder extrem duenne Frequenzraster)
  ueber die bereits in `mdat.py` etablierte `AllocationBudget`-Haertung
  hinaus.
- Pruefen, ob `grid.py`s R160-Interpolation an den Bandraendern (sehr
  niedrige/hohe Frequenzen) numerisch stabil bleibt, inkl. Tests mit
  bewusst knappen Ueberlappungsbereichen (`FrequencyGrid.overlap_with`).

## Prioritaet 6 - Nachvollziehbarkeit der Optimierer-Zielfunktionen (Vorschlag, noch nicht begonnen)

- `sub_optimizer.py` und `eq_solver.py`: dokumentierte Sensitivitaetstests
  (wie stark aendert sich das Optimum bei kleinen Aenderungen der
  Modell-Gewichtung `ObjectiveWeights`?), um die bereits als Modell
  gekennzeichnete 70/30-Gewichtung nicht unbeabsichtigt als robuster
  erscheinen zu lassen, als sie ist.
- Pruefen, ob `validate_no_harm()` (Do-No-Harm-Check) auch fuer den
  Mehrpositions-Fall mit sehr wenigen Positionen (z. B. nur 2) noch
  aussagekraeftige Streuungskennzahlen liefert.

## Prioritaet 7 - Dokumentations- und Test-Konsistenz (Vorschlag, noch nicht begonnen)

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
