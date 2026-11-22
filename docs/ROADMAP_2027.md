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

## Prioritaet 5 - Robustheit des Import-/Validierungspfads

Umgesetzt per PR, siehe Commit-Historie ab 2026-11-22.

1. **Fuzz-/Grenzwerttests fuer `validator.py` und `grid.py`:** systematisch
   mit ungueltigen, extremen und grenzwertigen Eingaben geprueft (NaN/Inf
   in Frequenzen/Pegeln/`sample_interval_s`, doppelte/invertierte
   Frequenzen, leere und einelementige Arrays, winzige Frequenzbereiche
   ohne R160-Gitterpunkt, nicht-numerische Impulsantwort-Samples, falsche
   Datentypen fuer `positions`, boolesche Werte an Zahlfeldern).
2. **Gefundene rohe Abstuerze behoben** (alle vorher live reproduziert):
   - `r160_grid()`: NaN/Inf als `start_hz`/`end_hz` schlug nicht am
     `<= 0`-Vorzeichen-Check an (NaN-Vergleiche sind immer `False`) und
     brach erst spaeter mit einem unklaren rohen
     `"cannot convert float NaN to integer"` ab.
   - `interpolate_to_grid()`: NaN/Inf im Quellraster (`source_freqs_hz`)
     schlug aus demselben Grund nicht an der Monotonie-Pruefung an und
     floss stillschweigend in `np.interp()` ein; ein leeres Quellraster
     brach mit numpys eigenem unklaren Fehler ab statt mit einer
     verstaendlichen Validierungsmeldung.
   - `FrequencyGrid.__post_init__()` (models.py): derselbe NaN-Fund wie in
     `grid.py` - ein NaN in `frequencies_hz` passierte die
     Monotonie-Pruefung unbemerkt.
   - `ImpulseResponseData.__post_init__()` (models.py): derselbe Fund fuer
     `sample_interval_s` - ein NaN passierte den `<= 0`-Check unbemerkt.
   - `validator.py`: nicht-numerische Impulsantwort-Samples (z. B.
     versehentlich Strings) wurden gar nicht geprueft und waeren erst in
     `ImpulseResponseData`/`np.asarray(..., dtype=float64)` mit einem
     rohen `ValueError` abgebrochen; NaN/Inf in `freq_hz`/`spl_db`/
     `sample_interval_s` wurden vorher nur unspezifisch als
     "ausserhalb des plausiblen Bereichs"-`warning` erfasst statt als
     klarer `error`.
3. **Keine neuen Funktionen, keine Kopplung an `pipeline.py`:** alle
   Aenderungen sind reine Eingabepruefungen in bereits bestehenden
   Funktionen/Methoden; `mdat.py` bleibt wie dokumentiert unabhaengig von
   `pipeline.py`.

## Prioritaet 6 - Nachvollziehbarkeit der Optimierer-Zielfunktionen (teilweise umgesetzt)

**Teil 1 (umgesetzt per PR, siehe Commit-Historie ab 2026-11-22):**
`sub_optimizer.py`: systematische, deterministische Sensitivitaetsanalyse der
70/30-Gewichtung (`ObjectiveWeights`) anhand mehrerer synthetischer
Mehrpunkt-Szenarien (Phasen-/Pegelstreuung zwischen Positionen) gegen 50/50,
60/40, 70/30, 80/20 und 90/10 (teils weitere Randwerte). Ergebnisse (siehe
`tests/test_sub_optimizer.py` `TestObjectiveWeightSensitivity` und
Docstring-Ergaenzung in `sub_optimizer.py`):

- **Stabiles Regime:** bei identischen Positionen (keine raeumliche
  Streuung) ist das Ergebnis exakt gewichtungsinvariant - der
  Streuungsterm ist ueberall 0.
- **Empfindliches Regime:** bei spuerbarer Phasen-/Pegelstreuung zwischen
  Positionen kann bereits eine Verschiebung von z. B. 80/20 auf 70/30
  einen anderen Delay-Kandidaten liefern. In einem Szenario liegt die
  Standardgewichtung 70/30 in der Mitte eines bis 50/50 stabilen
  Plateaus; direkt daneben (95/5 bzw. 40/60) kippt das Optimum auf einen
  anderen Delay-Kandidaten.
- Die 70/30-Standardgewichtung selbst wurde **nicht** veraendert; alle
  Befunde sind reine algorithmische Verifikation an synthetischen Daten,
  keine Aussage ueber reale Raum-/Fahrzeugdaten.

**Teil 2:** Der `validate_no_harm()`-Grenzfall mit sehr wenigen Positionen
ist in Haertungsrunde B (siehe unten) geschlossen. Offen bleibt die zu
Teil 1 analoge Sensitivitaetsbetrachtung fuer `eq_solver.py`.

## Prioritaet 7 - Dokumentations- und Test-Konsistenz

Umgesetzt per PR, siehe Commit-Historie ab 2026-11-22. Letzter Punkt des
Arbeitsplans - keine Code-Aenderung am Rechenkern, keine Aenderung an
`docs/SMART453_BASS_SPEC.md`.

1. **`README.md` aktualisiert:** der veraltete Status-Hinweis "Phase 1"
   (Stand Projektstart, siehe `docs/SPECIFICATION.md` Abschnitt 4) wurde
   durch den tatsaechlichen Reifegrad ersetzt - abgeschlossene Phasen 1-3,
   die seither ergaenzten Funktionen (Windows-Starterdatei, REW-Notes-
   Durchreichung, Schema-Validierung/`--dry-run`, A/B-Vergleichsmodus) und
   der vollstaendige Stand der Haertungsphase (Prioritaeten 1-6 Teil 1
   abgeschlossen, Prioritaet 6 Teil 2 und Prioritaet 7 bei Erstellung
   dieses Eintrags offen bzw. laufend). Die bis Fruehjahr 2027
   zurueckgestellten offenen GitHub-Issues (#4, #5, #6, #7) sind jetzt
   auch im README einzeln benannt, nicht nur in diesem Dokument.
2. **Docstring-Abgleich gegen `docs/SPECIFICATION.md`:** ein echter Fund -
   `fdw.py` implementiert Abschnitt 3.D.3 ("Frequenzabhaengige Fensterung
   (FDW)"), hatte aber im Gegensatz zu allen anderen D.x-Modulen
   (`bass_resonance.py`=D.1, `lr_sum.py`=D.2, `etc.py`=D.4,
   `biquad_preview.py`=D.5) KEINE Referenz auf diesen Abschnitt im
   Docstring. Ergaenzt, keine inhaltliche Aenderung. `complex_response.py`
   (gemeinsame Mathe-Infrastruktur mehrerer Abschnitte) und `validator.py`
   (spaetere Ergaenzung ohne eigenen Spec-Abschnitt, GitHub-Issue #10)
   bleiben bewusst ohne Einzelverweis - dokumentiert in
   `tests/test_docs_consistency.py`.
3. **`docs/REW_NOTES_TEMPLATE.md` vs. Notes-Durchreichung:** der dort
   referenzierte Berichts-Abschnittstitel ("Mess- & DSP-Notizen (aus
   REW)") stimmt weiterhin exakt mit `reporting.py` ueberein - keine
   Abweichung gefunden, jetzt aber als Regressionstest abgesichert statt
   nur von Hand geprueft. Compare-Modus und `--dry-run` beruehren die
   Notes-Durchreichung nicht (unabhaengige Funktionen), daher keine
   Aktualisierung dort notwendig.
4. **Automatisierte Konsistenztests** (`tests/test_docs_consistency.py`,
   26 neue Tests): Spec-Referenz-Vollstaendigkeit pro Modul,
   REW-Notes-Template-Abschnittstitel, README-Status (kein veralteter
   "Phase 1"-Wortlaut mehr, Verweis auf `ROADMAP_2027.md` vorhanden,
   zurueckgestellte Issue-Nummern genannt), sowie durchgehende
   Prioritaets-Nummerierung 1-7 in diesem Dokument selbst.

## Haertungsrunde B - Do-No-Harm-Statusmodell und reale L+R-Summenebene

Eigener PR nach Runde A (PR #22), Stand 2026-11-22. Keine Aenderung an
`docs/SMART453_BASS_SPEC.md`, kein `parse_bytes()` in `mdat.py`. Alle
Befunde sind algorithmische Verifikation an synthetischen Daten, keine
Raum- oder Fahrzeugvalidierung; die Toleranzen sind Arbeitswerte.

1. **Drei-Stufen-Statusmodell (`eq_solver.py`):** `validate_no_harm()`
   liefert statt `tuple[bool, float, float]` ein `NoHarmResult` mit
   `NoHarmStatus.SAFE / UNSAFE / INCONCLUSIVE`. Bei `N < 3` Positionen
   (Arbeitswert `MIN_POSITIONS_FOR_NO_HARM`) oder nicht-endlichen Werten ist
   das Ergebnis `INCONCLUSIVE`; der bisherige stille NaN-Vergleich
   (`np.std(ddof=1)` bei N=1, `NaN <= x` ist immer False) entfaellt. Neben
   der mittleren Streuung (0,5 dB) prueft er die groesste frequenzlokale
   Zunahme der Streuung (2,0 dB) und optional die RMS-Zielkurvenabweichung
   (0,5 dB). Nur `SAFE` gilt als freigegeben.
2. **Kennzeichnung nicht freigegebener Exporte (`pipeline.py`,
   `exporters.py`, `reporting.py`):** Bei `UNSAFE`/`INCONCLUSIVE` tragen
   Profilname (`kefplayer_profile.json`, Schema unveraendert) und
   REW-EQ-Dateien (`#`-Kommentarkopf) die Markierung "NICHT FREIGEGEBEN",
   zusaetzlich entsteht `NICHT_FREIGEGEBEN.txt`; Report (JSON und HTML)
   zeigt den Freigabestatus samt Grund. Ob REW `#`-Zeilen beim Import
   toleriert, ist **nicht verifiziert**; freigegebene Dateien bleiben
   unveraendert.
3. **Wirksame Regressionstests (`tests/test_eq_solver.py`):** Der schwache
   `test_detects_increased_spread_from_harmful_filter` wurde durch
   deterministische Tests ersetzt (kanalgetrennte L/R-Filter nahe
   Teilausloeschung, schmalbandiger Boost, Zielkurve, N=1, N=2).
4. **Reale L+R-Summenmessung (`validator.py`, `pipeline.py`,
   `reporting.py`):** optionaler Kanal `lr` an `sofa_mitte` (an anderen
   Positionen Warnung und Ignorieren; zaehlt nicht als L oder R). Er wird
   mit `predict_complex_sum()` / `compare_predicted_vs_measured()` im Band
   20-80 Hz als dritte Konsistenzebene verglichen (RMSE, max. Abweichung)
   und im Report ausgewiesen - als Konsistenz-Check ohne automatisches
   Urteil. Ohne `lr` bleibt das Verhalten unveraendert.

## Finaler Stand der Haertungsphase (Stand 2026-11-22)

Prioritaeten 1-5 und 6 (Teil 1) sowie 7 abgeschlossen (siehe jeweilige
Abschnitte oben und Commit-/PR-Historie #14-#20). Ein Punkt bleibt
bewusst offen fuer eine spaetere, eigene Haertungsrunde:

- **Prioritaet 6, Teil 2 (teilweise offen):** Der `validate_no_harm()`-
  Grenzfall mit sehr wenigen Positionen ist in Haertungsrunde B
  geschlossen. Offen bleibt eine zu Prioritaet 6/Teil 1 analoge
  Sensitivitaetsbetrachtung fuer `eq_solver.py`.

## Zurueckgestellt bis reale Messdaten vorliegen (ab Fruehjahr 2027)

Betrifft ausschliesslich die folgenden vier offenen GitHub-Issues - alle
vier sind reale Validierungs-/Kalibrierungsaufgaben, die ohne echte
Wohnzimmer- bzw. Fahrzeugmessdaten nicht sinnvoll abschliessbar sind:

- **Issue #7** - Jede Validierung gegen eine echte, unveraenderte
  REW-`.mdat`-Datei (`mdat.py` Abschnitt G.4 der Spezifikation;
  `parse_bytes()` bleibt bis dahin bewusst `NotImplementedError`; siehe
  auch Prioritaet 4 oben, die sich auf dieses Issue bezieht, es aber
  ausdruecklich nicht schliesst). Dazu gehoert auch die unabhaengige
  Plausibilisierung von `mdat.py`/`models.py` gegen den REW-Text-Export
  bzw. die REW-API (siehe `claude/secs_mdat_pruefbericht.md` Abschnitt 8).
- **Issue #5** - ETC-Reflexions-Schwellenwert-Kalibrierung anhand realer
  Wohnzimmer-Messdaten (`etc.py`, Abschnitt G.2 der Spezifikation).
- **Issue #4** - Lesende Pruefung der REW-API-Metadaten auf aktive
  FDW-/Window-Kennzeichnung (`fdw.py`/`rew_client.py`, Abschnitt G.1).
- **Issue #6** - Klaerung eines optionalen spaeteren Car-Audio-Exports
  (Mosconi-DSP mit Laufzeit in `ms`/`cm` und Phasentabelle, Abschnitt G.3)
  - diese Entscheidung haengt am ehesten vom zurueckgestellten
  Smart-453-Themenpaket ab, nicht primaer von Messdaten.

Zusaetzlich, ohne eigene Issue-Nummer:

- Jede reale Kontrollmessung fuer `sub_optimizer.py`-Ergebnisse (Messregel 1
  aus den Projekt-Vorgaben: Simulation vor Messung klar als
  "simulationsbasiert, noch nicht real verifiziert" kennzeichnen).
- Das gesamte Smart-453-Themenpaket (`docs/SMART453_BASS_SPEC.md`) - ruht
  unveraendert, siehe dort.
