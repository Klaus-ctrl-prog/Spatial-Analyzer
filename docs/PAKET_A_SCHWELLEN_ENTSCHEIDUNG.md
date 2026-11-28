# Paket A: Schwellenwerte, Entscheidungsvorlage

**Status: Entscheidungsvorlage, nicht freigegeben.** Dieses Dokument legt keine Grenzwerte fest. Es zeigt den Ist-Zustand im Code mit Fundstellen. Die fachliche Freigabe bleibt beim Projektleiter.

Stand: 28.11.2026, Basis `82888de`. Keine Code- oder Teständerungen, keine REW-Abfragen.

## 1. Modenkandidaten

| Schwelle | Wert | Fundstelle | Funktion | Auswirkung | Tests |
|---|---|---|---|---|---|
| Frequenztoleranz | 2,0 Hz | `etc.py:27` (`DEFAULT_FREQ_TOLERANCE_HZ`) | `classify_bass_dip`: Position zählt als gleich, wenn |f − f0| ≤ Toleranz; Nähe zum Modell | Bestimmt, welche Positionen einen Einbruch teilen | `tests/test_etc.py` |
| Abgleich Modenfrequenzen | 3,0 Hz | `compare.py:43` (`MODE_MATCH_TOLERANCE_HZ`) | Zuordnung von Moden zwischen zwei Berichten (`compare.py:199`) | Entscheidet, welche Moden vorher/nachher als gleich gelten | nicht geprüft |
| Theoretische Moden | Modell `RoomModel.axial_mode_candidates(max_order=2)` | `pipeline.py:294` | Einordnung als „modelliert“ | Unterscheidet `mode_candidate` von `mode_candidate_unmodeled` | `tests/test_etc.py` |

## 2. Bestätigung

| Schwelle | Wert | Fundstelle | Funktion | Auswirkung | Tests |
|---|---|---|---|---|---|
| Konsistenzanteil „ortsfest“ | 0,7 | `etc.py:28` (`DEFAULT_CONSISTENCY_THRESHOLD`) | `classify_bass_dip`: Anteil Positionen mit Einbruch ≥ 0,7 | Entscheidet zwischen Modenkandidat und Kammfilter | `tests/test_etc.py` |
| Abklingzeit-Schwelle | Modell (Stützpunkte): 32 Hz 0,90 s; 63 Hz 0,30 s; 100 Hz 0,27 s; 150 Hz 0,18 s; 200 Hz 0,17 s | `bass_resonance.py:74` (`PERCEPTUAL_THRESHOLD_MODEL`) | Interpolation log-log, außerhalb 32 bis 200 Hz konstant | `exceeds_perceptual_threshold = Abklingzeit > Schwelle` (`bass_resonance.py:342`), als `decay_confirmed` übernommen in `pipeline.py:310`, ohne Referenz-IR `None` (`pipeline.py:312`) | `tests/test_bass_resonance.py` |

**Herkunft der Abklingzeit-Schwelle (laut Modul-Docstring):** Die Stützpunkte stammen aus Einzelwerten einer Publikation. Die Interpolation ist eine eigene Modellentscheidung (Issue #2). Eine fachliche Bestätigung der Werte fehlt im Repo.

## 3. Kammfilter

Kein eigener Schwellenwert. Ein Einbruch wird als `comb_filter_candidate` eingestuft, wenn der Konsistenzanteil unter 0,7 liegt (`etc.py`, `classify_bass_dip`). Kammfilter-Kandidaten werden nie geschnitten (`tests/test_eq_solver.py:47`).

## 4. Cut-Vorschläge

| Schwelle | Wert | Fundstelle | Funktion | Auswirkung | Tests |
|---|---|---|---|---|---|
| Max. Absenkung pro Mode | 9,0 dB | `eq_solver.py:30` (`DEFAULT_MAX_MODE_CUT_DB`) | `gain_db = −min(excess, max_cut)` (`eq_solver.py:105`) | Begrenzt jeden Cut | `tests/test_eq_solver.py` |
| Max. Breitband-Korrektur | 6,0 dB | `eq_solver.py:33` (`DEFAULT_MAX_BROADBAND_GAIN_DB`) | Default für `max_boost_db` und `max_cut_db` (`eq_solver.py:125` bis `126`), angewendet in `eq_solver.py:168` | Begrenzt Anhebung und Absenkung | `tests/test_eq_solver.py` |
| Überschuss | `excess_db ≤ 0` → kein Cut | `eq_solver.py:103` | Keine Absenkung unter der Zielkurve | Logik, keine Schwelle | `tests/test_eq_solver.py` |
| Bedingungen für Cut | Klasse `mode_candidate*`, `decay_confirmed is True`, Frequenz im Band | `eq_solver.py:95` bis `97` | Auswahl der Kandidaten | Nur bestätigte Moden werden geschnitten | `tests/test_eq_solver.py` |

**Harm-Toleranzen (laut Code als „Arbeitswert“ bezeichnet):**

| Schwelle | Wert | Fundstelle |
|---|---|---|
| Räumliche Streuung | 0,5 dB | `eq_solver.py:35` |
| Lokale Streuung | 2,0 dB | `eq_solver.py:36` |
| RMS-Zielabweichung | 0,5 dB | `eq_solver.py:37` |

## 5. Offene Entscheidungen (deine Freigabe)

1. **Abklingzeit-Modell (`bass_resonance.py`):** Übernehmen als fachliche Grundlage oder erst die Quelle prüfen? Die Herkunft ist nur im Docstring beschrieben.
2. **Cut-Grenzen 9,0 dB und 6,0 dB:** Bestätigen oder ändern?
3. **Harm-Toleranzen 0,5, 2,0 und 0,5 dB:** Im Code als Arbeitswerte bezeichnet. Bestätigen, ersetzen oder als vorläufig festschreiben?
4. **Konsistenzanteil 0,7 und Frequenztoleranzen 2,0 und 3,0 Hz:** Bestätigen oder ändern?
5. **Weiterhin offen aus der Ableitung:** Vorverstärkung (A3) und Bedeutung von `teilstopfen` (A5).

Die Werte sind Ist-Zustand. Keiner davon ist im Repo als fachlich bestätigt belegt, außer dem, was du hier freigibst.

## 6. Quellenprüfung (Stand 28.11.2026)

Quelle: Fazenda, Stephenson, Goldberg (2015), *JASA* 137(3), S. 1088 bis 1098, DOI 10.1121/1.4908217. Volltext: [Repository der University of Huddersfield](https://eprints.hud.ac.uk/id/eprint/24317/1/FazendaStephensonGoldberg-PerceptualThresholdsModalDecay.pdf). Der Abruf über das Web-Werkzeug lieferte nur PDF-Rohdaten. Der Text wurde deshalb lokal mit `pdftotext` extrahiert und gelesen. Die Zitate stammen aus diesem extrahierten Text. Die Extraktion liegt nicht im Repo; Zitate und Paper-Angaben sind daher nur in jener Sitzung geprüft. Die fünf Stützwerte stimmen mit dem Docstring `bass_resonance.py:23` bis `27` überein.

### 6.1 Abklingzeit-Schwelle (`bass_resonance.py`, PERCEPTUAL_THRESHOLD_MODEL)

**BESTÄTIGT (wörtlich aus dem Paper, Zusammenfassung, Abschnitt zu den Ergebnissen):** „Average thresholds measured with artificial stimuli are 0.9s at 32Hz, 0.3s at 63Hz, 0.27s at 100Hz, 0.18s at 150Hz, and 0.17s at 200Hz.“

Die fünf Werte im Code entsprechen exakt den Mittelwerten für Sinus-Bursts. Das Paper nennt nur diese fünf Frequenzen und gibt keine Interpolation an (Suche nach „interpolat“: 0 Treffer).

**Einordnung:**
- **Belegt:** die Stützpunkte selbst (Messwerte aus dem Paper).
- **Eigene Übertragung:** die log-log-Interpolation zwischen den Stützpunkten und die konstante Fortsetzung außerhalb 32 bis 200 Hz. Der Code kennzeichnet das selbst als eigene Modellentscheidung (Issue #2). Das Paper stützt sie nicht.
- **Widerspruch im Paper:** Die Zusammenfassung nennt einen Knick bei 63 Hz, die Diskussion spricht von einem Knick bei etwa 100 Hz („A knee point at around 100Hz“). Das Paper ist an dieser Stelle nicht eindeutig.

**Messbedingungen (Paper):**
- Sinus-Bursts bei 32, 63, 100, 150 und 200 Hz, mit Pegeln von 70 und 85 dB SPL am Trommelfell.
- 17 Hörer bei Sinus-Bursts, 10 bei Musik (vier gemeinsame).
- Ruhige Hörräume, Lautsprecher und Kopfhörer.

**Definition der Abklingzeit (Paper):** Zeit bis zu einer Dämpfung um 60 dB. Das Paper betont, dass es sich nicht um eine diffuse Nachhallzeit handelt. Die Schwelle gilt so: Hat eine Mode eine kürzere Abklingzeit als die Schwelle, ist ihr Einzeleffekt für die Mehrheit der Hörer typischerweise nicht hörbar.

**Übertragbarkeit auf REW-Messungen (OFFEN):**
- Das Paper misst die Schwelle mit synthetisch erzeugten Abklingzeiten.
- Der Code schätzt die Abklingzeit aus gemessenen Impulsantworten (`narrowband_decay_time`, `bass_resonance.py:232`). Die Schätzung nutzt den Bereich −5 bis −25 dB und extrapoliert auf −60 dB, eine T20/T30-Konvention. Das Paper beschreibt dagegen eine Abklingzeit um 60 dB, die nicht aus einem Ausschnitt extrapoliert wird.
- Ob eine extrapolierte Schätzung aus einer REW-Messung mit einer synthetisch vorgegebenen Abklingzeit vergleichbar ist, ist nicht belegt.

### 6.2 Cut-Grenzen −9 dB und +6 dB

**Keine Quelle im Code.** `DEFAULT_MAX_MODE_CUT_DB = 9.0` (`eq_solver.py:30`) und `DEFAULT_MAX_BROADBAND_GAIN_DB = 6.0` (`eq_solver.py:33`) tragen keinen Kommentar mit Herkunft. Die Werte sind im Repo nicht begründet. Ein Papier oder eine Spezifikation wurde nicht gefunden.

**Status:** fachlich ungeklärt. Entweder Herkunft nachweisen oder als Arbeitswert festlegen.

### 6.3 Harm-Toleranzen

| Wert | Fundstelle | Herkunft laut Code |
|---|---|---|
| 0,5 dB räumlich | `eq_solver.py:35` | kein Inline-Kommentar; Docstring `eq_solver.py:262`: alle Toleranzen sind Arbeitswerte |
| 2,0 dB lokal | `eq_solver.py:36` | „Arbeitswert“ |
| 0,5 dB Ziel | `eq_solver.py:37` | „Arbeitswert“ |
| 3 Positionen | `eq_solver.py:38` | „Arbeitswert“ |

**Status:** Arbeitswerte im Code gekennzeichnet, drei davon ohne Begründung. Keine Quelle im Repo, keine fachliche Bestätigung.

### 6.4 Konsistenz 0,7 und Frequenztoleranzen 2 Hz und 3 Hz

- **0,7 (`etc.py:28`):** Kommentar „Anteil der Positionen fuer ortsfest“. Keine Quelle, keine Begründung.
- **2 Hz (`etc.py:27`):** keine Begründung.
- **3 Hz (`compare.py:43`):** keine Begründung. Verwendet in `compare.py:199` (Modenzuordnung vorher/nachher).

**Status:** ungeklärt. Diese Werte sind Festlegungen des Projekts ohne fachliche Herleitung im Repo.

## 7. Bewertung nach Kategorien

| Kategorie | Punkte |
|---|---|
| Wissenschaftlich belegt (Paper) | Die fünf Stützpunkte der Abklingzeit-Schwelle für Sinus-Bursts. |
| Eigene Übertragung oder Interpolation | Log-log-Interpolation, konstante Fortsetzung, Übertragung auf REW-Messungen über die Extrapolation aus dem Fit. |
| Arbeits- oder Implementierungswerte | Harm-Toleranzen (zum Teil als „Arbeitswert“ markiert), Mindestanzahl Positionen (3), Konsistenz 0,7, Frequenztoleranzen 2 und 3 Hz. |
| Fachlich ungeklärt | Cut-Grenzen −9 dB und +6 dB, Knick der Schwelle (63 oder 100 Hz im Paper), Übertragbarkeit auf REW-Messungen. |

## 8. Entscheidungsfragen (zusätzlich zu den Abschnitten 1 bis 5)

1. **Abklingzeit-Schwelle:** Die Stützpunkte als belegte Werte übernehmen und die Interpolation ausdrücklich als Projektentscheidung festschreiben? Oder die Interpolation ganz entfernen und nur die Stützpunkte nutzen?
2. **Übertragung auf REW:** Soll die Abklingzeit aus der Impulsantwort (Extrapolation aus −5 bis −25 dB) mit dem Paper-Verfahren vergleichbar sein, oder ist die Schwelle für REW-Messungen nicht belegt?
3. **Knick-Widerspruch im Paper:** Für die Dokumentation den Verweis auf Abschnitt 6.1 setzen?
4. **Cut-Grenzen:** Herkunft nachweisen oder als Arbeitswert festlegen?
5. **Harm-Toleranzen, Konsistenz, Frequenztoleranzen:** Arbeitswerte bestätigen oder ersetzen?
