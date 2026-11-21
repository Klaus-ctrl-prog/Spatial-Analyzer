# Projekt-Anweisung & Start-Auftrag: Spatial Analyzer (`Klaus-ctrl-prog/Spatial-Analyzer`)

## 1. Projektziel & Systemkontext
- **Repository:** `Klaus-ctrl-prog/Spatial-Analyzer` (eigenständiges Python-Projekt unter **MIT-Lizenz**, strikt getrennt von `Klaus-ctrl-prog/KEFplayer`).
- **Primärer Einsatzrahmen (Wohnzimmer):**
  - Raummaße: `5,72 × 3,72 × 2,40 m` (theoretische axiale Modenkandidaten als Modellwerte: ~30,0 Hz [1. Länge], ~46,1 Hz [1. Breite], ~60,0 Hz [2. Länge], ~71,5 Hz [1. Höhe]).
  - Wiedergabekette: KEF LS50 Wireless Gen1 + Saxx DS 120 DSP (Einzel-Subwoofer).
  - Messtechnik: UMIK-1 mit 90°-Kalibrierdatei, REW Sinus-Sweeps mit akustischer Timing-Referenz.
  - Messraster: Sofamitte + 10 Wanderpositionen (11 Positionen), jeweils getrennt für Links (`L`) und Rechts (`R`), plus `L + R`-Summenmessung an der Sofamitte und getrennte Einzelmessungen (`L`, `R`, `Saxx`) für das Subwoofer-Alignment.
- **Abgrenzung:** Ein möglicher späterer Auto-/Car-Audio-Export (z. B. Mosconi-DSP mit Laufzeit in `ms`/`cm` und Phasentabelle) ist ausdrücklich **nicht** Teil des Projekt-Kerns, sondern nur als offener späterer Erweiterungspunkt (GitHub-Issue) vorgemerkt.

## 2. Verbindliche Arbeits- & Sicherheitsregeln
- **Strikte Lizenz-Sauberkeit (MIT):** Keinen Code aus AGPL-/GPL-Projekten (`SECS 2R`, `pierreaubert/autoeq` / `sotf`) kopieren. Alle Kernmodule werden sauber in Python (`numpy`, `scipy`) nach Spezifikation selbst implementiert.
- **Zulässige Bibliotheken & Referenzen:**
  - `pyfar` (MIT) & `python-acoustics` (BSD-3-Clause) als optionale Signalverarbeitungs-Bausteine (die FDW-Fensterlogik selbst bleibt eigene Implementierung).
  - Unsere eigenen, gegen `np.longdouble` auf `-215 dBFS` verifizierten RBJ-Biquad-Filterformeln und das Export-Schema aus `Klaus-ctrl-prog/KEFplayer` (`kefplayer/dsp.py`).
- **REW-API-Sicherheit:** Ausschließlich schreibgeschützte `GET`-Aufrufe verwenden. Keine `POST`-, `PUT`- oder `DELETE`-Aufrufe und keine REW-Prozesskommandos ausführen. Messungen immer über stabile `UUID` identifizieren, niemals über veränderliche numerische Indizes.
- **Physikalische Ehrlichkeit:** Rechenmodelle (theoretische Raummoden, 70/30-Gewichtung, Zielkurven) immer ausdrücklich als Modell kennzeichnen, niemals als Messwert darstellen.

## 3. Fachliche Spezifikation (vollständig zu berücksichtigen)

### A. Datenimport & Validierung (Phase 1)
1. **Eigener `.mdat`-Archiv-Parser (nach `secs-mdat`-Spezifikation):**
   - Einlesen von REW-`.mdat`-Dateien (Java-Serialisierung) inklusive `FrequencySpacing`, `FrequencyResponse` (`spl_db`, `phase_deg`, `unwrapped_phase_deg`, `group_delay_s`), `ImpulseResponse` (`samples`, `sample_interval_s`, `measurement_delay_s`, `spl_offset_db`, `peak_index`), `RoomDimensions`, `notes`, `rew_version`.
   - Pflichtfeldprüfung, REW-Versions-Fallback auf Header-Version, Erhalt unbekannter Felder in `unknown_fields` und Speicherbudget-Schutz (`AllocationBudget`) gegen korrupte Dateien.
   - Automatische Unterscheidung zwischen echten Sinus-Sweeps (mit Impulsantwort) und reinen RTA-/Betragsmessungen.
2. **Schreibgeschützter REW-API-Client (`GET` only):**
   - Unterstützung für Big-Endian-Float32-Streams, Metadaten-, Frequenzgang-, Phasen- und Impulsantwort-Abruf per `UUID`.
   - **96-PPO-Regel:** 96 PPO enthält bei REW aus Anti-Aliasing-Gründen zwingend eine 1/48-Oktav-Glättung und darf nie als ungeglättete Rohauflösung fehlinterpretiert werden. Wo echte ungeglättete Rohdaten/Phase benötigt werden, ist der **Impulsantwort-Pfad (IR)** zu bevorzugen.

### B. Frequenzraster & Glättung
- Unterschiedliche Frequenzraster niemals über Array-Indizes mitteln.
- Gemeinsamen gültigen Überlappungsbereich (`valid_start_hz` bis `valid_end_hz`) bestimmen und ohne Extrapolation auf ein einheitliches logarithmisches **R160-Raster (ISO 3)** interpolieren; fehlende/ungültige Werte (`NaN`) nicht künstlich ersetzen.
- Strikte Trennung von Originalauflösung, Analyseauflösung (1/48 Oktave im Bass 20–200 Hz) und Darstellungsauflösung (1/12 bzw. 1/6 Oktave).

### C. Räumliche Mittelung & Statistik (getrennt für `L` und `R`)
- **Variante A (Absoluter energetischer Mittelwert):** `L_avg = 10 * log10((1/N) * sum(10^(L_i / 10)))` ohne Pegelnormalisierung.
- **Variante B (Normalisierter energetischer Mittelwert):** Vorherige Pegelangleichung im Bereich `400 Hz – 2 kHz` (analog REW *Align SPL*), danach energetische Mittelung.
- **Streuungs- & Unsicherheitsmaße:** Standardabweichung, Minimum, Maximum, Spannweite, gültige Punktanzahl je Frequenz sowie **Case-Bootstrap (Efron)** für Konfidenzbänder bei `N = 11` Positionen (unter Dokumentation der räumlichen Korrelation benachbarter Messpunkte; keine SVD).

### D. Gesamtspektrum-Analyse (`20 Hz – 20 kHz`) & 4 Profi-Erweiterungen
1. **Bassresonanz- & Abklinganalyse (`20–200 Hz`):**
   - Kombination aus Peak-Prominenz, Q-Faktor, räumlicher Häufigkeit über alle 11 Positionen, echtem Ausschwingverhalten (`Waterfall` / `CSD` aus der Impulsantwort, Bewertung angelehnt an perzeptive Hörschwellen nach *Fazenda, Stephenson & Goldberg 2015, JASA*) und Abgleich mit den Raummaßen (`5,72 × 3,72 × 2,40 m`).
2. **Dritte Prüfebene (`L + R` Summenmessung an Sofamitte):**
   - Primär reale akustische `L + R`-Summenmessung plus komplexe `L + R`-Addition aus den Einzelmessungen als Konsistenz-Check für Interferenzen im Bereich `20–80 Hz`.
3. **Frequenzabhängige Fensterung (`FDW`):**
   - Eigene, transparent parametrisierte FDW-Fensterung auf der ungeglätteten Roh-Impulsantwort (trennt oberhalb ~500 Hz den Direktschall der KEF LS50 Wireless von späten Wandreflexionen, während 20–200 Hz das volle Zeitfenster behält).
4. **Erstreflexions- & Kammfilter-Trennung (`ETC`):**
   - Auswertung früher Reflexionen (`2–15 ms`) aus der Impulsantwort in Kombination mit der räumlichen Konsistenz über alle 11 Punkte, um wandernde Kammfilter-Kerben (nicht per EQ korrigierbar) sicher von ortsfesten Raummoden (absenkbar) zu trennen.
5. **Mehrpunkt-Vorher/Nachher-Simulation (RBJ-Biquad-Vorschau):**
   - Simulation vorgeschlagener PEQ-Filter mit den exakten RBJ-Biquad-Formeln aus `kefplayer/dsp.py` auf allen 11 `L`- und `R`-Positionen sowie auf `L + R` (im allgemeinen Fall kanalgetrennt `H_L * L + H_R * R`, `H * (L + R)` nur bei identischen Filtern).

### E. Subwoofer-Übergangs- & Delay-Optimierer (`KEF + Saxx DS 120 DSP`)
- Aus zeitlich konsistenten Einzelmessungen (`L`, `R`, `Saxx` mit Timing-Referenz) berechnet eine vektorisierte `numpy`-Suche im Übergangsbereich (`50–120 Hz`) die komplexe Addition:
  `H_Gesamt(f, g, τ, p) = H_LR(f) + p * 10^(g/20) * H_Sub(f) * exp(-j * 2 * pi * f * τ)`
- **3D-Suchraum:** Pegel `g ∈ [-6, +6] dB`, Delay `τ ∈ [0, 20] ms`, Polarität `p ∈ {0°, 180°}`.
- **Zielfunktion (als bewusst gewählte Modell-Gewichtung gekennzeichnet):** `70 %` RMSE zur Zielkurve + `30 %` räumliche Streuung über alle 11 Positionen; bekannte Modenbänder (`60–62 Hz`) werden getrennt markiert, damit Interferenzen nicht als behobene Raummoden fehlinterpretiert werden.

### F. Zielkurve, Export-Module & Berichte
- **Zielkurve & Sicherheitsregeln:** Konfigurierbare psychoakustische Raum-Zielkurve (sanfter Bassanstieg unter 150 Hz, leichter Höhenabfall ab 2 kHz; als Modell gekennzeichnet), *Cut-only*-Regel im Modenbereich (`20–200 Hz`, keine schmalen Boosts in Auslöschungen) und begrenzte Filtergüte `Q` im Mittel-/Hochton.
- **Verbindlicher Export:** `KEFplayer`-kompatibles JSON-Profil (`L` und `R` getrennt, inkl. Preamp-Headroom) sowie REW-Generic-EQ-Textformat (`.txt`).
- **Berichte:** Nachvollziehbare CSV-, JSON- und HTML-Berichte mit klarer Trennung von Messwerten, Modellannahmen und Filterempfehlungen.

### G. Offene Punkte (als GitHub-Issues im neuen Repo anlegen)
1. Lesende Prüfung der REW-API-Metadaten auf aktive FDW-/Window-Kennzeichnung.
2. Kalibrierung des Reflexions-Schwellenwerts (`ETC`) anhand realer Wohnzimmer-Messdaten.
3. Klärung eines optionalen späteren Car-Audio-Exports (`Mosconi-DSP` mit Laufzeit in `ms`/`cm` und Phasentabelle).
4. Validierung des `.mdat`-Parsers mit einer echten, unveränderten REW-`.mdat`-Datei aus unserem Bestand sowie Abgleich gegen REW-Text-Exporte (spätestens ab März 2027).

## 4. Dein konkreter Start-Auftrag für Phase 1
Bitte richte jetzt das Repository `Klaus-ctrl-prog/Spatial-Analyzer` sauber ein und setze **Phase 1** um:
1. Lege die Projektstruktur (`spatial_analyzer/`, `tests/`, `docs/`, `pyproject.toml`, GitHub-Actions-CI für Ubuntu & Windows unter Python 3.12) an und speichere diese Spezifikation als `docs/SPECIFICATION.md`.
2. Implementiere das Kern-Datenmodell (`Measurement`, `FrequencyGrid`, `RoomModel`) und den **schreibgeschützten REW-API-Client (`GET` only)** inkl. `UUID`-Adressierung, Big-Endian-Float32-Dekodierung und `96-PPO`-Glättungs-Kennzeichnung.
3. Implementiere den **eigenen Python-`.mdat`-Parser** (nach `secs-mdat`-Spezifikation mit Pflichtfeldprüfung, Versions-Fallback, `AllocationBudget` und Sweep- vs. RTA-Erkennung).
4. Implementiere die **R160-Raster-Interpolation** (ohne Extrapolation, `NaN`-sicher) und **beide energetischen Mittelungsverfahren (Variante A & B)** getrennt für `L` und `R` inklusive Streuungsmetriken.
5. Schreibe vollständige Unit- und Regressionstests für alle Module von Phase 1, prüfe die CI und melde den Abschluss von Phase 1 zur Freigabe.
