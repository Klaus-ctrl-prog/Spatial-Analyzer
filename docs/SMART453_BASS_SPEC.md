# Smart 453 – Bass-/Grundtonoptimierer 20-300 Hz: Recherche & Spezifikationsentwurf

**Status: reiner Recherche- und Konzeptauftrag. Kein Code, keine Aenderung an
`spatial_analyzer/*.py`.** Dieses Dokument bewertet vorhandene Open-Source-
Projekte und Algorithmen und schlaegt eine Architektur fuer ein spaeter zu
implementierendes, vom Wohnzimmer-Kern strikt getrenntes Modul vor. Nichts
hier ist eine Freigabe zur Implementierung - das bleibt ein separater,
spaeterer Schritt mit eigener Freigabe.

Fahrzeug: Smart 453 Cabrio, Mosconi Gladen PICO 8|10DSP (Gen1).
Optimierungsbereich: **ausschliesslich 20-300 Hz**. Keine Vollbereichs- oder
automatische HT-Optimierung bei 2.800 Hz.

---

## 1. Verifizierte Befunde aus Code und Dokumentation

Durchgesehen wurden die GitHub-Dateiliste (`git/trees`-API), `README.md` und
`License.md` von `DIMOSUS/Resonalyze` sowie oeffentlich zugaengliche
Dokumentation zu `MSO` (Multi-Sub Optimizer) und `jaakkopasanen/AutoEq` bzw.
dessen Weiterentwicklung `pierreaubert/autoeq`/`SotF`. Es wurde **kein**
Quellcode kopiert oder im Detail nachgebaut; es wurden nur oeffentlich
beschriebene Konzepte und - soweit aus Dateinamen/README ablesbar - grobe
Architekturideen gesichtet.

### 1.1 `DIMOSUS/Resonalyze`

- **Lizenz: MIT** (`License.md`, Copyright (c) 2023 dimosus, Standard-MIT-Text
  "Permission is hereby granted, free of charge..."). Permissiv, mit unserem
  MIT-Projekt kompatibel - **aber** Kompatibilitaet der Lizenz ersetzt nicht
  die eigene Regel "keine Code-Uebernahme": wir implementieren unabhaengig,
  nicht aus dem Quelltext abgeschrieben, wie schon bei den KEFplayer-RBJ-
  Formeln gehandhabt.
- **Reale, nicht nur behauptete Code-Struktur verifiziert** (ueber die
  GitHub-Tree-API, nicht nur README): u. a. `dsp/AllPassFilter.cs`,
  `dsp/CrossoverFilter.cs`, `dsp/CrossoverJunctionTuner.cs`,
  `dsp/CrossoverAutoSetup.cs`, `dsp/AutoAlignmentEngine.cs`,
  `dsp/StereoPairSum.cs`, `dsp/StereoJunctionBranch.cs`,
  `dsp/JunctionPhaseAlignment.cs`, `dsp/JunctionPhaseLobe.cs`,
  `dsp/JunctionPhaseSpectra.cs`, `dsp/LowJunctionPolarity.cs`,
  `dsp/SpatialAverage.cs`, `dsp/TimeAlignmentAnalysis.cs`,
  `dsp/BilinearTransform.cs`, `dsp/FractionalSampleShift.cs`,
  `dsp/MicrophoneAngleModel.cs`, `dsp/PostCheckPolarity.cs`.
- **Tests sind real vorhanden** (nicht nur behauptet): `tests/Resonalyze.App.Tests/`
  enthaelt mehrere hundert Testdateien, darunter direkt relevante:
  `PeqSlotAllPassTests.cs`, `EqWizardAutoTuneAllPassTests.cs`,
  `SpatialAverageHybridTests.cs`, `SpatialAverageModeSettlingTests.cs`,
  `MeasuredSumTests.cs`, `OverlayMathTests.cs`, `DelayTableTextTests.cs`,
  `RewImportTimingTests.cs`, `RewMeasurementImportTests.cs`,
  `GDOptTests.cs` (vermutlich "Group Delay Optimizer" o. ae., Name allein
  nicht ausreichend zur Funktionspruefung). Ebenso `source/Tools/VirtualCrossover/`
  mit `VirtualCrossoverJunctionTuneSearch.cs`, `VirtualCrossoverPanel.AutoDelay.cs`,
  `VirtualCrossoverPanel.AutoCrossover.cs`, `VirtualCrossoverStereoSum.cs`,
  `VirtualCrossoverMetric(s).cs`.
  **Einschraenkung:** Dateinamen und Testnamen belegen, dass Code und Tests
  zu diesen Themen existieren - sie belegen NICHT die konkrete Formel, den
  Suchalgorithmus oder die Korrektheit. Dazu waere ein Lesen der Dateiinhalte
  noetig, was fuer eine reine Konzeptrecherche (kein Code-Checkout) bewusst
  nicht erfolgt ist.
- **Beschrieben, aber nicht am Code verifiziert** (laut README):
  - "Auto Delay" und "Auto Crossover" "search the actual settings against
    the phase-aware predicted sum" (Zitat-sinngemaess); beide sollen
    stereo-seitig gemeinsam arbeiten. **Objektivfunktion, Parametergrenzen,
    global vs. lokal: nicht dokumentiert.**
  - All-Pass-Baender sind als PEQ-Typ in "Virtual DSP" verfuegbar, neben
    Bell/Shelf. **Keine Erwaehnung einer automatischen Allpass-Optimierung**
    - das waere laut README ein manueller/EQ-Wizard-Vorgang, nicht Teil von
    Auto Delay/Auto Crossover.
  - Die komplexe Summe wird als "true complex (vector) sum" beschrieben, mit
    einer "Sum-Loss"-Kennzahl (Differenz zur phasenblinden Betragsaddition,
    Mittel + "worst dip"). **Keine Formel angegeben.**
  - Zeitausrichtung ("Time Alignment") nutzt laut README Sub-Sample-Delay-
    Schaetzung aus der Impulsantwort, verfeinert per GCC-PHAT-Kreuzkorrelation.
    Ob dies dieselbe Methode wie "Auto Delay" ist, bleibt laut README offen.
  - Raeumliche Mittelung ("Spatial Average") wird laut README tatsaechlich
    in "Virtual DSP" und dem "EQ Wizard" als Eingabe verwendet (nicht nur zur
    Anzeige) - **aber** ob der automatische Delay-/Crossover-Optimierer selbst
    diesen Mittelwert nutzt oder nur den Einzelpunkt, bleibt laut README
    **unklar** ("the prediction is valid at one microphone position").
- **Entscheidender methodischer Befund fuer unseren Anwendungsfall** (direkt
  aus dem README-FAQ, wortnah zusammengefasst): Resonalyze **lehnt USB-
  Mikrofone explizit ab**, mit Nennung von Grund "a USB microphone is its
  own audio device with its own free-running clock" - die Kombination mit
  einem separaten Loopback-Geraet erzeuge einen unbekannten Start-Offset
  plus kontinuierlichen Drift. Resonalyze verlangt stattdessen einen
  **elektrischen Loopback** (zurueckgefuehrtes Wiedergabesignal) als
  Zeitreferenz, nicht eine akustische Referenz. **Das ist ein fundamentaler
  Unterschied zu unserer eigenen Messkette** (UMIK-1 per USB + REW
  "Acoustic Timing Reference", siehe Abschnitt 4) - Resonalyzes
  Zeitausrichtungs-Methodik ist auf unsere Messkette NICHT direkt
  uebertragbar, nur das allgemeine Prinzip "eine gemeinsame Zeitreferenz
  ist zwingend" ist uebernehmbar.

### 1.2 MSO (Multi-Sub Optimizer, "Andy C")

- **Keine offene Lizenz, kein oeffentlicher Quellcode** - MSO ist laut
  mehreren unabhaengigen Foren-Threads (Audioholics, diyAudio, AVForums)
  kostenlose, aber **closed-source** Windows-Freeware. Damit entfaellt jede
  Lizenz- *und* Code-Pruefung von vornherein - es kann sich ausschliesslich
  um eine **Konzept-Inspiration** handeln, niemals um Code-Uebernahme (es
  gibt nichts zu uebernehmen).
- **Konzept (aus Foren-Beschreibungen, nicht aus Code):** Einzelmessungen
  von Hauptlautsprechern und jedem Subwoofer an mehreren Hoerpositionen;
  iterative Suche (ein Nutzer berichtet von "1 million iterations" bei vier
  Positionen) ueber Pegel/Delay/Filter je Sub; Export der gefundenen
  Biquad-Koeffizienten direkt in die "Advanced Biquad Programming"-
  Oberflaeche von miniDSP-Geraeten.
- **Praxis-Hinweis aus den Foren (Warncharakter, nicht fachlich relevant):**
  mehrere Nutzer berichten von unsauberer Windows-Integration/Deinstallation.
  Fuer unsere Bewertung irrelevant, da wir ohnehin kein Fremd-Binary
  einsetzen, sondern nur das Konzept "mehrere Messpositionen -> gemeinsame
  Pegel/Delay/Filter-Optimierung" uebernehmen, das in unserem eigenen
  `sub_optimizer.py` (Wohnzimmer-Projekt) bereits inhaltlich existiert.

### 1.3 `jaakkopasanen/AutoEq` und `pierreaubert/autoeq`/`SotF`

- `jaakkopasanen/AutoEq` (das urspruengliche, bekannte Projekt, 13k+ Stars):
  **MIT-Lizenz** (Copyright Jaakko Pasanen), Fokus Kopfhoerer-Einzel-EQ
  (Zielkurven-Fitting fuer Bell-Filter). Fachlich nicht direkt uebertragbar
  (kein Mehrwege-, kein Delay-/Allpass-Thema), bestaetigt aber methodisch
  nur das, was unser eigenes `eq_solver.py` bereits umsetzt.
- `pierreaubert/autoeq` (Rust-Neuimplementierung/-Erweiterung mit optionaler
  `SotF`-GUI) ist die fachlich naeher liegende Variante: das Projekt nennt
  explizit "Multi-Driver Speakers: Active crossover optimization with
  polarity inversion testing", "Multi-subwoofer alignment", globale +
  lokale Optimierungsalgorithmen (NLopt: COBYLA/ISRES/SLSQP/BOBYQA/DIRECT,
  Differential Evolution). **Lizenz: GNU General Public License v3**
  (Copyright-Header in `src/optim.rs`: "This program is free software...
  under the terms of the GNU General Public License ... version 3").
  **Das ist eine Copyleft-Lizenz, inkompatibel mit unserem MIT-Projekt fuer
  jede Form von Code-Uebernahme oder enger struktureller Nachbildung.**
  Fuer dieses Projekt gilt daher: **ausschliesslich oeffentlich beschriebene
  Algorithmus-Konzepte (z. B. "globale Suche + lokale Verfeinerung",
  "Polaritaetsumkehr als Testparameter") duerfen als Idee einfliessen, keine
  Codestruktur, keine Variablennamen, keine 1:1-Uebernahme von Formeln aus
  dem Quelltext.** Da wir fuer Filterformeln ohnehin auf den oeffentlich
  dokumentierten RBJ-"Audio-EQ-Cookbook" zurueckgreifen (wie bereits in
  `biquad_preview.py` fuer das Wohnzimmer-Projekt), ist diese Einschraenkung
  in der Praxis unkritisch - es gibt eine lizenzfreie Primaerquelle fuer die
  Mathematik.

### 1.4 REW "Acoustic Timing Reference" (fuer Abschnitt 4 relevant, hier vorab verifiziert)

Die REW-eigene Hilfe-Dokumentation (roomeqwizard.com) bestaetigt: ohne
gemeinsamen Takt zwischen Wiedergabe- und Aufnahmegeraet (USB-Mikrofon wie
UMIK-1) koennen unterschiedliche Sample-Clock-Raten "ueber einen langen
Sweep zu erheblichen Verzerrungen der Impulsantwortform fuehren und die
Phasenantwort beeinflussen"; REW korrigiert dies ueber "Adjust clock with
acoustic ref" (oder bei Hardware-Loopback: "Adjust clock with loopback").
Mehrere unabhaengige Nutzerberichte (avnirvana.com-Forum, diyaudio.com)
bestaetigen den Praxisbetrieb mit UMIK-1 und "Acoustic Timing Reference",
berichten aber auch **messbare Rest-Schwankungen** (Taktkorrektur in der
Groessenordnung einiger -15 ppm, Delay-Schaetzungen mit Streuung im
niedrigen ms- bzw. sub-ms-Bereich zwischen Wiederholungsmessungen) sowie den
Hinweis, dass der **Abstand zwischen Referenz-Treiber und Mikrofon ueber
alle Messungen hinweg auf wenige Millimeter konstant bleiben muss**, damit
Phasendaten zwischen verschiedenen Messungen (z. B. verschiedene Wege)
uebertragbar bleiben.

---

## 2. Bewertung: geeignet vs. zu verwerfen (fuer 20-300 Hz, Smart 453)

| Konzept | Quelle | Bewertung |
|---|---|---|
| Komplexe (phasenrichtige) Summenbildung mehrerer Wege, "Sum-Loss" als Kennzahl | Resonalyze (Konzept) | **Geeignet.** Mathematisch identisch zu dem, was wir im Wohnzimmer-Projekt in `complex_response.py`/`lr_sum.py`/`sub_optimizer.py` bereits unabhaengig umsetzen - direkt auf N Wege statt 2 verallgemeinerbar. |
| Allpass-Filter 1./2. Ordnung als Phasenstellglied an Uebergangsfrequenzen | Resonalyze (Konzept, als PEQ-Typ) | **Geeignet, aber eigenstaendig zu implementieren.** Die 2.-Ordnung-Allpass-Formel ist Teil des oeffentlichen RBJ-Cookbooks (wie unsere peq/shelf-Filter) - keine Code-Abhaengigkeit noetig. Die automatische *Optimierung* der Allpass-Parameter ist laut Resonalyze-README nicht automatisiert - muss hier eigenstaendig entworfen werden. |
| Grid-Search ueber Delay/Polaritaet (+ optional Allpass) an einer Uebergangsfrequenz, bewertet an Mehrpunkt-Messung | MSO-Konzept + eigenes `sub_optimizer.py` | **Geeignet, bereits precedented.** Direkte Erweiterung des bestehenden Wohnzimmer-Optimierer-Musters (Grid statt globaler Suche: nachvollziehbarer, fuer 20-300 Hz mit wenigen Uebergaengen rechnerisch unproblematisch, deckt sich mit der bestehenden Projekt-Praxis). |
| Globale Optimierer (COBYLA/DIRECT/Differential Evolution) fuer Mehrwege-Kreuzung | pierreaubert/autoeq (GPLv3, Konzept only) | **Verworfen fuer diese erste Ausbaustufe.** (a) Lizenzgrund: keine Code-Naehe zu einem GPL-Projekt gewuenscht, selbst bei eigenstaendiger Implementierung bleibt ein reines Nachbauen eines unter GPL stehenden Suchverfahrens vermeidbar, wenn ein einfacheres Verfahren (Grid-Search) ausreicht. (b) Fachgrund: bei nur 2 Wegen je Uebergang (Front-Sub+Sitz-Sub, "Bass"+Tuer-MT) und begrenztem Parameterraum (Delay, Polaritaet, wenige Allpass-Kandidaten) ist eine erschoepfende Grid-Search nachvollziehbarer und einfacher zu verifizieren als ein globaler Optimierer - passt besser zur Projektkultur "nachvollziehbar vor raffiniert". |
| Automatische HT-/MT-Phasenoptimierung bei 2.800 Hz | - | **Explizit ausgeschlossen** (Nutzervorgabe). Dieser Uebergang bleibt aussen vor. |
| Elektrischer Loopback als Zeitreferenz | Resonalyze | **Nicht uebertragbar ohne Zusatzhardware.** Unsere Messkette hat aktuell keinen Loopback-Pfad vom Mosconi-DSP zurueck zum PC. Uebernommen wird nur das *Prinzip* ("eine gemeinsame, über alle Einzelmessungen konstante Zeitreferenz ist Pflicht"), umgesetzt stattdessen ueber REWs eigene "Acoustic Timing Reference"-Funktion (siehe Abschnitt 4). |
| Spatial Average als Eingabe des Optimierers (nicht nur Anzeige) | Resonalyze (Konzept, laut README fuer Virtual DSP/EQ Wizard genutzt) | **Geeignet und bereits precedented** (das gesamte Wohnzimmer-Projekt ist genau darauf aufgebaut: 11-Positionen-Mittelung fliesst in jede Optimierung ein). Fuer Smart 453: 7-9 Kopf-Wuerfel-Punkte statt 11 Sofa-Positionen, gleiches Muster. |

---

## 3. Architekturvorschlag (eigenstaendiges Modul)

**Kein Code in diesem Schritt.** Vorschlag fuer eine *spaetere*, separat
freizugebende Umsetzung:

- **Eigenes Top-Level-Package `smart453_bass/`** im selben Repository, NICHT
  unter `spatial_analyzer/` und ohne Aenderung bestehender Dateien dort.
  Begruendung: andere Domaene (Fahrzeug statt Wohnzimmer), andere
  Wege-Namen (keine feste L/R-Symmetrieannahme, sondern benannte Wege wie
  `front_sub`, `seat_sub`, `door_mt_l`, `door_mt_r`), eigener
  Frequenzbereich (20-300 Hz statt 20 Hz-20 kHz), eigene Freigabehistorie.
- Erlaubt: **reiner Lesezugriff** (Import, keine Modifikation) auf die
  bereits verifizierten, generischen Low-Level-Bausteine aus
  `spatial_analyzer/complex_response.py` (SPL/Phase <-> komplexe
  Uebertragungsfunktion) und `spatial_analyzer/grid.py` (R160-Raster,
  NaN-sichere Interpolation) - beide sind reine, bereits getestete
  Mathematik ohne Bass-/Wohnzimmer-spezifische Annahmen. Alternative, falls
  auch das vermieden werden soll: Duplizieren dieser < 50-Zeilen-Bausteine
  im neuen Package, um jede Kopplung auszuschliessen. Diese Entscheidung
  kann bei der eigentlichen Implementierung getroffen werden.
- Geplante Module (Arbeitsnamen, Interfaces noch nicht final):
  - `crossover_filters.py`: Butterworth- und Linkwitz-Riley-Weichen
    (12/24 dB/Okt. gemaess Auftrag; 36/48 dB/Okt. zurueckgestellt, bis ein
    tatsaechlicher Bedarf am realen Mosconi-Kanal belegt ist - "keine
    ungepruefte Komplexitaet").
  - `allpass_filters.py`: Allpass 1. und 2. Ordnung (RBJ-Cookbook-Formel
    fuer 2. Ordnung, Standard-Bilineartransformation fuer 1. Ordnung -
    beide oeffentlich dokumentiert, keine Fremdcode-Abhaengigkeit).
  - `way_chain.py`: virtuelle DSP-Kette pro Weg (Weiche + Allpass + Gain +
    Delay + Polaritaet), komplexer Frequenzgang je Kette.
  - `multiway_sum.py`: phasenrichtige Summe ueber alle aktiven Wege, pro
    Kopf-Wuerfel-Position und gemittelt.
  - `transition_optimizer.py`: Grid-Search (Delay, Polaritaet, optionaler
    Allpass-Kandidat) fuer je einen Uebergang zwischen zwei benachbarten
    Wegen, bewertet gegen Summenpegel + raeumliche Streuung ueber die
    Kopf-Wuerfel-Punkte (Gewichtung analog `sub_optimizer.py`, aber als
    eigene, explizit als Modell gekennzeichnete Konstante fuer dieses
    Fahrzeug, nicht uebernommen von den Wohnzimmer-Werten).
  - `mosconi_table.py`: Erzeugung der Einstelltabelle (Abschnitt 6).
  - `report.py` + `cli.py`: analog zum Wohnzimmer-Projekt, aber als eigene,
    getrennte CLI (`python -m smart453_bass ...`), nicht als Erweiterung
    von `spatial_analyzer.cli`.
- **Nicht vorgesehen in dieser ersten Ausbaustufe:** automatische
  Mehrkanal-Uebertragung an die Mosconi-Software (Auftrag schliesst das
  explizit aus), FIR-Filter (keine belegte Notwendigkeit genannt), globale
  Optimierer (siehe Abschnitt 2).

---

## 4. Reproduzierbare Simulationsmethodik (inkl. Zeitreferenz-Grenzen)

1. **Zeitreferenz:** Messung jedes Wegs mit REW, **"Acoustic Timing
   Reference"** aktiv (ein fester Referenz-Treiber, z. B. ein Tuer-Hochtoener,
   dauerhaft als Referenzkanal mitlaufend) und Analyseoption **"Adjust clock
   with acoustic ref"** gesetzt (siehe REW-Hilfedokumentation, Abschnitt 1.4)
   - das ist exakt die von REW fuer USB-Mikrofone (UMIK-1) vorgesehene
     Loesung fuer den Taktdrift zwischen Wiedergabe- und Aufnahmegeraet.
2. **Geometrische Konstanz als Voraussetzung:** Der Abstand (und die
   Winkellage) zwischen Referenz-Treiber und Mikrofon muss ueber die
   gesamte Messserie (alle Wege, alle Kopf-Wuerfel-Punkte) konstant bleiben,
   sonst wird die Phasenreferenz selbst zur Fehlerquelle (siehe Abschnitt
   1.4). Praktische Folge: Referenz-Treiber und Mikrofonhalterung muessen in
   der Messsitzung fixiert sein, nicht zwischen Einzelwegen umgebaut werden.
3. **Restunsicherheit, die bestehen bleibt** (ehrlich auszuweisen, nicht
   wegzudefinieren):
   - Die USB-Taktkorrektur ist eine Schaetzung je Messung (REW gibt pro
     Messung eine eigene ppm-Korrektur aus); bei Wiederholungsmessungen ist
     laut Nutzerberichten mit einer Streuung der geschaetzten Verzoegerung
     im niedrigen Millisekunden- bzw. sub-Millisekunden-Bereich zu rechnen.
     Bei einem angestrebten Delay-Raster von 0,02 ms (Mosconi-Hardware-
     Granularitaet) kann diese Restunsicherheit **in der Groessenordnung
     mehrerer Rasterschritte** liegen - das optimierte Delay ist deshalb als
     Modell-Ergebnis mit dieser Unsicherheit zu kennzeichnen, nicht als
     exakter Messwert.
   - Jede einzelne Wegmessung (Front-Sub, Sitz-Sub, Tuer-MT L/R) muss mit
     *derselben* Referenzkette gemessen werden, damit die Phasenbeziehungen
     zwischen den Wegen ueberhaupt vergleichbar sind - eine Messung ohne
     durchgaengige Referenz (z. B. an einem anderen Tag, mit neu
     aufgebautem Referenzkanal) ist fuer die Mehrwege-Summenbildung
     **nicht verwendbar**, auch wenn die Einzelkurve fuer sich plausibel
     aussieht.
   - Die tatsaechliche Kanalbelegung (CH7/8 gebrueckt vs. CH9/10 Line-Out
     fuer den Front-Sub) ist laut Auftrag noch gegen das reale DSP-Preset
     zu pruefen, bevor irgendeine Simulation mit realen Zahlen gefuettert
     wird - das ist kein Software-, sondern ein Dokumentationsschritt vor
     der ersten Messung.
4. **Ablauf je Messserie:**
   1. DSP-Ist-Zustand (Kanalbelegung, vorhandene Weichen/Delays/Allpaesse)
      aus dem aktuellen Mosconi-Preset dokumentieren (nicht aendern).
   2. Referenzkanal fixieren, "Acoustic Timing Reference" + "Adjust clock
      with acoustic ref" aktivieren.
   3. Jeden Weg einzeln an jedem Kopf-Wuerfel-Punkt messen (Sweep, REW-
      Export mit Frequenzgang **und** Impulsantwort/Phase).
   4. Messdateien mit vollem DSP-Notizentext archivieren (siehe
      `docs/REW_NOTES_TEMPLATE.md`-Prinzip, fuer Smart 453 um Kanalbelegung
      und Referenz-Setup zu erweitern - eigener Vorlagenentwurf, falls
      gewuenscht, erst nach Freigabe dieser Spezifikation).
   5. Erst danach: Einspeisen in das (noch zu implementierende) Modul fuer
      Simulation/Optimierung.

---

## 5. Pruefstrategie

- **Synthetische Tests (sofort moeglich, keine Messdaten noetig):**
  - Weichen-Filterfunktionen gegen bekannte Butterworth-/LR-Eigenschaften
    (Flankensteilheit in dB/Oktave per `scipy.signal.sosfreqz`-Steigung
    pruefen; LR24 = zwei kaskadierte Butterworth-2.-Ordnung-Stufen - Summe
    HPF+LPF bei der Trennfrequenz ergibt -6 dB je Zweig, zusammen 0 dB bei
    symmetrischer Weiche, das ist eine pruefbare Eigenschaft, kein Modell).
  - Allpass-Filter: Betragsgang muss auf allen Frequenzen 0 dB sein (per
    Definition), nur die Phase dreht - direkt pruefbar ohne jede
    Messgrundlage.
  - Komplexe Summation: mit frei erfundenen, aber kontrollierten
    synthetischen Einzelwegen (z. B. zwei identische Quellen mit
    bekanntem Delay-Versatz) pruefen, ob die vorhergesagte Auslöschung/
    Addition an der erwarteten Frequenz auftritt.
  - Optimierer: synthetischer Fall mit bekanntem optimalem Delay/Polaritaet
    konstruieren und pruefen, dass der Grid-Search dieses Optimum
    wiederfindet (Regressionstest, analog zum bestehenden
    `narrowband_decay_time()`-Testmuster im Wohnzimmer-Projekt).
- **Reale Messdaten (erst nach den in Abschnitt 4 beschriebenen Messungen):**
  - Einzelwegmessungen gegen die dokumentierte DSP-Ist-Konfiguration
    gegenpruefen (Plausibilitaet: gemessene Trennfrequenz/Flankensteilheit
    soll zur dokumentierten Einstellung passen).
  - Vorgeschlagene Optimierer-Ergebnisse immer gegen eine **Kontrollmessung**
    mit den neuen Einstellungen verifizieren (gleiches Prinzip wie im
    Wohnzimmer-Projekt: "Annahmen/Modelle klar kennzeichnen, dann messen").
  - Keine Freigabe eines Zahlenwerts in die Mosconi-Einstelltabelle (siehe
    Abschnitt 6), ohne dass eine reale Messung diesen Wert stuetzt.

### Gate-Qualitaetspruefung (Leading-Edge Loss Guard, inspiriert von Resonalyze `GateLeadingEdgeLossDb`)

Verifiziert ueber GitHub-Code-Suche (`gh api search/code`, Treffer in
`dsp/DataHelper.Phase.cs`, `dsp/PhaseGatePlacement.cs`,
`docs/tech/phase-and-group-delay.md`, `docs/tech/junction-phase-and-group-placement.md`,
`source/Tools/VirtualCrossover/GatePlacementVerdict.cs`,
`tests/Resonalyze.Dsp.Tests/FrequencyDependentPhaseTests.cs` - letzteres
bestaetigt zugleich, dass `Resonalyze.Dsp.Tests` real existiert, was in
Abschnitt 1.1 als offene Frage markiert war) sowie durch Lesen von
`docs/tech/phase-and-group-delay.md`:

- **Zweck:** Entscheidet, ob ein Zeitfenster ("Gate") auf eine Kurve
  ueberhaupt gelegt werden darf, bevor deren Impulsantwort ausgewertet wird
  - insbesondere relevant fuer Front-Sub und Sitz-Sub (hohe Gruppenlaufzeit,
    Gate leicht zu fruh oder zu spaet gesetzt) und fuer die Tuer-MT (kurze,
    aber durch Kabinereflexionen verschmierte Anstiegsflanke).
- **Messgroesse:** Anteil der Energie **vor** dem Plateau-Beginn des Gates
  gegenueber der im Gate verbleibenden Energie, in dB. Bewusst **nur die
  Anstiegsflanke** ("leading edge"), nicht die Abklingflanke - das
  Abschneiden des Ausklingens ist der eigentliche Zweck eines Gates und
  wird nicht bestraft.
- **Referenzwerte:** `-∞ dB` = kein Verlust; `0 dB` = so viel verworfen wie
  behalten; `+∞ dB` = das Fenster enthaelt keinen Anteil des Kanals mehr
  (vollstaendige Fehlplatzierung - absichtlich nicht als `-∞ dB`
  kodiert, da sonst eine komplett daneben liegende Platzierung als die
  "sicherste" erschiene).
- **Zirkulaerer Vorlauf ("wrapped pre-roll"):** Die Berechnung liest Energie
  mit der eigenen zyklischen Adressierung der Extraktion - eine Flanke, die
  vor Beginn des Aufzeichnungsfensters beginnt, liest in den zyklischen
  Schwanz hinein, in dem bei einer Transferfunktions-Impulsantwort der
  Negativ-Zeit-Anteil liegt. Diesen Bereich als leer zu behandeln wuerde
  "ein Fenster voller ungesehener Inhalte durchlassen" (sinngemaess aus der
  Quelle). Fuer unsere eigene Umsetzung heisst das: bei kurzen FFT-Laengen
  oder knapp gesetztem Pre-Trigger muss explizit geprueft werden, ob die
  Gate-Vorlaufzeit in den zyklisch umgeschlagenen Bereich der Messung reicht,
  statt diesen Bereich stillschweigend als Nullsignal zu interpretieren.
- **Keine Wiederherstellung durch nachtraegliche Zeitverschiebung:** Laut
  Quelle gilt ausdruecklich "a window is not a time shift" - eine
  nachtraegliche Referenzierung auf einen gemeinsamen Zeitnullpunkt dreht
  nur das Spektrum, kann aber bereits vom Gate abgeschnittene
  Anstiegsenergie **nicht zurueckholen**. Fuer unsere Methodik (Abschnitt 4)
  bedeutet das: ein zu knapp gesetztes Gate bei einer Einzelwegmessung ist
  ein Messfehler, kein nachtraeglich im Rechenmodell korrigierbarer Fehler -
  bei Verdacht auf Anschnitt muss die Messung wiederholt werden, nicht
  rückwirkend verschoben.
- **Nicht verifiziert, offen:** Ob `dsp/PhaseGatePlacement.cs` und
  `source/Tools/VirtualCrossover/GatePlacementVerdict.cs` exakt diese Metrik
  implementieren oder nur thematisch verwandte Bausteine sind, ist aus dem
  gelesenen Dokument (`phase-and-group-delay.md`) nicht ablesbar - dort wird
  stattdessen eine Funktion `ResolveGatePlacement` als die Stelle
  beschrieben, an der ein Gate zu Samples wird (Extraktionsstart = ein
  linkes Schulterstueck vor dem Gate-Offset, Plateaustart = der Offset,
  Tukey-Fensterform ueber die Gate-Laenge). Der Quellcode dieser Dateien
  wurde nicht gelesen (reine Doku-/Konzeptrecherche, kein Checkout).
- **Geplante eigene Umsetzung (noch kein Code):** Eine vergleichbare,
  eigenstaendig implementierte Pruefgroesse (Arbeitsname
  `gate_leading_edge_loss_db()`) als Teil von `smart453_bass/` (Abschnitt 3),
  die vor jeder Weg-Auswertung automatisch warnt, wenn eine Gate-Platzierung
  relevante Anstiegsenergie verwirft - eigenstaendig nach der oeffentlich
  beschriebenen Definition (dB-Verhaeltnis, zyklische Adressierung,
  `+∞`/`-∞`-Randfaelle) neu geschrieben, nicht aus dem MIT-Quelltext
  uebernommen.
- **Integration erst nach Pruefung an synthetischen UND realen Messdaten**,
  gemaess Abschnitt 5 - fruehestens ab Fruehjahr 2027, wenn echte
  Einzelwegmessungen vorliegen.

---

## 6. Entwurf der spaeteren Mosconi-Einstelltabelle (Zielausgabe)

Eine Tabelle mit den Spalten `Weg | Gain (dB) | Delay (ms) | Polaritaet |
HPF (Fc/Typ/Flanke) | LPF (Fc/Typ/Flanke) | Allpass (Fc/Q je Stufe)` -
**ausschliesslich mit Werten, die direkt aus einer realen Messung plus
Kontrollmessung stammen.** Vorgesehene Regel (analog zum bestehenden
"Do-No-Harm"-Prinzip aus `eq_solver.py`): ein Feld bleibt leer bzw. zeigt
"nicht ausreichend durch Messung gestuetzt", wenn keine Kontrollmessung
vorliegt, statt eine unbelegte Zahl einzutragen - "keine Zahlen erfinden"
gilt hier genauso wie im Wohnzimmer-Projekt. Die Tabelle ist ein Vorschlag
zum manuellen Uebertragen durch den Nutzer in die Mosconi-Software; dieses
Projekt sendet zu keinem Zeitpunkt automatisch etwas an das Geraet.

---

## Grenzen (aus dem Auftrag, hier nochmals festgehalten)

Keine Aenderung an der Mosconi-Hardware/-Software. Keine Aenderung an
bestehenden DSP-Dateien. Keine automatische Uebertragung von Einstellungen.
Keine ungepruefte Filterwerte. Keine automatische HT-/MT-Phasenoptimierung
bei 2.800 Hz. Keine Aenderung an `spatial_analyzer/*.py`. Keine Uebernahme
von Fremdcode aus inkompatiblen Lizenzen (insbesondere: nichts aus
`pierreaubert/autoeq`/`SotF`, da GPLv3).

## Quellen

- `DIMOSUS/Resonalyze`: https://github.com/DIMOSUS/Resonalyze (README, License.md, Datei-/Testbaum)
- MSO (Multi-Sub Optimizer): Foren-Threads auf audioholics.com, diyaudio.com, avforums.com (siehe Recherche; kein offizielles Repository, da closed-source Freeware)
- `jaakkopasanen/AutoEq`: https://github.com/jaakkopasanen/AutoEq (MIT)
- `pierreaubert/autoeq`/SotF: Paketdokumentation auf docs.rs/crate/autoeq (Lizenz-Header in `src/optim.rs`: GPLv3)
- REW-Hilfedokumentation: https://www.roomeqwizard.com/betahelp/help/html/makingmeasurements.html
- Praxisberichte zu "Acoustic Timing Reference" mit UMIK-1: avnirvana.com- und diyaudio.com-Forenthreads (siehe Recherche)
