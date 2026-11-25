# Projekt-Verfassung & KI-Arbeitsteilung (`AGENTS.md`)

## 0. Aktueller Stand (nur hier pflegen)
- **Arbeits-Branch:** `fix/package-a-measurement-qc-20261124`
- **Arbeitspakete für Claude Code:** `Paket B`, `Paket C` (Umsetzung erst nach ausdrücklicher Freigabe)
- **Reale Messdaten:** Frühjahr 2027
- **Saxx-Hardware-Rundtest:** ab April 2027

## 1. Rollen & Zuständigkeiten

| Rolle | Aufgaben | Grenze |
|---|---|---|
| **ChatGPT / Codex** (Architekt, Spezifikations-Hüter, unabhängiger Reviewer) | Schnittstellenverträge (`filter_export.schema.json`), Tabellen-Spezifikationen, Offline-Validierungswerkzeuge und Checklisten erstellen/prüfen; Draft-PRs gegen Spezifikationen und diese Verfassung reviewen | Code-Reviews strikt read-only; keine Schreib- oder Merge-Rechte |
| **Claude Code** (Implementierer, Test-Ausführer) | Freigegebene Pakete im lokalen Git-Arbeitsbaum umsetzen, `pytest` ausführen, Review-Befunde beheben, Draft-PRs auf GitHub erstellen | Kein automatischer Arbeitsbeginn ohne Freigabe; kein stilles Refactoring geprüfter Funktionen/Tests |
| **Projektleiter** (Mensch) | Pakete freigeben, Vier-Augen-Review (Claude + ChatGPT) prüfen, finalen Merge durchführen | Einziger, der einen Merge in `main` freigibt und ausführt |

## 2. Ablauf je Arbeitspaket (Vier-Augen-Prinzip)
1. **Plan & Check (nur lesend):** Claude prüft den tatsächlichen lokalen Repository-Zustand, benennt die exakten Ansatzpunkte und wartet auf die ausdrückliche Freigabe des Projektleiters.
2. **Umsetzung & Test:** Claude implementiert das freigegebene Paket auf dem Arbeits-Branch ohne stilles Refactoring bestehender geprüfter Funktionen/Tests, führt betroffene `pytest`-Tests und vor dem **Draft-PR** die gesamte Test-Suite aus.
3. **Unabhängiges Review:** ChatGPT prüft den Draft-PR read-only gegen diese `AGENTS.md` und die Spezifikationen und klassifiziert das Ergebnis eindeutig als `APPROVED`, `CHANGES_REQUIRED` oder `INCONCLUSIVE`.
4. **Abschluss:** Claude behebt eventuelle `CHANGES_REQUIRED`-Befunde; nach grünem Testlauf und `APPROVED`-Review erteilt der Projektleiter die Merge-Freigabe.

## 3. Git- & Sicherheitsregeln (verbindlich)
1. **Branch-Schutz:** Entwicklung ausschließlich auf Arbeits-Branches. Kein direkter Push in `main`; Merge in `main` ausschließlich durch den Projektleiter.
2. **Strikt `GET-only`:** `ReadOnlyRewClient` führt ausschließlich `GET`-Anfragen aus. Keine Schreib-, Sweep- oder Konfigurationsänderungen in REW auslösen.
3. **Do-No-Harm-Gate:** Die Prüflogik (`SAFE`, `UNSAFE`, `INCONCLUSIVE`) in `eq_solver.py` und `pipeline.py` bleibt unverändert; nicht freigegebene Zustände müssen Exporte blockieren.
4. **Datenschutz & `.gitignore`:** Niemals Original-Executables (`.exe`), Firmware-Dumps oder echte Subwoofer-Backups (`.datdsp`, `.datgrp`) committen oder in öffentliche Repositories laden. Tests nutzen ausschließlich synthetische Testvektoren.

## 4. Fachliche Architektur-Grenzen

### 4.1 `Spatial-Analyzer`
- **Feature-Freeze (README):** Bis reale Messdaten vorliegen (siehe Abschnitt 0), keine neuen Zusatz-Features im gehärteten Kern (z. B. Web-UIs, Audio-Player, eigenständige Offline-Utilities).
- **Einmal messen, getrennt adaptieren (`filter_export.json`):** Akustik- und EQ-Berechnung ausschließlich im Kern. Geräte-Adapter unter `spatial_analyzer/adapters/` übersetzen und validieren nur.
- **Filtertrennung:** KEF-L/R-Satellitenfilter (`eq_proposals_l/r`) dürfen niemals auf den Subwoofer exportiert werden.

### 4.2 `Saxx DS 120` (Hardware- & Binärgrenzen)
- **Hardware-Gate & keine Scheinvalidierung (verbindlich):** Ein bestandener Offline-Roundtrip ist ausdrücklich keine Hardwarevalidierung. `hardware_validated: false` bleibt strikt aktiv bis zum physischen Rundtest am Gerät.
- **Nachvollziehbare Quantisierung:** Exakte Lookup-Tabellen mit `244` Frequenz-, `181` Gain- und `79` Q-Stufen; Sollwert, Tabellenwert, Index und Abweichung (`Sollwert -> Tabellenwert`) müssen dokumentiert werden.
- **Filtertypen:** `EQ 1–9`: `PEQ`, `LS6`, `LS12`, `HS6`, `HS12`. `EQ 10–14`: ausschließlich `PEQ`.
- **Weichen & Phase:** `HPF`/`LPF`: `0 = aktiv`, `1 = Bypass`. Phase: `8` Stufen zu `45°` (`0°..315°`).
- **Template-Patching:** Alle `opaque / unmapped` Bytes (`11, 16, 21, 25, 27–33`, `146–169`, `210–211`) unverändert aus der Vorlage übernehmen. `Bytes 146–169` niemals als bewiesene Extra-EQs behandeln.

### 4.3 `Mosconi Pico 8/10` & `Smart 453` (`SMART453_BASS_SPEC.md`)
- **Trennung:** Die Smart-453-Fahrzeuganalyse bleibt konzeptionell im eigenständigen Top-Level-Namespace `smart453_bass/`. `spatial_analyzer/adapters/mosconi_pico/` dient zunächst nur als Gerätefähigkeits- und Validierungsmodul.
- **Gerätegrenzen:** Delay-Raster **`0,02 ms`** (nicht `0,01 ms`), `12/24 dB LR`-Weichen, max. `6 PEQs`/Kanal, Phase `0°/180°`, Hochtöner-HPF-Schutz `>= 2.500 Hz`.

