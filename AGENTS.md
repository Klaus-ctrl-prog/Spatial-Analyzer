# Projekt-Verfassung & KI-Arbeitsteilung (`AGENTS.md`)

## 0. Aktueller Stand (nur hier pflegen)
- **Arbeits-Branch:** `main`. Der Branch `fix/package-a-measurement-qc-20261124` ist mit PR #31 gemergt und abgeschlossen.
- **Arbeitspakete für Claude Code:** `Paket A` (Mess-QC, Commits `90473f4`, `354bfbe`), `Paket B` (guided session QC, Commit `ce2bbb3`) und `Paket C` (Filter-Export, Saxx/Mosconi-Adapter) sind implementiert und Bestandteil von PR #31 (gemergt am 27.11.2026, laut GitHub `MERGED`).
- **Bezeichnung (Festlegung):** „Paket A/B/C“ bezieht sich auf die Pakete aus PR #31. Die Labels in Commit `b0abfb3` sind davon zu trennen: „Package A (release integrity)“ meint die Release-Bindung in `filter_export`, „Package B (Saxx mapping integrity)“ meint die Saxx-Mapping-Integrität. Für das zweite Label gilt künftig „Mapping-Integrität“.
- **Review-Stand Paket A/B:** Kein unabhängiges Review für Paket A oder B ist in Commit-Nachrichten, `docs/` oder `AGENTS.md` dokumentiert. Suche am 28.11.2026 (grep in `docs/`, `README*`; `git log --all --grep` nach „Paket A“, „Paket B“, „Package A“, „Package B“): keine Spezifikation für Paket A oder B gefunden. In `AGENTS.md` (seit `bbae768`) und in den Commit-Nachrichten kommen nur Arbeitspaket-Bezeichnungen vor.
- **Review-Stand PR #31:** Befund G1 (`build_preview`) ist behoben, committet und gepusht (`4253dee`). Der Review-Vermerk „APPROVED“ steht nur in diesem Dokument, nicht in der Commit-Nachricht. PR #31 ist gemergt.
- **G2 (Mosconi `delay_to_steps`/`delay_grid_value`):** implementiert, committet und gepusht (`3ed4fef`); Ubuntu- und Windows-CI erfolgreich. Der Review-Vermerk „APPROVED“ (Sonnet 5.5) steht nur in diesem Dokument.
- **G3 (`-0.0` im Digest, `filter_export.py`):** im Code adressiert durch Commit `b0abfb3` (Normierung in `filter_digest`, Regressionstest `test_filter_export.py`). `tests/test_filter_export.py` läuft grün (92 Tests). G3 ist unabhängig geprüft (`APPROVED`, Review am 28.11.2026). Ausgenommen ist der Legacy-Digest `compute_filters_digest` (nur Version 1.0), der laut Docstring absichtlich ohne `-0.0`-Normierung arbeitet.
- **Stand 28.11.2026 (uncommitted, Branch `docs/paket-a-entscheidung`):** Paket A mit vorläufigen Arbeitswerten entschieden (`docs/PAKET_A_SCHWELLEN_ENTSCHEIDUNG.md`, Abschnitt 9). Pipeline schreibt `filter_export_kef.json` und, bei Sub-Daten, `filter_export_saxx.json` (immer BLOCKED). Phase-Gate: `sub_optimizer.optimize_phase_only` (Saxx-Phase in 45°-Stufen, ohne Delay). `optimize()` lehnt andere Winkel als 0/180° ab. CLI `saxx-preview`.
- **Schritt 3 (unabhängiges Review A/B, 28.11.2026):** `APPROVED WITH FINDINGS`. F1 (gelockerte Toleranzen → keine Freigabe), F3 (KEF-RELEASED nur ohne `crossovers`) und F4 (Subwoofer-Exporte `do_no_harm: INCONCLUSIVE`) behoben (Branch `fix/schritt3-findings`). F5–F7 (Vorschau-only) nur dokumentiert.
- **F2 und Zielkurve (Branch `fix/f2-measurement-binding`):** KEF-RELEASED nur, wenn `measurement_set_id` dem Digest der geprüften Messdaten (`NoHarmResult.checked_data_sha256`) entspricht und das Zielkurvenkriterium gelaufen ist. Ohne Vertragsänderung (1.1). Grenze: Die Bindung wirkt beim Erzeugen; eine Datei enthält die Messdaten nicht und kann sie daher nicht selbst nachweisen.
- **Weiterhin offen:** Hardwarevalidierung (`hardware_validated: false`), Vorzeichen der Saxx-Phase (90/270° ggf. vertauscht), reales Saxx-Template, Subwoofer-PEQs im Kern, Validierung der Arbeitswerte mit realen Messdaten.
- **Reale Messdaten:** Frühjahr 2027
- **Saxx-Hardware-Rundtest:** ab April 2027

## 1. Rollen & Zuständigkeiten

| Rolle | Aufgaben | Grenze |
|---|---|---|
| **Claude Sonnet, Review-Sitzung** (unabhängiger Reviewer, ab 28.11.2026 statt ChatGPT/Codex) | Neue Sitzung ohne Kontext der Arbeitssitzung, `/code-review medium <PR>`; Schnittstellenverträge (`filter_export.schema.json`), Tabellen-Spezifikationen, Offline-Validierungswerkzeuge und Checklisten erstellen/prüfen; Draft-PRs gegen Spezifikationen und diese Verfassung reviewen | Code-Reviews strikt read-only; keine Schreib- oder Merge-Rechte |
| **Claude Code** (Implementierer, Test-Ausführer) | Freigegebene Pakete im lokalen Git-Arbeitsbaum umsetzen, `pytest` ausführen, Review-Befunde beheben, Draft-PRs auf GitHub erstellen | Kein automatischer Arbeitsbeginn ohne Freigabe; kein stilles Refactoring geprüfter Funktionen/Tests |
| **Projektleiter** (Mensch) | Pakete freigeben, Vier-Augen-Review (Arbeitssitzung + Review-Sitzung) prüfen, finalen Merge durchführen | Einziger, der einen Merge in `main` freigibt und ausführt |

## 2. Ablauf je Arbeitspaket (Vier-Augen-Prinzip)
1. **Plan & Check (nur lesend):** Claude prüft den tatsächlichen lokalen Repository-Zustand, benennt die exakten Ansatzpunkte und wartet auf die ausdrückliche Freigabe des Projektleiters.
2. **Umsetzung & Test:** Claude implementiert das freigegebene Paket auf dem Arbeits-Branch ohne stilles Refactoring bestehender geprüfter Funktionen/Tests, führt betroffene `pytest`-Tests und vor dem **Draft-PR** die gesamte Test-Suite aus.
3. **Unabhängiges Review:** Eine neue Claude-Sonnet-Sitzung (nicht die Arbeitssitzung) prüft den Draft-PR read-only gegen diese `AGENTS.md` und die Spezifikationen und klassifiziert das Ergebnis eindeutig als `APPROVED`, `CHANGES_REQUIRED` oder `INCONCLUSIVE`.
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

