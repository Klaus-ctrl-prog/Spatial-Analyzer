# Paket A (Mess-QC): aus Code und Historie abgeleitete Spezifikation

**Status: ABGELEITET, nicht freigegeben.** Dieses Dokument ist keine fachliche Vorgabe. Es beschreibt, was die Paket-A-Commits tatsächlich ändern. Fachliche Kriterien, Grenzwerte und Bewertungen sind nicht enthalten und bleiben **OFFEN** bis zur Entscheidung des Projektleiters.

**Stand:** 28.11.2026, Basis `82888de`, PR #31 gemergt.

## Umfang

Commits: `90473f4` (Hauptänderung), `354bfbe` (Regressionstests), Folgefixes `5d164af`, `0108d00`, `8399a10`, `6311e6a`, `545ed56`, `26e3b3c`, `64ba0f7`.

Betroffene Module: `compare.py`, `pipeline.py`, `reporting.py`, `exporters.py`, `guided_session.py`, `averaging.py`.

## Abgeleitete Anforderungen

Jede Anforderung nennt den Commit, aus dem sie abgeleitet wurde. Die Anforderungen beschreiben das Verhalten des Codes, nicht ein fachlich richtiges Verhalten.

- **A1 L/R-Trennung (`90473f4`):** Referenz-Impulsantwort, Modenkandidaten, Bestätigungen, Kammfilter-Ausschlüsse und Cut-Vorschläge werden je Kanal getrennt berechnet.
- **A2 Fehlende Referenz-IR (`6311e6a`):** Pro Kanal wird eine Hinweismeldung ausgegeben. Der betroffene Kandidat wird nicht für Cut-Vorschläge verwendet.
- **A3 Vorverstärkung (`90473f4`):** Ohne Angabe wird sie aus den Filtern automatisch bestimmt.
- **A4 REW-Messungsliste nach Wiederholung (`90473f4`, `0108d00`):** Ist sie nicht abrufbar, erscheint eine Warnung, und der UUID-Ausgangsstand bleibt unverändert. Ausnahme ist der Probelauf.
- **A5 QC-Export (`90473f4`):** Die Auswahl umfasst `stopfen_ein`, `teilstopfen` und `offen`.
- **A6 Berichtskompatibilität (`5d164af`, `0108d00`):** Konstruktor und Legacy-Modus bleiben erhalten.
- **A7 Sparse-Bins (`26e3b3c`, `64ba0f7`):** Streuungsmaße erzeugen keine Laufzeitwarnungen.
- **A8 Berichtsausgabe (`8399a10`):** API-Vergleich und Ausrichtung sind im Bericht beschrieben.

## Testabdeckung

Die Tests in `test_compare.py`, `test_exporters.py`, `test_guided_session.py`, `test_pipeline.py`, `test_reporting.py` und `test_averaging.py` decken A1 bis A8 zumindest teilweise ab. Die Abdeckung von A5 (`teilstopfen`) ist nicht im Detail geprüft.

## OFFEN (fachliche Entscheidung des Projektleiters)

- Grenzwerte und Toleranzen für Modenkandidaten, Bestätigung und Kammfilter-Erkennung.
- Richtigkeit der Kanalbewertung und der Cut-Vorschläge.
- Bedeutung der Auswahl `teilstopfen` im Vergleich zu `stopfen_ein` und `offen`.
- Ob die Vorverstärkung automatisch oder fest gesetzt werden soll.
- Akzeptanzkriterien für das Gesamtpaket.

## Nächster Schritt

Review dieses Dokuments durch den Projektleiter. Danach kann ein Review des Codes gegen diese Liste laufen. Eine fachliche Freigabe bleibt beim Projektleiter.
