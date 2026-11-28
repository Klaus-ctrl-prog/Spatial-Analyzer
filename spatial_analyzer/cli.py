"""Einfaches Kommandozeilen-Interface: `python -m spatial_analyzer analyze ...`.

Siehe docs/SPECIFICATION.md Abschnitt 3.F ("Einfaches Kommandozeilen-
Interface ... zur Auswertung von .mdat-Dateien oder REW-API-Sitzungen").
Da der echte .mdat-Byte-Parser noch nicht verifiziert ist (siehe mdat.py),
arbeitet der `analyze`-Befehl auf der dokumentierten JSON-Zwischenform
(siehe pipeline.py-Docstring), nicht auf echten Binaer-.mdat-Dateien.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from spatial_analyzer.compare import run_compare
from spatial_analyzer.pipeline import run_pipeline
from spatial_analyzer.validator import has_errors, validate_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spatial_analyzer",
        description="REW-Mehrpositions-Analyzer (KEF LS50 Wireless + Saxx DS 120 DSP).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser(
        "analyze",
        help="Analysiert eine JSON-Messdatei und schreibt Report + Exporte in ein Verzeichnis.",
    )
    analyze.add_argument("input", help="Pfad zur JSON-Eingabedatei (siehe pipeline.py-Docstring)")
    analyze.add_argument("output_dir", help="Zielverzeichnis fuer Report, CSV, JSON, HTML und Exporte")
    analyze.add_argument(
        "--dry-run",
        action="store_true",
        help="Nur die Eingabedatei pruefen (Schema/Plausibilitaet), keine Analyse rechnen.",
    )

    compare = subparsers.add_parser(
        "compare",
        help="Vergleicht zwei abgeschlossene 'analyze'-Durchlaeufe (vorher/nachher).",
    )
    compare.add_argument("before_dir", help="Ausgabeverzeichnis der Vorher-Messung")
    compare.add_argument("after_dir", help="Ausgabeverzeichnis der Nachher-Messung")
    compare.add_argument("output_dir", help="Zielverzeichnis fuer compare_report.json/html")

    saxx = subparsers.add_parser(
        "saxx-preview",
        help="Zeigt die Saxx-DS120-Abbildung eines filter_export.json (nur Vorschau, keine Datei, keine Hardware).",
    )
    saxx.add_argument("export_json", help="Pfad zu filter_export_saxx.json (aus 'analyze')")

    rew_import = subparsers.add_parser(
        "import-rew",
        help="Baut aus einem guided-session-Manifest und der laufenden REW-API (nur GET) die Eingabe-JSON fuer 'analyze'.",
    )
    rew_import.add_argument("manifest", help="Pfad zu session_manifest.json (aus der guided session)")
    rew_import.add_argument("output", help="Zielpfad der Eingabe-JSON fuer 'analyze'")
    rew_import.add_argument(
        "--room",
        nargs=3,
        type=float,
        metavar=("LAENGE", "BREITE", "HOEHE"),
        help="Raummasse in m (optional, z. B. 5.72 3.72 2.40)",
    )
    rew_import.add_argument("--base-url", default=None, help="REW-API-Adresse (Standard: http://localhost:4735)")

    return parser


def _run_import_rew(manifest: str, output: str, room: list[float] | None, base_url: str | None) -> int:
    from spatial_analyzer.rew_client import DEFAULT_BASE_URL, ReadOnlyRewClient, RewApiError
    from spatial_analyzer.rew_import import RewImportError, import_from_manifest, import_warnings

    client = ReadOnlyRewClient(base_url=base_url or DEFAULT_BASE_URL, timeout_s=60.0)
    try:
        result = import_from_manifest(manifest, output, client, tuple(room) if room else None)
    except (RewImportError, RewApiError) as exc:
        print(f"Import nicht moeglich: {exc}")
        return 1
    n_sub = sum(1 for p in result["positions"] if "sub" in p)
    print(f"Eingabe-JSON geschrieben nach: {output}")
    print(f"Positionen: {len(result['positions'])} (davon mit Sub: {n_sub})")
    try:
        manifest_data = json.loads(Path(manifest).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest_data = {}
    for warning in import_warnings(manifest_data):
        print(f"Hinweis: {warning}")
    return 0


def _run_saxx_preview(path: str) -> int:
    from spatial_analyzer.adapters.saxx_ds120 import SaxxAdapterError, build_preview, render_preview
    from spatial_analyzer.filter_export import FilterExportError, load_filter_export

    try:
        print(render_preview(build_preview(load_filter_export(path))), end="")
    except (FilterExportError, SaxxAdapterError, ValueError) as exc:
        print(f"Saxx-Vorschau nicht moeglich: {exc}")
        return 1
    return 0


def _run_dry_run(input_path: str) -> int:
    issues = validate_file(input_path)
    if not issues:
        print(f"Keine Beanstandungen: {input_path} sieht gueltig aus.")
        return 0

    for issue in issues:
        print(str(issue))

    n_errors = sum(1 for i in issues if i.severity == "error")
    n_warnings = len(issues) - n_errors
    print(f"\n{n_errors} Fehler, {n_warnings} Hinweis(e).")
    if has_errors(issues):
        print("Eingabedatei ist NICHT lauffaehig - bitte Fehler oben beheben.")
        return 1
    print("Keine Fehler gefunden, nur Hinweise - 'analyze' sollte laufen.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze":
        if args.dry_run:
            return _run_dry_run(args.input)

        issues = validate_file(args.input)
        if has_errors(issues):
            print("Eingabedatei enthaelt Fehler - Analyse wird nicht gestartet:\n")
            for issue in issues:
                if issue.severity == "error":
                    print(str(issue))
            print("\nTipp: 'analyze ... --dry-run' zeigt auch die Hinweise (Warnungen) an.")
            return 1

        report = run_pipeline(args.input, args.output_dir)
        print(f"Report geschrieben nach: {args.output_dir}")
        print(f"Modenkandidaten gefunden: {len(report.mode_confirmations)}")
        print(f"EQ-Vorschlaege L: {len(report.eq_proposals_l)}, R: {len(report.eq_proposals_r)}")
        for note in report.notes:
            print(f"Hinweis: {note}")
        return 0

    if args.command == "compare":
        compare_report = run_compare(args.before_dir, args.after_dir, args.output_dir)
        print(f"Vergleichsbericht geschrieben nach: {args.output_dir}")
        for note in compare_report.notes:
            print(f"Hinweis: {note}")
        return 0

    if args.command == "import-rew":
        return _run_import_rew(args.manifest, args.output, args.room, args.base_url)

    if args.command == "saxx-preview":
        return _run_saxx_preview(args.export_json)

    parser.error(f"Unbekannter Befehl: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
