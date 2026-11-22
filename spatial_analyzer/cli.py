"""Einfaches Kommandozeilen-Interface: `python -m spatial_analyzer analyze ...`.

Siehe docs/SPECIFICATION.md Abschnitt 3.F ("Einfaches Kommandozeilen-
Interface ... zur Auswertung von .mdat-Dateien oder REW-API-Sitzungen").
Da der echte .mdat-Byte-Parser noch nicht verifiziert ist (siehe mdat.py),
arbeitet der `analyze`-Befehl auf der dokumentierten JSON-Zwischenform
(siehe pipeline.py-Docstring), nicht auf echten Binaer-.mdat-Dateien.
"""

from __future__ import annotations

import argparse
import sys

from spatial_analyzer.pipeline import run_pipeline


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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze":
        report = run_pipeline(args.input, args.output_dir)
        print(f"Report geschrieben nach: {args.output_dir}")
        print(f"Modenkandidaten gefunden: {len(report.mode_confirmations)}")
        print(f"EQ-Vorschlaege L: {len(report.eq_proposals_l)}, R: {len(report.eq_proposals_r)}")
        for note in report.notes:
            print(f"Hinweis: {note}")
        return 0

    parser.error(f"Unbekannter Befehl: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
