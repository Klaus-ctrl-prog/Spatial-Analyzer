"""Textvorschau der Abbildung: Sollwert -> Tabellenwert, Index, Abweichung.

Zeigt auch nicht freigegebene Exporte (deutlich markiert); erzeugt keine
Datei und kennt keinerlei Hardware-Zugriff.
"""

from __future__ import annotations

from typing import Any

from spatial_analyzer.adapters.saxx_ds120.mapper import MappingResult, map_export_to_edit


def build_preview(
    doc: dict[str, Any],
    *,
    profile_name: str | None = None,
    reset_unused_bands: bool = True,
    tables: dict[str, tuple[float, ...]] | None = None,
) -> MappingResult:
    """Wie map_export_to_edit, aber ohne Freigabepflicht (Vorschau).

    `require_released` wird bewusst nicht akzeptiert: die Vorschau ist nie
    freigabepflichtig und darf nicht per Argument umkonfiguriert werden.
    """
    return map_export_to_edit(
        doc,
        profile_name=profile_name,
        require_released=False,
        reset_unused_bands=reset_unused_bands,
        tables=tables,
    )


def render_preview(result: MappingResult) -> str:
    lines = [
        "Saxx DS 120 - Abbildungsvorschau (Candidate Builder 0.15.0-qa, UNVALIDIERT)",
        f"Freigabe: {'RELEASED' if result.released else 'NICHT FREIGEGEBEN'}; "
        f"hardware_validated={str(result.hardware_validated).lower()}",
        "",
    ]
    for q in result.quantization:
        lines.append("  " + q.describe())
    for n in result.notes:
        lines.append("Hinweis: " + n)
    return "\n".join(lines) + "\n"
