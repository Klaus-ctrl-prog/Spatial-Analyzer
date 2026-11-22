"""Haertung (Prioritaet 7, Stand 2026-11-22): Konsistenz zwischen
docs/SPECIFICATION.md, docs/REW_NOTES_TEMPLATE.md und den Modul-Docstrings
als automatisierte Regressionstests, statt nur einmalig von Hand geprueft.

Ziel: ein zukuenftiges neues Modul oder eine Umbenennung soll nicht erneut
unbemerkt eine Spezifikations-/Docstring-Referenz verlieren (wie bei
fdw.py vor dieser Haertungsrunde - siehe ROADMAP_2027.md Prioritaet 7).
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SPATIAL_ANALYZER_DIR = REPO_ROOT / "spatial_analyzer"

# Module, die bewusst KEINE docs/SPECIFICATION.md-Abschnittsreferenz tragen:
# complex_response.py ist reine, von mehreren Spec-Abschnitten gemeinsam
# genutzte Mathe-Infrastruktur (siehe eigener Docstring, verweist auf die
# drei nutzenden Module statt auf einen einzelnen Spec-Abschnitt);
# validator.py ist eine spaetere Ergaenzung (GitHub-Issue #10, Schema-
# Validierung/--dry-run) ohne eigenen Abschnitt in der urspruenglichen
# Spezifikation. __init__.py/__main__.py sind reines Boilerplate.
MODULES_WITHOUT_SPEC_REFERENCE = {
    "__init__.py",
    "__main__.py",
    "complex_response.py",
    "validator.py",
}


def _module_files():
    return sorted(SPATIAL_ANALYZER_DIR.glob("*.py"))


class TestSpecificationReferenceCompleteness:
    """Jedes fachliche Modul soll (wie alle anderen) eine Referenz auf
    docs/SPECIFICATION.md im Docstring tragen - fdw.py fehlte das bis zu
    dieser Haertungsrunde, obwohl alle anderen 'D.x'-Module
    (bass_resonance.py, lr_sum.py, etc.py, biquad_preview.py) sie haben."""

    @pytest.mark.parametrize("path", _module_files(), ids=lambda p: p.name)
    def test_module_references_specification_unless_explicitly_exempt(self, path):
        if path.name in MODULES_WITHOUT_SPEC_REFERENCE:
            pytest.skip(f"{path.name} ist bewusst von dieser Pflicht ausgenommen")
        text = path.read_text(encoding="utf-8")
        assert "SPECIFICATION.md" in text, (
            f"{path.name} referenziert docs/SPECIFICATION.md nicht - "
            "entweder Referenz ergaenzen oder bewusst in "
            "MODULES_WITHOUT_SPEC_REFERENCE aufnehmen (mit Begruendung)."
        )

    def test_fdw_module_now_references_its_spec_section(self):
        # Regressionstest fuer den konkreten, in dieser Haertungsrunde
        # behobenen Fund: fdw.py implementiert docs/SPECIFICATION.md
        # Abschnitt 3.D.3, referenzierte das aber nirgends.
        text = (SPATIAL_ANALYZER_DIR / "fdw.py").read_text(encoding="utf-8")
        assert "SPECIFICATION.md" in text
        assert "3.D.3" in text

    def test_exempt_modules_list_matches_actual_files_on_disk(self):
        # Stellt sicher, dass die Ausnahmeliste nicht auf ein
        # umbenanntes/geloeschtes Modul zeigt und so unbemerkt zu weit wird.
        existing_names = {p.name for p in _module_files()}
        for name in MODULES_WITHOUT_SPEC_REFERENCE:
            assert name in existing_names, (
                f"{name} aus MODULES_WITHOUT_SPEC_REFERENCE existiert nicht "
                "mehr unter spatial_analyzer/ - Ausnahmeliste veraltet"
            )


class TestRewNotesTemplateConsistency:
    """docs/REW_NOTES_TEMPLATE.md verweist auf einen konkreten Berichts-
    Abschnittstitel in reporting.py - wird dieser Titel dort jemals
    umbenannt, soll das hier auffallen statt unbemerkt zu veralten."""

    def test_referenced_report_section_heading_exists_in_reporting_module(self):
        template_text = (REPO_ROOT / "docs" / "REW_NOTES_TEMPLATE.md").read_text(encoding="utf-8")
        assert "Mess- & DSP-Notizen (aus REW)" in template_text
        reporting_text = (SPATIAL_ANALYZER_DIR / "reporting.py").read_text(encoding="utf-8")
        assert "Mess- &amp; DSP-Notizen (aus REW)" in reporting_text


class TestReadmeRoadmapConsistency:
    """README.md darf keinen veralteten Phasenstand mehr behaupten und muss
    auf docs/ROADMAP_2027.md fuer den laufenden Haertungsstand verweisen."""

    def test_readme_does_not_claim_stale_phase_1_only_status(self):
        readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        # Der urspruengliche, seit Abschluss von Phase 3 veraltete Wortlaut
        # ("Phase 1 (...)" als alleiniger Status) darf nicht mehr vorkommen.
        assert "Phase 1 (Datenimport, Frequenzraster, energetische Mittelung)" not in readme_text

    def test_readme_references_roadmap_2027(self):
        readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        assert "ROADMAP_2027.md" in readme_text

    def test_readme_mentions_deferred_issue_numbers(self):
        # Die bis Fruehjahr 2027 zurueckgestellten offenen GitHub-Issues
        # (#4 FDW-Metadaten, #5 ETC-Schwellenwert, #7 .mdat-Binaervalidierung)
        # sollen im README nachvollziehbar benannt sein, nicht nur in
        # docs/ROADMAP_2027.md.
        readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        for issue_ref in ("Issue #4", "Issue #5", "Issue #7"):
            assert issue_ref in readme_text


class TestRoadmapDocumentConsistency:
    """docs/ROADMAP_2027.md muss alle sieben Prioritaeten in fortlaufender
    Nummerierung enthalten und darf docs/SMART453_BASS_SPEC.md nicht als
    veraenderbar darstellen."""

    def test_roadmap_lists_all_seven_priorities_in_order(self):
        text = (REPO_ROOT / "docs" / "ROADMAP_2027.md").read_text(encoding="utf-8")
        for n in range(1, 8):
            assert f"## Prioritaet {n} -" in text, f"Prioritaet {n} fehlt oder falsch nummeriert"

    def test_roadmap_does_not_propose_changes_to_smart453_spec(self):
        text = (REPO_ROOT / "docs" / "ROADMAP_2027.md").read_text(encoding="utf-8")
        assert "keine Aenderung an docs/SMART453_BASS_SPEC.md" in text or (
            "Aenderung von" in text and "SMART453_BASS_SPEC.md" in text
        )
