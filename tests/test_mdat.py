import numpy as np
import pytest

from spatial_analyzer.mdat import (
    AllocationBudgetExceeded,
    MdatParser,
    MdatValidationError,
)
from spatial_analyzer.models import FieldMeaning


def _base_sweep_dict():
    return {
        "frequency_spacing": {"valid_start_hz": 20.0, "valid_end_hz": 20000.0, "ppo": 48},
        "frequency_response": {
            "frequency_hz": [20.0, 100.0, 1000.0],
            "spl_db": [60.0, 70.0, 68.0],
            "phase_deg": [0.0, -30.0, -90.0],
        },
        "impulse_response": {
            "samples": [0.0, 0.01, 0.9, -0.1, 0.02],
            "sample_interval_s": 1 / 48000,
        },
        "room_dimensions": {"length_m": 5.72, "width_m": 3.72, "height_m": 2.40},
        "notes": "Sofamitte, L-Kanal",
        "rew_version": "5.40",
    }


class TestMdatParserRequiredFields:
    def test_missing_frequency_spacing_raises(self):
        data = _base_sweep_dict()
        del data["frequency_spacing"]
        with pytest.raises(MdatValidationError):
            MdatParser().parse_dict(data)

    def test_missing_both_response_types_raises(self):
        data = _base_sweep_dict()
        del data["frequency_response"]
        del data["impulse_response"]
        with pytest.raises(MdatValidationError):
            MdatParser().parse_dict(data)

    def test_non_dict_input_raises(self):
        with pytest.raises(MdatValidationError):
            MdatParser().parse_dict("not a dict")  # type: ignore[arg-type]


class TestMdatParserVersionFallback:
    def test_uses_declared_version_when_present(self):
        record = MdatParser().parse_dict(_base_sweep_dict())
        assert record.rew_version == "5.40"
        assert record.rew_version_meaning == FieldMeaning.CONFIRMED

    def test_falls_back_to_header_version_when_missing(self):
        data = _base_sweep_dict()
        del data["rew_version"]
        record = MdatParser().parse_dict(data, header_rew_version_fallback="5.30-beta")
        assert record.rew_version == "5.30-beta"
        assert record.rew_version_meaning == FieldMeaning.INFERRED


class TestMdatParserUnknownFields:
    def test_unexpected_top_level_keys_are_preserved(self):
        data = _base_sweep_dict()
        data["some_future_field"] = {"x": 1}
        record = MdatParser().parse_dict(data)
        assert record.unknown_fields == {"some_future_field": {"x": 1}}


class TestMdatParserAllocationBudget:
    def test_oversized_array_is_rejected(self):
        data = _base_sweep_dict()
        parser = MdatParser(max_allocation_bytes=1024)
        data["impulse_response"]["samples"] = list(np.zeros(10_000))
        with pytest.raises(AllocationBudgetExceeded):
            parser.parse_dict(data)

    def test_reasonable_array_within_default_budget(self):
        record = MdatParser().parse_dict(_base_sweep_dict())
        assert record.impulse_response is not None


class TestMdatParserSweepVsRta:
    def test_clear_peak_is_classified_as_sweep(self):
        record = MdatParser().parse_dict(_base_sweep_dict())
        assert record.is_sweep is True

    def test_flat_noise_without_peak_is_not_sweep(self):
        data = _base_sweep_dict()
        data["impulse_response"]["samples"] = [0.1, 0.11, 0.09, 0.1, 0.1]
        record = MdatParser().parse_dict(data)
        assert record.is_sweep is False

    def test_no_impulse_response_is_not_sweep(self):
        data = _base_sweep_dict()
        del data["impulse_response"]
        record = MdatParser().parse_dict(data)
        assert record.is_sweep is False

    def test_single_sample_impulse_response_is_not_sweep(self):
        # Grenzfall: genau ein Sample - kein "Rest" zum Vergleich moeglich,
        # per Definition (ir.samples.size <= 1) als nicht-Sweep behandelt.
        data = _base_sweep_dict()
        data["impulse_response"]["samples"] = [0.9]
        record = MdatParser().parse_dict(data)
        assert record.is_sweep is False

    def test_all_zero_samples_is_not_sweep(self):
        # median_rest <= 0 Zweig: ein durchgaengig stummer Kanal hat keinen
        # erkennbaren Peak (peak_value ebenfalls 0, also nicht > 0).
        data = _base_sweep_dict()
        data["impulse_response"]["samples"] = [0.0, 0.0, 0.0, 0.0, 0.0]
        record = MdatParser().parse_dict(data)
        assert record.is_sweep is False

    def test_peak_exactly_at_factor_three_threshold_is_sweep(self):
        # Peak genau beim dokumentierten Schwellenwert (Faktor >= 3 ueber
        # dem Median der restlichen Samples) - Grenzfall der Heuristik.
        data = _base_sweep_dict()
        data["impulse_response"]["samples"] = [0.3, 0.3, 0.9, 0.3, 0.3]
        record = MdatParser().parse_dict(data)
        assert record.is_sweep is True

    def test_peak_just_below_factor_three_threshold_is_not_sweep(self):
        data = _base_sweep_dict()
        data["impulse_response"]["samples"] = [0.3, 0.3, 0.89, 0.3, 0.3]
        record = MdatParser().parse_dict(data)
        assert record.is_sweep is False


class TestMdatParserStartTime:
    """Haertung (ab 2026-11-22): REWs Zeitbezug `start_time_s` darf beim
    Parsen weder stillschweigend verschluckt noch mit 0.0 verwechselt
    werden (siehe Modul-Docstring von mdat.py und
    claude/secs_mdat_pruefbericht.md Abschnitt 2)."""

    def test_start_time_s_is_preserved_when_present(self):
        data = _base_sweep_dict()
        data["impulse_response"]["start_time_s"] = 1732273200.5
        record = MdatParser().parse_dict(data)
        assert record.impulse_response is not None
        assert record.impulse_response.start_time_s == pytest.approx(1732273200.5)

    def test_start_time_s_is_none_not_zero_when_absent(self):
        record = MdatParser().parse_dict(_base_sweep_dict())
        assert record.impulse_response is not None
        assert record.impulse_response.start_time_s is None

    def test_start_time_s_of_exactly_zero_is_preserved_distinct_from_absent(self):
        data = _base_sweep_dict()
        data["impulse_response"]["start_time_s"] = 0.0
        record = MdatParser().parse_dict(data)
        assert record.impulse_response.start_time_s == 0.0
        assert record.impulse_response.start_time_s is not None

    def test_start_time_s_is_independent_of_measurement_delay_s(self):
        data = _base_sweep_dict()
        data["impulse_response"]["start_time_s"] = 1732273200.5
        data["impulse_response"]["measurement_delay_s"] = 0.0123
        record = MdatParser().parse_dict(data)
        ir = record.impulse_response
        assert ir.start_time_s == pytest.approx(1732273200.5)
        assert ir.measurement_delay_s == pytest.approx(0.0123)


class TestMdatParserRoomDimensions:
    def test_room_dimensions_parsed_with_confirmed_meaning(self):
        record = MdatParser().parse_dict(_base_sweep_dict())
        assert record.room_dimensions is not None
        assert record.room_dimensions.length_m == 5.72
        assert record.room_dimensions.meaning == FieldMeaning.CONFIRMED

    def test_missing_room_dimensions_is_none(self):
        data = _base_sweep_dict()
        del data["room_dimensions"]
        record = MdatParser().parse_dict(data)
        assert record.room_dimensions is None


class TestMdatParserBytesNotImplemented:
    def test_parse_bytes_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            MdatParser().parse_bytes(b"\x00\x01\x02")


class TestMdatParserMalformedStructures:
    """Haertung (Prioritaet 4, Issue #7): fehlerhafte/unvollstaendige
    Strukturen und falsche Datentypen muessen als klare
    `MdatValidationError` erkennbar sein statt als roher, nicht
    abgefangener TypeError/ValueError/AttributeError durchzuschlagen.

    Jeder Fall hier wurde vor dem Fix live reproduziert (siehe
    Commit-Historie) und schlug mit einer uncaught Exception fehl."""

    def test_frequency_response_wrong_type_raises_validation_error(self):
        data = _base_sweep_dict()
        data["frequency_response"] = ["not", "a", "dict"]
        with pytest.raises(MdatValidationError):
            MdatParser().parse_dict(data)

    def test_impulse_response_wrong_type_raises_validation_error(self):
        data = _base_sweep_dict()
        data["impulse_response"] = "not a dict"
        with pytest.raises(MdatValidationError):
            MdatParser().parse_dict(data)

    def test_room_dimensions_wrong_type_raises_validation_error(self):
        data = _base_sweep_dict()
        data["room_dimensions"] = "not a dict"
        with pytest.raises(MdatValidationError):
            MdatParser().parse_dict(data)

    def test_frequency_spacing_wrong_type_raises_validation_error(self):
        data = _base_sweep_dict()
        data["frequency_spacing"] = "not a dict"
        with pytest.raises(MdatValidationError):
            MdatParser().parse_dict(data)

    def test_frequency_hz_non_numeric_raises_validation_error(self):
        data = _base_sweep_dict()
        data["frequency_response"]["frequency_hz"] = ["abc", "def"]
        with pytest.raises(MdatValidationError):
            MdatParser().parse_dict(data)

    def test_peak_index_non_numeric_raises_validation_error(self):
        data = _base_sweep_dict()
        data["impulse_response"]["peak_index"] = "abc"
        with pytest.raises(MdatValidationError):
            MdatParser().parse_dict(data)

    def test_ppo_non_numeric_raises_validation_error(self):
        data = _base_sweep_dict()
        data["frequency_spacing"]["ppo"] = "abc"
        with pytest.raises(MdatValidationError):
            MdatParser().parse_dict(data)

    def test_measurement_delay_s_non_numeric_raises_validation_error(self):
        data = _base_sweep_dict()
        data["impulse_response"]["measurement_delay_s"] = "abc"
        with pytest.raises(MdatValidationError):
            MdatParser().parse_dict(data)

    def test_spl_offset_db_non_numeric_raises_validation_error(self):
        data = _base_sweep_dict()
        data["impulse_response"]["spl_offset_db"] = "abc"
        with pytest.raises(MdatValidationError):
            MdatParser().parse_dict(data)

    def test_start_time_s_non_numeric_raises_validation_error(self):
        data = _base_sweep_dict()
        data["impulse_response"]["start_time_s"] = "abc"
        with pytest.raises(MdatValidationError):
            MdatParser().parse_dict(data)

    def test_room_dimension_non_numeric_raises_validation_error(self):
        data = _base_sweep_dict()
        data["room_dimensions"]["length_m"] = "nicht-numerisch"
        with pytest.raises(MdatValidationError):
            MdatParser().parse_dict(data)

    def test_empty_frequency_arrays_raise_validation_error(self):
        data = _base_sweep_dict()
        data["frequency_response"]["frequency_hz"] = []
        data["frequency_response"]["spl_db"] = []
        with pytest.raises(MdatValidationError):
            MdatParser().parse_dict(data)

    def test_empty_impulse_response_samples_raise_validation_error(self):
        data = _base_sweep_dict()
        data["impulse_response"]["samples"] = []
        with pytest.raises(MdatValidationError):
            MdatParser().parse_dict(data)

    def test_phase_deg_length_mismatch_raises_validation_error(self):
        # phase_deg hat eine andere Laenge als frequency_hz/spl_db - ohne
        # diese Pruefung wuerden nachgelagerte punktweise Berechnungen
        # (z. B. complex_response.to_complex) stillschweigend falsche
        # Frequenz-Phase-Zuordnungen erhalten.
        data = _base_sweep_dict()
        data["frequency_response"]["phase_deg"] = [0.0, -30.0]  # 2 statt 3 Werte
        with pytest.raises(MdatValidationError):
            MdatParser().parse_dict(data)

    def test_group_delay_s_length_mismatch_raises_validation_error(self):
        data = _base_sweep_dict()
        data["frequency_response"]["group_delay_s"] = [0.001]
        with pytest.raises(MdatValidationError):
            MdatParser().parse_dict(data)


class TestMdatParserVersionEdgeCases:
    """Haertung (Prioritaet 4, Issue #7): Verhalten bei unbekannten/leeren
    REW-Versionsangaben explizit dokumentiert und abgesichert."""

    def test_empty_string_version_falls_back_like_missing(self):
        data = _base_sweep_dict()
        data["rew_version"] = ""
        record = MdatParser().parse_dict(data, header_rew_version_fallback="5.30-beta")
        assert record.rew_version == "5.30-beta"
        assert record.rew_version_meaning == FieldMeaning.INFERRED

    def test_unknown_future_version_string_is_accepted_as_confirmed(self):
        # Eine REW-Version, die dem Projekt unbekannt ist (z. B. neuer als
        # zum Zeitpunkt der Implementierung), wird nicht abgelehnt, sondern
        # unveraendert als CONFIRMED durchgereicht - Versions-Fallback ist
        # nur fuer FEHLENDE Angaben gedacht, nicht fuer Validierung gegen
        # eine bekannte Versionsliste.
        data = _base_sweep_dict()
        data["rew_version"] = "99.99-future"
        record = MdatParser().parse_dict(data)
        assert record.rew_version == "99.99-future"
        assert record.rew_version_meaning == FieldMeaning.CONFIRMED

    def test_very_old_version_string_is_accepted_as_confirmed(self):
        data = _base_sweep_dict()
        data["rew_version"] = "4.0"
        record = MdatParser().parse_dict(data)
        assert record.rew_version == "4.0"
        assert record.rew_version_meaning == FieldMeaning.CONFIRMED


class TestMdatParserAllocationBudgetEdgeCases:
    """Haertung (Prioritaet 4, Issue #7): Speicherbudget auch fuer die
    optionalen frequency_response-Nebenarrays (phase_deg etc.), nicht nur
    fuer frequency_hz/spl_db/samples."""

    def test_oversized_phase_deg_alone_is_rejected(self):
        data = _base_sweep_dict()
        parser = MdatParser(max_allocation_bytes=1024)
        n = 10_000
        data["frequency_response"]["frequency_hz"] = list(np.linspace(20.0, 20000.0, n))
        data["frequency_response"]["spl_db"] = list(np.zeros(n))
        data["frequency_response"]["phase_deg"] = list(np.zeros(n))
        with pytest.raises(AllocationBudgetExceeded):
            parser.parse_dict(data)
