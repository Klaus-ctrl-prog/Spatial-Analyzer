import numpy as np
import pytest

from spatial_analyzer.sub_optimizer import ObjectiveWeights, optimize


def _constant_complex(value_db, n_freqs, phase_deg=0.0):
    amp = 10.0 ** (value_db / 20.0)
    return np.full(n_freqs, amp * np.exp(1j * np.deg2rad(phase_deg)), dtype=np.complex128)


class TestObjectiveWeights:
    def test_rejects_weights_not_summing_to_one(self):
        with pytest.raises(ValueError):
            ObjectiveWeights(target_weight=0.5, spatial_weight=0.6)

    def test_default_weights_sum_to_one(self):
        w = ObjectiveWeights()
        assert w.target_weight + w.spatial_weight == pytest.approx(1.0)


class TestOptimize:
    def test_finds_zero_delay_zero_polarity_for_matched_identical_sources(self):
        # H_LR und H_Sub identisch und phasengleich -> beste Loesung sollte
        # 0 Grad Polaritaet und (wegen Periodizitaet) kleines Delay sein,
        # ausser das Optimum liegt bei einer Gain-Anpassung.
        freq = np.linspace(50.0, 120.0, 50)
        n_positions = 3
        h_lr = np.tile(_constant_complex(70.0, freq.size), (n_positions, 1))
        h_sub = np.tile(_constant_complex(70.0, freq.size), (n_positions, 1))
        target = np.full(freq.size, 76.0)  # Ziel: +6dB ueber Einzelquelle (konstruktive Summe)

        result = optimize(
            freq, h_lr, h_sub, target,
            band_hz=(50.0, 120.0),
            gain_range_db=(-6.0, 6.0), gain_step_db=1.0,
            delay_range_ms=(0.0, 2.0), delay_step_ms=0.5,
            polarities_deg=(0.0, 180.0),
        )
        assert result.best_polarity_deg == 0.0
        assert result.best_delay_ms == pytest.approx(0.0, abs=0.01)

    def test_mode_band_rmse_is_finite_when_band_present(self):
        freq = np.linspace(50.0, 120.0, 71)  # 1 Hz Schritte, enthaelt 60-62
        n_positions = 2
        h_lr = np.tile(_constant_complex(70.0, freq.size), (n_positions, 1))
        h_sub = np.tile(_constant_complex(65.0, freq.size), (n_positions, 1))
        target = np.full(freq.size, 70.0)
        result = optimize(
            freq, h_lr, h_sub, target,
            gain_range_db=(-2.0, 2.0), gain_step_db=2.0,
            delay_range_ms=(0.0, 1.0), delay_step_ms=1.0,
            polarities_deg=(0.0, 180.0),
        )
        assert np.isfinite(result.mode_band_rmse_at_best_db)
        assert result.mode_band_hz == (60.0, 62.0)

    def test_rejects_mismatched_position_counts(self):
        freq = np.linspace(50.0, 120.0, 10)
        h_lr = np.tile(_constant_complex(70.0, freq.size), (2, 1))
        h_sub = np.tile(_constant_complex(70.0, freq.size), (3, 1))
        target = np.full(freq.size, 70.0)
        with pytest.raises(ValueError):
            optimize(freq, h_lr, h_sub, target)

    def test_rejects_band_outside_freq_range(self):
        freq = np.linspace(200.0, 300.0, 10)
        h_lr = np.tile(_constant_complex(70.0, freq.size), (1, 1))
        h_sub = np.tile(_constant_complex(70.0, freq.size), (1, 1))
        target = np.full(freq.size, 70.0)
        with pytest.raises(ValueError):
            optimize(freq, h_lr, h_sub, target, band_hz=(50.0, 120.0))


class TestSubSampleDelayPrecision:
    """Haertung (ab 2026-11-22): die Delay-Suche darf auf keine ganze
    Sample-Periode runden - ein deterministisches Szenario mit bekanntem,
    bewusst sub-sample-grossem Delay (deutlich kleiner als eine Sample-
    Periode bei 48 kHz, ~0.0208 ms) muss exakt gefunden werden, nicht nur
    auf die naechste ganze Sample-Periode genaehert.

    Konstruktion: H_Sub ist eine reine, phasenverschobene Kopie von H_LR
    (exp(+j*2*pi*f*tau_true) - eine bekannte, exakte Vorauseilung). Bei
    Polaritaet 180 Grad und der korrekten Gegenverzoegerung tau_true loeschen
    sich beide exakt und frequenzunabhaengig aus (H_total = 0 an jeder
    Frequenz im Band) - jede andere Delay-Stufe im Suchraster ergibt eine
    deutlich schlechtere (frequenzabhaengige) Teil-Aufhebung. Das macht das
    Optimum scharf unterscheidbar und ohne Modellannahme pruefbar.
    """

    SAMPLE_RATE_HZ = 48000.0
    SAMPLE_PERIOD_MS = 1000.0 / SAMPLE_RATE_HZ  # ~0.02083 ms

    def _delayed_pair(self, freq, tau_true_s, level_db=70.0):
        amp = 10.0 ** (level_db / 20.0)
        h_ref_row = np.full(freq.size, amp, dtype=np.complex128)
        h_sub_row = h_ref_row * np.exp(2j * np.pi * freq * tau_true_s)
        return np.tile(h_ref_row, (2, 1)), np.tile(h_sub_row, (2, 1))

    def test_finds_exact_sub_sample_delay_below_one_sample_period(self):
        tau_true_ms = 0.0037  # < 1 Sample-Periode (~0.0208 ms) bei 48 kHz
        assert tau_true_ms < self.SAMPLE_PERIOD_MS
        tau_true_s = tau_true_ms / 1000.0

        freq = np.linspace(50.0, 120.0, 36)
        h_lr, h_sub = self._delayed_pair(freq, tau_true_s)
        target = np.full(freq.size, -200.0)  # erzwingt moeglichst vollstaendige Aufhebung

        result = optimize(
            freq, h_lr, h_sub, target,
            band_hz=(50.0, 120.0),
            gain_range_db=(0.0, 0.0), gain_step_db=1.0,
            delay_range_ms=(0.0, 0.02), delay_step_ms=tau_true_ms,
            polarities_deg=(0.0, 180.0),
        )

        assert result.best_polarity_deg == 180.0
        assert result.best_delay_ms == pytest.approx(tau_true_ms, abs=1e-9)
        # Score am Optimum muss klar (hier: > 100 Punkte) besser sein als an
        # den direkten Nachbar-Gitterpunkten - sonst waere die Unterscheidung
        # kein echter Beleg fuer Sub-Sample-Genauigkeit, sondern Zufall.
        neighbor_scores = [s for s in result.score_grid[1, 0, :] if s != result.best_score]
        assert all(s - result.best_score > 100.0 for s in neighbor_scores)

    def test_rejects_wrong_polarity_even_at_correct_delay(self):
        # Gegenprobe: bei richtigem Delay, aber Polaritaet 0 Grad (statt 180)
        # darf keine Aufhebung auftreten - stellt sicher, dass der obige Test
        # tatsaechlich Phase/Delay prueft und nicht zufaellig triviale Werte.
        tau_true_ms = 0.0037
        tau_true_s = tau_true_ms / 1000.0
        freq = np.linspace(50.0, 120.0, 36)
        h_lr, h_sub = self._delayed_pair(freq, tau_true_s)
        target = np.full(freq.size, -200.0)

        result = optimize(
            freq, h_lr, h_sub, target,
            band_hz=(50.0, 120.0),
            gain_range_db=(0.0, 0.0), gain_step_db=1.0,
            delay_range_ms=(0.0, 0.02), delay_step_ms=tau_true_ms,
            polarities_deg=(0.0,),
        )
        # Ohne Polaritaetsumkehr addieren sich beide Quellen weitgehend
        # konstruktiv statt destruktiv - der Score darf nicht in die Naehe
        # des destruktiven Optimums aus dem vorigen Test kommen.
        assert result.best_score > 50.0


class TestExactAnalyticConstructiveSum:
    """Prioritaet 2, Punkt 1: exakte analytische Referenz fuer die
    konstruktive Summe zweier phasengleicher, gleich starker Quellen
    (+6.0206 dB, siehe auch test_lr_sum.py fuer dieselbe Referenz in
    predict_complex_sum). Reine algorithmische Verifikation an
    synthetischen Daten - keine Fahrzeug- oder Raumvalidierung.
    """

    def test_in_phase_equal_sources_score_is_near_zero_at_exact_target(self):
        freq = np.linspace(50.0, 120.0, 20)
        h_lr = np.tile(_constant_complex(70.0, freq.size), (2, 1))
        h_sub = np.tile(_constant_complex(70.0, freq.size), (2, 1))
        exact_constructive_db = 70.0 + 20.0 * np.log10(2.0)  # == 70 + 6.0206...
        target = np.full(freq.size, exact_constructive_db)

        result = optimize(
            freq, h_lr, h_sub, target,
            band_hz=(50.0, 120.0),
            gain_range_db=(0.0, 0.0), gain_step_db=1.0,
            delay_range_ms=(0.0, 0.0), delay_step_ms=1.0,
            polarities_deg=(0.0, 180.0),
        )
        assert result.best_polarity_deg == 0.0
        # Score darf nur aus Gleitkomma-Rundung abweichen (dokumentierte
        # Fehlergrenze), da target exakt dem analytischen Wert entspricht.
        assert result.best_score == pytest.approx(0.0, abs=1e-9)


class TestNaNGuard:
    """Prioritaet 2, Punkt 2: ein NaN/Inf-Bin im Uebergangsband darf nicht
    stillschweigend zu einem bedeutungslosen Default-Ergebnis fuehren
    (siehe Docstring-Haertungshinweis in sub_optimizer.py `optimize()`)."""

    def test_nan_in_h_sub_raises_instead_of_silent_default(self):
        freq = np.linspace(50.0, 120.0, 10)
        h_lr = np.tile(_constant_complex(70.0, freq.size), (2, 1))
        h_sub = np.tile(_constant_complex(65.0, freq.size), (2, 1))
        h_sub[0, 3] = np.nan + 0j  # z. B. aus einer Interpolation ausserhalb des Quellbereichs
        target = np.full(freq.size, 74.0)
        with pytest.raises(ValueError, match="NaN/Inf"):
            optimize(freq, h_lr, h_sub, target, band_hz=(50.0, 120.0))

    def test_nan_in_target_raises_instead_of_silent_default(self):
        freq = np.linspace(50.0, 120.0, 10)
        h_lr = np.tile(_constant_complex(70.0, freq.size), (2, 1))
        h_sub = np.tile(_constant_complex(65.0, freq.size), (2, 1))
        target = np.full(freq.size, 74.0)
        target[0] = np.nan
        with pytest.raises(ValueError, match="NaN/Inf"):
            optimize(freq, h_lr, h_sub, target, band_hz=(50.0, 120.0))

    def test_nan_outside_band_does_not_raise(self):
        # NaN ausserhalb des gewaehlten Uebergangsbands ist unproblematisch -
        # der Guard darf nicht ueberschiessend den gesamten Array pruefen.
        freq = np.linspace(50.0, 130.0, 11)
        h_lr = np.tile(_constant_complex(70.0, freq.size), (2, 1))
        h_sub = np.tile(_constant_complex(65.0, freq.size), (2, 1))
        h_sub[0, -1] = np.nan + 0j  # letzter Punkt bei 130 Hz, ausserhalb 50-120
        target = np.full(freq.size, 74.0)
        result = optimize(freq, h_lr, h_sub, target, band_hz=(50.0, 120.0))
        assert np.isfinite(result.best_score)


class TestNarrowBandDelayAmbiguity:
    """Prioritaet 2, Punkt 3: 360-Grad-Periodizitaet bei schmalen Baendern
    (siehe Docstring-Hinweis in optimize()). Dokumentiert die numerische
    Fehlergrenze dieser Mehrdeutigkeit anhand des Modenbands um 61 Hz.
    Reine algorithmische Verifikation - keine Fahrzeug- oder
    Raumvalidierung.
    """

    def test_delays_separated_by_one_period_are_nearly_degenerate(self):
        freq = np.linspace(60.9, 61.1, 5)  # sehr schmales Band um 61 Hz
        f0 = 61.0
        period_ms = 1000.0 / f0  # ~16.3934 ms

        h_lr = np.tile(_constant_complex(70.0, freq.size), (2, 1))
        h_sub = np.tile(_constant_complex(70.0, freq.size), (2, 1))
        target = np.full(freq.size, 76.0)

        result = optimize(
            freq, h_lr, h_sub, target,
            band_hz=(60.9, 61.1),
            gain_range_db=(0.0, 0.0), gain_step_db=1.0,
            delay_range_ms=(0.0, period_ms * 2), delay_step_ms=period_ms,
            polarities_deg=(0.0,),
        )
        scores = result.score_grid[0, 0, :]
        # Scores bei 0, einer und zwei Perioden duerfen sich nur im Bereich
        # von 1e-3 Score-Einheiten unterscheiden (dokumentierte Fehlergrenze
        # fuer diese Bandbreite/Frequenz) - sie sind fast, aber bewusst
        # nicht exakt, entartet.
        assert np.max(scores) - np.min(scores) < 1e-3

    def test_half_period_delay_is_clearly_worse_not_ambiguous(self):
        # Gegenprobe: eine halbe Periode (destruktiv statt konstruktiv)
        # muss sich klar (um Groessenordnungen) vom Score bei 0/einer
        # vollen Periode unterscheiden - sonst waere der obige Test kein
        # Beleg fuer echte (Fast-)Periodizitaet, sondern fuer eine
        # degenerierte Zielfunktion, die ueberhaupt nicht unterscheidet.
        freq = np.linspace(60.9, 61.1, 5)
        f0 = 61.0
        period_ms = 1000.0 / f0
        half_period_ms = period_ms / 2.0

        h_lr = np.tile(_constant_complex(70.0, freq.size), (2, 1))
        h_sub = np.tile(_constant_complex(70.0, freq.size), (2, 1))
        target = np.full(freq.size, 76.0)

        result = optimize(
            freq, h_lr, h_sub, target,
            band_hz=(60.9, 61.1),
            gain_range_db=(0.0, 0.0), gain_step_db=1.0,
            delay_range_ms=(0.0, period_ms), delay_step_ms=half_period_ms,
            polarities_deg=(0.0,),
        )
        scores = result.score_grid[0, 0, :]
        score_at_zero = scores[0]
        score_at_half_period = scores[1]
        # Mindestens drei Groessenordnungen schlechter als bei 0/voller Periode.
        assert score_at_half_period > 1000.0 * score_at_zero
