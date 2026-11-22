"""Bassresonanz- & Abklinganalyse (20-200 Hz).

Siehe docs/SPECIFICATION.md Abschnitt 3.D.1 und Zwischenstand-Doku Abschnitt
5/5a: Kombination aus Peak-Prominenz, Q-Faktor, raeumlicher Haeufigkeit ueber
alle 11 Positionen, echtem Ausschwingverhalten aus der Impulsantwort und
Abgleich mit den theoretischen axialen Modenkandidaten (RoomModel - immer
als Modellwert gekennzeichnet, niemals als Messwert, siehe models.py).

Wichtiger Unterschied zur geprueften RoomEQ-Referenz (Abschnitt 5a der
Zwischenstand-Doku): dort fehlt trotz gegenteiliger Eigenaussage eine echte
raeumliche Konsistenzpruefung und jegliche Abkling-/Ringing-Analyse aus der
Zeitdomaene (nur eine indirekte Q-zu-Abklingzeit-Umrechnung). Dieses Modul
implementiert beides eigenstaendig:
- raeumliche Haeufigkeit ueber tatsaechlich uebergebene Mehrpositionsdaten
  (spatial_consistency()), nicht nur behauptet.
- eine echte, bandpassgefilterte Envelope-Abklinganalyse aus der IR
  (narrowband_decay_time()), zusaetzlich zur schnelleren Q-Naeherung.

Hinweis zu den perzeptiven Schwellen nach Fazenda, Stephenson & Goldberg
(2015, JASA 137(3), pp. 1088-1098, DOI 10.1121/1.4908217) - kalibriert am
2026-11-22, schliesst GitHub-Issue #2:

Die Stuetzpunkte in PERCEPTUAL_THRESHOLD_MODEL sind jetzt direkt aus dem
Fliesstext der Originalpublikation uebernommen (Abschnitt "Conclusion",
Satz "Average thresholds measured with artificial stimuli are 0.9s at
32Hz, 0.3s at 63Hz, 0.27s at 100Hz" plus die dort ebenfalls genannten
0.18s bei 150Hz und 0.17s bei 200Hz) - nicht aus einem Diagramm digitalisiert
und nicht selbst gewaehlt. Verwendet wird bewusst die "artificial stimuli"-
Kurve (absolute Schwelle mit Test-Toenen), nicht die "music stimuli"-Kurve
(praktische Schwelle, 0.51s bei 63Hz bis 0.12s bei 250Hz) - Begruendung:
`narrowband_decay_time()` misst eine schmalbandig gefilterte, toenerzeugte
Abklingzeit aus der Impulsantwort, methodisch naeher an den Test-Toenen der
Studie als an Musikmaterial.

Wichtige, durch die Quelle selbst verursachte Einschraenkung (nicht durch
uns aufgeloest, siehe Issue #2 und PR-Beschreibung): die Publikation
widerspricht sich selbst beim Knickpunkt - das Abstract nennt einen
"knee at 63Hz", der Ergebnisteil dagegen "A knee point at around 100Hz".
Wir uebernehmen deshalb beide umliegenden Stuetzpunkte (63Hz UND 100Hz)
unveraendert aus der Konklusion, statt diesen Widerspruch selbst zu
entscheiden - die Interpolationskurve zwischen ihnen (siehe
`perceptual_threshold_s()`) ist dadurch automatisch unser eigenes Modell,
nicht die Position des echten Knicks.

Messbereich der Studie: 32-250 Hz (artificial stimuli: 32-200 Hz). Fuer
Frequenzen AUSSERHALB dieses Bereichs (hier: < 32 Hz, im fuer dieses
Projekt relevanten 20-200-Hz-Fokus) wird in `perceptual_threshold_s()`
bewusst NICHT extrapoliert, sondern auf den jeweiligen Randwert geklemmt
(`np.clip`) - die Studie selbst warnt ausdruecklich davor, Schwellen ueber
den gemessenen Bereich hinaus zu extrapolieren ("the extrapolation of
obtained Q factor thresholds onto frequency dependent decay thresholds
cannot be assumed").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from spatial_analyzer.models import FieldMeaning, RoomModel

DEFAULT_PROMINENCE_DB = 3.0
DEFAULT_OCTAVE_WINDOW = 1.0
DEFAULT_SPATIAL_TOLERANCE_HZ = 2.0
DEFAULT_SPATIAL_PRESENCE_THRESHOLD = 0.7

# Kalibriert gegen Fazenda, Stephenson & Goldberg (2015), Abschnitt
# "Conclusion", "artificial stimuli"-Mittelwerte (siehe Modul-Docstring für
# die genauen Zitatstellen und die bewusst nicht aufgeloeste Knick-Frage).
# Stuetzpunkte (Frequenz_Hz, max. tolerierbare Abklingzeit_s auf -60dB,
# T20/T30-Konvention): direkt aus dem Text uebernommene Messwerte.
PERCEPTUAL_THRESHOLD_MODEL: tuple[tuple[float, float], ...] = (
    (32.0, 0.90),
    (63.0, 0.30),
    (100.0, 0.27),
    (150.0, 0.18),
    (200.0, 0.17),
)


@dataclass
class PeakCandidate:
    freq_hz: float
    level_db: float
    prominence_db: float
    q_factor: float | None


def find_peak_candidates(
    freq_hz: np.ndarray,
    spl_db: np.ndarray,
    band_hz: tuple[float, float] = (20.0, 200.0),
    prominence_db: float = DEFAULT_PROMINENCE_DB,
    octave_window: float = DEFAULT_OCTAVE_WINDOW,
) -> list[PeakCandidate]:
    """Lokale Maxima im Band mit Prominenz relativ zu einem +-octave_window
    Oktaven breiten Nachbarschaftsfenster (log-Frequenz), plus Q-Faktor-
    Schaetzung ueber die -3-dB-Breite um den Peak.
    """
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    spl_db = np.asarray(spl_db, dtype=np.float64)
    band_mask = (freq_hz >= band_hz[0]) & (freq_hz <= band_hz[1])
    indices = np.where(band_mask)[0]
    if indices.size < 3:
        return []

    log_f = np.log10(freq_hz)
    candidates: list[PeakCandidate] = []
    for idx in indices[1:-1]:
        if not (spl_db[idx] >= spl_db[idx - 1] and spl_db[idx] >= spl_db[idx + 1]):
            continue
        neighborhood_mask = np.abs(log_f - log_f[idx]) <= octave_window
        neighborhood_mask[idx] = False
        if not np.any(neighborhood_mask):
            continue
        neighborhood_mean = float(np.nanmean(spl_db[neighborhood_mask]))
        prominence = float(spl_db[idx] - neighborhood_mean)
        if prominence < prominence_db:
            continue
        q = _estimate_q_factor(freq_hz, spl_db, idx)
        candidates.append(
            PeakCandidate(
                freq_hz=float(freq_hz[idx]),
                level_db=float(spl_db[idx]),
                prominence_db=prominence,
                q_factor=q,
            )
        )
    return candidates


def _estimate_q_factor(freq_hz: np.ndarray, spl_db: np.ndarray, peak_idx: int) -> float | None:
    """Q = f0 / Bandbreite(-3dB). None, wenn keine -3dB-Breite innerhalb des
    verfuegbaren Rasters gefunden wird (sehr schmaler oder am Rand liegender
    Peak - bewusst als "nicht bestimmbar" statt als erfundener Wert markiert).
    """
    peak_level = spl_db[peak_idx]
    target = peak_level - 3.0

    lower_idx = peak_idx
    while lower_idx > 0 and spl_db[lower_idx] > target:
        lower_idx -= 1
    upper_idx = peak_idx
    while upper_idx < spl_db.size - 1 and spl_db[upper_idx] > target:
        upper_idx += 1

    if lower_idx == 0 and spl_db[lower_idx] > target:
        return None
    if upper_idx == spl_db.size - 1 and spl_db[upper_idx] > target:
        return None
    if upper_idx <= lower_idx:
        return None

    bandwidth = freq_hz[upper_idx] - freq_hz[lower_idx]
    if bandwidth <= 0:
        return None
    return float(freq_hz[peak_idx] / bandwidth)


def spatial_consistency(
    peaks_per_position: list[list[float]],
    tolerance_hz: float = DEFAULT_SPATIAL_TOLERANCE_HZ,
) -> list[tuple[float, float]]:
    """Gruppiert Peak-Frequenzen ueber mehrere Positionen und gibt pro
    gefundenem Cluster (mittlere Frequenz, Anteil der Positionen mit
    einem Peak im Cluster) zurueck.

    Simple Greedy-Clusterbildung: sortiert alle Peak-Frequenzen, fasst
    Werte innerhalb `tolerance_hz` zusammen. Ausreichend fuer die Groessen-
    ordnung von 11 Positionen mit wenigen Peaks je Position.
    """
    n_positions = len(peaks_per_position)
    all_freqs = sorted(f for freqs in peaks_per_position for f in freqs)
    if not all_freqs:
        return []

    clusters: list[list[float]] = [[all_freqs[0]]]
    for f in all_freqs[1:]:
        if f - clusters[-1][-1] <= tolerance_hz:
            clusters[-1].append(f)
        else:
            clusters.append([f])

    result = []
    for cluster in clusters:
        mean_freq = float(np.mean(cluster))
        # Anteil der Positionen, die MINDESTENS einen Peak in diesem Cluster
        # beigetragen haben (nicht die Gesamtzahl der Peaks, falls eine
        # Position zufaellig zwei nahe beieinanderliegende Peaks hat).
        positions_in_cluster = sum(
            1 for freqs in peaks_per_position if any(abs(f - mean_freq) <= tolerance_hz for f in freqs)
        )
        presence_fraction = positions_in_cluster / n_positions
        result.append((mean_freq, presence_fraction))
    return result


def compare_to_room_model(
    freq_hz: float, room: RoomModel, tolerance_hz: float = DEFAULT_SPATIAL_TOLERANCE_HZ, max_order: int = 2
) -> tuple[str, float, FieldMeaning] | None:
    """Vergleicht eine gefundene Frequenz mit den theoretischen axialen
    Modenkandidaten des RoomModel. Gibt (Kandidaten-Key, Kandidaten-Freq,
    FieldMeaning.MODEL) zurueck, oder None, wenn kein Kandidat innerhalb der
    Toleranz liegt - der Rueckgabewert ist IMMER ein Modellwert, nie ein
    Messwert.
    """
    candidates = room.axial_mode_candidates(max_order=max_order)
    best_key = None
    best_freq = None
    best_diff = tolerance_hz
    for key, cand_freq in candidates.items():
        diff = abs(freq_hz - cand_freq)
        if diff <= best_diff:
            best_diff = diff
            best_key = key
            best_freq = cand_freq
    if best_key is None:
        return None
    return best_key, best_freq, FieldMeaning.MODEL


@dataclass
class NarrowbandDecay:
    center_freq_hz: float
    decay_time_s: float
    exceeds_perceptual_threshold: bool | None
    threshold_s: float | None


def narrowband_decay_time(
    ir_samples: np.ndarray,
    sample_interval_s: float,
    center_freq_hz: float,
    bandwidth_oct: float = 1.0 / 3.0,
    fit_range_db: tuple[float, float] = (-5.0, -25.0),
) -> float:
    """Schaetzt die Abklingzeit (Sekunden bis -60 dB) in einem schmalen
    Frequenzband um `center_freq_hz`, direkt aus der Impulsantwort.

    Folgt der in der Raumakustik ueblichen T20/T30-Konvention (vgl. die in
    der Zwischenstand-Doku Abschnitt 7b referenzierte `room-eq-wizard-mcp`
    get_rt60-Funktion): der lineare Fit wird NICHT bis zum Rauschboden
    gezogen, sondern nur auf dem frueheren Abschnitt des Abklingvorgangs
    (Default -5 bis -25 dB relativ zum Peak) gebildet und dann linear auf
    -60 dB extrapoliert. Grund: bei einem schmalbandigen Filter dominiert
    im spaeteren Verlauf oft das Ausklingen des Filters selbst (Filter-
    "Ringing") statt des tatsaechlichen Signalabfalls - ein Fit bis zum
    Rauschboden wuerde dann systematisch eine zu lange Abklingzeit liefern.

    Dies ist eine ECHTE Zeitbereichsmessung (bandpassgefilterte
    Hilbert-Envelope + linearer Fit des fruehen Abfalls), keine Q-zu-Zeit-
    Umrechnung wie in der geprueften RoomEQ-Referenz (siehe Modul-Docstring).
    """
    ir_samples = np.asarray(ir_samples, dtype=np.float64)
    fs = 1.0 / sample_interval_s
    nyquist = fs / 2.0

    factor = 2.0 ** (bandwidth_oct / 2.0)
    low = center_freq_hz / factor
    high = center_freq_hz * factor
    low = max(low, 1.0)
    high = min(high, nyquist * 0.99)
    if low >= high:
        raise ValueError(f"Ungueltiges Bandpass-Band fuer {center_freq_hz} Hz bei fs={fs} Hz")

    sos = butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, ir_samples)
    envelope = np.abs(hilbert(filtered))
    peak_idx = int(np.argmax(envelope))
    peak_value = envelope[peak_idx]
    if peak_value <= 0:
        return float("nan")

    with np.errstate(divide="ignore"):
        decay_db = 20.0 * np.log10(np.maximum(envelope[peak_idx:], 1e-12) / peak_value)

    # Nur bis zum globalen Minimum der Abklingkurve betrachten: danach kann
    # die Hilbert-Envelope durch Rauschen/Filter-Randeffekte am Ende des
    # Arrays wieder ansteigen, was faelschlich als "spaete Abklingphase" in
    # den Fit-Bereich rutschen und das Ergebnis verfaelschen wuerde.
    global_min_idx = int(np.argmin(decay_db))
    decay_db = decay_db[: global_min_idx + 1]

    upper_db, lower_db = max(fit_range_db), min(fit_range_db)
    valid_mask = (decay_db <= upper_db) & (decay_db >= lower_db)
    if np.sum(valid_mask) < 2:
        return float("nan")

    t = np.arange(decay_db.size)[valid_mask] * sample_interval_s
    y = decay_db[valid_mask]
    # Linearer Fit dB(t) = slope*t + intercept; Abklingzeit bis -60dB:
    slope, _intercept = np.polyfit(t, y, 1)
    if slope >= 0:
        return float("nan")  # kein Abklingen erkennbar
    return float(-60.0 / slope)


def perceptual_threshold_s(freq_hz: float) -> float:
    """Interpoliert die Schwelle aus PERCEPTUAL_THRESHOLD_MODEL.

    Haertungshinweis (ab 2026-11-22, Issue #2 - siehe Modul-Docstring fuer
    die Herkunft der Stuetzpunkte):

    - **Zwischen den Stuetzpunkten:** linear in log10(Frequenz) UND
      log10(Zeit) (doppelt-logarithmische Gerade zwischen je zwei
      benachbarten Punkten). Das ist eine EIGENE Modellentscheidung fuer
      die Interpolationsform, nicht Teil der Originalpublikation - diese
      gibt laut eigener Aussage keine gefittete Kurve an, nur gemessene
      Einzelwerte (siehe Modul-Docstring). Log-log wurde gewaehlt, weil es
      die berichtete Form ("thresholds decrease rapidly... up to about
      100Hz where they appear to level out") qualitativ nachbildet, ohne
      selbst einen Wert zwischen den Stuetzpunkten zu erfinden, der nicht
      zumindest zwischen den beiden echten Nachbarwerten liegt.
    - **Ausserhalb des Stuetzpunktbereichs (hier: < 32 Hz oder > 200 Hz):**
      KEINE Extrapolation - der jeweilige Randwert wird per `np.clip`
      konstant fortgesetzt (0.9s fuer alles <= 32 Hz, 0.17s fuer alles
      >= 200 Hz). Das entspricht ausdruecklich der Warnung der
      Originalstudie vor Extrapolation ausserhalb des gemessenen Bereichs.
    """
    points = PERCEPTUAL_THRESHOLD_MODEL
    freqs = np.array([p[0] for p in points])
    times = np.array([p[1] for p in points])
    log_f = np.log10(freqs)
    log_t = np.log10(times)
    clamped = np.clip(freq_hz, freqs[0], freqs[-1])
    interp_log_t = np.interp(np.log10(clamped), log_f, log_t)
    return float(10.0 ** interp_log_t)


def evaluate_decay_against_threshold(decay_time_s: float, freq_hz: float) -> NarrowbandDecay:
    if np.isnan(decay_time_s):
        return NarrowbandDecay(
            center_freq_hz=freq_hz, decay_time_s=decay_time_s,
            exceeds_perceptual_threshold=None, threshold_s=None,
        )
    threshold = perceptual_threshold_s(freq_hz)
    return NarrowbandDecay(
        center_freq_hz=freq_hz,
        decay_time_s=decay_time_s,
        exceeds_perceptual_threshold=decay_time_s > threshold,
        threshold_s=threshold,
    )
