"""Export-Module: KEFplayer-JSON-Profil und REW-Generic-EQ-Textformat.

Siehe docs/SPECIFICATION.md Abschnitt 3.F. Das KEFplayer-JSON-Schema ist
bewusst identisch zum `Profile`/`Filter`-Schema aus
`Klaus-ctrl-prog/KEFplayer` (`kefplayer/dsp.py`, Methoden `to_dict()` /
`from_dict()`) nachgebaut (gegengeprueft), damit ein hier exportiertes
Profil direkt mit `Profile.from_dict()` in KEFplayer geladen werden kann.
Kein Code-Import aus KEFplayer - nur das Dict-Schema wird nachgebildet,
um Spatial-Analyzer als eigenstaendiges Projekt (siehe docs/SPECIFICATION.md
Abschnitt 1) ohne Abhaengigkeit auf das andere Repo zu halten.

Hinweis zum REW-Generic-EQ-Textformat: Das "PK"-Zeilenformat
("Filter N: ON PK Fc <fc> Hz Gain <gain> dB Q <q>") ist weithin bekannt und
dokumentiert und wird hier mit Zuversicht erzeugt. Fuer die anderen
Filtertypen (Shelf, Low-/Highpass) ist die exakte REW-interne Textsyntax in
dieser Recherchephase NICHT byte-genau gegen die REW-Dokumentation
verifiziert worden - diese Zeilen sind ein plausibler, aber nicht
hundertprozentig abgesicherter Best-Effort und sollten vor produktivem
Einsatz einmal gegen einen echten REW-Export abgeglichen werden (siehe
auch die analoge Einschraenkung beim .mdat-Byte-Parser in mdat.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spatial_analyzer.biquad_preview import PeakingFilter, response_complex

DEFAULT_FS_HZ = 48000.0

_REW_TYPE_CODE = {
    "peq": "PK",
    "lowshelf": "LS",
    "highshelf": "HS",
    "lowpass": "LP",
    "highpass": "HP",
}


def filter_to_kefplayer_dict(f: PeakingFilter) -> dict:
    """Identisch zu KEFplayer `Filter.to_dict()`: gain_db wird bei
    lowpass/highpass weggelassen."""
    d: dict = {"type": f.type, "fc": f.fc}
    if f.type not in ("lowpass", "highpass"):
        d["gain_db"] = f.gain_db
    d["q"] = f.q
    return d


def auto_preamp_db(
    filters_l: list[PeakingFilter], filters_r: list[PeakingFilter], fs: float = DEFAULT_FS_HZ
) -> float:
    """Preamp, der den hoechsten Betragsgang beider Kanaele auf <= 0 dB
    bringt - identischer Algorithmus zu KEFplayer `Profile.auto_preamp_db()`:
    Spitzenwert auf einem dichten Frequenzraster (plus den exakten fc-Werten
    der Filter) finden, dann mit math.ceil auf 2 Nachkommastellen aufrunden.
    """
    top = fs / 2 * 0.9999
    grid = np.geomspace(5.0, top, 8000)
    peaks = []
    for filters in (filters_l, filters_r):
        extra = [f.fc for f in filters if 0 < f.fc < top]
        this_grid = np.union1d(grid, np.array(extra)) if extra else grid
        h = response_complex(filters, fs, this_grid)
        with np.errstate(divide="ignore"):
            mag_db = 20.0 * np.log10(np.maximum(np.abs(h), 1e-12))
        peaks.append(float(np.max(mag_db)))
    peak = max(peaks)
    if peak <= 0:
        return 0.0
    import math

    return -math.ceil(peak * 100.0) / 100.0


def build_kefplayer_profile(
    filters_l: list[PeakingFilter],
    filters_r: list[PeakingFilter],
    name: str = "spatial_analyzer",
    preamp_db: float | None = None,
    gain_l_db: float = 0.0,
    gain_r_db: float = 0.0,
    delay_l_ms: float = 0.0,
    delay_r_ms: float = 0.0,
    fs: float = DEFAULT_FS_HZ,
) -> dict:
    """Baut ein KEFplayer-kompatibles Profil-Dict (Schema identisch zu
    `Profile.to_dict()` in kefplayer/dsp.py).

    `preamp_db=None` bedeutet "auto" (automatische Headroom-Berechnung via
    `auto_preamp_db()`, wie in KEFplayer selbst).
    """
    resolved_preamp: float | str
    if preamp_db is None:
        resolved_preamp = "auto"
    else:
        resolved_preamp = preamp_db

    return {
        "name": name,
        "preamp_db": resolved_preamp,
        "filters": [],  # keine globalen (beide-Kanaele-)Filter, alles kanalgetrennt
        "channels": {
            "L": {
                "gain_db": gain_l_db,
                "delay_ms": delay_l_ms,
                "filters": [filter_to_kefplayer_dict(f) for f in filters_l],
            },
            "R": {
                "gain_db": gain_r_db,
                "delay_ms": delay_r_ms,
                "filters": [filter_to_kefplayer_dict(f) for f in filters_r],
            },
        },
    }


def export_kefplayer_json(
    path: str | Path,
    filters_l: list[PeakingFilter],
    filters_r: list[PeakingFilter],
    name: str = "spatial_analyzer",
    preamp_db: float | None = None,
    gain_l_db: float = 0.0,
    gain_r_db: float = 0.0,
    delay_l_ms: float = 0.0,
    delay_r_ms: float = 0.0,
    fs: float = DEFAULT_FS_HZ,
) -> dict:
    """Schreibt das KEFplayer-Profil als JSON-Datei und gibt das Dict zurueck.

    `preamp_db=None` (="auto") wird vor dem Schreiben konkret berechnet
    (via `auto_preamp_db()`), damit die exportierte Datei bereits einen
    festen Zahlenwert enthaelt und nicht von einer spaeteren KEFplayer-
    Berechnung abhaengt - KEFplayer akzeptiert aber weiterhin auch "auto".
    """
    resolved_preamp = preamp_db
    if resolved_preamp is None:
        resolved_preamp = auto_preamp_db(filters_l, filters_r, fs)

    profile = build_kefplayer_profile(
        filters_l, filters_r, name=name, preamp_db=resolved_preamp,
        gain_l_db=gain_l_db, gain_r_db=gain_r_db,
        delay_l_ms=delay_l_ms, delay_r_ms=delay_r_ms, fs=fs,
    )
    Path(path).write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    return profile


def _rew_filter_line(index: int, f: PeakingFilter) -> str:
    code = _REW_TYPE_CODE[f.type]
    parts = [f"Filter {index}: ON {code} Fc {f.fc:.2f} Hz"]
    if f.type not in ("lowpass", "highpass"):
        parts.append(f"Gain {f.gain_db:.2f} dB")
    parts.append(f"Q {f.q:.3f}")
    return " ".join(parts)


def build_rew_generic_eq_text(filters: list[PeakingFilter], preamp_db: float = 0.0) -> str:
    """Erzeugt REW-Generic-EQ-Textinhalt. Siehe Modul-Docstring zur
    Verlaesslichkeit der einzelnen Filtertyp-Zeilenformate."""
    lines = [f"Preamp: {preamp_db:.2f} dB", ""]
    for i, f in enumerate(filters, start=1):
        if f.type not in _REW_TYPE_CODE:
            raise ValueError(f"Unbekannter Filtertyp fuer REW-Export: {f.type}")
        lines.append(_rew_filter_line(i, f))
    return "\n".join(lines) + "\n"


def export_rew_generic_eq_txt(
    path: str | Path,
    filters: list[PeakingFilter],
    preamp_db: float | None = None,
    header_comment: str | None = None,
) -> str:
    """`header_comment` (z. B. "NICHT FREIGEGEBEN / UNSAFE ...") wird als
    "#"-Kommentarzeilen vorangestellt. Ob REW "#"-Zeilen beim Import
    toleriert, ist NICHT verifiziert - der Kommentar wird deshalb nur fuer
    nicht freigegebene Exporte gesetzt (freigegebene Dateien bleiben
    byte-identisch zum bisherigen Format); scheitert ein Import wegen der
    Kommentarzeile, ist das fuer ein nicht freigegebenes Artefakt
    ausdruecklich akzeptabel."""
    if preamp_db is None:
        preamp_db = auto_preamp_db(filters, [])
    text = build_rew_generic_eq_text(filters, preamp_db=preamp_db)
    if header_comment:
        header = "".join(f"# {line}\n" for line in header_comment.splitlines())
        text = header + text
    Path(path).write_text(text, encoding="utf-8")
    return text
