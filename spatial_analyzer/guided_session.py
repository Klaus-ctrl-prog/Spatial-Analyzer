"""Geführte Python-Mess-Session (`python -m spatial_analyzer.guided_session`).

**Bewusste Ausnahme von raum_akustik.md Rev. 8** (Abschnitt 15.7: "Heute wird
kein Code geschrieben", kein PR vor realen Messdateien ab März 2027): Dieses
Modul wurde auf expliziten Nutzerwunsch jetzt umgesetzt, nicht erst ab März
2027. Es ändert nichts an den übrigen Phase-1-Modulen und greift nicht auf
`smart453_bass` zu (dieses Package existiert laut `docs/SMART453_BASS_SPEC.md`
noch nicht - reiner Recherchestand). Da vor März 2027 ohnehin keine neuen
Messungen möglich sind, kann dieses Modul bis dahin nur gegen Archivdaten,
synthetische Daten oder im `--dry-run`-Modus geprüft werden, nie gegen eine
echte REW-/KEF-Sitzung.

**Zweck:** Begleitet einen manuellen REW-Sweep (das Mikrofon-Umpositionieren
und das Starten des Sweeps bleiben Handarbeit, siehe
`claude/rew_automatisierung_befunde.md`), übernimmt aber die Schritte
daneben: Erkennung der neu entstandenen Messung über die REW-API, einheitliche
Benennung (`P01..P11` bzw. `ABA_A1/B/A2_P01..P11` für das Wohnzimmer-Profil,
`C01..C09` für das Smart-453-Profil), Sofort-QC (Clipping, Rauschboden,
A1/A2-Drift) und ein lokales Session-Manifest.

**GET-only-Architektur strikt eingehalten (docs/SPECIFICATION.md Abschnitt 2):**
Dieses Modul schreibt zu KEINEM Zeitpunkt in REW zurück. `ReadOnlyRewClient`
bietet bewusst keine Schreibmethode (siehe `rew_client.py`-Docstring und
`tests/test_rew_client.py::TestReadOnlyRewClientGetOnly`), und das bleibt
hier unverändert so. Das in der Projekt-Doku skizzierte "automatische
Notes-Logging über die REW-API" wurde deshalb NICHT umgesetzt. Stattdessen:
- Alles wird in ein lokales `session_manifest.json` protokolliert.
- Für jede Messung wird zusätzlich ein Text-Entwurf im Stil von
  `docs/REW_NOTES_TEMPLATE.md` als `.txt`-Datei abgelegt, den man wie bisher
  **von Hand** ins REW-Notes-Feld einfügt (REW kann DSP-Einstellungen
  externer Geräte laut Vorlage ohnehin nicht automatisch auslesen).

Der Sweep selbst wird nicht ausgelöst (kein Pro nötig, kein Risiko falscher
Kanalumschaltung über `rew_api_measurement_session`, siehe
`rew_automatisierung_befunde.md` Abschnitt 1b) - die Verantwortung dafür
bleibt beim Menschen, wie in jeder bisherigen Messserie.

Alle QC-Schwellenwerte (`QcThresholds`) sind bewusst gewählte
Arbeitshypothesen, keine aus Messungen abgeleiteten Werte (Projektregel
"keine Zahlen erfinden", docs/SPECIFICATION.md Abschnitt 2) - vor der ersten
echten Kampagne ab März 2027 zu überprüfen (analog zum offenen Punkt 22 in
`raum_akustik.md`).

**Zwischenspeicher & Fortsetzung (`--resume`):** `session_manifest.json` wird
nicht erst am Sessionende geschrieben, sondern nach JEDEM abgeschlossenen
("bestätigt") oder übersprungenen Schritt atomar (Schreiben in eine
Temp-Datei, danach `Path.replace`, siehe `SessionManifest.write`) neu
geschrieben - ein Absturz oder Stromausfall mitten in der Kampagne verliert
damit höchstens den gerade laufenden Schritt. `--resume <pfad>` liest dieses
Manifest wieder ein, übernimmt Profil/Portzustand/Messweg/`--aba` daraus und
überspringt jeden Schritt mit `status == "completed"`. Ein übersprungener
("skip") Schritt gilt dabei bewusst NICHT als abgeschlossen (`status ==
"skipped"`) und wird bei einem `--resume` erneut angeboten - "übersprungen"
heißt hier "jetzt nicht gemessen", nicht "endgültig verzichtet". Eine
A1/A2-Drift-Prüfung (Abschnitt 15.5) über die Grenze eines Resumes hinweg
(z. B. A1 vor dem Abbruch, A2 erst danach) ist nicht möglich, da die dafür
nötigen SPL-Rohdaten nur im Prozessspeicher zwischengehalten werden, nicht im
Manifest - das wird bewusst ausgelassen statt eine Prüfung ohne
Datengrundlage vorzutäuschen (Projektregel "keine Zahlen erfinden").

**Block-Auswahl (`--block 1..4`, nur Profil `wohnzimmer`):** bildet die vier
Blöcke des Messplans aus raum_akustik.md Abschnitt 16 (Rev. 17) ab. Der Block
legt Messfolge (L/R-Einzelmessung oder A1/B/A2) und Portzustand fest und gibt
beim Start einen Hinweis auf den zugehörigen REW-Sweep-Bereich aus. Der
Sweep-Bereich ist nur ein Hinweis: das Modul liest ihn nicht aus REW zurück
und prüft ihn nicht. Widersprechen explizit gesetzte `--aba`/`--port-state`
dem Block, bricht `main()` mit einer Fehlermeldung ab (statt still zu
überschreiben). Ohne `--block` verhält sich das Modul wie zuvor.

**Setup-Notiz (`--notes`):** freier Text (z. B. KEF-Phasenkorrektur EIN/AUS,
KEF-HP, Saxx-LP, Saxx-Phase, Saxx-Pegelregler-Wert), wird im Manifest
gespeichert und in jeden Notes-Entwurf übernommen. Ohne `--notes` fragt eine
interaktive Session (kein `--dry-run`, kein `--resume`) einmal beim Start
danach, `Enter` überspringt. Bei `--resume` bleibt die gespeicherte Notiz
erhalten, ein neues `--notes` ersetzt sie.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np

from spatial_analyzer.rew_client import DEFAULT_BASE_URL, ReadOnlyRewClient, RewApiError, RewMeasurementMeta

# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionProfile:
    """Eine Mess-Kampagnen-Konfiguration (Positionen, Kanäle, Benennung).

    `channels` wird nur für die Nicht-A-B-A-Einzelmessung verwendet
    (siehe `build_plan`). `supports_aba=False` bedeutet nicht "A-B-A ist
    fachlich sinnlos", sondern nur: für dieses Profil ist die A-B-A-Folge
    (noch) nicht spezifiziert (für `smart453` existiert dafür bisher keine
    Vorgabe in `docs/SMART453_BASS_SPEC.md`).
    """

    name: str
    position_ids: tuple[str, ...]
    channels: tuple[str, ...]
    supports_aba: bool
    aba_prefix: str = "ABA_"


def _wohnzimmer_profile() -> SessionProfile:
    # P01..P11: Sofamitte + 10 Wanderpositionen, Benennungsschema aus
    # raum_akustik.md Abschnitt 13 (Kampagne MIMO2703, "P01..P11").
    position_ids = tuple(f"P{i:02d}" for i in range(1, 12))
    return SessionProfile(
        name="wohnzimmer",
        position_ids=position_ids,
        channels=("L", "R"),
        supports_aba=True,
    )


def _smart453_profile() -> SessionProfile:
    # Kopf-Würfel-Positionen (docs/SMART453_BASS_SPEC.md, "7-9 Kopf-Würfel-
    # Punkte"): hier bewusst generisch als C01..C09 benannt, da die
    # eigentliche Wege-Benennung (front_sub, seat_sub, door_mt_l/r) laut
    # Spec noch nicht im Code existiert (reiner Recherchestand, kein
    # smart453_bass-Package) - dieses Modul referenziert dieses Package
    # bewusst nicht.
    position_ids = tuple(f"C{i:02d}" for i in range(1, 10))
    return SessionProfile(
        name="smart453",
        position_ids=position_ids,
        channels=("MIC",),
        supports_aba=False,
    )


PROFILES: dict[str, SessionProfile] = {
    "wohnzimmer": _wohnzimmer_profile(),
    "smart453": _smart453_profile(),
}


@dataclass(frozen=True)
class BlockSpec:
    """Ein Block des Wohnzimmer-Messplans (raum_akustik.md Abschnitt 16).

    `sweep_hint` ist nur ein Hinweis fuer den manuell gestarteten REW-Sweep
    und wird weder aus REW gelesen noch geprueft.
    """

    number: int
    use_aba: bool
    port_state: str
    sweep_hint: str
    description: str


BLOCKS: dict[int, BlockSpec] = {
    1: BlockSpec(
        1,
        False,
        "stopfen_ein",
        "20 Hz - 20.000 Hz",
        "KEF allein (2.0, Sub-Out aus, Saxx AUS), L und R einzeln, Stopfen drin",
    ),
    2: BlockSpec(
        2,
        True,
        "stopfen_ein",
        "15 Hz - 500 Hz",
        "A1/B/A2: A = KEF mit aktivem Hochpass, Saxx gemutet; B = KEF + Saxx aktiv; Stopfen drin",
    ),
    3: BlockSpec(
        3,
        True,
        "offen",
        "15 Hz - 500 Hz",
        "A1/B/A2 wie Block 2, Stopfen entfernt (nach einmaligem Stopfen-Ziehen), gleiche Kombination wie Block 2",
    ),
    4: BlockSpec(
        4,
        False,
        "offen",
        "20 Hz - 20.000 Hz",
        "KEF allein (2.0, Sub-Out aus, Saxx AUS), L und R einzeln, Stopfen entfernt",
    ),
}


def resolve_block(profile: SessionProfile, block: int) -> BlockSpec:
    """Liefert die Block-Vorgabe oder wirft `ValueError` (unbekannter Block
    oder Profil ohne Block-Plan; Bloecke sind nur fuer `wohnzimmer` definiert)."""
    if profile.name != "wohnzimmer":
        raise ValueError(f"--block ist nur fuer das Profil 'wohnzimmer' definiert (nicht {profile.name!r})")
    if block not in BLOCKS:
        raise ValueError(f"Unbekannter Block {block!r} (erwartet 1-4)")
    return BLOCKS[block]


# ---------------------------------------------------------------------------
# Mess-Plan & Benennung
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionStep:
    """Ein einzelner geplanter Schritt (eine erwartete neue REW-Messung)."""

    position_id: str
    channel: str | None
    aba_state: str | None  # "A1" | "B" | "A2" | None


def build_measurement_name(
    profile: SessionProfile,
    position_id: str,
    channel: str | None = None,
    aba_state: str | None = None,
) -> str:
    """Benennungsschema: `P01..P11`/`C01..C09` als Basis, `_L`/`_R`-Suffix
    für Einzelkanal-Messungen, `ABA_`-Präfix für die A-B-A-Folge (siehe
    raum_akustik.md Abschnitt 10 Punkt 19: 'Dateinamen getrennt halten')."""
    if aba_state is not None:
        if not profile.supports_aba:
            raise ValueError(f"Profil {profile.name!r} unterstuetzt keine A-B-A-Folge")
        if aba_state not in ("A1", "B", "A2"):
            raise ValueError(f"Unbekannter A-B-A-Zustand {aba_state!r} (erwartet A1/B/A2)")
        return f"{profile.aba_prefix}{aba_state}_{position_id}"
    if channel is not None:
        if channel not in profile.channels:
            raise ValueError(f"Kanal {channel!r} nicht in Profil {profile.name!r} vorgesehen")
        return f"{position_id}_{channel}"
    return position_id


def build_plan(profile: SessionProfile, use_aba: bool) -> list[SessionStep]:
    """Erzeugt die geordnete Liste erwarteter Schritte für eine volle
    Kampagne über alle Positionen des Profils."""
    if use_aba and not profile.supports_aba:
        raise ValueError(f"Profil {profile.name!r} unterstuetzt keine A-B-A-Folge")

    steps: list[SessionStep] = []
    for position_id in profile.position_ids:
        if use_aba:
            for state in ("A1", "B", "A2"):
                steps.append(SessionStep(position_id, None, state))
        else:
            for channel in profile.channels:
                steps.append(SessionStep(position_id, channel, None))
    return steps


# ---------------------------------------------------------------------------
# Benutzer-Interaktion
# ---------------------------------------------------------------------------


class UserAction(str, Enum):
    CONFIRM = "confirm"
    REPEAT = "repeat"
    SKIP = "skip"
    QUIT = "quit"


_ACTION_KEYS: dict[str, UserAction] = {
    "": UserAction.CONFIRM,
    "r": UserAction.REPEAT,
    "s": UserAction.SKIP,
    "q": UserAction.QUIT,
}

PROMPT_TEXT = "[Enter]=bestaetigen  [r]=wiederholen  [s]=ueberspringen  [q]=speichern&beenden > "


def parse_user_action(raw: str) -> UserAction | None:
    return _ACTION_KEYS.get(raw.strip().lower())


def prompt_user_action(input_fn: Callable[[str], str] = input, print_fn: Callable[[str], None] = print) -> UserAction:
    while True:
        raw = input_fn(PROMPT_TEXT)
        action = parse_user_action(raw)
        if action is not None:
            return action
        print_fn(f"Unbekannte Eingabe {raw!r} - bitte [Enter], r, s oder q verwenden.")


# ---------------------------------------------------------------------------
# ID-Erkennung (genau eine neue Messung)
# ---------------------------------------------------------------------------


@dataclass
class IdDetectionResult:
    new_measurements: list[RewMeasurementMeta]
    status: str  # "ok" | "none_found" | "multiple_found"


def detect_new_measurement(
    known_uuids: set[str], current: list[RewMeasurementMeta]
) -> IdDetectionResult:
    """Ordnet die genau eine neue Messung zu. Liefert eine Warnung (Status),
    falls 0 oder > 1 neue Messungen seit `known_uuids` erschienen sind -
    siehe Bauplan-Vorgabe und Risiko in rew_automatisierung_befunde.md
    Abschnitt 1b (Kanaltrennung nicht garantiert dokumentiert)."""
    new = [m for m in current if m.uuid not in known_uuids]
    if len(new) == 0:
        return IdDetectionResult(new, "none_found")
    if len(new) > 1:
        return IdDetectionResult(new, "multiple_found")
    return IdDetectionResult(new, "ok")


# ---------------------------------------------------------------------------
# Sofort-QC
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QcThresholds:
    """Arbeitshypothesen für die Sofort-QC (keine Messwerte, siehe Modul-
    Docstring). `aba_band_hz` folgt raum_akustik.md Abschnitt 15.5
    ('Drift-Check... Band 20-120 Hz')."""

    clip_margin_dbfs: float = 1.0
    min_pretrigger_snr_db: float = 20.0
    aba_drift_max_db: float = 1.0
    aba_band_hz: tuple[float, float] = (20.0, 120.0)
    pretrigger_fraction: float = 0.1  # Anteil der IR-Länge, der als Pre-Trigger gilt


@dataclass
class QcFinding:
    severity: str  # "ok" | "warning"
    check: str
    message: str

    def __str__(self) -> str:
        label = "OK" if self.severity == "ok" else "WARNUNG"
        return f"[{label}] {self.check}: {self.message}"


def check_clipping(ir_samples: np.ndarray, thresholds: QcThresholds) -> QcFinding:
    """Peak-Pruefung. Setzt voraus, dass die IR-Samples auf +-1.0 normiert
    sind ("dBFS"); diese Annahme ist laut raum_akustik.md Abschnitt 15.6
    (offener Punkt 21) fuer die echte REW-Export-Kette NICHT verifiziert -
    bis zur Verifikation ist das Ergebnis ein relativer Arbeitswert, keine
    belastbare Pegelangabe."""
    samples = np.asarray(ir_samples, dtype=np.float64)
    if samples.size == 0:
        return QcFinding("warning", "clipping", "Impulsantwort ist leer - Pruefung nicht moeglich")
    peak = float(np.max(np.abs(samples)))
    if peak <= 0.0:
        return QcFinding("warning", "clipping", "Peak-Amplitude ist 0 - Signal fehlt vermutlich")
    peak_dbfs = 20.0 * np.log10(peak)
    if peak_dbfs >= -thresholds.clip_margin_dbfs:
        return QcFinding(
            "warning",
            "clipping",
            f"Peak bei {peak_dbfs:.1f} dBFS, naeher als {thresholds.clip_margin_dbfs} dB an "
            "Vollausschlag - Wiederholung mit reduziertem Pegel empfohlen",
        )
    return QcFinding("ok", "clipping", f"Peak bei {peak_dbfs:.1f} dBFS, kein Clipping-Verdacht")


def check_pretrigger_noise(ir_samples: np.ndarray, thresholds: QcThresholds) -> QcFinding:
    """SNR zwischen Pre-Trigger-Segment (Rauschboden) und Peak (Messregel 5:
    'Stoergeraeusche beachten, verrauschte Messungen wiederholen')."""
    samples = np.asarray(ir_samples, dtype=np.float64)
    pretrigger_n = int(samples.size * thresholds.pretrigger_fraction)
    if samples.size < 4 or pretrigger_n < 2:
        return QcFinding("warning", "noise_floor", "zu wenige Samples fuer Pre-Trigger-Rauschanalyse")
    pre = samples[:pretrigger_n]
    peak = float(np.max(np.abs(samples)))
    noise_rms = float(np.sqrt(np.mean(pre**2)))
    if peak <= 0.0 or noise_rms <= 0.0:
        return QcFinding("ok", "noise_floor", "Rauschboden nicht messbar (Null-Signal) - keine Warnung ausgeloest")
    snr_db = 20.0 * np.log10(peak / noise_rms)
    if snr_db < thresholds.min_pretrigger_snr_db:
        return QcFinding(
            "warning",
            "noise_floor",
            f"SNR {snr_db:.1f} dB unter Schwelle {thresholds.min_pretrigger_snr_db} dB - "
            "Stoergeraeusche wahrscheinlich, Wiederholung empfohlen",
        )
    return QcFinding("ok", "noise_floor", f"SNR {snr_db:.1f} dB, kein Rausch-Verdacht")


def check_aba_drift(
    a1_spl_db: np.ndarray,
    a2_spl_db: np.ndarray,
    freqs_hz: np.ndarray,
    thresholds: QcThresholds,
) -> QcFinding:
    """|A2 - A1| im Band `thresholds.aba_band_hz` (raum_akustik.md Abschnitt
    15.5, 'Drift-Check'). Erwartet identisches Frequenzraster fuer A1/A2 -
    eine Interpolation auf ein gemeinsames Raster ist Aufgabe von `grid.py`,
    nicht dieser QC-Pruefung."""
    a1 = np.asarray(a1_spl_db, dtype=np.float64)
    a2 = np.asarray(a2_spl_db, dtype=np.float64)
    freqs = np.asarray(freqs_hz, dtype=np.float64)
    if a1.shape != freqs.shape or a2.shape != freqs.shape:
        return QcFinding("warning", "aba_drift", "A1/A2/freqs haben unterschiedliche Laenge - Pruefung nicht moeglich")

    band_lo, band_hi = thresholds.aba_band_hz
    mask = (freqs >= band_lo) & (freqs <= band_hi)
    if not np.any(mask):
        return QcFinding("warning", "aba_drift", f"keine Frequenzpunkte im Band {thresholds.aba_band_hz} Hz")

    drift = np.abs(a2[mask] - a1[mask])
    if not np.any(np.isfinite(drift)):
        return QcFinding("warning", "aba_drift", "Drift nicht berechenbar (nur NaN/Inf im Band)")
    max_drift = float(np.nanmax(drift))

    if max_drift > thresholds.aba_drift_max_db:
        return QcFinding(
            "warning",
            "aba_drift",
            f"max |A2-A1| = {max_drift:.2f} dB im Band {thresholds.aba_band_hz} Hz, ueber "
            f"Schwelle {thresholds.aba_drift_max_db} dB - A1/A2 gelten laut raum_akustik.md "
            "15.5 als nicht hinreichend stabil, komplette A1/B/A2-Dreiergruppe wiederholen",
        )
    return QcFinding("ok", "aba_drift", f"max |A2-A1| = {max_drift:.2f} dB im Band, innerhalb Toleranz")


# ---------------------------------------------------------------------------
# Optionaler KEF-DSP-Snapshot (aiokef) / Smart-453-Preset-Abfrage
# ---------------------------------------------------------------------------


def _default_kef_client_factory(host: str) -> Any:
    """Baut den aiokef-Client. **Unverifiziert** (siehe
    rew_automatisierung_befunde.md Abschnitt 3): Klassenname und
    Konstruktorsignatur von `aiokef` sind aus der Foren-/Home-Assistant-
    Dokumentation uebernommen, nie gegen die echte Bibliothek oder ein
    echtes Geraet geprueft - vor dem ersten realen Einsatz pruefen."""
    import aiokef  # type: ignore[import-not-found]

    return aiokef.AsyncKefSpeaker(host)


def read_kef_snapshot(
    host: str,
    client_factory: Callable[[str], Any] | None = None,
    print_fn: Callable[[str], None] = print,
) -> dict[str, Any] | None:
    """Liest den aktuellen KEF-LS50-Wireless-Gen1-DSP-Status ueber `aiokef`
    (siehe rew_automatisierung_befunde.md Abschnitt 3: Mode-namedtuple mit
    desk_mode/wall_mode/phase_correction/high_pass/sub_polarity/
    bass_extension). `client_factory` ist fuer Tests injizierbar (Fake-
    Objekt statt echtem aiokef/echtem Geraet). Jeder Fehler (fehlendes
    Paket, Netzwerk, Timeout) fuehrt zu `None` plus Konsolenhinweis - nie
    zu einem Abbruch der Session (siehe Projektregel: Fallback auf manuelle
    Abfrage statt Absturz)."""
    factory = client_factory or _default_kef_client_factory
    try:
        import asyncio

        async def _read() -> dict[str, Any]:
            speaker = factory(host)
            mode = await speaker.get_mode()
            return {
                "desk_mode": getattr(mode, "desk_mode", None),
                "wall_mode": getattr(mode, "wall_mode", None),
                "phase_correction": getattr(mode, "phase_correction", None),
                "high_pass": getattr(mode, "high_pass", None),
                "sub_polarity": getattr(mode, "sub_polarity", None),
                "bass_extension": getattr(mode, "bass_extension", None),
            }

        return asyncio.run(_read())
    except ImportError:
        print_fn(
            "Hinweis: 'aiokef' ist nicht installiert - KEF-Snapshot wird ausgelassen "
            "(bitte manuell nach docs/REW_NOTES_TEMPLATE.md eintragen)."
        )
        return None
    except Exception as exc:  # noqa: BLE001 - jeder Fehler ist hier ein Offline-Fallback, kein Absturz
        print_fn(f"Hinweis: KEF-Snapshot ueber {host!r} fehlgeschlagen ({exc}) - wird ausgelassen.")
        return None


def prompt_smart453_preset(input_fn: Callable[[str], str] = input) -> str:
    """Keine automatisierte Mosconi-Preset-Abfrage vorhanden (kein
    dokumentiertes Protokoll, siehe docs/SMART453_BASS_SPEC.md) - manuelle
    Eingabe, analog zum 'Auto'-Block in docs/REW_NOTES_TEMPLATE.md."""
    return input_fn("Aktueller Mosconi-Preset-Name (manuell, kein automatisches Auslesen) > ").strip()


def prompt_setup_notes(input_fn: Callable[[str], str] = input) -> str | None:
    """Fragt einmal beim Start nach einer freien Setup-Notiz (KEF-Phasenkorrektur,
    KEF-HP, Saxx-LP, Saxx-Phase, Saxx-Pegelregler). Leere Eingabe = keine Notiz."""
    answer = input_fn(
        "Setup-Notiz (KEF-Phasenkorrektur, KEF-HP, Saxx-LP, Saxx-Phase, Saxx-Pegelregler; Enter = ueberspringen): "
    ).strip()
    return answer or None


# ---------------------------------------------------------------------------
# Notes-Text (nur lokale Datei, kein REW-API-Write) & Manifest
# ---------------------------------------------------------------------------


def build_notes_text(
    position_id: str,
    channel: str | None,
    aba_state: str | None,
    kef_snapshot: dict[str, Any] | None,
    smart453_preset: str | None,
    port_state: str | None,
    signal_path: str | None,
    block: int | None = None,
    setup_notes: str | None = None,
) -> str:
    """Text-Entwurf im Stil von `docs/REW_NOTES_TEMPLATE.md` zum manuellen
    Einfuegen ins REW-Notes-Feld. Dieses Modul schreibt NICHT automatisch in
    REW (GET-only, siehe Modul-Docstring) - das Einfuegen bleibt wie bisher
    ein manueller Schritt."""
    lines = [f"Position: {position_id}"]
    if block is not None:
        lines.append(f"Block: {block}")
    if channel:
        lines.append(f"Kanal: {channel}")
    if aba_state:
        lines.append(f"A-B-A-Zustand: {aba_state}")
    if port_state:
        lines.append(f"KEF Bassreflex-Port: {port_state}")
    if signal_path:
        lines.append(f"Messweg: {signal_path}")
    if kef_snapshot:
        for key, value in kef_snapshot.items():
            lines.append(f"KEF {key}: {value}")
    elif channel or aba_state:
        lines.append("KEF-Snapshot: nicht verfuegbar - manuell nach REW_NOTES_TEMPLATE.md eintragen")
    if smart453_preset:
        lines.append(f"Mosconi-Preset: {smart453_preset}")
    if setup_notes:
        lines.append(f"Setup-Notiz: {setup_notes}")
    return "\n".join(lines)


def write_notes_sidecar(notes_dir: str | Path, measurement_name: str, text: str) -> Path:
    directory = Path(notes_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{measurement_name}.txt"
    path.write_text(text, encoding="utf-8")
    return path


@dataclass
class SessionManifestEntry:
    measurement_name: str
    position_id: str
    channel: str | None
    aba_state: str | None
    uuid: str | None
    detection_status: str | None
    qc_findings: list[dict[str, str]]
    kef_snapshot: dict[str, Any] | None
    smart453_preset: str | None
    notes_text: str
    timestamp: str
    # "completed" = bestaetigter, ausgewerteter Schritt; "skipped" = bewusst
    # ausgelassen und bei --resume erneut anzubieten (siehe Modul-Docstring,
    # Abschnitt "Zwischenspeicher & Fortsetzung"). Default haelt bestehende
    # Aufrufstellen ohne `status=` rueckwaertskompatibel.
    status: str = "completed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SessionManifest:
    profile: str
    dry_run: bool
    created_at: str
    entries: list[SessionManifestEntry] = field(default_factory=list)
    # Fuer --resume: Modus/Bedingungen der Session, damit eine Fortsetzung
    # sie nicht erneut abfragen oder erraten muss.
    use_aba: bool = False
    port_state: str | None = None
    signal_path: str | None = None
    # Messplan-Block (1-4, nur Profil wohnzimmer) und freie Setup-Notiz
    # (KEF-/Saxx-Einstellungen), beides optional und fuer --resume erhalten.
    block: int | None = None
    setup_notes: str | None = None

    def add(self, entry: SessionManifestEntry) -> None:
        self.entries.append(entry)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "dry_run": self.dry_run,
            "created_at": self.created_at,
            "use_aba": self.use_aba,
            "port_state": self.port_state,
            "signal_path": self.signal_path,
            "block": self.block,
            "setup_notes": self.setup_notes,
            "entries": [e.to_dict() for e in self.entries],
        }

    def write(self, path: str | Path) -> None:
        """Schreibt das Manifest atomar: zuerst in eine Temp-Datei im selben
        Verzeichnis, dann per `Path.replace` (= `os.replace`, atomarer
        Rename) an den Zielpfad - ein Absturz mitten im Schreiben hinterlaesst
        damit nie eine halb geschriebene/korrupte `session_manifest.json`."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_name(target.name + ".tmp")
        tmp_path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(target)


def load_manifest(path: str | Path) -> SessionManifest:
    """Liest ein zuvor geschriebenes `session_manifest.json` fuer `--resume`
    wieder ein. Fehlende Felder (z. B. `status` in einem vor dieser Funktion
    geschriebenen Manifest) werden mit einem sinnvollen Default aufgefuellt:
    ein Eintrag ohne `status`-Feld stammt aus einer Version, die nur
    bestaetigte Schritte protokolliert hat, gilt also als "completed"."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = [
        SessionManifestEntry(
            measurement_name=e["measurement_name"],
            position_id=e["position_id"],
            channel=e.get("channel"),
            aba_state=e.get("aba_state"),
            uuid=e.get("uuid"),
            detection_status=e.get("detection_status"),
            qc_findings=e.get("qc_findings", []),
            kef_snapshot=e.get("kef_snapshot"),
            smart453_preset=e.get("smart453_preset"),
            notes_text=e.get("notes_text", ""),
            timestamp=e.get("timestamp", ""),
            status=e.get("status", "completed"),
        )
        for e in data.get("entries", [])
    ]
    return SessionManifest(
        profile=data["profile"],
        dry_run=data.get("dry_run", False),
        created_at=data.get("created_at", _now_iso()),
        entries=entries,
        use_aba=data.get("use_aba", False),
        port_state=data.get("port_state"),
        signal_path=data.get("signal_path"),
        block=data.get("block"),
        setup_notes=data.get("setup_notes"),
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Hauptablauf
# ---------------------------------------------------------------------------


def _detect_and_qc(
    *,
    client: ReadOnlyRewClient,
    known_uuids: set[str],
    qc_thresholds: QcThresholds,
    dry_run: bool,
    print_fn: Callable[[str], None],
) -> tuple[IdDetectionResult, list[QcFinding], dict[str, Any], set[str]]:
    """Fuehrt ID-Erkennung und Sofort-QC fuer EINE bestaetigte Messung aus.

    Liefert zusaetzlich das aktualisierte `known_uuids`-Set zurueck (immer
    alle aktuell in REW sichtbaren UUIDs, nicht nur die zugeordnete eine) -
    das macht wiederholte Sweeps robust: eine verworfene Wiederholung taucht
    beim naechsten Diff nicht faelschlich als zweite "neue" Messung auf.
    """
    if dry_run:
        print_fn("(--dry-run: keine echte REW-Abfrage)")
        return IdDetectionResult([], "none_found"), [], {}, known_uuids

    try:
        current = client.list_measurements()
    except RewApiError as exc:
        print_fn(f"WARNUNG: REW-API nicht erreichbar ({exc}) - ID-Erkennung ausgelassen.")
        return IdDetectionResult([], "none_found"), [], {}, known_uuids

    id_result = detect_new_measurement(known_uuids, current)
    updated_known = {m.uuid for m in current}

    if id_result.status == "none_found":
        print_fn("WARNUNG: keine neue Messung in REW erkannt.")
        return id_result, [], {}, updated_known
    if id_result.status == "multiple_found":
        print_fn(
            f"WARNUNG: {len(id_result.new_measurements)} neue Messungen erkannt - "
            "bitte manuell zuordnen, automatische QC wird ausgelassen."
        )
        return id_result, [], {}, updated_known

    measurement = id_result.new_measurements[0]
    qc_findings: list[QcFinding] = []
    qc_data: dict[str, Any] = {}

    try:
        freq_response = client.get_frequency_response(measurement.uuid)
        qc_data["spl_db"] = np.asarray(freq_response.get("spl", []), dtype=np.float64)
        qc_data["freq_hz"] = np.asarray(freq_response.get("freq", []), dtype=np.float64)
    except RewApiError as exc:
        print_fn(f"Hinweis: Frequenzgang nicht abrufbar ({exc}) - SPL-QC ausgelassen.")

    try:
        ir = client.get_impulse_response_raw(measurement.uuid)
        qc_data["ir_samples"] = ir
        qc_findings.append(check_clipping(ir, qc_thresholds))
        qc_findings.append(check_pretrigger_noise(ir, qc_thresholds))
    except RewApiError as exc:
        print_fn(f"Hinweis: Impulsantwort nicht abrufbar ({exc}) - IR-QC ausgelassen.")

    for finding in qc_findings:
        print_fn(str(finding))

    return id_result, qc_findings, qc_data, updated_known


def run_guided_session(
    profile: SessionProfile,
    *,
    use_aba: bool,
    client: ReadOnlyRewClient,
    manifest_path: str | Path,
    notes_dir: str | Path,
    kef_host: str | None,
    kef_client_factory: Callable[[str], Any] | None = None,
    qc_thresholds: QcThresholds | None = None,
    dry_run: bool,
    port_state: str | None = "stopfen_ein",
    signal_path: str | None = "USB",
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
    resume_manifest: SessionManifest | None = None,
    block: int | None = None,
    setup_notes: str | None = None,
    ask_setup_notes: bool = False,
) -> SessionManifest:
    """Führt eine komplette geführte Session über alle Positionen des
    Profils aus (oder bis `[q]` gedrückt wird) und schreibt dabei laufend
    das Manifest, damit bei einem Abbruch nichts verloren geht.

    `resume_manifest` (optional, siehe `load_manifest`/`--resume`): ein
    zuvor geschriebenes Manifest, dessen bereits mit `status == "completed"`
    markierte Schritte übersprungen werden; alle anderen Schritte (auch
    zuvor übersprungene) laufen wie gewohnt. Die zurückgegebene
    `SessionManifest`-Instanz ist in diesem Fall dieselbe (erweiterte)
    Instanz wie `resume_manifest`, nicht eine neue.

    `block` (1-4, nur `wohnzimmer`, siehe `BLOCKS`): muss zu `use_aba` und
    `port_state` passen (sonst `ValueError`) und gibt einen Sweep-Hinweis aus.
    `setup_notes`: freie Notiz fuer Manifest und Notes-Entwuerfe; bei einem
    Resume ersetzt ein uebergebener Wert die gespeicherte Notiz.
    `ask_setup_notes`: fragt (nur wenn `setup_notes` fehlt und kein Resume
    vorliegt) einmal per `input_fn` nach einer Notiz."""
    qc_thresholds = qc_thresholds or QcThresholds()
    block_spec: BlockSpec | None = None
    if block is not None:
        block_spec = resolve_block(profile, block)
        if use_aba != block_spec.use_aba or port_state != block_spec.port_state:
            raise ValueError(
                f"Block {block} verlangt use_aba={block_spec.use_aba} und port_state={block_spec.port_state!r} "
                f"(erhalten: use_aba={use_aba}, port_state={port_state!r})"
            )
    plan = build_plan(profile, use_aba)

    if resume_manifest is not None:
        manifest = resume_manifest
        manifest.dry_run = dry_run
        manifest.use_aba = use_aba
        manifest.port_state = port_state
        manifest.signal_path = signal_path
        manifest.block = block
        if setup_notes is not None:
            manifest.setup_notes = setup_notes
        completed_keys = {
            (e.position_id, e.channel, e.aba_state) for e in manifest.entries if e.status == "completed"
        }
        remaining = [s for s in plan if (s.position_id, s.channel, s.aba_state) not in completed_keys]
        print_fn(f"Fortsetzung: {len(completed_keys)} von {len(plan)} Schritten bereits abgeschlossen.")
        if remaining:
            first = remaining[0]
            first_name = build_measurement_name(profile, first.position_id, first.channel, first.aba_state)
            print_fn(f"Fortsetzung bei: {first_name}")
        else:
            print_fn("Alle Schritte dieser Kampagne sind bereits abgeschlossen.")
    else:
        manifest = SessionManifest(
            profile=profile.name,
            dry_run=dry_run,
            created_at=_now_iso(),
            use_aba=use_aba,
            port_state=port_state,
            signal_path=signal_path,
            block=block,
            setup_notes=setup_notes,
        )
        if manifest.setup_notes is None and ask_setup_notes:
            manifest.setup_notes = prompt_setup_notes(input_fn)
        completed_keys = set()

    if block_spec is not None:
        print_fn(f"Block {block_spec.number}: {block_spec.description}")
        print_fn(f"REW-Sweep-Hinweis: {block_spec.sweep_hint} (nur Hinweis, wird nicht geprueft)")
        print_fn(f"Portzustand: {block_spec.port_state}")
    if manifest.setup_notes:
        print_fn(f"Setup-Notiz: {manifest.setup_notes}")

    # Bereits aus dem Manifest bekannte UUIDs zusaetzlich zur frischen
    # REW-Abfrage beruecksichtigen (robust, falls eine fruehere Messung aus
    # der REW-Liste verschwunden sein sollte - sie bleibt dann trotzdem als
    # "bekannt" markiert und wird nicht faelschlich erneut als "neu" erkannt).
    known_uuids: set[str] = {e.uuid for e in manifest.entries if e.uuid}
    if not dry_run:
        try:
            known_uuids |= {m.uuid for m in client.list_measurements()}
        except RewApiError as exc:
            print_fn(f"WARNUNG: REW-API nicht erreichbar ({exc}) - IDs werden nicht automatisch erkannt.")

    # Zwischenspeicher fuer die A1/A2-Drift-Pruefung je Position. Nach einem
    # --resume mitten in einer A-B-A-Dreiergruppe ist dieser Cache leer (die
    # SPL-Rohdaten eines vor dem Abbruch bestaetigten A1 liegen nicht im
    # Manifest, nur sein Protokoll) - eine A2-Drift-Pruefung wird dann bewusst
    # ausgelassen statt eine Pruefung ohne Datengrundlage vorzutaeuschen.
    aba_a1_cache: dict[str, dict[str, np.ndarray]] = {}

    for step in plan:
        key = (step.position_id, step.channel, step.aba_state)
        if key in completed_keys:
            continue

        name = build_measurement_name(profile, step.position_id, step.channel, step.aba_state)
        print_fn(f"\n=== {name} ===")

        action = UserAction.REPEAT
        while action == UserAction.REPEAT:
            action = UserAction.CONFIRM if dry_run else prompt_user_action(input_fn, print_fn)
            if dry_run:
                print_fn("(--dry-run: Bestaetigung wird simuliert)")

            if action == UserAction.QUIT:
                manifest.write(manifest_path)
                print_fn(f"Session abgebrochen, Manifest gespeichert nach: {manifest_path}")
                return manifest

            if action == UserAction.SKIP:
                # Auch ein uebersprungener Schritt wird sofort protokolliert
                # (Crash-/Abbruchsicherheit, siehe Modul-Docstring); er gilt
                # bei --resume nicht als abgeschlossen und wird erneut angeboten.
                manifest.add(
                    SessionManifestEntry(
                        measurement_name=name,
                        position_id=step.position_id,
                        channel=step.channel,
                        aba_state=step.aba_state,
                        uuid=None,
                        detection_status=None,
                        qc_findings=[],
                        kef_snapshot=None,
                        smart453_preset=None,
                        notes_text="",
                        timestamp=_now_iso(),
                        status="skipped",
                    )
                )
                manifest.write(manifest_path)
                break

            if action == UserAction.REPEAT:
                if not dry_run:
                    try:
                        known_uuids = {m.uuid for m in client.list_measurements()}
                    except RewApiError as exc:
                        print_fn(f"WARNUNG: REW-Messungsliste nach Wiederholung nicht abrufbar ({exc}); UUID-Ausgangsstand bleibt unveraendert.")
                continue

            # action == CONFIRM
            id_result, qc_findings, qc_data, known_uuids = _detect_and_qc(
                client=client,
                known_uuids=known_uuids,
                qc_thresholds=qc_thresholds,
                dry_run=dry_run,
                print_fn=print_fn,
            )

            if step.aba_state == "A1" and "spl_db" in qc_data:
                aba_a1_cache[step.position_id] = qc_data
            elif step.aba_state == "A2" and step.position_id in aba_a1_cache and "spl_db" in qc_data:
                a1_data = aba_a1_cache[step.position_id]
                drift_finding = check_aba_drift(
                    a1_data["spl_db"], qc_data["spl_db"], qc_data["freq_hz"], qc_thresholds
                )
                print_fn(str(drift_finding))
                qc_findings.append(drift_finding)

            kef_snapshot = None
            smart453_preset = None
            if profile.name == "wohnzimmer" and kef_host and not dry_run:
                kef_snapshot = read_kef_snapshot(kef_host, kef_client_factory, print_fn)
            if profile.name == "smart453" and not dry_run:
                smart453_preset = prompt_smart453_preset(input_fn)

            notes_text = build_notes_text(
                step.position_id,
                step.channel,
                step.aba_state,
                kef_snapshot,
                smart453_preset,
                port_state,
                signal_path,
                block,
                manifest.setup_notes,
            )
            write_notes_sidecar(notes_dir, name, notes_text)

            uuid = id_result.new_measurements[0].uuid if id_result.status == "ok" else None
            manifest.add(
                SessionManifestEntry(
                    measurement_name=name,
                    position_id=step.position_id,
                    channel=step.channel,
                    aba_state=step.aba_state,
                    uuid=uuid,
                    detection_status=id_result.status,
                    qc_findings=[{"severity": f.severity, "check": f.check, "message": f.message} for f in qc_findings],
                    kef_snapshot=kef_snapshot,
                    smart453_preset=smart453_preset,
                    notes_text=notes_text,
                    timestamp=_now_iso(),
                    status="completed",
                )
            )
            manifest.write(manifest_path)

    manifest.write(manifest_path)
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spatial_analyzer.guided_session",
        description=(
            "Geführte Mess-Session: begleitet einen manuellen REW-Sweep (ID-Erkennung, "
            "Benennung, Sofort-QC, lokales Manifest). Löst keinen Sweep aus und schreibt "
            "nicht in REW zurück (GET-only, siehe docs/SPECIFICATION.md Abschnitt 2)."
        ),
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), default="wohnzimmer")
    parser.add_argument(
        "--aba",
        action="store_true",
        help="A-B-A-Folge mit Saxx-Mute statt L/R-Einzelmessung (nur Profil 'wohnzimmer')",
    )
    parser.add_argument(
        "--block",
        type=int,
        choices=sorted(BLOCKS),
        default=None,
        help=(
            "Messplan-Block 1-4 (nur Profil 'wohnzimmer', raum_akustik.md Abschnitt 16): legt Messfolge und "
            "Portzustand fest (1: L/R, Stopfen ein; 2: A1/B/A2, Stopfen ein; 3: A1/B/A2, offen; 4: L/R, offen) "
            "und gibt den REW-Sweep-Hinweis aus. Widerspruechliche --aba/--port-state brechen ab."
        ),
    )
    parser.add_argument(
        "--notes",
        default=None,
        help=(
            "Freie Setup-Notiz (KEF-Phasenkorrektur, KEF-HP, Saxx-LP, Saxx-Phase, Saxx-Pegelregler), wird im "
            "Manifest gespeichert und in die Notes-Entwuerfe uebernommen. Ohne --notes fragt eine interaktive "
            "Session beim Start einmal nach (Enter = ueberspringen)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simuliert den Ablauf ohne echte REW-/KEF-Abfragen (fuer Tests/Vorfuehrung)",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="REW-API-Basis-URL")
    parser.add_argument(
        "--kef-host", default=None, help="Hostname/IP der KEF LS50 Wireless fuer den optionalen DSP-Snapshot"
    )
    parser.add_argument(
        "--port-state",
        default="stopfen_ein",
        choices=["stopfen_ein", "teilstopfen", "offen"],
        help="KEF-Bassreflex-Portzustand fuer das Notes-Protokoll (Standard: stopfen_ein, siehe raum_akustik.md Abschnitt 3)",
    )
    parser.add_argument(
        "--signal-path",
        default="USB",
        choices=["USB", "Bluetooth"],
        help="Messweg fuer das Notes-Protokoll (Standard: USB, siehe raum_akustik.md Abschnitt 15.1)",
    )
    parser.add_argument("--manifest", default="session_manifest.json", help="Pfad fuer das Session-Manifest (JSON)")
    parser.add_argument(
        "--resume",
        default=None,
        metavar="MANIFEST_PFAD",
        help=(
            "Setzt eine unterbrochene Session fort: liest das angegebene session_manifest.json, "
            "uebernimmt Profil, --aba, --port-state und --signal-path daraus (die entsprechenden "
            "CLI-Flags werden ignoriert), ueberspringt Schritte mit status 'completed' und schreibt "
            "weiter in dieselbe Datei (--manifest wird ignoriert)"
        ),
    )
    parser.add_argument(
        "--notes-dir",
        default="session_notes",
        help="Verzeichnis fuer die erzeugten Notes-Textbloecke (.txt, manuell in REW einzufuegen)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    resume_manifest: SessionManifest | None = None
    if args.resume:
        try:
            resume_manifest = load_manifest(args.resume)
        except (OSError, ValueError, KeyError) as exc:
            parser.error(f"--resume: Manifest {args.resume!r} nicht lesbar ({exc})")
        if resume_manifest.profile not in PROFILES:
            parser.error(f"--resume: unbekanntes Profil {resume_manifest.profile!r} im Manifest")
        profile_name = resume_manifest.profile
        use_aba = resume_manifest.use_aba
        port_state = resume_manifest.port_state or args.port_state
        signal_path = resume_manifest.signal_path or args.signal_path
        manifest_path = args.resume
        block = resume_manifest.block
    else:
        profile_name = args.profile
        use_aba = args.aba
        port_state = args.port_state
        signal_path = args.signal_path
        manifest_path = args.manifest
        block = args.block
        if block is not None:
            if profile_name != "wohnzimmer":
                parser.error("--block ist nur fuer --profile wohnzimmer definiert")
            spec = BLOCKS[block]
            tokens = list(sys.argv[1:] if argv is None else argv)
            port_explicit = any(t == "--port-state" or t.startswith("--port-state=") for t in tokens)
            if args.aba and not spec.use_aba:
                parser.error(f"--aba widerspricht --block {block} (Block {block} ist eine L/R-Einzelmessung)")
            if port_explicit and args.port_state != spec.port_state:
                parser.error(
                    f"--port-state {args.port_state} widerspricht --block {block} (verlangt {spec.port_state})"
                )
            use_aba = spec.use_aba
            port_state = spec.port_state

    if use_aba and profile_name != "wohnzimmer":
        parser.error("--aba ist derzeit nur fuer --profile wohnzimmer spezifiziert")

    profile = PROFILES[profile_name]
    client = ReadOnlyRewClient(base_url=args.base_url)

    manifest = run_guided_session(
        profile,
        use_aba=use_aba,
        client=client,
        manifest_path=manifest_path,
        notes_dir=args.notes_dir,
        kef_host=args.kef_host,
        qc_thresholds=QcThresholds(),
        dry_run=args.dry_run,
        port_state=port_state,
        signal_path=signal_path,
        resume_manifest=resume_manifest,
        block=block,
        setup_notes=args.notes,
        ask_setup_notes=args.notes is None and resume_manifest is None and not args.dry_run,
    )
    print(f"Manifest geschrieben nach: {manifest_path} ({len(manifest.entries)} Eintraege)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
