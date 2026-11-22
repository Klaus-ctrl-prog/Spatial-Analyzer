# Forschungsstudie: Adaptive Messpunktplanung & komplexe Schallfeldrekonstruktion mittels Complex GP und EIVR

> **Status:** Rein mathematische und synthetische Offline-Forschungsstudie.
> **Keine** Aenderung an `spatial_analyzer/*.py`, **keine** Integration in den
> laufenden Analyzer, **keine** Aenderung an `docs/SMART453_BASS_SPEC.md`. Die
> 342 bestehenden Tests auf `main` sind von diesem Dokument nicht betroffen.
> Alle numerischen Ergebnisse stammen aus einem temporaeren, deterministischen
> Python/NumPy-Skript, das nach dem Lauf geloescht wurde (siehe Abschnitt 4.1,
> "Reproduzierbarkeit"). Es ist kein reales Messergebnis und keine Validierung
> am tatsaechlichen Hoerraum - alle Zahlen in diesem Dokument sind Modell-
> bzw. Simulationsergebnisse.

## 0. Zusammenfassung

Expected Integrated Variance Reduction (EIVR) ist fuer einen komplexen,
zirkulaer-symmetrischen Gauss-Prozess (GP) mit reellem, isotropem Kernel
analytisch in geschlossener Form berechenbar; die Formel ist identisch zur
reellen GP-Variante, mit dem Betragsquadrat der komplexen Posterior-
Kovarianz anstelle des reellen Quadrats. Die synthetische Gegenprobe (fuenf
Szenarien, vier Vergleichsmethoden plus IDW) zeigt:

- Der **groesste** Genauigkeitsgewinn kommt aus dem **GP-Rekonstruktionsmodell
  selbst** (physikalisch motivierter Bessel-Kernel) gegenueber Complex IDW
  (Faktor 10-40 in dB-Fehler), **nicht** aus der Wahl der Punktauswahl-
  Strategie.
- EIVR, maximale punktweise Varianz und einfaches Uniform-/Farthest-Point-
  Sampling liegen in den getesteten kleinen, symmetrischen 2D-Szenarien
  dicht beieinander; Random Sampling ist in allen vier Szenarien messbar
  schlechter.
- Das erwartete Randeffekt-Verhalten (Forschungsfrage 4) ist in der
  Simulation nachweisbar, aber in dieser kleinen Domaene nur moderat
  ausgepraegt.
- Das "truegerische Sicherheit"-Phaenomen bei raeumlicher Unterabtastung
  (Forschungsfrage 6) ist klar und reproduzierbar nachgewiesen: Sobald der
  mittlere Punktabstand die Nyquist-Grenze λ/2 ueberschreitet, steigt das
  Kalibrierungsverhaeltnis eines geometrie-/datengetriebenen RBF-Kernels auf
  das 5- bis 14-fache (massive Unterschaetzung der Unsicherheit), waehrend
  ein Kernel mit der physikalisch korrekten Wellenzahl deutlich moderater
  bleibt (Faktor 1,0-1,8).

**Akzeptanzbewertung (Kurzfassung):** Fuer den vorliegenden Anwendungsfall
(ein Sofa-Hoerplatz, ein Subwoofer, Zimmerlautstarke, feste 8-12-Punkt-
Wedel-Messung) bietet das hier untersuchte EIVR/Complex-GP-Verfahren **keinen
belegbaren Mehrwert** gegenueber dem bestehenden Vorgehen. Der ggf. nuetzliche
Baustein ist nicht die aktive Punktauswahl, sondern die kalibrierte
Unsicherheitskarte einer GP-Rekonstruktion - und auch die nur unter der
Bedingung, dass der Kernel die wahre Wellenzahl kennt und der Punktabstand
die Nyquist-Grenze einhaelt. Details siehe Abschnitt 5.

## 1. Ziel & Abgrenzung

Diese Studie prueft ausschliesslich, ob ein Complex-Gaussian-Process-Modell
mit Expected-Integrated-Variance-Reduction-Kriterium fuer die adaptive Wahl
zusaetzlicher Messpunkte bei der raeumlichen Schallfeldrekonstruktion
(2D-Ausschnitt einer Hoerzone) mathematisch fundiert und praktisch
vorteilhaft ist. Es werden **keine** neuen Features fuer den
Spatial-Analyzer entwickelt, **keine** realen Messungen vorausgesetzt oder
ersetzt, und es wird **keine** Aussage ueber die tatsaechliche Akustik des
Wohnzimmers getroffen. Alle Modelle (ebene Wellen, Bessel-Kernel, RBF-Kernel)
sind explizit als Modelle gekennzeichnet.

## 2. Lizenz- und Code-Pruefung: `d-caviedes/acoustic_gps`

**Lizenz:** MIT (Copyright (c) 2020 Diego Caviedes Nozal), vollstaendig im
Repository als `LICENSE.md` enthalten. Permissiv, erlaubt Nutzung,
Veraenderung und Weitergabe inkl. kommerzieller Nutzung. Fuer diese Studie
wurde **kein Code aus dem Repository uebernommen** - es wurden ausschliesslich
Lizenz, Kernel-Definitionen und die mathematische Modellierungs-Konvention
gelesen und in eigenen Worten referenziert; die GP-Implementierung in
Abschnitt 4 ist eine eigene, von Grund auf neu geschriebene, vereinfachte
Umsetzung.

**Umfang der Bibliothek:** `acoustic_gps` ist eine Bayes'sche GP-Regressions-
/Rekonstruktionsbibliothek fuer Schallfelder (Hyperparameter-Inferenz per
Stan/MCMC), **kein** aktives Lern- oder Sampling-Werkzeug. Eine gezielte
Durchsicht von `example/example.py` und `example/helper.py` (Stichwort-Suche
nach `variance`, `active`, `candidate`, `EIVR`, `entropy`, `argmax`,
`argmin`, `next_point`, `sample`) ergab **keinen** Code fuer
Varianzreduktions-basierte naechste-Punkt-Auswahl. Die EIVR-Herleitung in
Abschnitt 4 ist daher eigenstaendige mathematische Arbeit; das Repository
dient nur als Referenz fuer uebliche Kernel-Konventionen bei
Schallfeld-GPs.

**Kernel-Katalog** (`acoustic_gps/kernels.py`, neun Funktionen): isotrope und
anisotrope quadratisch-exponentielle (RBF-)Kernel, eine periodische
RBF-Variante mit sin(k·|Δx|/2)² statt |Δx|², ein `sinc`-Kernel
(klassischer 3D-isotroper Diffusfeld-Kernel), ein `bessel0`-Kernel
σ²·J₀(k·|Δx|) (das 2D-Analogon des Diffusfeld-Kernels), sowie
richtungsparametrisierte `cosine`/`sine`/`sine_neg`-Kernel fuer bekannte
oder geschaetzte Einfallsrichtungen ebener Wellen, plus ein
Null-Platzhalterkernel.

**Komplexe Darstellung - die zentrale Erkenntnis fuer Abschnitt 4:**
`acoustic_gps/utils.py` modelliert ein komplexes Schallfeld p = p_r + i·p_i
als **gemeinsamen reellen GP** ueber dem gestapelten Vektor [p_r; p_i] mit
Blockkovarianz [[K_rr, K_ri], [K_ir, K_ii]] und rekonstruiert daraus explizit
sowohl die eigentliche komplexe Kovarianz
K = K_rr + K_ii + i·(K_ri^T − K_ri) als auch die **Pseudo-Kovarianz**
K_p = K_rr − K_ii + i·(K_ri^T + K_ri). K_p ≠ 0 zeigt einen nicht-zirkulaeren
("improper") komplexen Prozess an; K_p = 0 genau dann, wenn K_rr = K_ii und
K_ri antisymmetrisch ist. Diese Pseudo-Kovarianz-Buchhaltung ist in der
Bibliothek allgemein angelegt, wird aber in der Praxis nicht ausgenutzt:
**alle** bereitgestellten Kernel-Kombinationen (`rbf_isotropic`,
`rbf_anisotropic`, `rbf_anisotropic_periodic`, `bessel_isotropic`:
jeweils `[K, K, "zero", "zero"]`; `plane_wave_hierarchical`:
`["cosine", "cosine", "sine", "sine_neg"]`) erfuellen bei Pruefung die
Zirkularitaetsbedingung (K_rr = K_ii, K_ri antisymmetrisch) exakt. Die
Bibliothek unterstuetzt improper komplexe GPs also strukturell, nutzt in
allen Beispielen aber durchgaengig die zirkulaer-symmetrische
Vereinfachung - physikalisch plausibel fuer Felder aus vielen ueberlagerten
ebenen Wellen mit zufaelligen Phasen (ein bekanntes Grenzwertargument fuer
komplex-Gauss'sche Zirkularitaet bei Ueberlagerung vieler unabhaengiger
Phasenlagen). Diese Beobachtung ist eine eigene Einordnung, kein Zitat aus
dem Repository.

## 3. Mathematische Herleitung

### 3.1 Modellannahmen

Sei p(x) ein komplexwertiges, zentriertes (mittelwertbereinigtes),
**zirkulaer-symmetrisches** GP mit hermitischer Kovarianzfunktion
k(x, x′) = E[p(x)·p(x′)*] (k(x′,x) = k(x,x′)*) und verschwindender
Pseudo-Kovarianz. Messrauschen ε sei ebenfalls zirkulaer-symmetrisch
komplex, unabhaengig, Var(ε) = σ_n² (Gesamtvarianz, d. h. σ_n²/2 je Real-
und Imaginaerteil), unabhaengig von p. Fuer die in Abschnitt 4 verwendeten
**isotropen reellen** Kernel (Bessel J₀, RBF) ist k(x,x′) reell und
symmetrisch in |x−x′| - das entspricht exakt der Kernel-Kombination
`[K, K, "zero", "zero"]` aus Abschnitt 2 (K_rr = K_ii = K, K_ri = 0).

### 3.2 Posterior und Rang-1-Update

Fuer bereits gemessene Punkte X mit Beobachtungen y = p(X) + ε ist der
Posterior-Prozess wieder ein zirkulaer-symmetrisches komplexes GP mit

```
k_post(z, z′) = k(z, z′) − k(z, X) [K(X,X) + σ_n² I]⁻¹ k(X, z′)
```

(k(X,z′) = k(z′,X)^H, konjugierte Transposition). Wird **ein** zusaetzlicher
hypothetischer Messpunkt x betrachtet, liefert dieselbe Projektionsformel -
jetzt mit k_post als Basiskovarianz - ein Rang-1-Update (skalarer Nenner, da
nur ein neuer Punkt):

```
k_post,neu(z, z′) = k_post(z, z′) − k_post(z, x) · k_post(x, z′) / (k_post(x, x) + σ_n²)
```

Mit z′ = z und k_post(x,z) = k_post(z,x)* (Hermitizitaet) folgt
k_post(z,x)·k_post(x,z) = |k_post(z,x)|². Die Varianzreduktion an Punkt z
durch Messung bei x ist damit

```
Δσ²(z | x) = σ²_post(z) − σ²_post,neu(z) = |k_post(z, x)|² / (k_post(x, x) + σ_n²)
```

- exakt die im Arbeitsauftrag vorgegebene Formel, hier fuer den komplexen
Fall mit Betragsquadrat statt reellem Quadrat hergeleitet.

### 3.3 Beantwortung der sieben Forschungsfragen

**1) Analytische Berechenbarkeit.** Δσ²(z|x) selbst ist immer
geschlossen-analytisch (siehe 3.2) und haengt **nicht** vom beobachteten Wert
y_x ab, sondern nur von der Kernelstruktur und den Punktlagen - eine
zentrale GP-Eigenschaft. Das Wort "Expected" ist fuer einen GP mit
**fixierten** Hyperparametern daher formal redundant (es gibt nichts ueber
zufaellige Ausgaenge zu mitteln); es wird erst sinnvoll, wenn (a) die
Kernel-Hyperparameter selbst unsicher sind und ueber deren Posterior
gemittelt wird (bei `acoustic_gps` via Stan/MCMC-Hyperparameter-Samples
genau dieser Fall), oder (b) man "Integrated" als raeumliches Integral
∫_Ω Δσ²(z|x) dz liest - zwei verschiedene, leicht zu verwechselnde
Bedeutungen von Mittelung. Das Integral selbst ist fuer Gauss-Kernel analytisch
(Fehlerfunktionen), fuer oszillierende akustische Kernel (Bessel J₀, sinc)
i. d. R. nur per deterministischer numerischer Quadratur - aber **nicht**
als Monte-Carlo-Erwartungswert, da keine Zufallsgroesse gemittelt wird.

**2) Einfluss komplexer Beobachtungen.** Fuer das zirkulaer-symmetrische
Modell mit reellem isotropem Kernel ist k_post(z,z) rein reell (siehe 3.1) -
die Posterior-**Varianz** ist formelgleich zu einer reellen GP-Regression;
nur der Posterior-**Mittelwert** wird komplex. Bei nicht-zirkulaerer
(improper) Modellierung (K_p ≠ 0, wie von `acoustic_gps`s Pseudo-Kovarianz-
Mechanik unterstuetzt, aber in keinem Beispielkernel genutzt) muss man auf
die augmentierte ("widely linear") Darstellung z = [p; p*] mit
Block-Kovarianz aus K **und** K_p wechseln; die EIVR-Formel wird dann zu
einem 2×2-Schur-Komplement statt einem Skalarverhaeltnis. Amplitude |p| und
Phase φ=arg(p) sind **nichtlineare** Transformationen von (Re p, Im p) und
daher nicht direkt GP-vertraeglich: Unsicherheit auf Amplitude/Phase
erfordert Linearisierung (Delta-Methode, nur bei hohem SNR brauchbar) oder
Monte-Carlo-Stichproben aus dem komplexen Posterior. Nahe einer
Interferenz-Nullstelle ist die Phase einer (naeherungsweise) zirkulaer-
symmetrischen komplexen Normalverteilung **gleichverteilt** auf [−π,π] -
keine brauchbare Phasen-Unsicherheitsangabe moeglich. Dieser Punkt ist
direkt mit Frage 6 verknuepft.

**3) Einfluss von σ_n² und Kandidatenposition x.** Δσ² ∝ 1/(k_post(x,x)+σ_n²)
faellt monoton mit steigendem Messrauschen (abnehmender Grenznutzen,
Obergrenze |k_post(z,x)|²/k_post(x,x) fuer σ_n²→0). Die Abhaengigkeit von x
folgt der Kernelgeometrie: fuer eine glatte RBF-Kovarianz mit Laengenskala ρ
faellt Δσ² mit dem **Quadrat** der Kernel-Korrelation, also ueber eine
effektiv halbe Laengenskala - ein raeumlich lokalisierter,
"myopischer" Reduktionsbereich. Fuer oszillierende akustische Kernel
(Bessel J₀(kr), sinc(kr)) oszilliert |k_post(z,x)|² selbst mit
Nebenmaxima bei Vielfachen der halben Wellenlaenge - Δσ² faellt also
**nicht** monoton mit dem Abstand, anders als beim glatten RBF-Fall.

**4) EIVR vs. maximale punktweise Varianz.** Das Max-Varianz-Kriterium
waehlt z* = argmax_z σ²_post(z) und bewertet Kandidaten nur nach ihrem
Effekt an diesem einen Punkt - typischerweise am **Rand/in Ecken** des
Gebiets (dort hat ein Punkt per Definition weniger umgebende Stuetzpunkte,
ein bekannter GP-Randeffekt). EIVR integriert dagegen ueber das gesamte
(ggf. gewichtete) Gebiet und wiegt "einen schlechtesten Punkt verbessern"
gegen "viele maessig unsichere Punkte verbessern" ab. Die Simulation
(Abschnitt 4.4, Randabstand) bestaetigt die Richtung qualitativ: In der hier
getesteten kleinen, symmetrischen 1×1-m-Domaene platzierte Max-Varianz
**alle** zehn Zusatzpunkte exakt auf dem Rand (mittlerer Randabstand 0,0 m),
EIVR einen messbar, aber moderat von Null verschiedenen mittleren
Randabstand (0,045 m) - der Effekt ist nachweisbar, in dieser kleinen,
stark symmetrischen Domaene aber nicht dramatisch.

**5) Frequenzabhaengige EIVR-Gewichtung.** Da k von der Wellenzahl
k_wave=2πf/c abhaengt, liefert EIVR bei unterschiedlichen Frequenzen i. A.
unterschiedliche optimale naechste Punkte (Interferenzmuster verschieben
sich mit f). Ein plausibles Mehrfrequenz-Kriterium ist eine gewichtete
Summe/Integral: EIVR_total(x) = Σ_f w(f)·EIVR_f(x) bzw.
∫ w(f)·EIVR_f(x) df, mit w(f) als projektspezifischer Gewichtung
(z. B. Betonung des Raummodenbereichs 20-80 Hz, analog zur in diesem Projekt
ohnehin gewuenschten Hauskurve - **als Modellvorschlag**, nicht validiert).
Wichtig: ein gepooltes/gewichtetes EIVR ueber Frequenzen darf nicht einen
einzigen breitbandigen Kernel verwenden, sonst erbt es exakt das
Fehlbild aus Frage 6 bei den hoeheren Frequenzen des Bands.

**6) Unterabtastung & truegerische Sicherheit.** Das feinste darstellbare
raeumliche Merkmal eines aus ebenen Wellen ueberlagerten Feldes hat im
ungünstigsten Fall (zwei gegenlaeufige Wellen, stehende Welle) eine
Halbwellenlaenge λ/2 = c/(2f). Ein GP mit **festem, datengetrieben
gefittetem** Kernel (z. B. RBF, Laengenskala aus dem Punktabstand selbst
abgeleitet, ohne Kenntnis von k_wave) kann bei Punktabstand Δx > λ/2 nur
eine Laengenskala lernen, die der verfuegbaren (notwendig glatten, da
unterabgetasteten) Datenlage entspricht - **nicht** der wahren Physik. Die
berichtete Varianz ist dann niedrig, weil sie korrekt wiedergibt "wie gut
diese Kernelklasse durch die Daten festgelegt ist", aber **nicht**, "wie
weit diese Kernelannahme von der wahren Feldstruktur entfernt ist"
(Modellfehlspezifikation ist fuer ein GP mit fixierter Kernelfamilie
unsichtbar). Ein physikalisch motivierter oszillierender Kernel
(Bessel J₀(k·r) mit bekanntem/gemessenem k) kodiert die λ/2-Struktur
direkt - bleibt aber ebenfalls angreifbar, wenn k selbst aus unterabgetasteten
Daten geschaetzt werden muss (Wellenzahl-Aliasing, dieselbe
Nyquist-Grenze eine Ebene hoeher). Die Simulation in Abschnitt 4.5
bestaetigt dieses Verhalten quantitativ.

**Vorschlag fuer ein Nyquist-/Laengenskalen-Warnsignal:**
Sei Δx_min der mittlere naechste-Nachbar-Abstand im Messraster und
λ/2 = c/(2f) die physikalische Mindest-Feature-Groesse bei Frequenz f.
Definiere das **Unterabtastungsverhaeltnis**
`r(f) = Δx_min / (λ/2) = Δx_min · f / (c/2)`.
- **Warnung**, falls r(f) > 1: das Raster kann die feinste moegliche
  Interferenzstruktur bei f grundsaetzlich nicht aufloesen - unabhaengig
  davon, was der gefittete Kernel an Unsicherheit berichtet.
- Zusaetzlich, falls der GP eine eigene Laengenskala ρ̂ (bzw. ein ĥk → λ̂=2π/k̂)
  fittet: Warnung, falls ρ̂ > λ/2(f) - der Kernel hat eine Glattheit "gelernt",
  die physikalisch bei dieser Frequenz nicht moeglich ist, unabhaengig von
  Fit-Guetemassen wie der marginalen Log-Likelihood.

**7) Abbruchkriterien fuer zusaetzliche Messpunkte.**
a) **Absolute Schwelle** auf integrierter Unsicherheit
∫_Ω σ²_post(z)·w(z) dz (bzw. deren Wurzel als RMS-Unsicherheit in dB/Grad).
b) **Grenznutzen-Schwelle**: Abbruch, wenn max_x EIVR(x) unter einen kleinen
Anteil (z. B. 1-5 %) der aktuellen integrierten Gesamtvarianz faellt.
c) **Budget/physische Grenze** (Zeit, Mikrofonpositionierung) als harte
Obergrenze, innerhalb der a)/b) als Vorzeitig-Abbruch wirken.
d) **Nyquist-Gate** (verknuepft mit Frage 6): **bevor** ein
varianzbasiertes Abbruchkriterium ueberhaupt als vertrauenswuerdig gilt,
muss r(f) ≤ 1 fuer die hoechste interessierende Frequenz erfuellt sein -
sonst ist niedrige berichtete Varianz kein gueltiges Abbruchsignal.
e) **Kreuzvalidierung/Calibration-Check**: periodisch einen bereits
gemessenen Punkt auslassen, aus den uebrigen vorhersagen und den
standardisierten Residualwert pruefen (bzw. ein Coverage-/
PIT-Histogramm); systematisch zu grosse Residuen relativ zur berichteten
Unsicherheit sind ein datengetriebenes Signal, der Varianz **nicht** zu
vertrauen, und stattdessen nach rein geometrischer Abdeckung weiter zu
messen.

## 4. Synthetische Gegenprobe

### 4.1 Methodik & Reproduzierbarkeit

Ein temporaeres, deterministisches Python/NumPy-Skript (`sim.py`, nicht Teil
dieses Pull Requests, nach dem Lauf geloescht) implementierte eine eigene,
vereinfachte komplexe GP-Regression (kein Code aus `acoustic_gps`). Fixe
Parameter fuer volle Reproduzierbarkeit:

| Parameter | Wert |
|---|---|
| RNG-Seed (Basis) | `20261122` (`numpy.random.default_rng`) |
| Domaene Ω | Quadrat 1,0 m × 1,0 m (synthetischer Ausschnitt, keine Raumgeometrie) |
| Schallgeschwindigkeit c | 343,0 m/s |
| Seed-Messpunkte (fix, 5) | (0,0) (1,0) (0,1) (1,1) (0.5,0.5) |
| Zusaetzliche Punkte je Methode | 10 (→ 15 Punkte gesamt) |
| Kandidatenraster | 21×21-Gitter ueber Ω (Seeds entfernt) |
| Auswertungsraster | 41×41 = 1681 Punkte (dicht, fuer RMSE & EIVR-Integral) |
| Messrauschen σ_n | 0,02 (komplexe Amplitude, je Re/Im σ_n/√2) |
| Kernel (Hauptvergleich) | Bessel-isotrop: k(r) = J₀(k_wave·r) |
| Kernel (Nyquist-Test, "naiv") | RBF: k(r) = exp(−r²/(2ρ²)), ρ = mittlerer Punktabstand |
| Betrags-Floor fuer dB-Metriken | 1·10⁻⁶ |
| IDW-Exponent / Floor | 2 / 1·10⁻⁶ |
| Wiederholungen je Messung (S1-S4 / S5) | 15 / 10 (gemittelt, Rauschen bzw. Zufallsauswahl variiert je Wiederholung) |

Fuenf Szenarien (ebene Wellen, 150 Hz fuer S1-S4 sofern nicht anders
angegeben):

- **S1** - eine ebene Welle (30°, Amplitude 1,0): Grundkontrolle, kein Nulldurchgang.
- **S2** - zwei gegenlaeufige Wellen (0°/180°, Amplituden 1,0/0,9, Phasenversatz 25°): periodisches Interferenzmuster mit regelmaessigen Nullstellen.
- **S3** - fuenf ebene Wellen, zufaellige Richtungen/Amplituden/Phasen (seedbasiert): realistischere Mehrfachreflexions-Situation mit isolierten Nullstellen.
- **S4** - drei ebene Wellen mit stark unterschiedlichen Amplituden (1,0/0,25/0,08): Test der Dominanz-/Schwaeche-Abbildung.
- **S5** - wie S3, aber Frequenz von 40 Hz bis 1000 Hz variiert bei **fester** Messpunktdichte (Uniform-Auswahl, Δx_min = 0,305 m) zur gezielten Pruefung der Unterabtastungs-/Kalibrierungs-Frage.

Fuenf verglichene Pipelines (siehe Methodik-Hinweis: EIVR/MaxVar/Uniform
waehlen Punkte unabhaengig vom Messwert - deterministisch je Szenario; nur
Random variiert die Punktlage pro Wiederholung):

1. **GP_EIVR** - Punktauswahl per EIVR-Kriterium, GP-Rekonstruktion (Bessel-Kernel).
2. **GP_MaxVar** - Punktauswahl per maximaler punktweiser Posterior-Varianz, GP-Rekonstruktion.
3. **Random** - zufaellige Punktauswahl, GP-Rekonstruktion.
4. **Uniform** - Farthest-Point-(Greedy-Coverage-)Punktauswahl, GP-Rekonstruktion.
5. **IDW** - Complex Inverse Distance Weighting **auf denselben Punkten wie Uniform** (IDW hat kein eigenes Auswahlkriterium; diese Zuordnung ist explizit und transparent gewaehlt, nicht ergebnisoptimiert).

Metriken auf dem 41×41-Auswertungsraster: Complex RMSE
(√mean(|p̂−p|²)), Magnitude RMSE in dB (RMS der Differenz 20·log₁₀|p|,
Floor 1e-6), zirkulaerer Phasenfehler in Grad (RMS von
angle(p̂·conj(p))), Null-Fehler (Magnitude-RMSE-dB beschraenkt auf das
unterste 5-%-Perzentil von |p_true|), und Kalibrierungsverhaeltnis
RMS(|p̂−p|)/RMS(σ_post) (nur fuer GP-Methoden definiert; ≈1 bedeutet
kalibriert, >1 ueberkonfident/zu kleine Varianz, <1 konservativ).

### 4.2 Ergebnisse S1-S4 (gemittelt ueber 15 Wiederholungen, 15 Messpunkte)

| Szenario | Methode | Complex RMSE | Mag-RMSE (dB) | Phase (°) | Null-RMSE (dB) | Kalibrierung |
|---|---|---|---|---|---|---|
| S1 | GP_EIVR | 0,0114 | 0,07 | 0,44 | 0,07 | 0,91 |
| S1 | GP_MaxVar | 0,0121 | 0,07 | 0,50 | 0,07 | 0,96 |
| S1 | Uniform | 0,0127 | 0,08 | 0,49 | 0,08 | 0,97 |
| S1 | Random | 0,0184 | 0,11 | 0,72 | 0,11 | 0,99 |
| S1 | IDW | 0,2317 | 1,67 | 9,96 | 1,69 | n/a |
| S2 | GP_EIVR | 0,0137 | 0,21 | 1,40 | 0,61 | 1,10 |
| S2 | GP_MaxVar | 0,0125 | 0,19 | 1,23 | 0,59 | 1,00 |
| S2 | Uniform | 0,0138 | 0,19 | 1,23 | 0,60 | 1,05 |
| S2 | Random | 0,0199 | 0,27 | 1,72 | 0,77 | 0,91 |
| S2 | IDW | 0,3453 | 3,71 | 16,60 | 4,44 | n/a |
| S3 | GP_EIVR | 0,0125 | 0,12 | 0,78 | 0,45 | 1,00 |
| S3 | GP_MaxVar | 0,0129 | 0,11 | 0,71 | 0,42 | 1,03 |
| S3 | Uniform | 0,0121 | 0,12 | 0,74 | 0,44 | 0,93 |
| S3 | Random | 0,0211 | 0,17 | 1,18 | 0,64 | 1,05 |
| S3 | IDW | 0,4847 | 2,38 | 15,91 | 4,77 | n/a |
| S4 | GP_EIVR | 0,0122 | 0,07 | 0,47 | 0,10 | 0,97 |
| S4 | GP_MaxVar | 0,0123 | 0,07 | 0,43 | 0,09 | 0,98 |
| S4 | Uniform | 0,0118 | 0,07 | 0,46 | 0,11 | 0,90 |
| S4 | Random | 0,0186 | 0,11 | 0,67 | 0,15 | 0,92 |
| S4 | IDW | 0,2198 | 1,30 | 9,52 | 0,51 | n/a |

**Befund:** GP_EIVR, GP_MaxVar und Uniform liegen in allen vier Szenarien
innerhalb von ca. 10-15 % Streuung zueinander und sind durchweg gut
kalibriert (0,90-1,10, mit Ausreisser S2 1,10). Random ist in jedem
Szenario messbar schlechter (um 30-70 % hoeherer Complex RMSE). IDW ist
um den Faktor 10-40 schlechter als jede GP-Variante in Mag-RMSE/dB - mit
Abstand der groesste Effekt in dieser Tabelle.

### 4.3 EIVR vs. Max-Varianz: Randeffekt (Forschungsfrage 4, Illustration an S3)

Fuer eine Realisierung von S3 wurden die 10 zusaetzlichen Punkte beider
Kriterien gegen den mittleren Abstand zum Domaenenrand ausgewertet:

| Kriterium | mittlerer Randabstand (m) | Punkte exakt auf dem Rand |
|---|---|---|
| GP_MaxVar | 0,000 | 10 / 10 |
| GP_EIVR | 0,045 | 8 / 10 |

Bestaetigt die theoretische Erwartung (Abschnitt 3.3, Frage 4) in der
Richtung - Max-Varianz konzentriert sich staerker auf den Rand als EIVR -
der Effekt ist in dieser kleinen, stark symmetrischen 1×1-m-Domaene mit nur
5 Seed-Punkten jedoch moderat, nicht dramatisch.

### 4.4 S5 - Unterabtastung und Kalibrierung (Forschungsfrage 6)

Feste Messpunktmenge (15 Punkte, Uniform-Auswahl, Δx_min = 0,305 m),
Frequenz von 40 bis 1000 Hz variiert; Vergleich Bessel-Kernel (kennt die
wahre Wellenzahl) gegen RBF-Kernel (Laengenskala = Δx_min, "naiv"
datengetrieben, kennt die Wellenzahl nicht). Gemittelt ueber 10
Wiederholungen je Frequenz.

| f (Hz) | λ/2 (m) | r(f)=Δx_min/(λ/2) | Bessel Mag-RMSE (dB) | Bessel Kalib. | RBF Mag-RMSE (dB) | RBF Kalib. |
|---|---|---|---|---|---|---|
| 40 | 4,288 | 0,071 | 0,06 | 0,96 | 0,24 | 0,23 |
| 80 | 2,144 | 0,142 | 0,11 | 0,97 | 0,45 | 0,26 |
| 120 | 1,429 | 0,214 | 0,10 | 0,99 | 0,46 | 0,30 |
| 160 | 1,072 | 0,285 | 0,13 | 1,18 | 0,55 | 0,42 |
| 200 | 0,858 | 0,356 | 0,18 | 1,31 | 0,64 | 0,46 |
| 250 | 0,686 | 0,445 | 0,20 | 1,09 | 0,73 | 0,67 |
| 300 | 0,572 | 0,534 | 0,31 | 1,12 | 0,97 | 0,66 |
| 400 | 0,429 | 0,712 | 0,80 | 1,27 | 2,93 | 2,34 |
| 500 | 0,343 | 0,890 | 3,40 | 1,34 | 5,38 | 4,95 |
| 600 | 0,286 | **1,068** | 3,74 | 1,44 | 7,23 | 8,74 |
| 700 | 0,245 | 1,246 | 4,46 | 1,60 | 6,62 | 11,51 |
| 800 | 0,214 | 1,424 | 5,51 | 1,56 | 6,90 | 12,06 |
| 1000 | 0,172 | 1,780 | 6,43 | 1,74 | 7,17 | 13,70 |

**Befund:** Sobald r(f) die Marke 1 ueberschreitet (zwischen 500 und 600 Hz
bei diesem Punktabstand), steigt das Kalibrierungsverhaeltnis des naiven
RBF-Kernels von moderaten Werten (0,23-0,67 bei r<1, tendenziell zu
konservativ) auf 8,7 bis 13,7 (massiv ueberkonfident - die berichtete
Unsicherheit ist 9-14× zu klein gegenueber dem tatsaechlichen Fehler). Der
Bessel-Kernel mit der korrekten Wellenzahl bleibt im gesamten getesteten
Bereich bei 0,96-1,74 - sein absoluter Rekonstruktionsfehler waechst bei
hohen Frequenzen ebenfalls stark (physikalische Informationsgrenze bei
diesem Punktabstand), aber seine **Selbsteinschaetzung** (Kalibrierung)
bleibt deutlich ehrlicher. Dies ist die direkte quantitative Bestaetigung
der "truegerischen Sicherheit" aus Forschungsfrage 6 und des darauf
basierenden Nyquist-Warnsignals r(f) = Δx_min/(λ/2).

## 5. Akzeptanzbewertung

**Wo das untersuchte Verfahren theoretisch Wert haben kann:** 2D-Kartierung
unterhalb der Nyquist-Grenze (r(f) ≤ 1) mit einer bekannten oder zuverlaessig
schaetzbaren Wellenzahl, wenn eine **kalibrierte Unsicherheitskarte**
gebraucht wird (z. B. um zu entscheiden, wo eine Messkampagne noch
Rekonstruktionsfehler verbergen koennte) und ausreichend Messpunkte/Zeit fuer
einen iterativen Prozess mit Zwischen-Refits vorhanden sind.

**Wo es abgelehnt werden muss, fuer den vorliegenden Anwendungsfall:**

1. **Kein belegter Genauigkeitsgewinn der aktiven Auswahl selbst.** In allen
   vier Testszenarien lagen EIVR, Max-Varianz und einfaches
   Uniform-/Farthest-Point-Sampling innerhalb der Streuung beieinander.
   Der ueberlegene Baustein war die **GP-Rekonstruktion an sich**
   (physikalisch motivierter Kernel) gegenueber Complex IDW - das laesst
   sich mit einer deutlich einfacheren, nicht-adaptiven Methode erreichen
   (fester Punktsatz + GP-Fit statt IDW), ohne die Komplexitaet eines
   EIVR-Auswahlkriteriums.
2. **Ein Subwoofer, Zimmerlautstarke, feste 8-12-Punkt-Wedel-Messung.** Das
   Projekt misst bereits raeumlich gemittelt an einer festen, kleinen
   Punktzahl um die Hoerposition - eine iterative, adaptive
   Online-Messplanung mit Zwischen-Refit passt nicht zum bestehenden
   Messablauf (REW-Sweep, UMIK-1, eine Serie je Einstellung) und wuerde
   den Messaufwand erhoehen, nicht senken.
3. **Nyquist-Risiko im relevanten Frequenzbereich.** Bei typischen
   Sofa-/Hoerplatz-Abstaenden und den fuer Bassoptimierung relevanten
   Frequenzen oberhalb der tiefsten Raummoden liegt man schnell im
   unterabgetasteten Bereich (Abschnitt 4.4) - exakt dort, wo ein GP mit
   falsch spezifiziertem oder datengetrieben gefittetem Kernel am
   unzuverlaessigsten kalibriert ist. Ohne eine zuverlaessige, unabhaengig
   bekannte Wellenzahl pro Frequenzband ist das Risiko einer truegerischen
   Sicherheitsangabe real und durch die Simulation belegt.
4. **Komplexitaet/Aufwand vs. Nutzen.** Eine GP-Implementierung mit
   Kernel-Wahl, Hyperparameter-Schaetzung und EIVR-Berechnung ist erheblich
   aufwendiger als die bestehende REW/UMIK-1-Messkette und das
   IDW-/Mittelungs-Vorgehen - bei im besten Fall marginalem,
   nicht konsistent nachgewiesenem Zusatznutzen fuer diesen konkreten
   Hoerplatz.

**Fazit:** Fuer die kostenguenstige Optimierung eines einzelnen Subwoofers
an einem Sofa-Hoerplatz bei Zimmerlautstarke ist dieses Verfahren **nicht
empfehlenswert**. Es waere allenfalls fuer ein deutlich groesseres,
unregelmaessigeres Abhoerareal mit mehreren Hoerpositionen und bekannter,
stabiler Wellenzahl pro Frequenzband denkbar - ein Szenario, das fuer dieses
Projekt laut Zielsetzung nicht vorliegt.

## 6. Nächster Schritt

Keine Weiterverfolgung im Spatial-Analyzer-Code vorgesehen. Falls
gewuenscht, koennte als rein dokumentarische Ergaenzung ein kurzer Verweis
in `docs/ROADMAP_2027.md` auf dieses Ergebnis aufgenommen werden (als
"gepruefte und abgelehnte Option"), ohne den bestehenden Siebenpunkte-Plan
inhaltlich zu aendern.
