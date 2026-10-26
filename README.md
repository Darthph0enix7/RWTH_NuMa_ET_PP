# RWTH — Numerische Mathematik für Elektrotechnik (Programmierprojekte)

Gruppen-Repo für die Programmierprojekte. Ein Ordner pro Projekt (`PP1`, später `PP2`, ...),
darin ein eigenständiges Skript pro Aufgabe. Jeder arbeitet seine Aufgabe selbst durch —
gemeinsame Module o.ä. erst, wenn wir merken, dass wir sie wirklich brauchen.

**Aktuell: PP1 — Entzerrung eines Funkkanals mit FFT (OFDM)**

---

## Setup (einmal pro Rechner)

```bash
git clone https://github.com/Darthph0enix7/RWTH_NuMa_ET_PP.git
cd RWTH_NuMa_ET_PP

python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

pip install -r requirements.txt
python sync.py name "Dein Vorname"
```

Voraussetzung: Python 3.10+ und Git. Auf Windows reicht der Installer von python.org
(Häkchen bei „Add Python to PATH“) und <https://git-scm.com/download/win>.

---

## Damit arbeiten (das ist alles, was man sich merken muss)

```bash
# im Projektordner (dort liegt sync.py) — bevor du anfaengst:
python sync.py pull

# arbeiten:
cd PP1
python aufgabe1.py

# fertig: zurueck in den Projektordner und hochladen (pull + commit + push):
cd ..
python sync.py "aufgabe1 a) und b) fertig"
```

Mehr braucht niemand. Wer will, kann zusätzlich:

| Befehl | Zweck |
| --- | --- |
| `python sync.py status` | Was ist lokal neu, was fehlt vom Server? |
| `python sync.py save "text"` | Nur lokal speichern, noch nicht hochladen |
| `python sync.py abort` | Konflikt aufgeben: Server-Fassung nehmen (dein Stand bleibt als Sicherheitskopie in `git stash list`) |
| `python sync.py name "Vorname"` | Name für die Commit-Historie ändern |

`sync.py` ist absichtlich ein Python-Skript: läuft auf Windows, macOS und Linux gleich.
Falls `python` bei dir nicht gefunden wird (macOS), einfach `python3 sync.py ...` schreiben.

---

## Wer macht was

| Aufgabe | Inhalt | Datei | Wer |
| --- | --- | --- | --- |
| 1 | Kanalschätzung + Entzerrung (a, b, c) | `PP1/aufgabe1.py` | _eintragen_ |
| 2 | DFT vs. FFT Laufzeitexperiment | `PP1/aufgabe2.py` | _eintragen_ |
| 3 | Kondition der Entzerrungsmatrix | `PP1/aufgabe3.py` | _eintragen_ |

Kurze Zwischenstände/ToDos könnt ihr direkt in die Commit-Nachricht schreiben
(`python sync.py "a) läuft, b) hänge ich"`) — das ist unser Protokoll.

---

## Aufbau

```
.
├── sync.py              # Git für Menschen, die kein Git können wollen
├── requirements.txt     # numpy, matplotlib - gleiche Versionen für alle
├── PP1/
│   ├── aufgabe1.py      # Vorlage: Kanalschätzung + Entzerrung
│   ├── aufgabe2.py      # Vorlage: DFT vs. FFT
│   ├── aufgabe3.py      # Vorlage: Kondition
│   ├── datensatz*.csv   # gegeben
│   ├── pilotsymbole.csv # gegeben
│   ├── Testat1.pdf      # Aufgabenblatt
│   └── figures/         # fertige Plots hier ablegen (kommen mit ins Repo)
```

Die drei Skripte sind leere Vorlagen: Kopfkommentar mit den Teilschritten, Imports,
`N`/`CP`, ein `TODO`. Einfach losschreiben.

---

## Drei Regeln

1. **Nur die eigene Datei anfassen.** Wer Aufgabe 1 hat, editiert `aufgabe1.py`.
   Neue Dateien (z.B. `aufgabe1_test.py`, `figures/a1_spektrum.png`) sind ok,
   bestehende Dateien von anderen bitte nicht umschreiben.
2. **Skripte, keine Notebooks.** `.ipynb` macht beim Mergen nur Ärger — Plots als PNG
   in `figures/` reichen.
3. **Vor dem Loslegen `pull`, nach dem Arbeiten `sync`.** Dann gibt es praktisch keine
   Konflikte.
