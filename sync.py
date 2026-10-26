#!/usr/bin/env python3
"""sync.py - Git ohne Git-Gefummel fuer unsere Gruppe.

    python sync.py                   # holen + speichern + hochladen (Standard)
    python sync.py "text"            # dasselbe, mit eigener Notiz
    python sync.py pull              # nur die Aenderungen der anderen holen
    python sync.py save "text"       # nur lokal speichern (noch nicht hochladen)
    python sync.py status            # was ist neu? was fehlt noch?
    python sync.py abort             # nach einem Konflikt: alles zuruecksetzen
    python sync.py name "Vorname"    # einmalig: Name fuer die Commit-Historie

Braucht nur Python und Git - kein pip install.

Regel, die 90% der Probleme verhindert:
    morgens  ->  python sync.py pull
    abends   ->  python sync.py "was ich gemacht habe"
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path


class GitFehler(Exception):
    pass


# --------------------------------------------------------------------------
# Kleinkram
# --------------------------------------------------------------------------
def git(*args, check=True):
    try:
        r = subprocess.run(["git", *args], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
    except FileNotFoundError:
        raise GitFehler("Git ist nicht installiert. Siehe README.md (Abschnitt Setup).")
    if check and r.returncode != 0:
        raise GitFehler((r.stderr or r.stdout).strip())
    return r


def ok(text):
    print(f"[ok]   {text}")


def info(text):
    print(f"[info] {text}")


def warnung(text):
    print(f"[!]    {text}")


def repo_root():
    try:
        return Path(git("rev-parse", "--show-toplevel").stdout.strip())
    except GitFehler:
        raise GitFehler("Hier ist kein Git-Repository. Bitte in den Projektordner wechseln.")


# --------------------------------------------------------------------------
# Identitaet einmalig setzen (sonst kann git nicht committen)
# --------------------------------------------------------------------------
def name_holen():
    return git("config", "user.name", check=False).stdout.strip()


def identitaet_sichern(frage=True):
    name = name_holen()
    email = git("config", "user.email", check=False).stdout.strip()

    if not name:
        if not frage:
            raise GitFehler("Kein Git-Name gesetzt. Bitte einmal: python sync.py name \"Vorname\"")
        print("Einmalig: wie heisst du? (erscheint in der Commit-Historie)")
        eingabe = input("Name: ").strip()
        if not eingabe:
            raise GitFehler("Ohne Namen geht es nicht.")
        git("config", "--local", "user.name", eingabe)
        name = eingabe
        ok(f"Name gesetzt: {name}")

    if not email:
        slug = "".join(c for c in name.lower() if c.isalnum()) or "gruppe"
        git("config", "--local", "user.email", f"{slug}@gruppe.local")
        info(f"E-Mail gesetzt: {slug}@gruppe.local")
    return name


# --------------------------------------------------------------------------
# Zustand pruefen
# --------------------------------------------------------------------------
def hat_aenderungen():
    """True, wenn es (ausser ignorierter Dateien) etwas zu committen gibt."""
    if git("status", "--porcelain").stdout.strip():
        return True
    # untracked, aber ignoriert zaehlt nicht -> status --porcelain zeigt es nicht
    return False


def unmerged():
    """Dateien, die der Index noch als ungeloesten Konflikt fuehrt."""
    return [z for z in git("diff", "--name-only", "--diff-filter=U",
                           check=False).stdout.splitlines() if z.strip()]


def im_konflikt():
    gitdir = Path(git("rev-parse", "--git-dir").stdout.strip())
    if not gitdir.is_absolute():
        gitdir = repo_root() / gitdir
    if any((gitdir / n).exists() for n in ("rebase-merge", "rebase-apply", "MERGE_HEAD")):
        return True
    return bool(unmerged())


def konflikt_erklaeren():
    warnung("Es gibt einen Konflikt - zwei Leute haben dieselbe Stelle geaendert.")
    dateien = git("diff", "--name-only", "--diff-filter=U").stdout.strip()
    if dateien:
        print("Betroffene Dateien:")
        for zeile in dateien.splitlines():
            print(f"   {zeile}")
    print()
    print("So loesen (in dieser Reihenfolge probieren):")
    print("  1) python sync.py abort     -> alles zuruecksetzen, dann neu anfangen")
    print("  2) betroffene Datei oeffnen, die Markierungen <<<<<<< ======= >>>>>>>")
    print("     entfernen, dann: python sync.py \"konflikt geloest\"")
    print()
    print("Am einfachsten vermeiden: jeder arbeitet in seiner eigenen Datei.")


def konflikt_marker():
    """Dateien, in denen noch <<<<<<< / >>>>>>> Markierungen stehen."""
    r = git("grep", "-l", "-I", "-e", "^<<<<<<< ", "--", ".", check=False)
    return [z for z in r.stdout.splitlines() if z.strip()]


def pruefe_marker():
    dateien = konflikt_marker()
    if not dateien:
        return True
    warnung("In diesen Dateien stecken noch Konflikt-Markierungen (<<<<<<<):")
    for datei in dateien:
        print(f"   {datei}")
    print()
    print("Bitte die Datei oeffnen, eine sinnvolle Fassung daraus machen und die")
    print("Zeilen mit <<<<<<<  =======  >>>>>>> loeschen. Danach nochmal sync.")
    print()
    info("Sicherheitskopie deines Standes von vorher: git stash list")
    return False


# --------------------------------------------------------------------------
# Befehle
# --------------------------------------------------------------------------
def hat_upstream():
    """Gibt es schon einen Server-Branch, von dem man ziehen kann?"""
    return git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}",
               check=False).returncode == 0


def upstream_verbinden():
    """Falls der lokale Branch noch keinem Server-Branch folgt: einmal verbinden."""
    if hat_upstream():
        return True
    git("fetch", "origin", check=False)

    ziel = git("symbolic-ref", "--short", "refs/remotes/origin/HEAD",
               check=False).stdout.strip()
    if not ziel:
        refs = git("for-each-ref", "--format=%(refname:short)", check=False).stdout.split()
        ziel = next((r for r in refs if r.startswith("origin/") and not r.endswith("HEAD")), "")

    if not ziel:
        return False
    git("branch", "--set-upstream-to", ziel, check=False)
    return hat_upstream()


def cmd_pull(leise=False):
    # kein Identitaets-Check: pullen darf jeder, auch ohne Namen
    offen = unmerged()
    if offen:
        marker = konflikt_marker()
        behoben = [f for f in offen if f not in marker]
        if behoben:
            git("add", *behoben)
            ok("Als geloest markiert: " + ", ".join(behoben))
        if unmerged():
            konflikt_erklaeren()
            return 1

    if not upstream_verbinden():
        if not leise:
            info("Noch nichts auf dem Server - der erste Upload legt den Branch an.")
        return 0

    # Bewusst NICHT "git pull": das raet selbst herum, auf welchen Branch es
    # rebasen soll ("Cannot rebase onto multiple branches"). Hier ist es explizit.
    branch = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if branch in ("", "HEAD"):
        info("Du bist auf keinem Branch (detached HEAD) - bitte: git switch main")
        return 1
    ziel = f"origin/{branch}"

    r = git("fetch", "origin", check=False)
    if r.returncode != 0:
        warnung("Konnte nicht vom Server holen (Internet? Zugang?).")
        return 1

    if git("rev-parse", "--verify", "--quiet", ziel, check=False).returncode != 0:
        if not leise:
            info(f"Server hat noch keinen Branch {branch} - der erste Upload legt ihn an.")
        return 0

    vorher = git("rev-parse", "HEAD").stdout.strip()
    try:
        neu = int(git("rev-list", "--count", f"{vorher}..{ziel}").stdout.strip() or 0)
    except ValueError:
        neu = 0

    r = git("rebase", "--autostash", ziel, check=False)
    if r.returncode != 0:
        if im_konflikt():
            konflikt_erklaeren()
        else:
            text = (r.stderr or r.stdout).strip()
            warnung(text.splitlines()[-1] if text else "rebase fehlgeschlagen")
            info("Nichts kaputt: dein Stand ist noch da. Nochmal probieren oder fragen.")
        return 1

    if not leise:
        if neu == 0:
            ok("War schon aktuell.")
        else:
            ok(f"{neu} neue Commit(s) von den anderen geholt.")
    return 0 if pruefe_marker() else 1


def cmd_save(nachricht=None):
    name = identitaet_sichern()

    if not pruefe_marker():
        return 1

    if not hat_aenderungen():
        info("Nichts zu speichern - Arbeitsstand ist schon gesichert.")
        return 0

    git("add", "-A")
    if not git("diff", "--cached", "--quiet", check=False).returncode:
        info("Nur ignorierte Dateien geaendert - nichts zu speichern.")
        return 0

    text = (nachricht or "").strip() or datetime.now().strftime("%d.%m. %H:%M")
    git("commit", "-m", f"{name}: {text}")
    ok(f"Gespeichert: \"{name}: {text}\"")
    return 0


def cmd_push():
    r = git("push", check=False)
    if r.returncode == 0:
        ok("Hochgeladen.")
        return 0

    fehler = (r.stderr or r.stdout).strip()
    if "no upstream branch" in fehler or "has no upstream" in fehler:
        git("push", "-u", "origin", "HEAD")
        ok("Hochgeladen (Branch verbunden).")
        return 0

    # Jemand war schneller: einmal neu holen und nochmal
    info("Jemand war schneller - hole kurz nach und versuche es nochmal.")
    if cmd_pull(leise=True) != 0:
        return 1
    r = git("push", check=False)
    if r.returncode != 0:
        warnung((r.stderr or r.stdout).strip().splitlines()[-1])
        return 1
    ok("Hochgeladen.")
    return 0


def cmd_sync(nachricht=None):
    if cmd_pull() != 0:
        return 1
    if cmd_save(nachricht) != 0:
        return 1
    return cmd_push()


def cmd_status():
    root = repo_root()
    print(f"Ordner: {root}")
    print(f"Du bist: {name_holen() or '(kein Name gesetzt)'}")
    print()

    branch = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    print(f"Branch: {branch}")

    aenderungen = git("status", "--short").stdout.strip()
    if aenderungen:
        print("Geaenderte Dateien (noch nicht gespeichert):")
        for zeile in aenderungen.splitlines():
            print(f"   {zeile}")
    else:
        print("Keine ungespeicherten Aenderungen.")

    git("fetch", "--quiet", check=False)
    vorne = git("rev-list", "--count", f"HEAD..@{{u}}", check=False).stdout.strip()
    hinten = git("rev-list", "--count", f"@{{u}}..HEAD", check=False).stdout.strip()
    if vorne and vorne != "0":
        print(f"Server hat {vorne} neue(n) Commit(s) -> python sync.py pull")
    if hinten and hinten != "0":
        print(f"Du hast {hinten} Commit(s) nur lokal -> python sync.py")

    print("\nLetzte Commits:")
    print(git("log", "--oneline", "-5", check=False).stdout.strip() or "   (noch keine)")
    return 0


def cmd_abort():
    if im_konflikt():
        for befehl in (("rebase", "--abort"), ("merge", "--abort")):
            if git(*befehl, check=False).returncode == 0:
                ok("Zurueckgesetzt. Arbeitsstand ist wie vorher.")
                return 0

        offen = unmerged()
        up = git("rev-parse", "--abbrev-ref", "--symbolic-full-name",
                 "@{u}", check=False).stdout.strip()
        if offen and up:
            git("checkout", up, "--", *offen, check=False)
            git("add", "-A", check=False)
            ok("Server-Fassung uebernommen: " + ", ".join(offen))
            info("Deine Fassung von vor dem pull liegt weiter in: git stash list")
            return 0

        warnung("Konnte nicht automatisch zuruecksetzen - siehe git status.")
        return 1
    ok("Kein Konflikt vorhanden - nichts zu tun.")
    return 0


def cmd_name(argumente):
    if not argumente:
        print(f"Aktueller Name: {name_holen() or '(nicht gesetzt)'}")
        print('Setzen mit: python sync.py name "Vorname"')
        return 0
    git("config", "--local", "user.name", " ".join(argumente))
    ok(f"Name gesetzt: {' '.join(argumente)}")
    identitaet_sichern(frage=False)
    return 0


HILFE = __doc__


def main(argv):
    try:
        repo_root()
    except GitFehler as e:
        warnung(str(e))
        return 1

    befehl = argv[0].lower() if argv else "sync"
    rest = argv[1:]

    try:
        if befehl in ("sync", "go"):
            return cmd_sync(" ".join(rest) or None)
        if befehl == "pull":
            return cmd_pull()
        if befehl == "save":
            return cmd_save(" ".join(rest) or None)
        if befehl == "push":
            return cmd_push()
        if befehl == "status":
            return cmd_status()
        if befehl == "abort":
            return cmd_abort()
        if befehl == "name":
            return cmd_name(rest)
        if befehl in ("help", "-h", "--help"):
            print(HILFE)
            return 0
        # Alles andere ist eine Notiz: python sync.py "hab aufgabe1 angefangen"
        return cmd_sync(" ".join(argv))
    except GitFehler as e:
        warnung(str(e))
        return 1
    except KeyboardInterrupt:
        print()
        return 130


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
