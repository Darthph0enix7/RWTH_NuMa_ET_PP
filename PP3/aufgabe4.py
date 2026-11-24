"""Aufgabe 4: Kondensatorenergie mit möglichst wenigen Messungen."""

import numpy as np

from aufgabe1 import gauss_formel, quadratur, summierte_formel
from aufgabe2 import nc_formel


def fehler(x, w):
    kapazitaet = 10 / (1 + (x / 8)**2)
    gemessen = np.round(kapazitaet, 2)
    energie = 320 * np.log1p((12 / 8)**2)
    q_exakt = quadratur(x * kapazitaet, w)
    q_gemessen = quadratur(x * gemessen, w)

    e_quad = q_exakt - energie
    e_daten = q_gemessen - q_exakt
    e_gesamt = abs(q_gemessen - energie)
    schranke = abs(e_quad) + 0.005 * np.sum(np.abs(w) * x)
    return e_quad, e_daten, e_gesamt, schranke


def main():
    print("Fehler und Schranke in µJ; Ziel: Fehler < 1 µJ.")
    for name, formel, randpunkte in (
        ("Gauß-Legendre", gauss_formel, False), ("Newton-Cotes", nc_formel, True)
    ):
        ergebnisse = []
        for n in range(2 if randpunkte else 1, 41):
            x_hat, w_hat = formel(n)
            max_m = 39 // (n - 1) if randpunkte else 40 // n
            for m in range(1, max_m + 1):
                x, w = summierte_formel(x_hat, w_hat, 0, 12, m)
                ergebnisse.append((len(x), n, m, *fehler(x, w)))

        print(f"\n{name}:")
        for titel, kandidaten in (
            ("Ein Intervall", [r for r in ergebnisse if r[2] == 1]),
            ("Mehrere Intervalle", [r for r in ergebnisse if r[2] > 1]),
        ):
            for kriterium, spalte in (("im Versuch", 5), ("garantiert", 6)):
                passend = [r for r in kandidaten if r[spalte] < 1]
                if not passend:
                    print(f"  {titel}, {kriterium}: nicht mit bis zu 40 Messungen")
                    continue
                kosten, n, m, quad, daten, gesamt, schranke = min(
                    passend, key=lambda r: (r[0], r[spalte])
                )
                print(f"  {titel}, {kriterium}: {kosten} Messungen (N={n}, M={m})")
                print(f"    Quadratur={quad:+.4f}, Daten={daten:+.4f}, "
                      f"Gesamtbetrag={gesamt:.4f}, Schranke={schranke:.4f}")


if __name__ == "__main__":
    main()
