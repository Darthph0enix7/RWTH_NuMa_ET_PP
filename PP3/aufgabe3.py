"""Aufgabe 3: Summierte Quadratur mit höchstens 48 Auswertungen."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aufgabe1 import gauss_formel, quadratur, summierte_formel
from aufgabe2 import nc_formel


def main():
    budget = 48
    funktionen = [
        (r"$e^x$", np.exp, np.expm1(1)),
        ("Schmaler Hügel", lambda x: 1 / (1 + 400 * (x - 0.3)**2),
         (np.arctan(14) + np.arctan(6)) / 20),
        (r"$|\sin(5x)|$", lambda x: np.abs(np.sin(5 * x)), (3 + np.cos(5)) / 5),
    ]
    fig, achsen = plt.subplots(1, 3, figsize=(13, 4))

    for name, formel, randpunkte in (
        ("Gauß-Legendre", gauss_formel, False), ("Newton-Cotes", nc_formel, True)
    ):
        n_werte = range(2 if randpunkte else 1, budget + 1)
        aufteilungen = []
        fehler = [[] for _ in funktionen]

        for n in n_werte:
            m = (budget - 1) // (n - 1) if randpunkte else budget // n
            x, w = summierte_formel(*formel(n), 0, 1, m)
            aufteilungen.append((n, m, len(x)))
            for werte, (_, f, integral) in zip(fehler, funktionen):
                werte.append(abs(quadratur(f(x), w) - integral))

        for ax, werte, (titel, _, _) in zip(achsen, fehler, funktionen):
            i = np.argmin(werte)
            n, m, kosten = aufteilungen[i]
            print(f"{titel}, {name}: N={n}, M={m}, Auswertungen={kosten}, Fehler={werte[i]:.3e}")
            ax.semilogy(n_werte, np.maximum(werte, 1e-16), ".-", label=name)

    for ax, (titel, _, _) in zip(achsen, funktionen):
        ax.set(title=titel, xlabel="Stützstellen N pro Teilintervall")
        ax.grid(True, alpha=0.3)
        ax.legend()
    achsen[0].set_ylabel("Absoluter Fehler (Plot-Untergrenze $10^{-16}$)")
    fig.tight_layout()
    fig_dir = Path(__file__).resolve().parent / "figures"
    fig_dir.mkdir(exist_ok=True)
    fig.savefig(fig_dir / "aufgabe3_summiert.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
