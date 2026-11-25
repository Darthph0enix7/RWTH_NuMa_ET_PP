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
    fig, (kurven, fehlerachsen, zoomachsen) = plt.subplots(
        3, 3, figsize=(13, 9), height_ratios=(1, 1.25, 1.25), layout="constrained"
    )
    x_plot = np.linspace(0, 1, 500)
    for ax, (titel, f, _), markierung in zip(
        kurven, funktionen, (None, (0.3, "Spitze bei 0,3"), (np.pi / 5, "Knick bei π/5"))
    ):
        ax.plot(x_plot, f(x_plot), color="0.2")
        ax.set(title=titel, xlabel="x", xlim=(0, 1))
        if markierung is not None:
            stelle, text = markierung
            ax.axvline(stelle, color="0.5", linestyle=":", label=text)
            ax.legend(loc="upper right", fontsize="small")

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

        stil = "s--" if randpunkte else "o-"
        farbe = "tab:orange" if randpunkte else "tab:blue"
        klein = range(n_werte.start, 11)
        for ax, zoom, werte, (titel, _, _) in zip(fehlerachsen, zoomachsen, fehler, funktionen):
            i = np.argmin(werte)
            n, m, kosten = aufteilungen[i]
            print(f"{titel}, {name}: N={n}, M={m}, Auswertungen={kosten}, Fehler={werte[i]:.3e}")
            y_plot = np.maximum(werte, 1e-16)
            ax.semilogy(n_werte, y_plot, stil, color=farbe, markersize=3, label=name)
            zoom.semilogy(klein, y_plot[:len(klein)], stil, color=farbe, markersize=4)

            if titel == "Schmaler Hügel":
                i_zwei = n_werte.index(2)
                zoom.scatter(2, y_plot[i_zwei], s=90, facecolors="none", edgecolors=farbe, zorder=4)
                print(f"  {name} bei N=2: M={aufteilungen[i_zwei][1]}, "
                      f"Auswertungen={aufteilungen[i_zwei][2]}, Fehler={werte[i_zwei]:.3e}")

    for ax, zoom in zip(fehlerachsen, zoomachsen):
        ax.set(title="Ganzer Bereich", xlabel="Punkte N pro Teilintervall", xlim=(0, budget + 1))
        zoom.set(title="Zoom: N bis 10", xlabel="Punkte N pro Teilintervall", xticks=range(1, 11))
        ax.grid(True, alpha=0.25)
        zoom.grid(True, alpha=0.25)
    zoomachsen[1].set_title("Zoom: Hügel bei N = 2 hervorgehoben")
    kurven[0].set_ylabel("Funktionswert f(x)")
    fehlerachsen[0].set_ylabel("Absoluter Fehler (log.)")
    zoomachsen[0].set_ylabel(r"Absoluter Fehler (log.; min. $10^{-16}$)")
    fehlerachsen[0].legend(loc="upper left", fontsize="small")
    fig.suptitle(f"Aufgabe 3: Summierte Quadratur mit höchstens {budget} Auswertungen")
    fig_dir = Path(__file__).resolve().parent / "figures"
    fig_dir.mkdir(exist_ok=True)
    fig.savefig(fig_dir / "aufgabe3_summiert.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
