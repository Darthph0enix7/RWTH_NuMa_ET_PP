"""Programmierprojekt 3: Numerische Integration.

Aufgabe 5 (Option A): Systematischer Fehler – konstanter Offset des Messgeräts.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from aufgabe1 import gauss_formel, quadratur, transformiere
from aufgabe2 import nc_formel


def kapazitaet(v):
    """Exakte Kapazität C(v) in µF."""
    return 10.0 / (1.0 + (v / 8.0) ** 2)


def berechne_fehler_mit_offset(x, w, offset, w_exakt):
    """Berechnet Quadratur-, Daten-, Gesamtfehler und die Worst-Case-Schranke."""
    c_exakt = kapazitaet(x)
    c_offset = c_exakt + offset  # Konstanter systematischer Fehler

    q_exakt = quadratur(x * c_exakt, w)
    q_offset = quadratur(x * c_offset, w)

    e_quad = abs(q_exakt - w_exakt)
    e_daten = abs(q_offset - q_exakt)
    e_gesamt = abs(q_offset - w_exakt)
    schranke = abs(offset) * np.sum(np.abs(w) * x)

    return e_quad, e_daten, e_gesamt, schranke


def main():
    a, b = 0.0, 12.0
    w_exakt = 320.0 * np.log1p((12.0 / 8.0) ** 2)  # ~ 377.170 µJ
    offset = 0.01  # Konstanter Offset: +0,01 µF
    theorie_datenfehler = abs(offset) * 0.5 * (b**2 - a**2)  # 0.01 * 72 = 0.72 µJ

    n_werte = np.arange(2, 41)

    # Ergebnisse speichern
    ergebnisse = {
        "Gauß-Legendre": {"quad": [], "daten": [], "gesamt": [], "schranke": []},
        "Newton-Cotes": {"quad": [], "daten": [], "gesamt": [], "schranke": []},
    }


    for N in n_werte:
        for name, formel in (("Gauß-Legendre", gauss_formel), ("Newton-Cotes", nc_formel)):
            x_hat, w_hat = formel(N)
            x, w = transformiere(x_hat, w_hat, a, b)
            eq, ed, eg, schr = berechne_fehler_mit_offset(x, w, offset, w_exakt)

            ergebnisse[name]["quad"].append(eq)
            ergebnisse[name]["daten"].append(ed)
            ergebnisse[name]["gesamt"].append(eg)
            ergebnisse[name]["schranke"].append(schr)

    # Konsolenausgabe
    print("=" * 78)
    print(f"Aufgabe 5A: Konstanter Offset (Delta C = +{offset:.2f} µF)")
    print(f"Theoretischer Datenfehler: Delta C * 72 V^2 = {theorie_datenfehler:.4f} µJ")
    print("=" * 78)
    print(f"{'N':>3} | {'Gauß e_daten':>13} | {'Gauß e_ges':>11} | {'NC e_daten':>12} | {'NC Schranke':>12}")
    print("-" * 78)
    for i, N in enumerate(n_werte):
        if N in (2, 3, 4, 5, 8, 10, 15, 20, 25, 30, 35, 40):
            gd = ergebnisse["Gauß-Legendre"]["daten"][i]
            gg = ergebnisse["Gauß-Legendre"]["gesamt"][i]
            nd = ergebnisse["Newton-Cotes"]["daten"][i]
            ns = ergebnisse["Newton-Cotes"]["schranke"][i]
            print(f"{N:3d} | {gd:13.6f} | {gg:11.6f} | {nd:12.6f} | {ns:12.2e}")
    print("=" * 78)

    # Visualisierung
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Plot 1: Gauß-Legendre
    g = ergebnisse["Gauß-Legendre"]
    ax1.semilogy(n_werte, np.maximum(g["quad"], 1e-16), "b.-", label=r"$e_{\mathrm{Quad}}$ (exakte Daten)")
    ax1.semilogy(n_werte, g["daten"], "s-", color="orange", markersize=4, label=r"$e_{\mathrm{Daten}}$ (Offset)")
    ax1.semilogy(n_werte, g["gesamt"], "r--", linewidth=1.8, label=r"$e_{\mathrm{Gesamt}}$")
    ax1.axhline(theorie_datenfehler, color="black", linestyle=":", label=rf"Theorie $\Delta C \cdot 72 = {theorie_datenfehler:.2f}\,\mu$J")
    ax1.axhline(1.0, color="green", linestyle="--", alpha=0.7, label=r"Ziel: $1\,\mu$J")
    ax1.set_title(rf"Gauß-Legendre mit Offset ($\Delta C = +{offset}\,\mu$F)")
    ax1.set_xlabel("Anzahl Stützstellen $N$")
    ax1.set_ylabel(r"Fehler [$\mu$J] (log. Skala)")
    ax1.grid(True, which="both", linestyle=":", alpha=0.6)
    ax1.legend(loc="lower left")

    # Plot 2: Newton-Cotes (Offset vs. Worst-Case-Schranke)
    nc = ergebnisse["Newton-Cotes"]
    ax2.semilogy(n_werte, np.maximum(nc["quad"], 1e-16), "b.-", label=r"$e_{\mathrm{Quad}}$ (exakte Daten)")
    ax2.semilogy(n_werte, nc["daten"], "s-", color="orange", markersize=4, label=r"$e_{\mathrm{Daten}}$ (Offset)")
    ax2.semilogy(n_werte, nc["schranke"], "k--", label=r"Schranke $|\Delta C| \sum |w_i| v_i$")
    ax2.axhline(theorie_datenfehler, color="gray", linestyle=":", label=rf"Theorie ${theorie_datenfehler:.2f}\,\mu$J")
    ax2.set_title(rf"Newton-Cotes mit Offset ($\Delta C = +{offset}\,\mu$F)")
    ax2.set_xlabel("Anzahl Stützstellen $N$")
    ax2.set_ylabel(r"Fehler [$\mu$J] (log. Skala)")
    ax2.grid(True, which="both", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left")

    plt.tight_layout()

    fig_dir = Path(__file__).resolve().parent / "figures"
    fig_dir.mkdir(exist_ok=True)
    plt.savefig(fig_dir / "aufgabe5_offset.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()