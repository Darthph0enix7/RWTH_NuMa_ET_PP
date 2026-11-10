"""Aufgabe 2 - Tschebyscheff-Stützstellen: Vandermonde vs. Baryzentrisch."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def horner(a, x):
    res = np.zeros_like(x, dtype=float)
    for c in reversed(a):
        res = res * x + c
    return res


def baryzentrische_gewichte(x):
    m = len(x)
    w = np.ones(m, dtype=float)
    for j in range(m):
        for k in range(m):
            if k != j:
                w[j] *= x[j] - x[k]
        w[j] = 1.0 / w[j]
    return w


def baryzentrische_auswertung(x_nodes, y_nodes, w, x_eval):
    zaehler = np.zeros_like(x_eval, dtype=float)
    nenner = np.zeros_like(x_eval, dtype=float)
    for j in range(len(x_nodes)):
        term = w[j] / (x_eval - x_nodes[j])
        zaehler += term * y_nodes[j]
        nenner += term
    return zaehler / nenner


def main():
    data_dir = Path(__file__).resolve().parent / "Datensätze-20261110"
    pruefraster = np.loadtxt(data_dir / "pruefraster.csv")
    stuetzwerte = np.loadtxt(data_dir / "stuetzwerte_tschebyscheff.csv")

    E_pruef = pruefraster[:, 0]
    T_exakt = pruefraster[:, 1]

    n_werte = list(range(2, 41))
    emax_vander = []
    emax_bary = []
    kappas = []

    for n in n_werte:
        zeilen = stuetzwerte[stuetzwerte[:, 0] == n]
        x_nodes = zeilen[:, 2]      # E_mV
        y_nodes = zeilen[:, 4]      # T_kalibrator_C

        # Vandermonde-Matrix + Horner-Schema
        V = np.vander(x_nodes, increasing=True)
        a = np.linalg.solve(V, y_nodes)
        p_vander = horner(a, E_pruef)
        emax_vander.append(np.max(np.abs(p_vander - T_exakt)))
        kappas.append(np.linalg.cond(V, np.inf))

        # Baryzentrische Interpolation
        w = baryzentrische_gewichte(x_nodes)
        p_bary = baryzentrische_auswertung(x_nodes, y_nodes, w, E_pruef)
        emax_bary.append(np.max(np.abs(p_bary - T_exakt)))

    print(f"{'n':>3} | {'e_max Vander [°C]':>17} | {'e_max Bary [°C]':>15} | {'kappa_inf(V)':>12}")
    print("-" * 56)
    for i, n in enumerate(n_werte):
        if n in (2, 5, 8, 9, 10, 15, 20, 22, 24, 29, 32, 40):
            print(f"{n:3d} | {emax_vander[i]:17.4e} | {emax_bary[i]:15.4e} | {kappas[i]:12.2e}")

    plt.figure(figsize=(8, 5))
    plt.semilogy(n_werte, emax_vander, "o-", label="Vandermonde + Horner")
    plt.semilogy(n_werte, emax_bary, "s-", label="Baryzentrisch")
    plt.axhline(0.1, color="red", linestyle="--", label="Ziel: 0,1 °C")
    plt.xlabel("Polynomgrad n")
    plt.ylabel(r"Maximaler Fehler $e_{\max}$ [°C]")
    plt.title("Aufgabe 2: Fehlervergleich mit Tschebyscheff-Stützstellen")
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    fig_dir = Path(__file__).resolve().parent / "figures"
    fig_dir.mkdir(exist_ok=True)
    plt.savefig(fig_dir / "a2_fehlervergleich.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
