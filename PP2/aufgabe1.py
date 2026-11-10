"""Programmierprojekt 2: Interpolation einer Thermoelement-Kennlinie.

Aufgabe 1a: Interpolation über die Vandermonde-Matrix mit Horner-Schema.
Aufgabe 1b: Baryzentrische Interpolation, Fehlervergleich und Randeffekte bei n = 20.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np



# Mathematische Hilfsfunktionen


def horner(a, x):
    """Wertet ein Polynom p(x) = a0 + a1*x + ... + an*x^n mit dem Horner-Schema aus.

    Parameter:
        a: Array-like, Koeffizienten [a0, a1, ..., an] in aufsteigender Potenz
        x: Skalar oder ndarray, Auswertungsstellen

    Rückgabe:
        Werte p(x) als ndarray
    """
    res = np.full_like(x, a[-1], dtype=float)
    for c in reversed(a[:-1]):
        res = res * x + c
    return res


def baryzentrische_gewichte(x):
    """Berechnet die baryzentrischen Gewichte w_j = prod_{k != j} (x_j - x_k)^(-1).

    Parameter:
        x: ndarray, Stützstellen x0, ..., xn

    Rückgabe:
        w: ndarray, Gewichte w0, ..., wn
    """
    n = len(x) - 1
    w = np.empty(n + 1, dtype=float)
    for j in range(n + 1):
        diff = x[j] - np.delete(x, j)
        w[j] = 1.0 / np.prod(diff)
    return w


def baryzentrische_auswertung(x_nodes, y_nodes, w, x_eval):
    """Wertet das Interpolationspolynom mit der 2. baryzentrischen Formel aus.

    Formel:
        p(x) = sum_j (w_j / (x - x_j) * y_j) / sum_j (w_j / (x - x_j))

    Hinweis:
        Gültig für Auswertungsstellen x != x_j (Prüfraster enthält keine Stützstellen).
    """
    # Matrix der Differenzen (x_eval_i - x_node_j) der Form (M, n+1)
    diff = x_eval[:, np.newaxis] - x_nodes[np.newaxis, :]
    terms = w[np.newaxis, :] / diff
    numerator = np.sum(terms * y_nodes[np.newaxis, :], axis=1)
    denominator = np.sum(terms, axis=1)
    return numerator / denominator


# =============================================================================
# Daten laden
# =============================================================================

def lade_daten():
    """Lädt Prüfraster und äquidistante Stützwerte aus den CSV-Dateien."""
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "Datensätze-20261110"

    pruefraster = np.loadtxt(data_dir / "pruefraster.csv")
    E_pruef = pruefraster[:, 0]
    T_exakt = pruefraster[:, 1]

    stuetzwerte = np.loadtxt(data_dir / "stuetzwerte_aequidistant.csv")

    return E_pruef, T_exakt, stuetzwerte


# =============================================================================
# Hauptprogramm mit Durchführung von 1a und 1b
# =============================================================================

def main():
    # 1. Daten einlesen
    E_pruef, T_exakt, stuetzwerte = lade_daten()

    fig_dir = Path(__file__).resolve().parent / "figures"
    fig_dir.mkdir(exist_ok=True)

    n_values = np.arange(2, 41)
    emax_vander = []
    emax_bary = []
    cond_vander = []

    print("=" * 70)
    print("Programmierprojekt 2: Interpolation - Aufgaben 1a & 1b")
    print("=" * 70)

    # 2. Schleife über die Grade n = 2 bis 40
    for n in n_values:
        # Relevante Zeilen für Grad n filtern
        mask = stuetzwerte[:, 0] == n
        data_n = stuetzwerte[mask]
        x_nodes = data_n[:, 2]      # E_mV
        y_nodes = data_n[:, 4]      # T_kalibrator_C

        # --- Aufgabe 1a: Vandermonde-Interpolation ---
        V = np.vander(x_nodes, increasing=True)
        cond = np.linalg.cond(V, p=np.inf)
        cond_vander.append(cond)

        # Lineares Gleichungssystem V * a = y lösen
        a = np.linalg.solve(V, y_nodes)
        p_vander = horner(a, E_pruef)
        err_v = np.max(np.abs(p_vander - T_exakt))
        emax_vander.append(err_v)

        # --- Aufgabe 1b: Baryzentrische Interpolation ---
        w = baryzentrische_gewichte(x_nodes)
        p_bary = baryzentrische_auswertung(x_nodes, y_nodes, w, E_pruef)
        err_b = np.max(np.abs(p_bary - T_exakt))
        emax_bary.append(err_b)

    emax_vander = np.array(emax_vander)
    emax_bary = np.array(emax_bary)
    cond_vander = np.array(cond_vander)

    # =========================================================================
    # Ausgabe der Auswertung
    # =========================================================================
    print(f"\n--- Ergebnisse für ausgewählte Grade n ---")
    print(f"{'n':>3} | {'e_max (Vander) [°C]':>20} | {'e_max (Bary) [°C]':>20} | {'kappa_inf(V)':>15}")
    print("-" * 65)
    for idx in [0, 8, 12, 14, 18, 28, 38]:  # n = 2, 10, 14, 16, 20, 30, 40
        print(f"{n_values[idx]:3d} | {emax_vander[idx]:20.4e} | {emax_bary[idx]:20.4e} | {cond_vander[idx]:15.2e}")

    min_vander_idx = np.argmin(emax_vander)
    print(f"\nMinimaler Fehler Vandermonde : {emax_vander[min_vander_idx]:.4f} °C bei n = {n_values[min_vander_idx]}")
    min_bary_idx = np.argmin(emax_bary)
    print(f"Minimaler Fehler Baryzentrisch: {emax_bary[min_bary_idx]:.4f} °C bei n = {n_values[min_bary_idx]}")
    print(f"Wird das Ziel e_max < 0.1 °C erreicht? -> {'JA' if np.min(emax_vander) < 0.1 else 'NEIN'}")

    # =========================================================================
    # Plots erstellen
    # =========================================================================

    # Plot 1: Aufgabe 1a - e_max und Konditionszahl über n
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color_err = "tab:blue"
    ax1.set_xlabel("Polynomgrad n")
    ax1.set_ylabel(r"Maximaler Fehler $e_{\max}$ [°C]", color=color_err)
    line1 = ax1.semilogy(n_values, emax_vander, "o-", color=color_err, label=r"$e_{\max}$ (Vandermonde)")
    line_target = ax1.axhline(0.1, color="red", linestyle="--", label="Ziel: 0.1 °C")
    ax1.tick_params(axis="y", labelcolor=color_err)
    ax1.grid(True, which="both", linestyle=":", alpha=0.6)

    ax2 = ax1.twinx()
    color_cond = "tab:orange"
    ax2.set_ylabel(r"Konditionszahl $\kappa_\infty(V)$", color=color_cond)
    line2 = ax2.semilogy(n_values, cond_vander, "s--", color=color_cond, label=r"$\kappa_\infty(V(X))$")
    ax2.tick_params(axis="y", labelcolor=color_cond)

    # Legenden beider Achsen zusammenführen
    lines = line1 + [line_target] + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center left", bbox_to_anchor=(0.12, 0.78))

    plt.title("Aufgabe 1a: Vandermonde-Interpolation (äquidistante Stützstellen)")
    fig.tight_layout()
    plot1_path = fig_dir / "a1a_vandermonde.png"
    plt.savefig(plot1_path, dpi=150)
    print(f"\nPlot 1a gespeichert: {plot1_path}")

    # Plot 2: Aufgabe 1b - Vergleich e_max Vandermonde vs. Baryzentrisch
    plt.figure(figsize=(8, 5))
    plt.semilogy(n_values, emax_vander, "o-", label="Vandermonde + Horner")
    plt.semilogy(n_values, emax_bary, "s-", label="Baryzentrische Interpolation")
    plt.axhline(0.1, color="red", linestyle="--", label="Zielwert 0.1 °C")
    plt.xlabel("Polynomgrad n")
    plt.ylabel(r"Maximaler Fehler $e_{\max}$ [°C]")
    plt.title("Aufgabe 1b: Fehlervergleich bei äquidistanten Stützstellen")
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plot2_path = fig_dir / "a1b_fehlervergleich.png"
    plt.savefig(plot2_path, dpi=150)
    print(f"Plot 1b (Vergleich) gespeichert: {plot2_path}")

    # Plot 3: Aufgabe 1b - Abweichung p(E) - T_exakt(E) über E für n = 20
    data_20 = stuetzwerte[stuetzwerte[:, 0] == 20]
    x_nodes_20 = data_20[:, 2]
    y_nodes_20 = data_20[:, 4]

    # Baryzentrisch bei n = 20
    w_20 = baryzentrische_gewichte(x_nodes_20)
    p_bary_20 = baryzentrische_auswertung(x_nodes_20, y_nodes_20, w_20, E_pruef)
    diff_bary_20 = p_bary_20 - T_exakt

    # Vandermonde bei n = 20 zum direkten Vergleich
    V_20 = np.vander(x_nodes_20, increasing=True)
    a_20 = np.linalg.solve(V_20, y_nodes_20)
    p_vander_20 = horner(a_20, E_pruef)
    diff_vander_20 = p_vander_20 - T_exakt

    plt.figure(figsize=(8, 5))
    plt.plot(E_pruef, diff_bary_20, label=r"Baryzentrisch: $p(E) - T_{\mathrm{exakt}}(E)$")
    plt.plot(E_pruef, diff_vander_20, "--", label=r"Vandermonde: $p(E) - T_{\mathrm{exakt}}(E)$", alpha=0.8)
    plt.axhline(0.1, color="red", linestyle=":", label="Toleranz ±0.1 °C")
    plt.axhline(-0.1, color="red", linestyle=":")
    plt.xlabel("Thermospannung E [mV]")
    plt.ylabel(r"Abweichung $p(E) - T_{\mathrm{exakt}}(E)$ [°C]")
    plt.title("Aufgabe 1b: Abweichung über den Spannungsbereich für n = 20")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plot3_path = fig_dir / "a1b_abweichung_n20.png"
    plt.savefig(plot3_path, dpi=150)
    print(f"Plot 1b (Abweichung n=20) gespeichert: {plot3_path}")

    # Plots anzeigen
    plt.show()


if __name__ == "__main__":
    main()
