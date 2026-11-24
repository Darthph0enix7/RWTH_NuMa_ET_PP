"""Programmierprojekt 3: Numerische Integration.

Aufgabe 2: Abgeschlossene Newton-Cotes-Formeln, baryzentrische Auswertung
           der Lagrange-Polynome und Vergleich mit Gauß-Legendre.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Funktionen aus Aufgabe 1 importieren
from aufgabe1 import exaktheitsgrad, gauss_formel

# Baryzentrische Lagrange-Auswertung & Newton-Cotes-Formel

def baryzentrische_gewichte(x):
    """Berechnet die baryzentrischen Gewichte w_j = prod_{k != j} (x_j - x_k)^(-1) wie in PP2."""
    n = len(x)
    w = np.empty(n, dtype=float)
    for j in range(n):
        diff = x[j] - np.delete(x, j)
        w[j] = 1.0 / np.prod(diff)
    return w


def nc_formel(N):
    """Berechnet die abgeschlossene Newton-Cotes-Formel auf [-1, 1].

    - Stützstellen: N äquidistante Punkte auf [-1, 1]
    - Gewichte: w_hat_i = int_{-1}^1 ell_i(x) dx
      Die Lagrange-Polynome ell_i werden über die 2. baryzentrische Interpolationsformel
      (PP2) an den Stützstellen einer Gauß-Quadratur ausgewertet und integriert.
    """
    if N < 2:
        raise ValueError("Abgeschlossene Newton-Cotes-Formeln erfordern mindestens N >= 2 Stützstellen.")

    # Äquidistante Stützstellen auf [-1, 1]
    x_nc = np.linspace(-1.0, 1.0, N)

    # Gauß-Legendre-Formel mit ausreichend vielen Punkten zur exakten Integration
    # Ein Polynom vom Grad N - 1 wird ab N_gauss >= N/2 exakt integriert
    N_gauss = max(N, 60)
    x_g, w_g = gauss_formel(N_gauss)

    # Baryzentrische Gewichte der Newton-Cotes-Stützstellen
    w_bary = baryzentrische_gewichte(x_nc)

    # 2. baryzentrische Formel für die Basispolynome ell_i(x):
    # Da ell_i(x_nc_j) = delta_{ij}, ist der Zähler w_bary[i] / (x - x_nc[i])
    # diff hat Dimension (N_gauss, N)
    diff = x_g[:, None] - x_nc[None, :]
    inv_diff = w_bary[None, :] / diff
    denom = np.sum(inv_diff, axis=1, keepdims=True)
    L = inv_diff / denom  # L[k, i] = ell_i(x_g[k])

    # Integration der Lagrange-Polynome über die Gauß-Formel
    w_nc = np.dot(w_g, L)

    # Numerische Symmetrisierung zur Wahrung der theoretischen Spiegelsymmetrie
    w_nc = 0.5 * (w_nc + w_nc[::-1])

    return x_nc, w_nc


# Hauptprogramm / Vergleich Gauß vs. Newton-Cotes


def main():
    n_werte = np.arange(2, 41)

    # Gauß-Legendre Daten
    m_gauss_gemessen = []
    m_gauss_theorie = 2 * n_werte - 1
    kond_gauss = []

    # Newton-Cotes Daten
    m_nc_gemessen = []
    # Bei NC ist der Exaktheitsgrad n = N - 1, bei ungeradem N (gerader Grad n) sogar n + 1 = N
    m_nc_theorie = [N if (N % 2 != 0) else (N - 1) for N in n_werte]
    kond_nc = []
    anzahl_negative_gewichte = []

    for i, N in enumerate(n_werte):
        # Gauß
        x_g, w_g = gauss_formel(N)
        m_gauss_gemessen.append(exaktheitsgrad(x_g, w_g, tol=1e-11))
        kond_gauss.append(np.sum(np.abs(w_g)))

        # Newton-Cotes
        x_nc, w_nc = nc_formel(N)
        m_nc_gemessen.append(exaktheitsgrad(x_nc, w_nc, tol=1e-11))
        kond_nc.append(np.sum(np.abs(w_nc)))
        anzahl_negative_gewichte.append(int(np.sum(w_nc < 0)))

    # Konsolenausgabe des Vergleichs
    print("=" * 80)
    print("Aufgabe 2: Vergleich Gauß-Legendre vs. Newton-Cotes (N = 2 bis 40)")
    print("=" * 80)
    print(f"{'N':>3} | {'Gauß Theo':>9} | {'Gauß Gem':>9} | {'NC Theo':>7} | {'NC Gem':>7} | {'Kond. Gauß':>10} | {'Kond. NC':>11} | {'NC w<0':>6}")
    print("-" * 80)
    for i, N in enumerate(n_werte):
        if N in (2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 32, 35, 40):
            print(f"{N:3d} | {m_gauss_theorie[i]:9d} | {m_gauss_gemessen[i]:9d} | {m_nc_theorie[i]:7d} | {m_nc_gemessen[i]:7d} | {kond_gauss[i]:10.2f} | {kond_nc[i]:11.2e} | {anzahl_negative_gewichte[i]:6d}")
    print("=" * 80)

    # Analyse und Beantwortung der Fragen
    print("\nErkenntnisse und numerische Interpretation (Aufgabe 2):")
    print("1. Exaktheitsgrad:")
    print("   - Gauß erreicht mit N Stützstellen einen Exaktheitsgrad von 2N - 1.")
    print("   - Newton-Cotes erreicht nur N - 1 (für gerades N) bzw. N (für ungerades N).")
    print("   - Für große N bricht der gemessene Exaktheitsgrad bei Newton-Cotes völlig ein,")
    print("     weil die Rundungsfehler aufgrund katastrophaler Auslöschung die Toleranz überschreiten.")
    print("2. Stabilität und Konditionszahl (sum_i |w_i|):")
    print("   - Gauß: Alle Gewichte sind positiv -> Kondition ist konstant sum|w_i| = 2.")
    print("   - Newton-Cotes: Ab N >= 9 treten negative Gewichte auf (Runge-Phänomen).")
    print("     Die Konditionszahl explodiert exponentiell (bis > 10^7 bei N = 40).")
    print("   - Dadurch verstärken sich kleinste Rundungsfehler (eps ~ 10^(-16)) dramatisch,")
    print("     was Newton-Cotes für höhere Ordnungen unbrauchbar macht.\n")

    # Visualisierung (Ergänzung der Plots aus Aufgabe 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Plot 1: Exaktheitsgrad Vergleich
    ax1.plot(n_werte, m_gauss_theorie, "b--", alpha=0.7, label=r"Gauß (Theorie $2N-1$)")
    ax1.plot(n_werte, m_gauss_gemessen, "bo-", markersize=4, label="Gauß (gemessen)")
    ax1.plot(n_werte, m_nc_theorie, "r--", alpha=0.7, label="Newton-Cotes (Theorie)")
    ax1.plot(n_werte, m_nc_gemessen, "rs-", markersize=4, label="Newton-Cotes (gemessen)")
    ax1.set_xlabel("Anzahl Stützstellen $N$")
    ax1.set_ylabel("Exaktheitsgrad $m$")
    ax1.set_title("Exaktheitsgrad im Vergleich")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # Plot 2: Konditionszahl (logarithmisch)
    ax2.semilogy(n_werte, kond_gauss, "b.-", label=r"Gauß: $\sum |\hat{w}_i| = 2$")
    ax2.semilogy(n_werte, kond_nc, "r.-", label=r"Newton-Cotes: $\sum |\hat{w}_i|$")
    ax2.axhline(2.0, color="blue", linestyle=":", alpha=0.5)
    ax2.set_xlabel("Anzahl Stützstellen $N$")
    ax2.set_ylabel(r"Konditionszahl $\sum_i |\hat{w}_i|$ (log. Skala)")
    ax2.set_title("Konditionszahl der Quadraturformeln")
    ax2.grid(True, which="both", linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()

    fig_dir = Path(__file__).resolve().parent / "figures"
    fig_dir.mkdir(exist_ok=True)
    plt.savefig(fig_dir / "aufgabe2_vergleich.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
