"""Aufgabe 3 - Kondition der Entzerrungsmatrix."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

N = 16


def berechne_kondition(h, N=16):
    """Berechnet kappa_2(C) = max|H_k| / min|H_k| ueber FFT."""
    H = np.fft.fft(h, N)
    betraege = np.abs(H)
    return np.max(betraege) / np.min(betraege)


def bestimme_h_aus_aufgabe1b(data_dir):
    """Echovektor h aus Datensatz 1 und 2 ermitteln."""
    d1 = np.loadtxt(data_dir / "datensatz1.csv", comments="#")
    d2 = np.loadtxt(data_dir / "datensatz2.csv", comments="#")

    # Praefix abschneiden (die ersten 4 Werte)
    x1 = d1[4:, 0] + 1j * d1[4:, 1]
    x2 = d2[4:, 0] + 1j * d2[4:, 1]

    # Piloten liegen bei k = 0, 4, 8, 12
    piloten = [0, 4, 8, 12]
    X1 = np.fft.fft(x1)
    X2 = np.fft.fft(x2)

    # Kanal an Pilotstellen berechnen und per IDFT nach Zeitbereich h ruecktransformieren
    H_piloten = X2[piloten] / X1[piloten]
    return np.fft.ifft(H_piloten).real


def main():
    data_dir = Path(__file__).resolve().parent

    # 1. h aus 1b und Kondition berechnen
    h_1b = bestimme_h_aus_aufgabe1b(data_dir)
    kappa_1b = berechne_kondition(h_1b, N)

    print("=== Aufgabe 3: Kondition der Entzerrungsmatrix ===")
    print(f"Echovektor aus Aufgabe 1b: h = {np.round(h_1b, 4)}")
    print(f"Konditionszahl kappa_2(C) : {kappa_1b:.4f}\n")

    # 2. Familie h = (1, a) untersuchen (-0.999 wegen Polstelle bei a = -1)
    a_werte = np.linspace(-0.999, 0.0, 500)
    kappas = [berechne_kondition([1.0, a], N) for a in a_werte]

    # 3. Plot erstellen und speichern
    fig_dir = data_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(a_werte, kappas, label=r"$\kappa_2(C)$", color="tab:blue", lw=2)
    plt.axvline(-1.0, color="tab:red", linestyle="--", alpha=0.7, label=r"Polstelle bei $a = -1$")
    plt.scatter([0.0], [1.0], color="tab:green", zorder=5, label=r"Idealfall $a = 0$ ($\kappa = 1$)")

    plt.title(r"Konditionszahl $\kappa_2(C)$ für Echovektor $h = (1, a)$")
    plt.xlabel("Echoparameter $a$")
    plt.ylabel(r"Konditionszahl $\kappa_2(C)$")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    plot_path = fig_dir / "aufgabe3_kondition.png"
    plt.savefig(plot_path, dpi=200)
    print(f"Plot gespeichert unter: {plot_path.name}")
    plt.show()

    # 4. Kurze Auswertung
    print("\n--- Auswertung ---")
    print(
        "1. Wann wird kappa gross?\n"
        "   Wenn min|H_k| gegen 0 geht. Bei h = (1, a) fuer a -> -1 loeschen sich\n"
        "   Direktpfad und Echo bei k=0 fast aus (H_0 = 1 + a -> 0), kappa explodiert.\n\n"
        "2. Bedeutung bei Rauschen:\n"
        "   Beim Entzerren teilen wir durch H_k (X_dach = Y/H = X + W/H).\n"
        "   Ist ein H_k winzig, wird das Rauschen W extrem verstaerkt\n"
        "   und zerstoert die Daten auf diesem Traeger."
    )


if __name__ == "__main__":
    main()
