"""Aufgabe 2 - DFT vs. FFT: Laufzeitexperiment."""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np


def dft_direkt(x):
    """Direkte DFT über Matrix-Vektor-Multiplikation in O(N^2)."""
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    F = np.exp(-2j * np.pi * k * n / N)
    return F @ x


def main():
    Ns = 2 ** np.arange(4, 13)
    t_dft = []
    t_fft = []

    print("Starte Laufzeitmessung (N = 16 bis 4096)...")
    for N in Ns:
        x = np.random.randn(N) + 1j * np.random.randn(N)

        # FFT messen (mehr Wiederholungen bei kleinem N für stabile Werte)
        reps_f = max(50, int(100_000 / N))
        t0 = time.perf_counter()
        for _ in range(reps_f):
            np.fft.fft(x)
        t_f = (time.perf_counter() - t0) / reps_f
        t_fft.append(t_f)

        # Direkte DFT messen
        reps_d = max(1, int(5000 / (N * (N / 16)))) if N <= 1024 else 2
        t0 = time.perf_counter()
        for _ in range(reps_d):
            dft_direkt(x)
        t_d = (time.perf_counter() - t0) / reps_d
        t_dft.append(t_d)

        print(f"  N = {N:4d} | DFT: {t_d * 1e3:8.3f} ms | FFT: {t_f * 1e6:7.2f} µs")

    t_dft = np.array(t_dft)
    t_fft = np.array(t_fft)

    # Empirische Steigungen im log-log-Bereich bestimmen
    steigung_dft, _ = np.polyfit(np.log(Ns), np.log(t_dft), 1)
    steigung_fft, _ = np.polyfit(np.log(Ns), np.log(t_fft), 1)
    steigung_fft_asymp, _ = np.polyfit(np.log(Ns[4:]), np.log(t_fft[4:]), 1)

    print("\n--- Empirische Steigungen ---")
    print(f"Direkte DFT : {steigung_dft:.2f}  (Theorie: 2.0 für O(N^2))")
    print(f"FFT         : {steigung_fft:.2f}  (ab N=256: {steigung_fft_asymp:.2f}, Theorie: ~1.0 für O(N log N))")
    print("Hinweis: Bei kleinen N flacht der konstante Python-Overhead die FFT-Kurve ab.")

    # Vergleich mit realen Mobilfunksystemen
    print("\n--- Mobilfunk-Vergleich (OFDM-Symbole pro Sekunde) ---")
    print(f"{'System':<10} | {'N':<5} | {'Bedarf [Symb/s]':<16} | {'Rechner DFT':<15} | {'Rechner FFT':<15}")
    print("-" * 72)

    vergleiche = [
        ("Projekt", 16, None),
        ("LTE", 2048, 14_000),
        ("5G NR", 4096, 28_000),
    ]

    for name, n_val, bedarf in vergleiche:
        idx = int(np.where(Ns == n_val)[0][0])
        symb_dft = 1.0 / t_dft[idx]
        symb_fft = 1.0 / t_fft[idx]
        bedarf_str = f"{bedarf:d}" if bedarf else "–"
        print(f"{name:<10} | {n_val:<5d} | {bedarf_str:<16} | {symb_dft:12.1f} /s | {symb_fft:12.1f} /s")

    # Plot erstellen
    plt.figure(figsize=(8, 5))
    plt.loglog(Ns, t_dft * 1e3, "o-", label=f"Direkte DFT (Steigung {steigung_dft:.2f})")
    plt.loglog(Ns, t_fft * 1e3, "s-", label=f"FFT (Steigung {steigung_fft:.2f})")
    plt.xlabel("Blocklänge N")
    plt.ylabel("Laufzeit [ms]")
    plt.title("Laufzeitvergleich: Direkte DFT vs. FFT (log-log)")
    plt.xticks(Ns, [str(n) for n in Ns])
    plt.grid(True, which="both", ls=":", lw=0.5)
    plt.legend()
    plt.tight_layout()

    fig_dir = Path(__file__).resolve().parent / "figures"
    fig_dir.mkdir(exist_ok=True)
    out_path = fig_dir / "a2_laufzeit.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot gespeichert: {out_path.relative_to(Path.cwd())}")

    plt.show()


if __name__ == "__main__":
    main()

"""
Sample Logs von MacBook air m4:

Starte Laufzeitmessung (N = 16 bis 4096)...
  N =   16 | DFT:    0.011 ms | FFT:    5.27 µs
  N =   32 | DFT:    0.032 ms | FFT:    3.35 µs
  N =   64 | DFT:    0.103 ms | FFT:    4.14 µs
  N =  128 | DFT:    0.508 ms | FFT:    4.71 µs
  N =  256 | DFT:    1.604 ms | FFT:    5.36 µs
  N =  512 | DFT:    6.520 ms | FFT:    7.36 µs
  N = 1024 | DFT:   31.030 ms | FFT:   10.87 µs
  N = 2048 | DFT:  105.731 ms | FFT:   19.51 µs
  N = 4096 | DFT:  426.626 ms | FFT:   37.79 µs

--- Empirische Steigungen ---
Direkte DFT : 1.94  (Theorie: 2.0 für O(N^2))
FFT         : 0.37  (ab N=256: 0.70, Theorie: ~1.0 für O(N log N))
Hinweis: Bei kleinen N flacht der konstante Python-Overhead die FFT-Kurve ab.

--- Mobilfunk-Vergleich (OFDM-Symbole pro Sekunde) ---
System     | N     | Bedarf [Symb/s]  | Rechner DFT     | Rechner FFT    
------------------------------------------------------------------------
Projekt    | 16    | –                |      94289.5 /s |     189774.0 /s
LTE        | 2048  | 14000            |          9.5 /s |      51253.6 /s
5G NR      | 4096  | 28000            |          2.3 /s |      26460.3 /s
"""
