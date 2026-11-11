import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ==============================================================================
# HILFSFUNKTIONEN
# ==============================================================================

def baryzentrische_gewichte(x_nodes):
    n = len(x_nodes)
    w = np.ones(n)
    for j in range(n):
        diffs = x_nodes[j] - x_nodes
        diffs[j] = 1.0  
        w[j] = 1.0 / np.prod(diffs)
    return w

def baryzentrische_interpolation(x_nodes, y_nodes, w, x_eval):
    diff = x_eval[:, None] - x_nodes[None, :]
    terme = w / diff
    zaehler = np.sum(terme * y_nodes, axis=1)
    nenner = np.sum(terme, axis=1)
    return zaehler / nenner

def berechne_lebesgue_konstante(x_nodes, w, x_eval):
    diff = x_eval[:, None] - x_nodes[None, :]
    terme = w / diff
    nenner = np.sum(terme, axis=1)
    l_basis = terme / nenner[:, None]
    lambda_x = np.sum(np.abs(l_basis), axis=1)
    return np.max(lambda_x)

def vandermonde_matrix(x):
    n = len(x) - 1
    V = np.zeros((len(x), n + 1))
    for j in range(n + 1):
        V[:, j] = x**j
    return V

def horner_schema(a, x_eval):
    p = np.zeros_like(x_eval, dtype=float)
    for i in range(len(a) - 1, -1, -1):
        p = p * x_eval + a[i]
    return p


# ==============================================================================
# HAUPTPROGRAMM
# ==============================================================================

def main():
    # --- PFADE AUTOMATISCH FINDEN ---
    basis_ordner = os.path.dirname(os.path.abspath(__file__))
    daten_ordner = os.path.join(basis_ordner, "Datensätze-20261110")
    if not os.path.exists(daten_ordner):
        daten_ordner = basis_ordner

    # --- GRUNDDATEN EINLESEN ---
    pruefraster = pd.read_csv(os.path.join(daten_ordner, "pruefraster.csv"), comment="#", sep=r'\s+', names=["E_mV", "T_exakt_C"])
    E_pruef = pruefraster["E_mV"].values
    T_exakt_pruef = pruefraster["T_exakt_C"].values

    # KORRIGIERTE SPALTENREIHENFOLGE: Erst T_exakt_C, dann T_kalibrator_C!
    spalten_stuetz = ["n", "i", "E_mV", "T_exakt_C", "T_kalibrator_C"]
    df_aequi = pd.read_csv(os.path.join(daten_ordner, "stuetzwerte_aequidistant.csv"), comment="#", sep=r'\s+', names=spalten_stuetz)
    df_tscheby = pd.read_csv(os.path.join(daten_ordner, "stuetzwerte_tschebyscheff.csv"), comment="#", sep=r'\s+', names=spalten_stuetz)

    N_werte = np.arange(2, 41)


    # ==========================================================================
    # TEIL 1: AUFGABE 3a - DER INTERPOLATIONSFEHLER (Nur exakte Daten)
    # ==========================================================================
    e_interp_aequi = []
    e_interp_tscheby = []

    for n in N_werte:
        # 1. Äquidistante Stützstellen
        daten_aequi = df_aequi[df_aequi["n"] == n]
        x_aequi = daten_aequi["E_mV"].values
        y_exakt_aequi = daten_aequi["T_exakt_C"].values 
        
        w_aequi = baryzentrische_gewichte(x_aequi)
        p_exakt_aequi = baryzentrische_interpolation(x_aequi, y_exakt_aequi, w_aequi, E_pruef)
        e_interp_aequi.append(np.max(np.abs(p_exakt_aequi - T_exakt_pruef)))

        # 2. Tschebyscheff-Stützstellen
        daten_tscheby = df_tscheby[df_tscheby["n"] == n]
        x_tscheby = daten_tscheby["E_mV"].values
        y_exakt_tscheby = daten_tscheby["T_exakt_C"].values 
        
        w_tscheby = baryzentrische_gewichte(x_tscheby)
        p_exakt_tscheby = baryzentrische_interpolation(x_tscheby, y_exakt_tscheby, w_tscheby, E_pruef)
        e_interp_tscheby.append(np.max(np.abs(p_exakt_tscheby - T_exakt_pruef)))

    plt.figure(1, figsize=(9, 5))
    plt.semilogy(N_werte, e_interp_aequi, 'o-', label="Äquidistant")
    plt.semilogy(N_werte, e_interp_tscheby, 's-', label="Tschebyscheff")
    plt.title("Aufgabe 3a: Äquidistant vs. Tschebyscheff")
    plt.xlabel("Polynomgrad $n$")
    plt.ylabel("Interpolationsfehler (log.)")
    plt.grid(True, which="both", linestyle="-", alpha=0.5)
    plt.legend()
    plt.tight_layout()


    # ==========================================================================
    # TEIL 2: AUFGABE 3b - DER DATENFEHLER
    # ==========================================================================
    e_daten_aequi, schranke_aequi = [], []
    e_daten_tscheby, schranke_tscheby = [], []

    for n in N_werte:
        # 1. Äquidistante Stützstellen
        daten_aequi = df_aequi[df_aequi["n"] == n]
        x_aequi = daten_aequi["E_mV"].values
        y_exakt_aequi = daten_aequi["T_exakt_C"].values
        y_kali_aequi = daten_aequi["T_kalibrator_C"].values
        
        w_aequi = baryzentrische_gewichte(x_aequi)
        p_exakt_aequi = baryzentrische_interpolation(x_aequi, y_exakt_aequi, w_aequi, E_pruef)
        p_kali_aequi = baryzentrische_interpolation(x_aequi, y_kali_aequi, w_aequi, E_pruef)
        
        e_daten_aequi.append(np.max(np.abs(p_kali_aequi - p_exakt_aequi)))
        schranke_aequi.append(berechne_lebesgue_konstante(x_aequi, w_aequi, E_pruef) * 0.005)

        # 2. Tschebyscheff-Stützstellen
        daten_tscheby = df_tscheby[df_tscheby["n"] == n]
        x_tscheby = daten_tscheby["E_mV"].values
        y_exakt_tscheby = daten_tscheby["T_exakt_C"].values
        y_kali_tscheby = daten_tscheby["T_kalibrator_C"].values
        
        w_tscheby = baryzentrische_gewichte(x_tscheby)
        p_exakt_tscheby = baryzentrische_interpolation(x_tscheby, y_exakt_tscheby, w_tscheby, E_pruef)
        p_kali_tscheby = baryzentrische_interpolation(x_tscheby, y_kali_tscheby, w_tscheby, E_pruef)
        
        e_daten_tscheby.append(np.max(np.abs(p_kali_tscheby - p_exakt_tscheby)))
        schranke_tscheby.append(berechne_lebesgue_konstante(x_tscheby, w_tscheby, E_pruef) * 0.005)

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.semilogy(N_werte, e_interp_aequi, 'o-', label=r"$e_{interp}$ (aus 3a)")
    ax1.semilogy(N_werte, e_daten_aequi, 's-', label=r"$e_{daten}$ (Datenfehler)")
    ax1.semilogy(N_werte, schranke_aequi, 'k--', label=r"Schranke $\Lambda_n \cdot 0,005$")
    ax1.set_title("Aufgabe 3b: Datenfehler (Äquidistant)")
    ax1.set_xlabel("Polynomgrad $n$")
    ax1.set_ylabel("Fehler [°C]")
    ax1.grid(True, which="both", linestyle="--", alpha=0.7)
    ax1.legend()
    
    ax2.semilogy(N_werte, e_interp_tscheby, 'o-', label=r"$e_{interp}$ (aus 3a)")
    ax2.semilogy(N_werte, e_daten_tscheby, 's-', label=r"$e_{daten}$ (Datenfehler)")
    ax2.semilogy(N_werte, schranke_tscheby, 'k--', label=r"Schranke $\Lambda_n \cdot 0,005$")
    ax2.set_title("Aufgabe 3b: Datenfehler (Tschebyscheff)")
    ax2.set_xlabel("Polynomgrad $n$")
    ax2.set_ylabel("Fehler [°C]")
    ax2.grid(True, which="both", linestyle="--", alpha=0.7)
    ax2.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()