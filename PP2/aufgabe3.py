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
    """Berechnet die Gewichte w_j für die baryzentrische Interpolation (3a & 3b)."""
    n = len(x_nodes)
    w = np.ones(n)
    for j in range(n):
        diffs = x_nodes[j] - x_nodes
        diffs[j] = 1.0  
        w[j] = 1.0 / np.prod(diffs)
    return w

def baryzentrische_interpolation(x_nodes, y_nodes, w, x_eval):
    """Wertet das Polynom an den 2000 Prüfpunkten x_eval aus (3a & 3b)."""
    diff = x_eval[:, None] - x_nodes[None, :]
    terme = w / diff
    zaehler = np.sum(terme * y_nodes, axis=1)
    nenner = np.sum(terme, axis=1)
    return zaehler / nenner

def berechne_lebesgue_konstante(x_nodes, w, x_eval):
    """Berechnet die Lebesgue-Konstante Lambda_n(X) für die Fehlerschranke (nur 3b)."""
    diff = x_eval[:, None] - x_nodes[None, :]
    terme = w / diff
    nenner = np.sum(terme, axis=1)
    l_basis = terme / nenner[:, None]
    lambda_x = np.sum(np.abs(l_basis), axis=1)
    return np.max(lambda_x)

def vandermonde_matrix(x):
    """Erstellt die Vandermonde-Matrix V(X) (nur 3c)."""
    n = len(x) - 1
    V = np.zeros((len(x), n + 1))
    for j in range(n + 1):
        V[:, j] = x**j
    return V

def horner_schema(a, x_eval):
    """Wertet ein Monom-Polynom mit Koeffizienten a aus (nur 3c)."""
    p = np.zeros_like(x_eval, dtype=float)
    for i in range(len(a) - 1, -1, -1):
        p = p * x_eval + a[i]
    return p


# ==============================================================================
# HAUPTPROGRAMM
# ==============================================================================

def main():

    basis_ordner = os.path.dirname(os.path.abspath(__file__))
    daten_ordner = os.path.join(basis_ordner, "Datensätze-20261110")
    if not os.path.exists(daten_ordner):
        daten_ordner = basis_ordner

    # --- GRUNDDATEN EINLESEN ---
    pruefraster = pd.read_csv(os.path.join(daten_ordner, "pruefraster.csv"), comment="#", sep=r'\s+', names=["E_mV", "T_exakt_C"])
    E_pruef = pruefraster["E_mV"].values
    T_exakt_pruef = pruefraster["T_exakt_C"].values

    spalten_stuetz = ["n", "i", "E_mV", "T_kalibrator_C", "T_exakt_C"]
    df_aequi = pd.read_csv(os.path.join(daten_ordner, "stuetzwerte_aequidistant.csv"), comment="#", sep=r'\s+', names=spalten_stuetz)
    df_tscheby = pd.read_csv(os.path.join(daten_ordner, "stuetzwerte_tschebyscheff.csv"), comment="#", sep=r'\s+', names=spalten_stuetz)
    N_werte = np.arange(2, 41)


    # ==========================================================================
    # TEIL 1: AUFGABE 3a - DER INTERPOLATIONSFEHLER (Nur exakte Daten)
    # ==========================================================================
    e_interp_aequi = []
    e_interp_tscheby = []

    for n in N_werte:
        # Äquidistante Stützstellen (exakte Temperaturen T_exakt_C)
        daten_aequi = df_aequi[df_aequi["n"] == n]
        x_aequi = daten_aequi["E_mV"].values
        y_exakt_aequi = daten_aequi["T_exakt_C"].values 
        
        w_aequi = baryzentrische_gewichte(x_aequi)
        p_exakt_aequi = baryzentrische_interpolation(x_aequi, y_exakt_aequi, w_aequi, E_pruef)
        
        # Größte Abweichung zur realen Kurve T_exakt_pruef speichern
        fehler_a = np.max(np.abs(p_exakt_aequi - T_exakt_pruef))
        e_interp_aequi.append(fehler_a)

        # Tschebyscheff-Stützstellen (exakte Temperaturen T_exakt_C)
        daten_tscheby = df_tscheby[df_tscheby["n"] == n]
        x_tscheby = daten_tscheby["E_mV"].values
        y_exakt_tscheby = daten_tscheby["T_exakt_C"].values 
        
        w_tscheby = baryzentrische_gewichte(x_tscheby)
        p_exakt_tscheby = baryzentrische_interpolation(x_tscheby, y_exakt_tscheby, w_tscheby, E_pruef)
        
        fehler_t = np.max(np.abs(p_exakt_tscheby - T_exakt_pruef))
        e_interp_tscheby.append(fehler_t)

    # --- Plot für Aufgabe 3a ---
    plt.figure(1, figsize=(9, 5))
    plt.semilogy(N_werte, e_interp_aequi, 'o-', label="Äquidistante Stützstellen")
    plt.semilogy(N_werte, e_interp_tscheby, 's-', label="Tschebyscheff-Stützstellen")
    plt.axhline(0.1, color='red', linestyle='--', label=r"Ziel: $e_{max} < 0,1^\circ C$")
    plt.title("Aufgabe 3a: Interpolationsfehler $e_{interp}(n)$")
    plt.xlabel("Polynomgrad $n$")
    plt.ylabel("Fehler [°C]")
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()


    # ==========================================================================
    # TEIL 2: AUFGABE 3b - DER DATENFEHLER (Exakt vs. Gerundeter Kalibrator)
    # ==========================================================================
    e_daten_aequi = []
    schranke_aequi = []
    
    e_daten_tscheby = []
    schranke_tscheby = []

    for n in N_werte:
        # Äquidistante Stützstellen
        daten_aequi = df_aequi[df_aequi["n"] == n]
        x_aequi = daten_aequi["E_mV"].values
        y_exakt_aequi = daten_aequi["T_exakt_C"].values
        y_kali_aequi = daten_aequi["T_kalibrator_C"].values  # Gerundete Messwerte!
        
        w_aequi = baryzentrische_gewichte(x_aequi)
        p_exakt_aequi = baryzentrische_interpolation(x_aequi, y_exakt_aequi, w_aequi, E_pruef)
        p_kali_aequi = baryzentrische_interpolation(x_aequi, y_kali_aequi, w_aequi, E_pruef)
        
        # Datenfehler = Unterschied zwischen Kalibrator-Polynom und exaktem Polynom
        e_daten_aequi.append(np.max(np.abs(p_kali_aequi - p_exakt_aequi)))
        
        # Theoretische Schranke = Lebesgue-Konstante * 0.005 °C
        lambda_aequi = berechne_lebesgue_konstante(x_aequi, w_aequi, E_pruef)
        schranke_aequi.append(lambda_aequi * 0.005)

        # Tschebyscheff-Stützstellen
        daten_tscheby = df_tscheby[df_tscheby["n"] == n]
        x_tscheby = daten_tscheby["E_mV"].values
        y_exakt_tscheby = daten_tscheby["T_exakt_C"].values
        y_kali_tscheby = daten_tscheby["T_kalibrator_C"].values  # Gerundete Messwerte!
        
        w_tscheby = baryzentrische_gewichte(x_tscheby)
        p_exakt_tscheby = baryzentrische_interpolation(x_tscheby, y_exakt_tscheby, w_tscheby, E_pruef)
        p_kali_tscheby = baryzentrische_interpolation(x_tscheby, y_kali_tscheby, w_tscheby, E_pruef)
        
        e_daten_tscheby.append(np.max(np.abs(p_kali_tscheby - p_exakt_tscheby)))
        
        lambda_tscheby = berechne_lebesgue_konstante(x_tscheby, w_tscheby, E_pruef)
        schranke_tscheby.append(lambda_tscheby * 0.005)

    # --- Plots für Aufgabe 3b (nutzt zusätzlich e_interp aus 3a zum Vergleich) ---
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


    # ==========================================================================
    # AUFGABE 3c - KOEFFIZIENTENFEHLER (Vandermonde vs. Baryzentrisch)
    # ==========================================================================
    rel_koeff_aenderung = []
    e_koeff_liste = []
    kondition_V = []

    for n in N_werte:
        # Aufgabe 3c wird laut Blatt nur für Tschebyscheff gemacht
        daten_tscheby = df_tscheby[df_tscheby["n"] == n]
        x_tscheby = daten_tscheby["E_mV"].values
        y_exakt_tscheby = daten_tscheby["T_exakt_C"].values 
        y_kali_tscheby = daten_tscheby["T_kalibrator_C"].values 
        
        # Referenz-Polynom (baryzentrisch mit Kalibratorwerten)
        w_tscheby = baryzentrische_gewichte(x_tscheby)
        p_bary_kali = baryzentrische_interpolation(x_tscheby, y_kali_tscheby, w_tscheby, E_pruef)

        # Vandermonde-Matrix aufstellen und Konditionszahl berechnen
        V = vandermonde_matrix(x_tscheby)
        kondition_V.append(np.linalg.cond(V, np.inf))
        
        try:
            # Gleichungssystem V*a = y lösen (einmal exakt, einmal Kalibrator)
            a = np.linalg.solve(V, y_exakt_tscheby)
            a_tilde = np.linalg.solve(V, y_kali_tscheby)
            
            # Relative Änderung der Koeffizienten: ||a_tilde - a||_inf / ||a||_inf
            rel_aenderung = np.max(np.abs(a_tilde - a)) / np.max(np.abs(a))
            rel_koeff_aenderung.append(rel_aenderung)
            
            # Vandermonde-Polynom auswerten und mit Baryzentrisch vergleichen
            p_vander_kali = horner_schema(a_tilde, E_pruef)
            e_koeff = np.max(np.abs(p_vander_kali - p_bary_kali))
            e_koeff_liste.append(e_koeff)
            
        except np.linalg.LinAlgError:
            rel_koeff_aenderung.append(np.nan)
            e_koeff_liste.append(np.nan)

    # --- Plots für Aufgabe 3c ---
    fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax3.semilogy(N_werte, rel_koeff_aenderung, 'm^-', label=r"Rel. Änderung $||\tilde{a}-a||_\infty / ||a||_\infty$")
    ax3.set_title("Aufgabe 3c: Instabilität der Koeffizienten")
    ax3.set_xlabel("Polynomgrad $n$")
    ax3.grid(True, which="both", linestyle="--", alpha=0.7)
    ax3.legend()
    
    ax4.semilogy(N_werte, e_koeff_liste, 'rD-', label=r"$e_{koeff}$ (Polynomfehler)")
    ax4.set_xlabel("Polynomgrad $n$")
    ax4.set_ylabel("Fehler [°C]", color='red')
    ax4.tick_params(axis='y', labelcolor='red')
    ax4.grid(True, which="both", linestyle="--", alpha=0.7)
    
    ax4_twin = ax4.twinx()
    ax4_twin.semilogy(N_werte, kondition_V, 'k--', label=r"Kondition $\kappa_\infty(V(X))$")
    ax4_twin.set_ylabel("Konditionszahl", color='black')
    
    lines_1, labels_1 = ax4.get_legend_handles_labels()
    lines_2, labels_2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
    plt.title("Aufgabe 3c: Koeffizientenfehler & Konditionszahl")
    plt.tight_layout()

    # Alle Fenster anzeigen
    plt.show()

if __name__ == "__main__":
    main()