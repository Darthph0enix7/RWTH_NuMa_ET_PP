"""Aufgabe 1 - Kanalschaetzung und Entzerrung

a) datensatz1.csv: Praefix weg, FFT, Pilotwerte bei k = 0, 4, 8, 12 pruefen
b) datensatz2.csv: Echovektor h (Laenge 4) ueber die Piloten bestimmen
c) datensatz3.csv: mit h entzerren und die 16 Symbole plotten

Piloten: k = 0, 4, 8, 12  ->  X = -3-3j, 1+1j, -1-1j, 3+3j  (pilotsymbole.csv)
Laden: zwei Spalten "re im", erste Zeile Kommentar.
"""

import numpy as np
import matplotlib.pyplot as plt

N, CP = 16, 4

if __name__ == "__main__":

    print("\n---Aufgabe 1a: datensatz1.csv einlesen, Präfix entfernen, FFT, Pilotwerte prüfen--- \n")
    # Datei einlesen 
    data = np.loadtxt('datensatz1.csv')

    # Komplexes Signal aus den beiden Spalten erstellen
    signal = data[:, 0] + 1j * data[:, 1]

    # Präfix entfernen
    ofdm_symbol = signal[CP:]

    print(f"Anzahl der Werte nach Präfix-Entfernung: {len(ofdm_symbol)} \n")

    # FFT auf das ofdm_symbol ohne Präfix anwenden
    X = np.fft.fft(ofdm_symbol)

    # Runden
    X_gerundet = np.round(X, 2)

    # Prüfung der bekannten Pilotwerte
    print("--- Bekannte Pilotwerte ---")
    print(f"k=0 : {X_gerundet[0]} (Erwartet: -3-3j)")
    print(f"k=4 : {X_gerundet[4]} (Erwartet:  1+1j)")
    print(f"k=8 : {X_gerundet[8]} (Erwartet: -1-1j)")
    print(f"k=12: {X_gerundet[12]} (Erwartet:  3+3j)")

    # Überprüfen aller Unterträger
    print("\n--- Alle 16 Unterträger im Überblick ---")
    for k in range(16):
        print(f"k={k:2d}: {X_gerundet[k]}")


    print("\n---Aufgabe 1b: datensatz2.csv einlesen, Echovektor h bestimmen--- \n")

    #kopierter Code für den zweiten Datensatz
    data = np.loadtxt('datensatz2.csv')

    signal = data[:, 0] + 1j * data[:, 1]
    ofdm_symbol = signal[CP:]
    X = np.fft.fft(ofdm_symbol)
    X_gerundet = np.round(X, 2)

    # Faltungssatz anwenden (H = Y / X)
    H = np.zeros(4, dtype=complex)

    H[0] = X[0]  / (-3 - 3j)
    H[1] = X[4]  / (1 + 1j)
    H[2] = X[8]  / (-1 - 1j)
    H[3] = X[12] / (3 + 3j)

    # Inverse FFT (IFFT) anwenden
    h = np.fft.ifft(H)

    print("Der unbekannte Echovektor h lautet (gerundet):")
    print(np.round(h, 2))


    print("\n---Aufgabe 1c: datensatz3.csv einlesen, mit h entzerren und die 16 Symbole plotten--- \n")

    #Datensatz 3 einlesen, Präfix entfernen, FFT anwenden
    data3 = np.loadtxt('datensatz3.csv')
    signal3 = data3[:, 0] + 1j * data3[:, 1]
    ofdm_symbol3 = signal3[4:]
    Y3 = np.fft.fft(ofdm_symbol3)

    # Kanal für alle 16 Unterträger berechnen
    H_full = np.fft.fft(h, 16)

    # Das Signal entzerren (Faltungssatz anwenden)
    X_entzerrt = Y3 / H_full

    # Die 16 übertragenen Symbole plotten
    plt.figure(figsize=(6, 6))

    plt.scatter(X_entzerrt.real, X_entzerrt.imag, color='blue', marker='x', s=100, label='Entzerrte Symbole')

    # Formateirung der Achsen und des Plots
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.grid(True, linestyle='--')
    plt.title('Entzerrte unbekannte Werte (Datensatz 3)')
    plt.xlabel('Realteil')
    plt.ylabel('Imaginärteil')
    plt.legend()
    plt.show()