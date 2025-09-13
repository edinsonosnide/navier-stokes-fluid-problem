import numpy as np
import matplotlib.pyplot as plt

# ============================================
# Volumen teórico de hiperesfera 4D (r=1)
# V = (pi^2 * r^4) / 2  -> con r=1: V = pi^2 / 2
# Estimación MC: V_hat = Vol(cubo)* (#dentro/N), cubo = [-r,r]^4
# ============================================

def volumen_hiperesfera_teorico(r: float = 1.0) -> float:
    return (np.pi**2) * (r**4) / 2.0

def estimar_volumen_mc(N: int, r: float = 1.0, rng: np.random.Generator | None = None):
    """
    Estima el volumen de la hiperesfera 4D de radio r en [-r,r]^4 usando N puntos.
    Devuelve: V_hat, dentro, p_hat
    """
    if rng is None:
        rng = np.random.default_rng()

    # Muestreo uniforme en el hipercubo [-r, r]^4
    X = rng.uniform(-r, r, size=(N, 4))

    # Puntos dentro de la hiperesfera: x1^2 + x2^2 + x3^2 + x4^2 <= r^2
    radios2 = np.sum(X**2, axis=1)
    dentro_mask = radios2 <= (r**2)
    dentro = int(np.sum(dentro_mask))

    p_hat = dentro / N
    volumen_cubo = (2.0 * r) ** 4
    V_hat = volumen_cubo * p_hat
    return V_hat, dentro, p_hat

def main():
    r = 1.0
    V_teo = volumen_hiperesfera_teorico(r)

    # Tamaños de simulación a evaluar (ajusta si deseas)
    Ns = [10**k for k in range(2, 6)]  # 1e2, 1e3, 1e4, 1e5
    rng = np.random.default_rng(12345)

    resultados = []
    for N in Ns:
        V_hat, dentro, p_hat = estimar_volumen_mc(N, r=r, rng=rng)
        err_abs = abs(V_hat - V_teo)
        err_rel = err_abs / V_teo
        resultados.append({
            "N": N,
            "V_hat": V_hat,
            "dentro": dentro,
            "p_hat": p_hat,
            "error_abs": err_abs,
            "error_rel": err_rel
        })

    # Tabla en consola
    print(f"Volumen teórico (r={r}): V = pi^2 r^4 / 2 = {V_teo:.6f}")
    print("-" * 72)
    print(f"{'N':>10}  {'V_hat':>12}  {'#dentro':>10}  {'p_hat':>9}  {'err_abs':>10}  {'err_rel':>10}")
    for row in resultados:
        print(f"{row['N']:>10}  {row['V_hat']:>12.6f}  {row['dentro']:>10d}  "
              f"{row['p_hat']:>9.6f}  {row['error_abs']:>10.6f}  {row['error_rel']:>10.6f}")

    # -------- Gráfica 1: Volumen estimado vs N --------
    Ns_arr = np.array([row["N"] for row in resultados])
    Vh_arr = np.array([row["V_hat"] for row in resultados])

    plt.figure()
    plt.plot(Ns_arr, Vh_arr, marker='o', linewidth=2, label="Estimación MC")
    plt.hlines(V_teo, Ns_arr[0], Ns_arr[-1], linestyles='--', label="Teórico")
    plt.xscale('log')
    plt.xlabel("Número de simulaciones N (escala log)")
    plt.ylabel("Volumen estimado")
    plt.title("Volumen de hiperesfera 4D (r=1): estimación vs N")
    plt.grid(True)
    plt.legend()
    plt.show()

    # -------- Gráfica 2: Error relativo vs N --------
    err_rel_arr = np.array([row["error_rel"] for row in resultados])

    plt.figure()
    plt.plot(Ns_arr, err_rel_arr, marker='o', linewidth=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Número de simulaciones N (escala log)")
    plt.ylabel("Error relativo |V̂ - V| / V")
    plt.title("Precisión de Monte Carlo vs N (log-log)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
