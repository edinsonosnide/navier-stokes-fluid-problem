# montecarlo_exponencial_matplotlib.py
# Estima E[T] para T ~ Exp(lambda) con n = 50, 100, 150, 200, 250
# Imprime una tabla simple y grafica la estimación vs el valor teórico 1/lambda.

import numpy as np
import matplotlib.pyplot as plt

def main(lam: float = 5.0, base: int = 50, max_mult: int = 30, seed: int = 42):
    rng = np.random.default_rng(seed)
    ns = [base * m for m in range(1, max_mult + 1)]
    means = []

    for n in ns:
        # Exponencial con escala = 1/lam
        T = rng.exponential(scale=1/lam, size=n)
        mu_hat = T.mean()
        means.append(mu_hat)

    # ---- Imprimir resultados (tabla simple) ----
    theo = 1.0 / lam
    print(" n   E[T] Monte Carlo   1/lambda (teórico)   error absoluto")
    for n, m in zip(ns, means):
        print(f"{n:3d}      {m: .6f}            {theo: .6f}          {abs(m-theo): .6f}")

    # ---- Gráfica ----
    plt.figure()
    plt.plot(ns, means, marker='o', label='Estimación Monte Carlo')
    plt.axhline(theo, linestyle='--', label='Valor teórico 1/λ')
    plt.xlabel('Número de iteraciones (n)')
    plt.ylabel('Estimación de E[T]')
    plt.title('Convergencia de Monte Carlo para E[T] con T ~ Exp(λ)')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
