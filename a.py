import numpy as np

def spectral_radius(A, method="auto", max_iter=1000, tol=1e-12, random_state=0, size_threshold=256):
    """
    Calcula el radio espectral rho(A) = max(|lambda_i|).
    
    Parámetros
    ----------
    A : array_like (n x n)
        Matriz (real o compleja).
    method : {"auto","eig","power"}
        - "eig": usa np.linalg.eigvals (exacto, O(n^3)).
        - "power": método de potencia (aprox., O(n^2 * iter)).
        - "auto": "eig" si n <= size_threshold, si no "power".
    max_iter : int
        Máximo de iteraciones para el método de potencia.
    tol : float
        Tolerancia de convergencia para el método de potencia.
    random_state : int
        Semilla para el vector inicial en el método de potencia.
    size_threshold : int
        Límite de tamaño para decidir en "auto".
        
    Retorna
    -------
    rho : float
        Estimación (o valor) del radio espectral.
    info : dict
        Metadatos: {"method":..., "converged":..., "iters":..., "estimate_history":...}
    """
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A debe ser cuadrada (n x n).")
    if not np.all(np.isfinite(A)):
        raise ValueError("A contiene NaN/Inf.")
    n = A.shape[0]

    # Selección de método
    if method == "auto":
        method = "eig" if n <= size_threshold else "power"

    if method == "eig":
        # Eigenvalores (complejos en general)
        lambdas = np.linalg.eigvals(A)
        rho = float(np.max(np.abs(lambdas)))
        info = {"method": "eig", "converged": True, "iters": 0, "estimate_history": [rho]}
        return rho, info

    elif method == "power":
        # Método de potencia sobre A (apunta al autovalor de mayor |lambda|)
        # Nota: para matrices muy no normales puede converger lento.
        rng = np.random.default_rng(random_state)
        x = rng.standard_normal(n) + 1j * rng.standard_normal(n) if np.iscomplexobj(A) else rng.standard_normal(n)
        x /= np.linalg.norm(x) or 1.0

        history = []
        lam_old = None
        converged = False

        for k in range(1, max_iter + 1):
            y = A @ x
            normy = np.linalg.norm(y)
            if normy == 0:
                # Vector cayó en el subespacio nulo → radio espectral es 0
                rho = 0.0
                history.append(rho)
                converged = True
                iters = k
                break
            x = y / normy
            # Estimación por cociente de Rayleigh
            lam = (x.conj().T @ (A @ x)).item()
            rho_k = float(abs(lam))
            history.append(rho_k)

            if lam_old is not None and abs(rho_k - abs(lam_old)) < tol * max(1.0, rho_k):
                converged = True
                iters = k
                break
            lam_old = lam

        else:
            # no break
            iters = max_iter

        rho = history[-1]
        info = {"method": "power", "converged": converged, "iters": iters, "estimate_history": history}
        return rho, info

    else:
        raise ValueError("method debe ser 'auto', 'eig' o 'power'.")


# --- Ejemplos de uso ---
if __name__ == "__main__":
    A = np.array([[2, -1, 0],
                  [1,  3, 0],
                  [0,  0, 0.5]], dtype=float)

    rho_eig, info_eig = spectral_radius(A, method="eig")
    print("rho (eig)   =", rho_eig, info_eig)

    rho_pow, info_pow = spectral_radius(A, method="power", max_iter=500, tol=1e-10)
    print("rho (power) =", rho_pow, info_pow)
