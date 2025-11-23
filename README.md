# 🌀 Navier–Stokes Fluid Simulation (Python)

This repository contains a numerical simulation of an incompressible 2D fluid flowing through a domain with obstacles, using the Navier–Stokes equations discretized with finite differences.  
The project includes:

- A **nonlinear solver** (Newton–Raphson)
- Multiple **iterative linear solvers** (Jacobi, Gauss–Seidel, Richardson, SOR, Gradient Descent, Conjugate Gradient)
- A module for **bidimensional cubic spline interpolation** to refine the solution from a coarse grid to a high-resolution grid

---

## 📁 Project Structure

```
sistema_no_lineal/
 └── newton_raphson.py          # Nonlinear solver for Navier–Stokes

sistemas_lineales/
 ├── conjugate_gradient.py
 ├── gauss_sidel.py
 ├── gradient_descent.py
 ├── jacobi.py
 ├── richardson.py
 ├── sor.py
 └── utils.py                   # Matrix and helper functions

splines/
 ├── cubic_spline.py            # Main 2D spline interpolation
 ├── cubic_spline_backup.py
 └── transposing_matrix.py

random_tests_sistemas_lineales.py
transposing_matrix.py
workshop1/
```

---

## 🔧 Installing Dependencies

### 🟣 Using Anaconda (Recommended)

You can run this entire project using **Anaconda** with an isolated environment.

### 1. Create a Conda environment with Python **3.12.7**

```bash
conda create -n navier python=3.12.7
```

### 2. Activate the environment

```bash
conda activate navier
```

### 3. Install dependencies from `requirements.txt`

```bash
pip install -r requirements.txt
```



---

## 🚀 Running the Code

### ▶️ Nonlinear Solver (Newton–Raphson)

```bash
python -m sistema_no_lineal.newton_raphson
```

---

## ▶️ Run Linear Solvers Individually

```bash
python -m sistemas_lineales.jacobi
python -m sistemas_lineales.gauss_sidel
python -m sistemas_lineales.sor
python -m sistemas_lineales.richardson
python -m sistemas_lineales.gradient_descent
python -m sistemas_lineales.conjugate_gradient
```

---

## ▶️ Run the 2D Spline Interpolation

```bash
python -m splines.cubic_spline
```

---

## ▶️ Random Linear-System Tests

```bash
python -m random_tests_sistemas_lineales
```

---

## 📌 Notes

- Modular design: solvers can be swapped or reused.
- The spline system works for general 2D grid refinement.
- Only NumPy and Matplotlib are required.

---

## 📄 License

This project is for academic and research purposes.  
You may modify or extend it freely.
