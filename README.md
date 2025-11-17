# Max3CUT_Ising

This repository provides Python implementations for formulating and simulating **Max-3-Cut** problems using **analog Ising machines**. 
It currently supports two Ising-based formulations, which are studied in the following research work:

**Related paper:**  
*“A More Convex Ising Formulation of Max-3-Cut Using Higher-Order Spin Interactions”*  
<https://arxiv.org/pdf/2508.00565v2>

---

## **Formulations Included**
- **Quadratic Ising formulation**  
  Standard quadratic construction; linear terms can optionally be rescaled for improved performance (see paper).
  
- **Higher-order Ising formulation**  
  Newly proposed approach using higher-order spin interactions.

---

## **Package Information**
- **Package name:** `max3cut_ising`  
- **Python:** ≥ 3.8  
- **Key dependencies:** `numpy`, `numba`, `pandas`, `scipy`

---

## **Installation**

You can install the package locally using `pip`:

```pip install -e .```

Quick usage
-----------
A minimal working example is available in: 

```tests/basic_usage.py```

Repository structure
--------------------
- `max3cut_ising/` — package source
- `graphs/` — sample graph instances
- `solutions/optimal_solutions.csv` — lookup for optimal Max-3-Cut solutions (i.e. the number of edges connecting vertices of equal state/color)
- `tests/` — small usage example
