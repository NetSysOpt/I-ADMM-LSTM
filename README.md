# I-ADMM-LSTM: A Learning-Based Inexact ADMM for Solving Quadratic Programs
This represitory is an implementation of the paper: A Learning-Based Inexact ADMM for Solving Quadratic Programs.
## Introduction
Quadratic programs (QPs) constitute a fundamental class of constrained optimization problems with broad applications spanning multiple disciplines, including finance, engineering, and machine learning. The development of efficient and reliable algorithms for solving QPs continues to be an important research direction due to their widespread utility. This paper focuses on solving the following convex quadratic program:

$$
    \begin{algined}
        \underset{x\in\mathbb{R}^n}{\operatorname{minimize}} & \frac{1}{2} x^{\top} Q x + p^{\top} x \\
        \text{subject to} & l \leq A x \leq u,
    \end{aligned}
$$

where the optimization variable $x \in \mathbb{R}^n$ minimizes a quadratic objective function characterized by a positive semidefinite matrix $Q \in \mathbb{S}^n_+$ and a linear term $p \in \mathbb{R}^n$. The problem constraints are linear inequalities defined by the matrix $A \in \mathbb{R}^{m \times n}$, with each constraint $i = 1,...,m$ having associated lower and upper bounds $l_i \in \mathbb{R} \cup \{-\infty\}$ and $u_i \in \mathbb{R} \cup \{+\infty\}$, respectively.

