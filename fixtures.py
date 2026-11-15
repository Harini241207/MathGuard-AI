"""Realistic AI-written solutions used by the parser / verifier tests.

They are hand-written in the styles that ChatGPT, Gemini and Claude typically use
(Markdown pipe tables, LaTeX arrays, plain text), with deliberate mistakes where
the name says so."""

S01_Q = "Maximize Z = 3x + 5y subject to x + y <= 4, x <= 2, y <= 3, x, y >= 0"

S01_CORRECT_MD = """**Step 1: Convert to standard form**

Maximize Z = 3x + 5y + 0s1 + 0s2 + 0s3
subject to
x + y + s1 = 4
x + s2 = 2
y + s3 = 3

**Initial Simplex Tableau**

| Basis | C_B | x | y | s1 | s2 | s3 | RHS | Ratio |
|---|---|---|---|---|---|---|---|---|
| s1 | 0 | 1 | 1 | 1 | 0 | 0 | 4 | 4/1 = 4 |
| s2 | 0 | 1 | 0 | 0 | 1 | 0 | 2 | — |
| s3 | 0 | 0 | 1 | 0 | 0 | 1 | 3 | 3/1 = 3 |
| Z_j | | 0 | 0 | 0 | 0 | 0 | 0 | |
| C_j − Z_j | | 3 | 5 | 0 | 0 | 0 | | |

The most positive Cj − Zj is 5, so **y is the entering variable**. The minimum ratio is 3, so **s3 is the leaving variable**. Pivot element = 1.

Row operations: R3 → R3, R1 → R1 − R3.

**Iteration 1**

| Basis | C_B | x | y | s1 | s2 | s3 | RHS | Ratio |
|---|---|---|---|---|---|---|---|---|
| s1 | 0 | 1 | 0 | 1 | 0 | -1 | 1 | 1 |
| s2 | 0 | 1 | 0 | 0 | 1 | 0 | 2 | 2 |
| y | 5 | 0 | 1 | 0 | 0 | 1 | 3 | — |
| Z_j | | 0 | 5 | 0 | 0 | 5 | 15 | |
| C_j − Z_j | | 3 | 0 | 0 | 0 | -5 | | |

x enters (Cj − Zj = 3) and s1 leaves (minimum ratio 1). Pivot element = 1.
R2 → R2 − R1

**Iteration 2**

| Basis | C_B | x | y | s1 | s2 | s3 | RHS |
|---|---|---|---|---|---|---|---|
| x | 3 | 1 | 0 | 1 | 0 | -1 | 1 |
| s2 | 0 | 0 | 0 | -1 | 1 | 1 | 1 |
| y | 5 | 0 | 1 | 0 | 0 | 1 | 3 |
| Z_j | | 3 | 5 | 3 | 0 | 2 | 18 |
| C_j − Z_j | | 0 | 0 | -3 | 0 | -2 | |

All Cj − Zj ≤ 0, so the optimal solution is reached.

**Final answer:** x = 1, y = 3, maximum Z = 18.
"""

# same working, but the RHS of row s2 in iteration 2 is wrong (2 - 1 = 1 written as 3) and Z is reported wrongly
S01_ROWOP_ERROR = S01_CORRECT_MD.replace("| s2 | 0 | 0 | 0 | -1 | 1 | 1 | 1 |", "| s2 | 0 | 0 | 0 | -1 | 1 | 1 | 3 |")

# wrong entering variable (x chosen although y has the larger Cj - Zj) - still a valid improving path
S01_WRONG_ENTERING = """Initial tableau:

| Basis | CB | x | y | s1 | s2 | s3 | RHS |
|---|---|---|---|---|---|---|---|
| s1 | 0 | 1 | 1 | 1 | 0 | 0 | 4 |
| s2 | 0 | 1 | 0 | 0 | 1 | 0 | 2 |
| s3 | 0 | 0 | 1 | 0 | 0 | 1 | 3 |
| Cj-Zj | | 3 | 5 | 0 | 0 | 0 | |

Entering variable: x. Ratios: s1: 4/1 = 4, s2: 2/1 = 2. Leaving variable: s2.

| Basis | CB | x | y | s1 | s2 | s3 | RHS |
|---|---|---|---|---|---|---|---|
| s1 | 0 | 0 | 1 | 1 | -1 | 0 | 2 |
| x | 3 | 1 | 0 | 0 | 1 | 0 | 2 |
| s3 | 0 | 0 | 1 | 0 | 0 | 1 | 3 |
| Cj-Zj | | 0 | 5 | 0 | -3 | 0 | |

Entering variable: y. Leaving variable: s1 (ratio 2/1 = 2).

| Basis | CB | x | y | s1 | s2 | s3 | RHS |
|---|---|---|---|---|---|---|---|
| y | 5 | 0 | 1 | 1 | -1 | 0 | 2 |
| x | 3 | 1 | 0 | 0 | 1 | 0 | 2 |
| s3 | 0 | 0 | 0 | -1 | 1 | 1 | 1 |
| Cj-Zj | | 0 | 0 | -5 | 2 | 0 | |

Entering variable: s2. Leaving variable: s3.

| Basis | CB | x | y | s1 | s2 | s3 | RHS |
|---|---|---|---|---|---|---|---|
| y | 5 | 0 | 1 | 0 | 0 | 1 | 3 |
| x | 3 | 1 | 0 | 1 | 0 | -1 | 1 |
| s2 | 0 | 0 | 0 | -1 | 1 | 1 | 1 |
| Cj-Zj | | 0 | 0 | -3 | 0 | -2 | |

Optimal: x = 1, y = 3, Z = 18.
"""

S01_LATEX = r"""We introduce slack variables $s_1, s_2, s_3$.

$$
\begin{array}{c|c|ccccc|c}
\text{Basis} & C_B & x & y & s_1 & s_2 & s_3 & \text{RHS} \\ \hline
s_1 & 0 & 1 & 1 & 1 & 0 & 0 & 4 \\
s_2 & 0 & 1 & 0 & 0 & 1 & 0 & 2 \\
s_3 & 0 & 0 & 1 & 0 & 0 & 1 & 3 \\ \hline
C_j - Z_j & & 3 & 5 & 0 & 0 & 0 & 
\end{array}
$$

$y$ enters, $s_3$ leaves (ratios $4/1 = 4$ and $3/1 = 3$), pivot element $= 1$.

$$
\begin{array}{c|c|ccccc|c}
\text{Basis} & C_B & x & y & s_1 & s_2 & s_3 & \text{RHS} \\ \hline
s_1 & 0 & 1 & 0 & 1 & 0 & -1 & 1 \\
s_2 & 0 & 1 & 0 & 0 & 1 & 0 & 2 \\
y & 5 & 0 & 1 & 0 & 0 & 1 & 3 \\ \hline
C_j - Z_j & & 3 & 0 & 0 & 0 & -5 &
\end{array}
$$

$x$ enters and $s_1$ leaves.

$$
\begin{array}{c|c|ccccc|c}
\text{Basis} & C_B & x & y & s_1 & s_2 & s_3 & \text{RHS} \\ \hline
x & 3 & 1 & 0 & 1 & 0 & -1 & 1 \\
s_2 & 0 & 0 & 0 & -1 & 1 & 1 & 1 \\
y & 5 & 0 & 1 & 0 & 0 & 1 & 3 \\ \hline
C_j - Z_j & & 0 & 0 & -3 & 0 & -2 &
\end{array}
$$

All $C_j - Z_j \le 0$, so the solution is optimal: $x = 1$, $y = 3$, $Z_{max} = 18$.
"""

S01_PLAIN = """Initial table
Basis  CB   x   y   s1  s2  s3  RHS
s1     0    1   1   1   0   0   4
s2     0    1   0   0   1   0   2
s3     0    0   1   0   0   1   3
Cj-Zj       3   5   0   0   0

y enters, s3 leaves.

Basis  CB   x   y   s1  s2  s3  RHS
s1     0    1   0   1   0   -1  1
s2     0    1   0   0   1   0   2
y      5    0   1   0   0   1   3
Cj-Zj       3   0   0   0   -5

x enters, s1 leaves.

Basis  CB   x   y   s1  s2  s3  RHS
x      3    1   0   1   0   -1  1
s2     0    0   0   -1  1   1   1
y      5    0   1   0   0   1   3
Cj-Zj       0   0   -3  0   -2

Optimal solution x = 1, y = 3, Z = 18.
"""

G01_Q = "Maximize Z = 3x + 2y subject to x + y <= 4, x <= 2, y <= 3, x, y >= 0"

G01_CORRECT = """Boundary lines: x + y = 4 meets the axes at (4, 0) and (0, 4); x = 2; y = 3.
Intersection of x + y = 4 and x = 2: solving gives (2, 2).
Intersection of x + y = 4 and y = 3: (1, 3).
Corner points of the feasible region: (0, 0), (2, 0), (2, 2), (1, 3), (0, 3).

| Corner | (x, y) | Z = 3x + 2y |
|---|---|---|
| O | (0, 0) | 0 |
| A | (2, 0) | 6 |
| B | (2, 2) | 10 |
| C | (1, 3) | 9 |
| D | (0, 3) | 6 |

Therefore the maximum is Z = 10 at x = 2, y = 2.
"""

G01_WRONG = """Boundary lines: x + y = 4 meets the axes at (4, 0) and (0, 5); x = 2; y = 3.
Intersection of x + y = 4 and x = 2: (2, 3).
Corner points: (0, 0), (2, 0), (2, 3), (1, 3), (0, 3).
At (0,0): Z = 0. At (2,0): Z = 6. At (2,3): Z = 3(2) + 2(3) = 12. At (1,3): Z = 9. At (0,3): Z = 6.
Therefore x = 2, y = 3, Z = 12.
"""

MIN_GE_Q = "Minimize Z = 2x + 3y subject to x + y >= 4, x + 2y >= 6, x, y >= 0"

BIGM_CORRECT = """Standard form: x + y - s1 + A1 = 4, x + 2y - s2 + A2 = 6.
Minimize Z = 2x + 3y + 0s1 + 0s2 + MA1 + MA2.

| Cj | | 2 | 3 | 0 | 0 | M | M | |
|---|---|---|---|---|---|---|---|---|
| Basis | CB | x | y | s1 | s2 | A1 | A2 | RHS |
| A1 | M | 1 | 1 | -1 | 0 | 1 | 0 | 4 |
| A2 | M | 1 | 2 | 0 | -1 | 0 | 1 | 6 |
| Zj | | 2M | 3M | -M | -M | M | M | 10M |
| Cj - Zj | | 2 - 2M | 3 - 3M | M | M | 0 | 0 | |

y enters (most negative Cj - Zj = 3 - 3M). Ratios: 4/1 = 4, 6/2 = 3, so A2 leaves. Pivot element = 2.

| Basis | CB | x | y | s1 | s2 | A1 | A2 | RHS |
|---|---|---|---|---|---|---|---|---|
| A1 | M | 1/2 | 0 | -1 | 1/2 | 1 | -1/2 | 1 |
| y | 3 | 1/2 | 1 | 0 | -1/2 | 0 | 1/2 | 3 |
| Zj | | M/2 + 3/2 | 3 | -M | M/2 - 3/2 | M | -M/2 + 3/2 | M + 9 |
| Cj - Zj | | 1/2 - M/2 | 0 | M | 3/2 - M/2 | 0 | 3M/2 - 3/2 | |

x enters, A1 leaves (ratios 1/(1/2) = 2 and 3/(1/2) = 6). Pivot element = 1/2.

| Basis | CB | x | y | s1 | s2 | A1 | A2 | RHS |
|---|---|---|---|---|---|---|---|---|
| x | 2 | 1 | 0 | -2 | 1 | 2 | -1 | 2 |
| y | 3 | 0 | 1 | 1 | -1 | -1 | 1 | 2 |
| Zj | | 2 | 3 | -1 | -1 | 1 | 1 | 10 |
| Cj - Zj | | 0 | 0 | 1 | 1 | M - 1 | M - 1 | |

All Cj - Zj >= 0, so the solution is optimal: x = 2, y = 2, minimum Z = 10.
"""

# Big M with a wrong penalty sign on the artificial costs (uses -M in a minimisation problem)
BIGM_WRONG_PENALTY = BIGM_CORRECT.replace("| Cj | | 2 | 3 | 0 | 0 | M | M | |", "| Cj | | 2 | 3 | 0 | 0 | -M | -M | |")

TWO_PHASE_MIN_W = """**Phase I**: Minimize W = A1 + A2.

| Cj | | 0 | 0 | 0 | 0 | 1 | 1 | |
|---|---|---|---|---|---|---|---|---|
| Basis | CB | x | y | s1 | s2 | A1 | A2 | RHS |
| A1 | 1 | 1 | 1 | -1 | 0 | 1 | 0 | 4 |
| A2 | 1 | 1 | 2 | 0 | -1 | 0 | 1 | 6 |
| Zj | | 2 | 3 | -1 | -1 | 1 | 1 | 10 |
| Cj - Zj | | -2 | -3 | 1 | 1 | 0 | 0 | |

y enters, A2 leaves.

| Basis | CB | x | y | s1 | s2 | A1 | A2 | RHS |
|---|---|---|---|---|---|---|---|---|
| A1 | 1 | 1/2 | 0 | -1 | 1/2 | 1 | -1/2 | 1 |
| y | 0 | 1/2 | 1 | 0 | -1/2 | 0 | 1/2 | 3 |
| Zj | | 1/2 | 0 | -1 | 1/2 | 1 | -1/2 | 1 |
| Cj - Zj | | -1/2 | 0 | 1 | -1/2 | 0 | 3/2 | |

x enters, A1 leaves.

| Basis | CB | x | y | s1 | s2 | A1 | A2 | RHS |
|---|---|---|---|---|---|---|---|---|
| x | 0 | 1 | 0 | -2 | 1 | 2 | -1 | 2 |
| y | 0 | 0 | 1 | 1 | -1 | -1 | 1 | 2 |
| Zj | | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Cj - Zj | | 0 | 0 | 0 | 0 | 1 | 1 | |

W = 0, so a feasible solution exists. **Phase II**: drop A1, A2 and use the original objective.

| Basis | CB | x | y | s1 | s2 | RHS |
|---|---|---|---|---|---|---|
| x | 2 | 1 | 0 | -2 | 1 | 2 |
| y | 3 | 0 | 1 | 1 | -1 | 2 |
| Zj | | 2 | 3 | -1 | -1 | 10 |
| Cj - Zj | | 0 | 0 | 1 | 1 | |

All Cj - Zj >= 0: optimal. x = 2, y = 2, Z = 10.
"""

MIN_LE_Q = "Minimize Z = x - 3y subject to x + y <= 4, y <= 3, x, y >= 0"

MIN_AS_MAX = """Convert: Maximize Z' = -Z = -x + 3y.

| Cj | | -1 | 3 | 0 | 0 | |
|---|---|---|---|---|---|---|
| Basis | CB | x | y | s1 | s2 | RHS |
| s1 | 0 | 1 | 1 | 1 | 0 | 4 |
| s2 | 0 | 0 | 1 | 0 | 1 | 3 |
| Cj - Zj | | -1 | 3 | 0 | 0 | |

y enters, s2 leaves (ratios 4 and 3).

| Basis | CB | x | y | s1 | s2 | RHS |
|---|---|---|---|---|---|---|
| s1 | 0 | 1 | 0 | 1 | -1 | 1 |
| y | 3 | 0 | 1 | 0 | 1 | 3 |
| Cj - Zj | | -1 | 0 | 0 | -3 | |

Optimal. Max Z' = 9, so the minimum Z = -9 at x = 0, y = 3.
"""

CLAIMS_ONLY = """Using the simplex method: initially y enters and s3 leaves. Next x enters and s1 leaves.
All net evaluations are now non-positive, so the optimum is x = 1, y = 3 with Z = 18."""

CONTRADICTORY = S01_CORRECT_MD.replace("so **y is the entering variable**", "so **x is the entering variable**")

E08_PREMATURE = """Initial tableau:

| Basis | CB | x | y | s1 | s2 | s3 | RHS |
|---|---|---|---|---|---|---|---|
| s1 | 0 | 1 | 1 | 1 | 0 | 0 | 4 |
| s2 | 0 | 1 | 0 | 0 | 1 | 0 | 2 |
| s3 | 0 | 0 | 1 | 0 | 0 | 1 | 3 |
| Cj - Zj | | 3 | 5 | 0 | 0 | 0 | |

The current solution is optimal: x = 0, y = 0, Z = 0.
"""

E10_UNBOUNDED_Q = "Maximize Z = 3x + 2y subject to x - y <= 1, x, y >= 0"
E10_FINITE_CLAIM = """| Basis | CB | x | y | s1 | RHS |
|---|---|---|---|---|---|
| s1 | 0 | 1 | -1 | 1 | 1 |
| Cj - Zj | | 3 | 2 | 0 | |

x enters, s1 leaves.

| Basis | CB | x | y | s1 | RHS |
|---|---|---|---|---|---|
| x | 3 | 1 | -1 | 1 | 1 |
| Cj - Zj | | 0 | 5 | -3 | |

The optimal solution is x = 1, y = 0 with Z = 3.
"""
E10_CORRECT = E10_FINITE_CLAIM.replace("The optimal solution is x = 1, y = 0 with Z = 3.",
                                       "y can enter but every entry in its column is non-positive, so the problem is unbounded.")

E09_Q = "Maximize Z = 2x + 4y subject to x + 2y <= 8, x + y <= 6, x, y >= 0"
E09_UNIQUE_CLAIM = """| Basis | CB | x | y | s1 | s2 | RHS |
|---|---|---|---|---|---|---|
| s1 | 0 | 1 | 2 | 1 | 0 | 8 |
| s2 | 0 | 1 | 1 | 0 | 1 | 6 |
| Cj - Zj | | 2 | 4 | 0 | 0 | |

y enters, s1 leaves.

| Basis | CB | x | y | s1 | s2 | RHS |
|---|---|---|---|---|---|---|
| y | 4 | 1/2 | 1 | 1/2 | 0 | 4 |
| s2 | 0 | 1/2 | 0 | -1/2 | 1 | 2 |
| Cj - Zj | | 0 | 0 | -2 | 0 | |

Optimal. The unique optimal solution is x = 0, y = 4, Z = 16.
"""
