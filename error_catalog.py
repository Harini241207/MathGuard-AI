"""Error categories used by the step verifier.

Each category has a short title, the kind of mistake (used for the green / red /
amber summary) and a *general* algorithm note.  General notes are written by
MathGuard and are always labelled as such in the report; passages from an
uploaded textbook are shown separately, with their page numbers.
"""

KINDS = {
    "calculation": "Incorrect calculation",
    "method": "Incorrect method application",
    "variable_selection": "Incorrect variable selection",
    "intermediate": "Invalid intermediate result",
    "final_answer": "Incorrect final answer",
    "incomplete": "Incomplete working",
    "unverifiable": "Unverifiable step",
}

CATALOG = {
    # ---- simplex family (codes requested in the specification)
    "E01": dict(title="Incorrect objective coefficients", kind="intermediate",
                note="The Cj row (and the CB column) must contain the objective coefficients exactly as in the problem: "
                     "decision variables keep their coefficients, slack and surplus variables get 0, and artificial variables "
                     "get the penalty (-M for maximisation, +M for minimisation) or the Phase I cost.",
                query="objective function coefficients Cj row cost coefficient basic variables CB"),
    "E02": dict(title="Incorrect slack / surplus / artificial variable signs", kind="method",
                note="A '<=' constraint gets +s (slack). A '>=' constraint gets -s (surplus) and +A (artificial). An equation "
                     "gets +A only. The initial basis must have a +1 coefficient in its own row.",
                query="slack variable surplus variable artificial variable standard form inequality"),
    "E03": dict(title="Incorrect entering variable", kind="variable_selection",
                note="Maximisation: the entering variable has the largest positive Cj - Zj. Minimisation: the most negative "
                     "Cj - Zj. A variable whose Cj - Zj does not improve Z must never enter.",
                query="entering variable key column largest positive Cj - Zj net evaluation optimality condition"),
    "E04": dict(title="Incorrect leaving variable or ratio test", kind="variable_selection",
                note="Divide each RHS by the corresponding entry of the pivot column, using strictly positive entries only. "
                     "The row with the smallest non-negative ratio leaves. Rows with zero or negative entries are ignored.",
                query="minimum ratio test leaving variable key row departing variable replacement ratio"),
    "E05": dict(title="Incorrect pivot-row normalisation", kind="calculation",
                note="Divide every entry of the pivot row, including the RHS, by the pivot element so the pivot becomes 1.",
                query="pivot row divide by pivot element key element new row"),
    "E06": dict(title="Incorrect row operation", kind="calculation",
                note="Every other row: new row = old row - (its entry in the pivot column) x (new pivot row). This makes the "
                     "rest of the pivot column zero.",
                query="row operations new row old row pivot column entry elementary row operation"),
    "E07": dict(title="Incorrect updated tableau (Zj, Cj - Zj, basis or Z)", kind="calculation",
                note="Zj = sum over rows of CB x (column entry); Cj - Zj = Cj minus Zj; Z = sum of CB x RHS. After a pivot the "
                     "entering variable replaces the leaving variable in the basis and its CB changes too.",
                query="Zj row Cj - Zj computation simplex tableau update basic variables"),
    "E08": dict(title="Incorrect standard-form conversion", kind="method",
                note="Each constraint must appear as an equation with its slack / surplus / artificial variable, the same "
                     "coefficients and a non-negative RHS. The initial tableau must copy these equations exactly.",
                query="standard form linear programming equation slack variables canonical form initial basic feasible solution"),
    "E09": dict(title="Incorrect constraint interpretation", kind="intermediate",
                note="The constraints restated in the answer must match the problem: same coefficients, same direction "
                     "(<=, >=, =) and the same right-hand side.",
                query="constraints formulation inequality direction right hand side"),
    "E10": dict(title="Incorrect feasibility assessment", kind="method",
                note="A point (or basic solution) is feasible only if it satisfies every constraint and non-negativity. In the "
                     "Big M and Two-Phase methods, an artificial variable that stays positive at the optimum means there is "
                     "no feasible solution.",
                query="feasible solution infeasible artificial variable remains positive basis feasibility"),
    "E11": dict(title="Incorrect pivot element", kind="calculation",
                note="The pivot element is the entry where the pivot (entering) column meets the pivot (leaving) row.",
                query="pivot element key element intersection key row key column"),
    "E12": dict(title="Incorrect final objective value", kind="final_answer",
                note="The optimal Z must equal the objective function evaluated at the optimal solution (sum of CB x RHS in "
                     "the final tableau).",
                query="optimal value objective function value optimal solution"),
    "E13": dict(title="Incorrect optimality conclusion", kind="method",
                note="Stop only when no Cj - Zj can improve Z (all <= 0 for maximisation, all >= 0 for minimisation). If a "
                     "non-basic variable has Cj - Zj = 0 in the optimal tableau, alternative optima exist.",
                query="optimality condition optimal tableau alternative optimal solution zero net evaluation"),
    "E14": dict(title="Incorrect unboundedness or infeasibility conclusion", kind="final_answer",
                note="Unbounded: an entering variable has no positive entry in its column. Infeasible: an artificial "
                     "variable stays positive at the optimum (or Phase I ends with W < 0). Otherwise there is a finite "
                     "optimum.",
                query="unbounded solution no positive entry infeasible solution artificial variable"),
    # ---- graphical method
    "G01": dict(title="Incorrect boundary equation or intercept", kind="calculation",
                note="Replace each inequality by an equation to get its boundary line. Intercepts: put y = 0 to get the "
                     "x-intercept and x = 0 to get the y-intercept.",
                query="graphical method boundary line intercepts plotting constraints"),
    "G02": dict(title="Incorrect intersection point", kind="calculation",
                note="An intersection (corner) point must satisfy both boundary equations at the same time. Solve them "
                     "simultaneously by elimination, substitution or Cramer's rule.",
                query="point of intersection simultaneous equations corner point graphical"),
    "G03": dict(title="Incorrect feasible region / corner points", kind="intermediate",
                note="Corner points are the intersection points that satisfy every constraint and non-negativity. Points that "
                     "violate a constraint are not part of the feasible region.",
                query="feasible region corner points extreme points vertices feasible solution graphical"),
    "G04": dict(title="Incorrect objective evaluation at a corner", kind="calculation",
                note="Substitute the coordinates of the corner into the objective function: Z = c1 x + c2 y.",
                query="evaluate objective function at corner points extreme point theorem"),
    "G05": dict(title="Incorrect selection of the optimal corner", kind="method",
                note="For maximisation choose the corner with the largest Z, for minimisation the smallest. If two adjacent "
                     "corners tie, every point on the edge between them is optimal.",
                query="optimal corner point maximum minimum value extreme point theorem graphical"),
    # ---- transportation / assignment
    "T01": dict(title="Incorrect initial basic feasible solution", kind="method",
                note="North-West Corner, Least Cost and Vogel's method each allocate min(supply, demand) to a specific cell; "
                     "the allocations must respect every row and column total.",
                query="initial basic feasible solution north west corner least cost vogel approximation"),
    "T02": dict(title="Incorrect transportation cost / allocation", kind="calculation",
                note="Total cost = sum of (allocation x unit cost) over occupied cells. Allocations must add up to each "
                     "supply and demand.",
                query="transportation cost allocation supply demand total cost"),
    "T03": dict(title="Incorrect optimality (MODI) conclusion", kind="method",
                note="MODI: ui + vj = cij on occupied cells; the opportunity cost dij = cij - (ui + vj) must be >= 0 for every "
                     "empty cell at the optimum (minimisation).",
                query="MODI method u-v method opportunity cost optimality test transportation"),
    "H01": dict(title="Incorrect assignment / Hungarian step", kind="method",
                note="Hungarian method: subtract row minima, then column minima, cover all zeros with the minimum number of "
                     "lines, adjust by the smallest uncovered value until n lines are needed, then assign on zeros.",
                query="Hungarian method assignment problem row reduction column reduction covering zeros"),
    # ---- final-answer categories kept from the original MathGuard
    "wrong_status": dict(title="Wrong problem status", kind="final_answer",
                         note="Decide carefully between a unique optimum, multiple optima, unbounded and infeasible.",
                         query="infeasible region no feasible solution unbounded objective alternative optima"),
    "wrong_value": dict(title="Wrong objective value", kind="final_answer",
                        note="Evaluate Z at the optimal solution.",
                        query="objective function value substitute optimal solution evaluate Z"),
    "suboptimal_solution": dict(title="Feasible but not optimal", kind="final_answer",
                                note="The reported point satisfies the constraints but another feasible point gives a better Z.",
                                query="optimality condition corner point extreme point largest objective value"),
    "infeasible_solution": dict(title="Reported point violates a constraint", kind="final_answer",
                                note="Substitute the reported point into every constraint and non-negativity condition.",
                                query="feasible region constraints satisfied feasible solution non-negativity"),
    "missed_multiple_optima": dict(title="Missed alternative optima", kind="final_answer",
                                   note="If the objective line is parallel to a binding constraint (or a non-basic variable has "
                                        "Cj - Zj = 0 at the optimum), there are infinitely many optimal solutions.",
                                   query="alternative optimal solutions zero reduced cost objective parallel binding constraint"),
}


def info(code):
    c = CATALOG.get(code, {})
    return {"code": code, "title": c.get("title", code), "kind": c.get("kind", "unverifiable"),
            "kind_text": KINDS.get(c.get("kind", ""), ""), "note": c.get("note", "")}


FINAL_TO_CODE = {
    "wrong_status": "E14", "wrong_value": "E12", "suboptimal_solution": "E13",
    "infeasible_solution": "E10", "missed_multiple_optima": "E13",
}
