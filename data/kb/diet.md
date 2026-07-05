# Diet and Blending Problems (LP)

The diet problem (equivalently, the blending problem) chooses how much of each
available food or ingredient to include in a mix so that total cost is
minimized while the mix meets nutritional or compositional requirements. It is
a pure linear program: all decision variables are continuous quantities, and
both cost and nutrient content scale linearly with quantity.

## Decision variables

Declare one continuous variable per food or ingredient, representing the
quantity purchased or included: `x_food >= 0`, in whatever unit the prices are
quoted in (kg, servings, ounces). Non-negativity is essential — a negative
quantity of an ingredient is meaningless, so every variable gets
`lower_bound = 0`. If the problem caps how much of one ingredient may be used
(e.g. "no more than 4 servings of milk per day"), encode that as the
variable's `upper_bound` rather than as a separate constraint; it keeps the
model smaller and the bound is reported directly on the variable. Variables
are continuous, never integer, unless the problem explicitly says items must
be bought in whole units.

## Objective function

Minimize total cost: the sum over foods of (unit cost) × (quantity variable),
e.g. `0.30*x_corn + 0.90*x_soy`. Every coefficient must come from the stated
prices. A common slip is mixing units — if corn is priced per kg but the
protein requirement is per 100 g, convert everything to one consistent unit
before writing coefficients. If the problem instead asks to maximize some
nutritional score subject to a budget, the same structure applies with the
objective and the budget constraint swapping roles.

## Constraint patterns

Each nutritional requirement becomes one linear constraint over all foods:

* Minimum requirement: `(content per unit of food A)*x_A + (content per unit
  of food B)*x_B + ... >= required_amount` — e.g. protein, vitamins, calories
  floors.
* Maximum allowance: same left-hand side with `<=` — e.g. sodium, fat,
  fiber ceilings.
* Exact composition: use `==` only when the problem says "exactly"; otherwise
  prefer inequalities, which keep the model feasible more often.

When the problem fixes the batch size ("per kg of blend", "a 100 kg batch"),
add a batch-size constraint like `x_corn + x_soy == 1` (or `== 100`) so that
percentage requirements are measured against a known total.

## Percentage requirements in blending

"The blend must be at least 30% protein" is a statement about the *ratio* of
protein to total weight. With a fixed batch size B and per-unit protein
contents p_i, it linearizes directly: `p_corn*x_corn + p_soy*x_soy >= 0.30*B`,
where B is the constant batch size. If the batch size is itself a variable T,
move everything to one side to keep it linear:
`p_corn*x_corn + p_soy*x_soy - 0.30*T >= 0`. Never write the requirement as a
division (`protein/total >= 0.30`) — that is nonlinear as written and will be
rejected by any LP solver; always multiply through by the denominator first.

## Worked example: two-ingredient feed blend (full formulation)

A feed mill blends corn ($0.30/kg, 9% protein, 2% fiber) and soybean meal
($0.90/kg, 48% protein, 6% fiber) into 1 kg of feed that needs at least 30%
protein and at most 5% fiber.

Variables: `x_corn >= 0`, `x_soy >= 0` (kg of each per kg of blend).

Minimize `0.30*x_corn + 0.90*x_soy`
subject to:
* batch: `x_corn + x_soy == 1`
* protein floor: `0.09*x_corn + 0.48*x_soy >= 0.30`
* fiber cap: `0.02*x_corn + 0.06*x_soy <= 0.05`

Substituting `x_corn = 1 - x_soy` into the protein floor gives
`0.09 + 0.39*x_soy >= 0.30`, so `x_soy >= 0.5385` (7/13). Cost rises with
`x_soy`, so the optimum is `x_soy = 7/13 ≈ 0.5385`, `x_corn = 6/13 ≈ 0.4615`,
cost ≈ `$0.6231/kg`, with the protein constraint binding and the fiber
constraint slack (0.0415 ≤ 0.05).

## Common pitfalls in diet/blending models

* Forgetting non-negativity: without `x >= 0` the LP can "sell" a cheap
  ingredient short to fake nutritional content, giving nonsense optima.
* Percentage constraints written against the wrong base: 30% protein means
  30% of the *blend weight*, not 30% of a kilogram of one ingredient. Fix the
  batch size or multiply through by the total.
* Unit mismatches between price units and nutrient units (per kg vs per 100 g
  vs per serving) — normalize before writing coefficients.
* Using `==` for requirements phrased as "at least"/"at most": over-constrains
  the model and commonly causes false infeasibility.
* Omitting an ingredient from a nutrient constraint because its content is
  small but nonzero — every food contributes to every nutrient row it has
  data for.

## Infeasibility diagnosis in diet problems

If the solver reports Infeasible, the requirements are mutually inconsistent
with the available ingredients. Typical causes: a nutrient floor higher than
what even the richest ingredient can deliver at the fixed batch size; a
ceiling (like fiber or sodium) that every ingredient individually violates; or
upper bounds on ingredient quantities that cap total nutrient supply below the
floor. Check feasibility by testing the extreme blend (100% of the richest
ingredient) against each floor and the leanest blend against each ceiling
before concluding the model itself is wrong.
