# Transportation and Assignment Problems (LP / MILP)

The transportation problem routes a single commodity from supply nodes
(plants, warehouses) to demand nodes (stores, customers) so that total
shipping cost is minimized, each supplier ships no more than its capacity,
and each demand point receives what it needs. The assignment problem is the
special case where every supply and every demand equals one unit.

## Decision variables

Declare one continuous variable per (supply node, demand node) pair:
`ship_i_j >= 0` is the quantity sent from source i to destination j. With m
sources and n destinations there are m×n variables — enumerate them all
explicitly (e.g. `ship_A_1, ship_A_2, ship_B_1, ship_B_2`), even for pairs
that look unattractive; the solver decides which routes to leave at zero. Use
continuous variables unless shipments must be whole truckloads/units, in
which case mark them integer. Non-negativity (`lower_bound = 0`) is required:
a negative shipment would mean reverse flow, which this model does not
represent.

## Objective function

Minimize the total shipping cost: sum over all (i, j) pairs of
(unit shipping cost on route i→j) × `ship_i_j`, e.g.
`4*ship_A_1 + 6*ship_A_2 + 5*ship_B_1 + 3*ship_B_2`. Every route that has a
variable needs a cost coefficient, even if zero. If a route is forbidden
("plant B cannot serve store 3"), either omit the variable entirely or give
it `upper_bound = 0` — do not fake it with a huge cost, which distorts
sensitivity information.

## Constraint patterns

Two families of constraints, one row per node:

* Supply (capacity) constraints, one per source i:
  `ship_i_1 + ship_i_2 + ... + ship_i_n <= supply_i`.
* Demand (requirement) constraints, one per destination j:
  `ship_1_j + ship_2_j + ... + ship_m_j >= demand_j`.

Use `<=` on supply and `>=` on demand as the default; this stays feasible
whenever total supply exceeds total demand. Only use `==` on demand if the
problem insists demand is met exactly — with minimization and non-negative
costs the solver will not over-ship anyway. When total supply equals total
demand exactly (a "balanced" problem), the `<=`/`>=` version yields the same
optimum as the classical all-equality formulation.

## Balanced vs unbalanced problems

Check the balance before modeling: total supply vs total demand. If supply <
demand, the model as stated is infeasible — the problem must say what demand
may go unmet (add a shortage variable with a penalty cost) or the data is
wrong. If supply > demand, the surplus simply stays at the sources under the
`<=` supply form; no dummy destination is needed in an LP solver (dummy nodes
are a hand-solution-method artifact from the transportation simplex). State
in your notes which case applies, since a false "infeasible" verdict very
often traces back to an unnoticed supply shortfall.

## Worked example: two plants, three stores (full formulation)

Plant A can ship 100 units, plant B 80. Stores 1, 2, 3 need 60, 70, 40.
Costs per unit: A→1 $4, A→2 $6, A→3 $9; B→1 $5, B→2 $3, B→3 $2.

Variables: `ship_A_1, ship_A_2, ship_A_3, ship_B_1, ship_B_2, ship_B_3`,
all `>= 0`.

Minimize
`4*ship_A_1 + 6*ship_A_2 + 9*ship_A_3 + 5*ship_B_1 + 3*ship_B_2 + 2*ship_B_3`
subject to:
* supply A: `ship_A_1 + ship_A_2 + ship_A_3 <= 100`
* supply B: `ship_B_1 + ship_B_2 + ship_B_3 <= 80`
* demand 1: `ship_A_1 + ship_B_1 >= 60`
* demand 2: `ship_A_2 + ship_B_2 >= 70`
* demand 3: `ship_A_3 + ship_B_3 >= 40`

Optimum: A ships 60 to store 1 and 30 to store 2; B ships 40 to store 2 and
40 to store 3. Cost = 4·60 + 6·30 + 3·40 + 2·40 = $620. Supply B and all
demand constraints are binding; plant A has 10 units of slack capacity.

## Assignment problems

Assigning n workers to n tasks (one each) is a transportation problem with
all supplies and demands equal to 1: `assign_w_t` in [0, 1], each worker row
`== 1`, each task column `== 1`, minimize total cost or time. The constraint
matrix is totally unimodular, so the LP relaxation already yields 0/1 optima
— binary variables are correct but not required. If the problem has side
constraints ("worker 2 cannot do task 3"), fix that variable's upper bound
to 0.

## Common pitfalls in transportation models

* Swapping the inequality directions — `>=` on supply or `<=` on demand
  produces absurd solutions (shipping nothing is then "feasible").
* Missing route variables: forgetting one (i, j) pair silently forbids that
  route and can inflate cost or cause infeasibility.
* Transposed cost data: reading the cost table as destination×source instead
  of source×destination scrambles all coefficients — sanity-check one cell.
* Double-counting demand when a destination appears in the prompt twice under
  different names (e.g. "the downtown store" and "store 1").
* Declaring shipments integer out of habit: it makes the model a MILP for no
  benefit — with integral supplies/demands the LP optimum is already integral.
