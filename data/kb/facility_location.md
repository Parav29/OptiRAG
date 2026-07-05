# Fixed-Charge Facility Location (MILP)

The facility location problem decides (a) which candidate facilities to open,
paying a one-time fixed cost per opened site, and (b) how much each opened
facility ships to each customer, paying a per-unit service cost. The objective
minimizes fixed opening costs plus variable service costs. The open/close
decision makes this a mixed-integer program: it cannot be modeled with
continuous variables alone.

## Decision variables

Two families of variables:

* `open_f` — binary, 1 if facility f is opened, 0 otherwise. One per
  candidate site. Declare `var_type = "binary"`; do not model open/close as a
  continuous variable in [0, 1], because a "half-open" facility would pay
  half the fixed cost, which understates true cost.
* `serve_f_c` — continuous `>= 0`, the quantity facility f ships to customer
  c (or, in the fractional-assignment variant, the fraction of customer c's
  demand served by f, in [0, 1]).

Enumerate every (facility, customer) pair explicitly, exactly as in a
transportation model — the binary layer sits on top of an ordinary
transportation structure.

## Objective function

Minimize fixed cost plus variable cost:
`fixed_A*open_A + fixed_B*open_B + ... + cost_A_1*serve_A_1 + cost_A_2*serve_A_2 + ...`
Both cost types belong in one objective. A frequent error is dropping the
fixed-cost terms after linking constraints are written — then opening every
facility is free and the binaries become meaningless. Every `open_f` variable
should appear in the objective with its fixed cost, and every `serve_f_c`
with its unit service cost.

## Linking constraints (the heart of the model)

A facility can serve customers only if it is open. Two standard encodings:

* Aggregate (big-M) linking, one per facility:
  `serve_f_1 + serve_f_2 + ... - M*open_f <= 0`, where M is the facility's
  capacity, or total demand for an uncapacitated model. Written with the
  binary moved to the left side to keep the expression linear and the rhs
  constant (0).
* Disaggregated linking, one per (facility, customer) pair:
  `serve_f_c - demand_c*open_f <= 0`. Tighter LP relaxation (solves faster in
  branch-and-bound) at the price of more rows.

Either is correct; the aggregate form is smaller and fine at this scale. The
key invariant: if `open_f = 0`, every `serve_f_*` is forced to 0.

## Demand and capacity constraints

* Demand satisfaction, one per customer c:
  `serve_A_c + serve_B_c + ... >= demand_c` (or `== 1` per customer in the
  fraction-based variant where serve variables are fractions of demand).
* Capacity, one per facility f (capacitated variant only):
  `serve_f_1 + ... + serve_f_n <= capacity_f` — note this can double as the
  big-M linking row by writing `... - capacity_f*open_f <= 0` instead;
  writing both a plain capacity row AND a linking row is redundant but
  harmless.

Choosing M: use the smallest valid value (the facility's capacity, or the sum
of all demands if uncapacitated). A sloppy huge M (like 1e9) weakens the LP
relaxation and can cause numerical trouble in CBC.

## Worked example: two candidate warehouses, three regions (full formulation)

Warehouses: Dallas (fixed $500, capacity 120) and Reno (fixed $400, capacity
100). Regions 1, 2, 3 demand 50, 60, 40. Unit service costs: Dallas→1 $2,
Dallas→2 $3, Dallas→3 $8; Reno→1 $6, Reno→2 $4, Reno→3 $3.

Variables: `open_dallas`, `open_reno` binary; `serve_d_1, serve_d_2,
serve_d_3, serve_r_1, serve_r_2, serve_r_3 >= 0`.

Minimize `500*open_dallas + 400*open_reno + 2*serve_d_1 + 3*serve_d_2 +
8*serve_d_3 + 6*serve_r_1 + 4*serve_r_2 + 3*serve_r_3`
subject to:
* demand 1: `serve_d_1 + serve_r_1 >= 50`
* demand 2: `serve_d_2 + serve_r_2 >= 60`
* demand 3: `serve_d_3 + serve_r_3 >= 40`
* link Dallas: `serve_d_1 + serve_d_2 + serve_d_3 - 120*open_dallas <= 0`
* link Reno: `serve_r_1 + serve_r_2 + serve_r_3 - 100*open_reno <= 0`

Total demand is 150, above either single capacity (120, 100), so both must
open. Optimal service: Dallas serves regions 1 and 2 (50 + 60 = 110), Reno
serves region 3 (40). Cost = 500 + 400 + 2·50 + 3·60 + 3·40 = $1300. Both
demand rows bind; Dallas link has 10 slack, Reno link 60.

## Single-facility feasibility check

Before solving, compare total demand with individual and combined capacities:
if no subset of facilities can cover total demand, the model is infeasible as
posed. Conversely, if one cheap facility can cover everything, expect the
optimizer to open exactly one — a solution opening zero facilities while
still serving demand signals broken linking constraints (the single most
common facility-location modeling bug).

## Common pitfalls in facility location models

* Missing linking constraints: serve variables become independent of the
  binaries, the solver serves customers from closed facilities, and fixed
  costs are simply never paid. Always trace: closed ⇒ all its serves zero.
* Off-by-one between facility and customer indices (e.g. `serve_1_2` meaning
  facility 2 → customer 1 in one row and the transpose in another). Fix a
  naming convention (`serve_<facility>_<customer>`) and apply it everywhere.
* Continuous "binaries": `0 <= open_f <= 1` without integrality lets the
  model pay fractional fixed costs — the classic source of impossibly low
  objective values.
* Fixed costs in constraints instead of the objective, or applied per unit
  served rather than per facility opened.
* Big-M smaller than real capacity, silently capping throughput below what
  the problem states.
