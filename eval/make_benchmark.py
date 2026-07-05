"""Generate eval/benchmark_problems.json.

Each benchmark problem carries:
  * the natural-language prompt (as a user would type it),
  * a hand-written ground-truth OptimizationSpec,
  * the ground-truth status/objective obtained by solving that hand-written
    spec directly with PuLP — fully independent of the agent pipeline.

Ambiguous problems (``ambiguous: true``) deliberately omit critical data to
exercise the Clarifier; they carry no objective ground truth and are scored on
clarifier behavior + hallucination only.

Run: python -m eval.make_benchmark
"""

import json
from pathlib import Path

from src.schemas import Constraint, DecisionVariable, OptimizationSpec
from src.solver import solve_spec

OUT = Path(__file__).parent / "benchmark_problems.json"


def var(name, var_type="continuous", lb=0.0, ub=None):
    return DecisionVariable(name=name, var_type=var_type, lower_bound=lb,
                            upper_bound=ub)


def con(name, expression, sense, rhs):
    return Constraint(name=name, expression=expression, sense=sense, rhs=rhs)


def spec(family, sense_, objective, variables, constraints):
    return OptimizationSpec(
        problem_family=family, objective_sense=sense_,
        objective_expression=objective, variables=variables,
        constraints=constraints, raw_llm_notes="ground truth (hand-written)",
    )


PROBLEMS: list[dict] = []


def add(pid, family, prompt, gt_spec=None, ambiguous=False, expect=None):
    entry = {"id": pid, "family": family, "prompt": prompt,
             "ambiguous": ambiguous, "ground_truth_spec": None,
             "ground_truth_status": None, "ground_truth_objective": None}
    if gt_spec is not None:
        result = solve_spec(gt_spec)
        entry["ground_truth_spec"] = gt_spec.model_dump()
        entry["ground_truth_status"] = result.status
        entry["ground_truth_objective"] = result.objective_value
        if expect is not None:
            assert result.status == "Optimal", f"{pid}: {result.status}"
            assert abs(result.objective_value - expect) <= 1e-4 * max(1, abs(expect)), (
                f"{pid}: solved {result.objective_value}, expected {expect}"
            )
    PROBLEMS.append(entry)


# =========================== DIET (9) =======================================

add("diet_01", "diet",
    "A feed mill blends corn and soybean meal into 1 kg of feed. Corn costs "
    "$0.30 per kg and is 9% protein and 2% fiber. Soybean meal costs $0.90 per "
    "kg and is 48% protein and 6% fiber. The blend must contain at least 30% "
    "protein and at most 5% fiber. Find the cheapest 1 kg blend.",
    spec("diet", "minimize", "0.30*x_corn + 0.90*x_soy",
         [var("x_corn"), var("x_soy")],
         [con("batch", "x_corn + x_soy", "==", 1),
          con("protein_min", "0.09*x_corn + 0.48*x_soy", ">=", 0.30),
          con("fiber_max", "0.02*x_corn + 0.06*x_soy", "<=", 0.05)]),
    expect=8.1 / 13)

add("diet_02", "diet",
    "Plan the cheapest daily diet from bread, milk, and cheese. Bread costs "
    "$0.20 per serving with 65 calories and 2 g protein. Milk costs $0.45 per "
    "serving with 120 calories and 8 g protein. Cheese costs $1.10 per serving "
    "with 110 calories and 7 g protein. The diet needs at least 2000 calories "
    "and at least 55 g protein per day.",
    spec("diet", "minimize", "0.20*x_bread + 0.45*x_milk + 1.10*x_cheese",
         [var("x_bread"), var("x_milk"), var("x_cheese")],
         [con("calories", "65*x_bread + 120*x_milk + 110*x_cheese", ">=", 2000),
          con("protein", "2*x_bread + 8*x_milk + 7*x_cheese", ">=", 55)]))

add("diet_03", "diet",
    "Build the cheapest daily meal plan from chicken, rice, beans, and "
    "spinach. Per serving: chicken costs $2.50 with 250 calories, 30 g "
    "protein, 400 mg sodium; rice costs $0.50 with 200 calories, 4 g protein, "
    "5 mg sodium; beans cost $0.80 with 220 calories, 14 g protein, 10 mg "
    "sodium; spinach costs $1.20 with 40 calories, 5 g protein, 80 mg sodium. "
    "Require at least 2200 calories, at least 70 g protein, and at most "
    "1500 mg sodium.",
    spec("diet", "minimize",
         "2.50*x_chicken + 0.50*x_rice + 0.80*x_beans + 1.20*x_spinach",
         [var("x_chicken"), var("x_rice"), var("x_beans"), var("x_spinach")],
         [con("calories",
              "250*x_chicken + 200*x_rice + 220*x_beans + 40*x_spinach",
              ">=", 2200),
          con("protein",
              "30*x_chicken + 4*x_rice + 14*x_beans + 5*x_spinach",
              ">=", 70),
          con("sodium",
              "400*x_chicken + 5*x_rice + 10*x_beans + 80*x_spinach",
              "<=", 1500)]))

add("diet_04", "diet",
    "A mill must produce exactly a 100 kg batch of animal feed from wheat, "
    "barley, and pea meal. Wheat costs $0.22 per kg with 11% protein, barley "
    "costs $0.19 per kg with 9% protein, and pea meal costs $0.35 per kg with "
    "22% protein. The batch must average at least 14% protein, and at most "
    "40 kg of barley may be used. Minimize the batch cost.",
    spec("diet", "minimize", "0.22*x_wheat + 0.19*x_barley + 0.35*x_pea",
         [var("x_wheat"), var("x_barley", ub=40.0), var("x_pea")],
         [con("batch", "x_wheat + x_barley + x_pea", "==", 100),
          con("protein", "0.11*x_wheat + 0.09*x_barley + 0.22*x_pea",
              ">=", 14)]))

add("diet_05", "diet",
    "Choose a cheapest breakfast from orange juice, cereal, and granola bars. "
    "Juice costs $0.60 per glass with 120 mg vitamin C and 110 calories, at "
    "most 3 glasses. Cereal costs $0.35 per bowl with 40 mg vitamin C and 150 "
    "calories, at most 4 bowls. A granola bar costs $0.50 with 30 mg vitamin C "
    "and 200 calories, at most 5 bars. Get at least 90 mg vitamin C and at "
    "least 500 calories.",
    spec("diet", "minimize", "0.60*x_juice + 0.35*x_cereal + 0.50*x_bar",
         [var("x_juice", ub=3.0), var("x_cereal", ub=4.0), var("x_bar", ub=5.0)],
         [con("vitamin_c", "120*x_juice + 40*x_cereal + 30*x_bar", ">=", 90),
          con("calories", "110*x_juice + 150*x_cereal + 200*x_bar", ">=", 500)]))

add("diet_06", "diet",
    "I have a budget of $5.00 for protein foods. Lentils cost $0.90 per "
    "serving with 18 g protein (at most 3 servings). Tuna costs $1.50 per "
    "serving with 25 g protein (at most 2 servings). Tofu costs $1.10 per "
    "serving with 15 g protein (at most 3 servings). Maximize total protein "
    "without exceeding the budget.",
    spec("diet", "maximize", "18*x_lentils + 25*x_tuna + 15*x_tofu",
         [var("x_lentils", ub=3.0), var("x_tuna", ub=2.0), var("x_tofu", ub=3.0)],
         [con("budget", "0.90*x_lentils + 1.50*x_tuna + 1.10*x_tofu",
              "<=", 5.00)]),
    expect=3 * 18 + (5.0 - 2.7) / 1.5 * 25)

add("diet_07", "diet",
    "Plan the cheapest weekly diet using rice, beans, and chicken so that I "
    "get enough protein and calories.",
    ambiguous=True)

add("diet_08", "diet",
    "Blend two iron ores into a 50-ton batch that must average at least 40% "
    "iron. Ore A costs $30 per ton and is 25% iron. Ore B costs $50 per ton. "
    "Minimize the cost of the batch.",
    ambiguous=True)  # Ore B iron content is missing

add("diet_09", "diet",
    "Make a 1 kg flour blend from corn flour and rice flour. Corn flour costs "
    "$0.40 per kg and is 9% protein; rice flour costs $0.55 per kg and is 7% "
    "protein. The blend must be at least 20% protein. Minimize cost.",
    spec("diet", "minimize", "0.40*x_corn + 0.55*x_rice",
         [var("x_corn"), var("x_rice")],
         [con("batch", "x_corn + x_rice", "==", 1),
          con("protein", "0.09*x_corn + 0.07*x_rice", ">=", 0.20)]))
# diet_09 ground truth is Infeasible (max attainable protein is 9%).

# ====================== TRANSPORTATION (9) ==================================

add("transport_01", "transportation",
    "Plant A can ship up to 100 units and plant B up to 80 units. Store 1 "
    "needs 60 units, store 2 needs 70, and store 3 needs 40. Per-unit "
    "shipping costs: A to store 1 is $4, A to 2 is $6, A to 3 is $9; B to 1 "
    "is $5, B to 2 is $3, B to 3 is $2. Minimize total shipping cost.",
    spec("transportation", "minimize",
         "4*ship_A_1 + 6*ship_A_2 + 9*ship_A_3 + "
         "5*ship_B_1 + 3*ship_B_2 + 2*ship_B_3",
         [var(n) for n in ["ship_A_1", "ship_A_2", "ship_A_3",
                           "ship_B_1", "ship_B_2", "ship_B_3"]],
         [con("supply_A", "ship_A_1 + ship_A_2 + ship_A_3", "<=", 100),
          con("supply_B", "ship_B_1 + ship_B_2 + ship_B_3", "<=", 80),
          con("demand_1", "ship_A_1 + ship_B_1", ">=", 60),
          con("demand_2", "ship_A_2 + ship_B_2", ">=", 70),
          con("demand_3", "ship_A_3 + ship_B_3", ">=", 40)]),
    expect=620.0)

add("transport_02", "transportation",
    "Three quarries supply two construction sites with gravel. Quarry 1 has "
    "30 tons, quarry 2 has 40 tons, quarry 3 has 50 tons. Site 1 needs 55 "
    "tons and site 2 needs 45 tons. Hauling costs per ton: quarry 1 to site 1 "
    "$6, to site 2 $8; quarry 2 to site 1 $7, to site 2 $5; quarry 3 to site "
    "1 $4, to site 2 $9. Minimize total hauling cost.",
    spec("transportation", "minimize",
         "6*ship_1_1 + 8*ship_1_2 + 7*ship_2_1 + 5*ship_2_2 + "
         "4*ship_3_1 + 9*ship_3_2",
         [var(n) for n in ["ship_1_1", "ship_1_2", "ship_2_1", "ship_2_2",
                           "ship_3_1", "ship_3_2"]],
         [con("supply_1", "ship_1_1 + ship_1_2", "<=", 30),
          con("supply_2", "ship_2_1 + ship_2_2", "<=", 40),
          con("supply_3", "ship_3_1 + ship_3_2", "<=", 50),
          con("demand_1", "ship_1_1 + ship_2_1 + ship_3_1", ">=", 55),
          con("demand_2", "ship_1_2 + ship_2_2 + ship_3_2", ">=", 45)]))

add("transport_03", "transportation",
    "Warehouse A holds 25 pallets and warehouse B holds 35 pallets. Customer "
    "1 needs 30 pallets and customer 2 needs 30 pallets. Delivery costs per "
    "pallet: A to customer 1 $3, A to customer 2 $7, B to customer 1 $6, B to "
    "customer 2 $2. Minimize delivery cost.",
    spec("transportation", "minimize",
         "3*ship_A_1 + 7*ship_A_2 + 6*ship_B_1 + 2*ship_B_2",
         [var(n) for n in ["ship_A_1", "ship_A_2", "ship_B_1", "ship_B_2"]],
         [con("supply_A", "ship_A_1 + ship_A_2", "<=", 25),
          con("supply_B", "ship_B_1 + ship_B_2", "<=", 35),
          con("demand_1", "ship_A_1 + ship_B_1", ">=", 30),
          con("demand_2", "ship_A_2 + ship_B_2", ">=", 30)]),
    expect=165.0)

add("transport_04", "transportation",
    "Assign three technicians to three jobs, one job each and every job "
    "covered. Completion times in hours: tech 1 takes 4, 7, 3 hours on jobs "
    "1, 2, 3; tech 2 takes 2, 6, 5; tech 3 takes 8, 3, 4. Minimize total "
    "completion time.",
    spec("transportation", "minimize",
         "4*a_1_1 + 7*a_1_2 + 3*a_1_3 + 2*a_2_1 + 6*a_2_2 + 5*a_2_3 + "
         "8*a_3_1 + 3*a_3_2 + 4*a_3_3",
         [var(f"a_{w}_{t}", var_type="binary") for w in (1, 2, 3)
          for t in (1, 2, 3)],
         [con("tech_1", "a_1_1 + a_1_2 + a_1_3", "==", 1),
          con("tech_2", "a_2_1 + a_2_2 + a_2_3", "==", 1),
          con("tech_3", "a_3_1 + a_3_2 + a_3_3", "==", 1),
          con("job_1", "a_1_1 + a_2_1 + a_3_1", "==", 1),
          con("job_2", "a_1_2 + a_2_2 + a_3_2", "==", 1),
          con("job_3", "a_1_3 + a_2_3 + a_3_3", "==", 1)]),
    expect=8.0)

add("transport_05", "transportation",
    "Two bakeries supply three cafes with bread. Bakery North can bake 70 "
    "loaves, bakery South 50 loaves. Cafes 1, 2, 3 need 40, 35, 30 loaves. "
    "Delivery cost per loaf: North to cafes 1/2/3 costs $0.50/$0.80/$0.60; "
    "South to cafes 1/2/3 costs $0.90/$0.40/$0.70. South cannot deliver to "
    "cafe 1 on Sundays, and this plan is for a Sunday, so that route is "
    "unavailable. Minimize delivery cost.",
    spec("transportation", "minimize",
         "0.50*ship_N_1 + 0.80*ship_N_2 + 0.60*ship_N_3 + "
         "0.90*ship_S_1 + 0.40*ship_S_2 + 0.70*ship_S_3",
         [var("ship_N_1"), var("ship_N_2"), var("ship_N_3"),
          var("ship_S_1", ub=0.0), var("ship_S_2"), var("ship_S_3")],
         [con("supply_N", "ship_N_1 + ship_N_2 + ship_N_3", "<=", 70),
          con("supply_S", "ship_S_1 + ship_S_2 + ship_S_3", "<=", 50),
          con("demand_1", "ship_N_1 + ship_S_1", ">=", 40),
          con("demand_2", "ship_N_2 + ship_S_2", ">=", 35),
          con("demand_3", "ship_N_3 + ship_S_3", ">=", 30)]))

add("transport_06", "transportation",
    "Ship gravel in whole truckloads from two pits to two sites. Pit 1 can "
    "send 7 truckloads, pit 2 can send 6. Site A needs 5 truckloads, site B "
    "needs 6. Costs per truckload: pit 1 to A $40, pit 1 to B $55, pit 2 to "
    "A $50, pit 2 to B $35. Shipments must be whole truckloads. Minimize "
    "cost.",
    spec("transportation", "minimize",
         "40*ship_1_A + 55*ship_1_B + 50*ship_2_A + 35*ship_2_B",
         [var(n, var_type="integer") for n in
          ["ship_1_A", "ship_1_B", "ship_2_A", "ship_2_B"]],
         [con("supply_1", "ship_1_A + ship_1_B", "<=", 7),
          con("supply_2", "ship_2_A + ship_2_B", "<=", 6),
          con("demand_A", "ship_1_A + ship_2_A", ">=", 5),
          con("demand_B", "ship_1_B + ship_2_B", ">=", 6)]),
    expect=40 * 5 + 35 * 6)

add("transport_07", "transportation",
    "Two warehouses ship to four retail stores. Shipping costs vary by "
    "route. How should we route shipments to minimize total cost?",
    ambiguous=True)

add("transport_08", "transportation",
    "Plant X has 90 units and plant Y has 60. Store 1 needs 50 units and "
    "store 2 needs 45. Store 3 also needs to be served but its requirement "
    "hasn't been confirmed yet. Shipping costs per unit: X to stores 1/2/3 "
    "are $3/$5/$4; Y to stores 1/2/3 are $6/$2/$7. Minimize shipping cost.",
    ambiguous=True)  # store 3 demand missing

add("transport_09", "transportation",
    "Depot 1 has 40 units and depot 2 has 30 units. Region A requires 50 "
    "units and region B requires 40 units — both must be fully met. Costs "
    "per unit: depot 1 to A $2, depot 1 to B $4, depot 2 to A $3, depot 2 to "
    "B $1. Minimize cost.",
    spec("transportation", "minimize",
         "2*ship_1_A + 4*ship_1_B + 3*ship_2_A + 1*ship_2_B",
         [var(n) for n in ["ship_1_A", "ship_1_B", "ship_2_A", "ship_2_B"]],
         [con("supply_1", "ship_1_A + ship_1_B", "<=", 40),
          con("supply_2", "ship_2_A + ship_2_B", "<=", 30),
          con("demand_A", "ship_1_A + ship_2_A", ">=", 50),
          con("demand_B", "ship_1_B + ship_2_B", ">=", 40)]))
# transport_09 ground truth is Infeasible (supply 70 < demand 90).

# ================== FACILITY LOCATION (9) ===================================

add("facility_01", "facility_location",
    "We may open a warehouse in Dallas (fixed cost $500, capacity 120 units) "
    "and/or Reno (fixed cost $400, capacity 100 units). Regions 1, 2, 3 "
    "demand 50, 60, 40 units. Unit service costs: Dallas to regions 1/2/3 "
    "cost $2/$3/$8; Reno to regions 1/2/3 cost $6/$4/$3. Decide which "
    "warehouses to open and how to serve demand at minimum total cost.",
    spec("facility_location", "minimize",
         "500*open_dallas + 400*open_reno + 2*serve_d_1 + 3*serve_d_2 + "
         "8*serve_d_3 + 6*serve_r_1 + 4*serve_r_2 + 3*serve_r_3",
         [var("open_dallas", var_type="binary"),
          var("open_reno", var_type="binary"),
          var("serve_d_1"), var("serve_d_2"), var("serve_d_3"),
          var("serve_r_1"), var("serve_r_2"), var("serve_r_3")],
         [con("demand_1", "serve_d_1 + serve_r_1", ">=", 50),
          con("demand_2", "serve_d_2 + serve_r_2", ">=", 60),
          con("demand_3", "serve_d_3 + serve_r_3", ">=", 40),
          con("link_dallas",
              "serve_d_1 + serve_d_2 + serve_d_3 - 120*open_dallas", "<=", 0),
          con("link_reno",
              "serve_r_1 + serve_r_2 + serve_r_3 - 100*open_reno", "<=", 0)]),
    expect=1300.0)

add("facility_02", "facility_location",
    "Three candidate depots can each handle up to 200 orders: depot A costs "
    "$100 to open with delivery costs of $2 per order to zone 1 and $2 to "
    "zone 2; depot B costs $120 to open with delivery costs of $1 and $1; "
    "depot C costs $150 to open with delivery costs of $3 and $3. Zone 1 has "
    "40 orders and zone 2 has 60 orders. Which depots should open to "
    "minimize total cost?",
    spec("facility_location", "minimize",
         "100*open_a + 120*open_b + 150*open_c + 2*serve_a_1 + 2*serve_a_2 + "
         "1*serve_b_1 + 1*serve_b_2 + 3*serve_c_1 + 3*serve_c_2",
         [var("open_a", var_type="binary"), var("open_b", var_type="binary"),
          var("open_c", var_type="binary"),
          var("serve_a_1"), var("serve_a_2"), var("serve_b_1"),
          var("serve_b_2"), var("serve_c_1"), var("serve_c_2")],
         [con("demand_1", "serve_a_1 + serve_b_1 + serve_c_1", ">=", 40),
          con("demand_2", "serve_a_2 + serve_b_2 + serve_c_2", ">=", 60),
          con("link_a", "serve_a_1 + serve_a_2 - 200*open_a", "<=", 0),
          con("link_b", "serve_b_1 + serve_b_2 - 200*open_b", "<=", 0),
          con("link_c", "serve_c_1 + serve_c_2 - 200*open_c", "<=", 0)]),
    expect=220.0)

add("facility_03", "facility_location",
    "Two uncapacitated fulfillment centers are proposed: center East costs "
    "$300 to open and center West costs $250. Customers 1, 2, 3 order 20, "
    "30, 25 units. Per-unit shipping: East to customers 1/2/3 costs "
    "$1/$4/$6; West to customers 1/2/3 costs $5/$2/$2. Either center could "
    "serve all demand. Minimize fixed plus shipping cost.",
    spec("facility_location", "minimize",
         "300*open_east + 250*open_west + 1*serve_e_1 + 4*serve_e_2 + "
         "6*serve_e_3 + 5*serve_w_1 + 2*serve_w_2 + 2*serve_w_3",
         [var("open_east", var_type="binary"),
          var("open_west", var_type="binary"),
          var("serve_e_1"), var("serve_e_2"), var("serve_e_3"),
          var("serve_w_1"), var("serve_w_2"), var("serve_w_3")],
         [con("demand_1", "serve_e_1 + serve_w_1", ">=", 20),
          con("demand_2", "serve_e_2 + serve_w_2", ">=", 30),
          con("demand_3", "serve_e_3 + serve_w_3", ">=", 25),
          con("link_east",
              "serve_e_1 + serve_e_2 + serve_e_3 - 75*open_east", "<=", 0),
          con("link_west",
              "serve_w_1 + serve_w_2 + serve_w_3 - 75*open_west", "<=", 0)]))

add("facility_04", "facility_location",
    "We can open plants in Ohio (fixed cost $900, capacity 150) and Georgia "
    "(fixed cost $700, capacity 140). Markets 1 and 2 demand 120 and 100 "
    "units. Unit production-plus-shipping costs: Ohio to market 1 $3, Ohio "
    "to market 2 $6, Georgia to market 1 $7, Georgia to market 2 $2. Total "
    "demand exceeds either single plant's capacity. Minimize total cost.",
    spec("facility_location", "minimize",
         "900*open_ohio + 700*open_georgia + 3*serve_o_1 + 6*serve_o_2 + "
         "7*serve_g_1 + 2*serve_g_2",
         [var("open_ohio", var_type="binary"),
          var("open_georgia", var_type="binary"),
          var("serve_o_1"), var("serve_o_2"),
          var("serve_g_1"), var("serve_g_2")],
         [con("demand_1", "serve_o_1 + serve_g_1", ">=", 120),
          con("demand_2", "serve_o_2 + serve_g_2", ">=", 100),
          con("link_ohio", "serve_o_1 + serve_o_2 - 150*open_ohio", "<=", 0),
          con("link_georgia",
              "serve_g_1 + serve_g_2 - 140*open_georgia", "<=", 0)]))

add("facility_05", "facility_location",
    "Choose among three candidate clinics to serve three neighborhoods. "
    "Opening costs: clinic A $200, clinic B $180, clinic C $260. Capacities: "
    "A 80 patients, B 60, C 100. Neighborhood demands: 45, 50, 40 patients. "
    "Per-patient travel costs: A to neighborhoods 1/2/3 cost $2/$5/$4; B "
    "costs $6/$1/$3; C costs $4/$4/$2. Minimize opening plus travel cost.",
    spec("facility_location", "minimize",
         "200*open_a + 180*open_b + 260*open_c + 2*serve_a_1 + 5*serve_a_2 + "
         "4*serve_a_3 + 6*serve_b_1 + 1*serve_b_2 + 3*serve_b_3 + "
         "4*serve_c_1 + 4*serve_c_2 + 2*serve_c_3",
         [var("open_a", var_type="binary"), var("open_b", var_type="binary"),
          var("open_c", var_type="binary"),
          var("serve_a_1"), var("serve_a_2"), var("serve_a_3"),
          var("serve_b_1"), var("serve_b_2"), var("serve_b_3"),
          var("serve_c_1"), var("serve_c_2"), var("serve_c_3")],
         [con("demand_1", "serve_a_1 + serve_b_1 + serve_c_1", ">=", 45),
          con("demand_2", "serve_a_2 + serve_b_2 + serve_c_2", ">=", 50),
          con("demand_3", "serve_a_3 + serve_b_3 + serve_c_3", ">=", 40),
          con("link_a", "serve_a_1 + serve_a_2 + serve_a_3 - 80*open_a",
              "<=", 0),
          con("link_b", "serve_b_1 + serve_b_2 + serve_b_3 - 60*open_b",
              "<=", 0),
          con("link_c", "serve_c_1 + serve_c_2 + serve_c_3 - 100*open_c",
              "<=", 0)]))

add("facility_06", "facility_location",
    "Company policy allows opening at most one of two candidate hubs. Hub "
    "North costs $350 to open with capacity 90; hub South costs $300 with "
    "capacity 90. Customers 1 and 2 demand 30 and 50 units. Unit delivery "
    "costs: North to customers 1/2 cost $2/$3; South costs $4/$1. Minimize "
    "total cost while meeting all demand.",
    spec("facility_location", "minimize",
         "350*open_north + 300*open_south + 2*serve_n_1 + 3*serve_n_2 + "
         "4*serve_s_1 + 1*serve_s_2",
         [var("open_north", var_type="binary"),
          var("open_south", var_type="binary"),
          var("serve_n_1"), var("serve_n_2"),
          var("serve_s_1"), var("serve_s_2")],
         [con("demand_1", "serve_n_1 + serve_s_1", ">=", 30),
          con("demand_2", "serve_n_2 + serve_s_2", ">=", 50),
          con("link_north", "serve_n_1 + serve_n_2 - 90*open_north", "<=", 0),
          con("link_south", "serve_s_1 + serve_s_2 - 90*open_south", "<=", 0),
          con("at_most_one", "open_north + open_south", "<=", 1)]),
    expect=300 + 4 * 30 + 1 * 50)

add("facility_07", "facility_location",
    "We're deciding whether to open distribution centers in Chicago and "
    "Atlanta to serve our Midwest and Southeast regions. Shipping is cheaper "
    "from the closer center. Which should we open?",
    ambiguous=True)  # no fixed costs, capacities, or demands

add("facility_08", "facility_location",
    "Candidate warehouses W1 and W2 have fixed opening costs of $600 and "
    "$450. Customer zones A and B need 70 and 90 units. W1 can hold 200 "
    "units but W2's capacity is still being negotiated. Unit shipping: W1 "
    "to A/B costs $3/$5; W2 to A/B costs $6/$2. Minimize total cost.",
    ambiguous=True)  # W2 capacity missing

add("facility_09", "facility_location",
    "Two kitchens can be opened to prepare meals: kitchen 1 (fixed cost "
    "$150, capacity 40 meals) and kitchen 2 (fixed cost $130, capacity 35 "
    "meals). Sites A and B require 50 and 45 meals — all demand must be "
    "met. Per-meal delivery: kitchen 1 to A/B costs $1/$2; kitchen 2 to A/B "
    "costs $2/$1. Minimize total cost.",
    spec("facility_location", "minimize",
         "150*open_k1 + 130*open_k2 + 1*serve_1_a + 2*serve_1_b + "
         "2*serve_2_a + 1*serve_2_b",
         [var("open_k1", var_type="binary"), var("open_k2", var_type="binary"),
          var("serve_1_a"), var("serve_1_b"),
          var("serve_2_a"), var("serve_2_b")],
         [con("demand_a", "serve_1_a + serve_2_a", ">=", 50),
          con("demand_b", "serve_1_b + serve_2_b", ">=", 45),
          con("link_1", "serve_1_a + serve_1_b - 40*open_k1", "<=", 0),
          con("link_2", "serve_2_a + serve_2_b - 35*open_k2", "<=", 0)]))
# facility_09 ground truth is Infeasible (capacity 75 < demand 95).


def main() -> None:
    OUT.write_text(json.dumps(PROBLEMS, indent=2))
    n_amb = sum(1 for p in PROBLEMS if p["ambiguous"])
    n_inf = sum(1 for p in PROBLEMS if p["ground_truth_status"] == "Infeasible")
    print(f"Wrote {len(PROBLEMS)} problems to {OUT} "
          f"({n_amb} ambiguous, {n_inf} infeasible).")
    for p in PROBLEMS:
        print(f"  {p['id']:14s} {p['ground_truth_status'] or 'AMBIGUOUS':10s} "
              f"{p['ground_truth_objective']}")


if __name__ == "__main__":
    main()
