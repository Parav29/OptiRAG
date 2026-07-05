"""Phase 2 de-risk: three hand-built specs (one per family) with independently
known optima, plus expression-safety and edge-status coverage."""

import pytest

from src.expr import ExpressionError, parse_linear_expression
from src.schemas import Constraint, DecisionVariable, OptimizationSpec
from src.solver import solve_spec

# --- hand-built specs, one per family --------------------------------------

DIET_SPEC = OptimizationSpec(
    problem_family="diet",
    objective_sense="minimize",
    objective_expression="0.30*x_corn + 0.90*x_soy",
    variables=[DecisionVariable(name="x_corn"), DecisionVariable(name="x_soy")],
    constraints=[
        Constraint(name="batch", expression="x_corn + x_soy", sense="==", rhs=1),
        Constraint(name="protein", expression="0.09*x_corn + 0.48*x_soy",
                   sense=">=", rhs=0.30),
        Constraint(name="fiber", expression="0.02*x_corn + 0.06*x_soy",
                   sense="<=", rhs=0.05),
    ],
)
DIET_OPTIMUM = 0.30 * (6 / 13) + 0.90 * (7 / 13)  # 0.62307...

TRANSPORT_SPEC = OptimizationSpec(
    problem_family="transportation",
    objective_sense="minimize",
    objective_expression=(
        "4*ship_A_1 + 6*ship_A_2 + 9*ship_A_3 + "
        "5*ship_B_1 + 3*ship_B_2 + 2*ship_B_3"
    ),
    variables=[DecisionVariable(name=n) for n in
               ["ship_A_1", "ship_A_2", "ship_A_3",
                "ship_B_1", "ship_B_2", "ship_B_3"]],
    constraints=[
        Constraint(name="supply_A", expression="ship_A_1 + ship_A_2 + ship_A_3",
                   sense="<=", rhs=100),
        Constraint(name="supply_B", expression="ship_B_1 + ship_B_2 + ship_B_3",
                   sense="<=", rhs=80),
        Constraint(name="demand_1", expression="ship_A_1 + ship_B_1",
                   sense=">=", rhs=60),
        Constraint(name="demand_2", expression="ship_A_2 + ship_B_2",
                   sense=">=", rhs=70),
        Constraint(name="demand_3", expression="ship_A_3 + ship_B_3",
                   sense=">=", rhs=40),
    ],
)
TRANSPORT_OPTIMUM = 620.0

FACILITY_SPEC = OptimizationSpec(
    problem_family="facility_location",
    objective_sense="minimize",
    objective_expression=(
        "500*open_dallas + 400*open_reno + 2*serve_d_1 + 3*serve_d_2 + "
        "8*serve_d_3 + 6*serve_r_1 + 4*serve_r_2 + 3*serve_r_3"
    ),
    variables=[
        DecisionVariable(name="open_dallas", var_type="binary"),
        DecisionVariable(name="open_reno", var_type="binary"),
        *[DecisionVariable(name=n) for n in
          ["serve_d_1", "serve_d_2", "serve_d_3",
           "serve_r_1", "serve_r_2", "serve_r_3"]],
    ],
    constraints=[
        Constraint(name="demand_1", expression="serve_d_1 + serve_r_1",
                   sense=">=", rhs=50),
        Constraint(name="demand_2", expression="serve_d_2 + serve_r_2",
                   sense=">=", rhs=60),
        Constraint(name="demand_3", expression="serve_d_3 + serve_r_3",
                   sense=">=", rhs=40),
        Constraint(name="link_dallas",
                   expression="serve_d_1 + serve_d_2 + serve_d_3 - 120*open_dallas",
                   sense="<=", rhs=0),
        Constraint(name="link_reno",
                   expression="serve_r_1 + serve_r_2 + serve_r_3 - 100*open_reno",
                   sense="<=", rhs=0),
    ],
)
FACILITY_OPTIMUM = 1300.0


def test_diet_lp_solves_to_known_optimum():
    result = solve_spec(DIET_SPEC)
    assert result.status == "Optimal"
    assert result.objective_value == pytest.approx(DIET_OPTIMUM, rel=1e-4)
    assert result.variable_values["x_soy"] == pytest.approx(7 / 13, rel=1e-4)
    assert "protein" in result.binding_constraints
    assert "fiber" not in result.binding_constraints


def test_transportation_lp_solves_to_known_optimum():
    result = solve_spec(TRANSPORT_SPEC)
    assert result.status == "Optimal"
    assert result.objective_value == pytest.approx(TRANSPORT_OPTIMUM, rel=1e-6)
    assert "supply_B" in result.binding_constraints
    assert "supply_A" not in result.binding_constraints


def test_facility_milp_solves_to_known_optimum():
    result = solve_spec(FACILITY_SPEC)
    assert result.status == "Optimal"
    assert result.objective_value == pytest.approx(FACILITY_OPTIMUM, rel=1e-6)
    assert result.variable_values["open_dallas"] == pytest.approx(1.0)
    assert result.variable_values["open_reno"] == pytest.approx(1.0)


def test_infeasible_detected():
    spec = OptimizationSpec(
        problem_family="diet",
        objective_sense="minimize",
        objective_expression="x",
        variables=[DecisionVariable(name="x", lower_bound=0, upper_bound=1)],
        constraints=[Constraint(name="impossible", expression="x", sense=">=", rhs=5)],
    )
    result = solve_spec(spec)
    assert result.status == "Infeasible"
    assert result.objective_value is None


def test_unbounded_detected():
    spec = OptimizationSpec(
        problem_family="diet",
        objective_sense="maximize",
        objective_expression="x",
        variables=[DecisionVariable(name="x")],
        constraints=[Constraint(name="floor", expression="x", sense=">=", rhs=0)],
    )
    result = solve_spec(spec)
    assert result.status in ("Unbounded", "Undefined", "Not Solved")
    assert result.objective_value is None


def test_maximize_sense():
    spec = OptimizationSpec(
        problem_family="diet",
        objective_sense="maximize",
        objective_expression="3*x + 2*y",
        variables=[DecisionVariable(name="x"), DecisionVariable(name="y")],
        constraints=[Constraint(name="cap", expression="x + y", sense="<=", rhs=10)],
    )
    result = solve_spec(spec)
    assert result.status == "Optimal"
    assert result.objective_value == pytest.approx(30.0)


# --- expression parser safety ----------------------------------------------

def test_parser_expands_linear_terms():
    terms = parse_linear_expression("2*x + 3*(y - x) + 1", {"x", "y"})
    assert terms["x"] == pytest.approx(-1.0)
    assert terms["y"] == pytest.approx(3.0)
    assert terms[""] == pytest.approx(1.0)


def test_parser_rejects_undeclared_names():
    with pytest.raises(ExpressionError):
        parse_linear_expression("x + hacker", {"x"})


@pytest.mark.parametrize("bad", [
    "__import__('os')",
    "x ** 2",
    "x * y",
    "1 / x",
    "open('/etc/passwd')",
    "x if True else 0",
    "[i for i in range(10)]",
    "x; import os",
    "",
])
def test_parser_rejects_unsafe_or_nonlinear(bad):
    with pytest.raises(ExpressionError):
        parse_linear_expression(bad, {"x", "y"})
