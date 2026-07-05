from src.schemas import Constraint, DecisionVariable, OptimizationSpec
from src.validator import validate_spec


def make_spec(**overrides) -> OptimizationSpec:
    base = dict(
        problem_family="diet",
        objective_sense="minimize",
        objective_expression="2*x + 3*y",
        variables=[DecisionVariable(name="x"), DecisionVariable(name="y")],
        constraints=[Constraint(name="c1", expression="x + y", sense=">=", rhs=1)],
    )
    base.update(overrides)
    return OptimizationSpec(**base)


def test_valid_spec_passes():
    result = validate_spec(make_spec())
    assert result.is_valid and result.errors == []


def test_undeclared_variable_in_constraint():
    spec = make_spec(constraints=[
        Constraint(name="c1", expression="x + z", sense=">=", rhs=1)])
    result = validate_spec(spec)
    assert not result.is_valid
    assert any("z" in e for e in result.errors)


def test_undeclared_variable_in_objective():
    spec = make_spec(objective_expression="2*w")
    result = validate_spec(spec)
    assert not result.is_valid


def test_objective_with_no_variables():
    spec = make_spec(objective_expression="5")
    result = validate_spec(spec)
    assert not result.is_valid


def test_inconsistent_bounds():
    spec = make_spec(variables=[
        DecisionVariable(name="x", lower_bound=10, upper_bound=1),
        DecisionVariable(name="y"),
    ])
    result = validate_spec(spec)
    assert not result.is_valid
    assert any("lower_bound" in e for e in result.errors)


def test_no_constraints():
    spec = make_spec(constraints=[])
    result = validate_spec(spec)
    assert not result.is_valid


def test_no_variables():
    spec = make_spec(variables=[], objective_expression="1",
                     constraints=[Constraint(name="c", expression="1",
                                             sense=">=", rhs=0)])
    result = validate_spec(spec)
    assert not result.is_valid


def test_duplicate_variable_names():
    spec = make_spec(variables=[DecisionVariable(name="x"),
                                DecisionVariable(name="x"),
                                DecisionVariable(name="y")])
    result = validate_spec(spec)
    assert not result.is_valid


def test_nonlinear_expression_rejected():
    spec = make_spec(objective_expression="x * y")
    result = validate_spec(spec)
    assert not result.is_valid
    assert any("Nonlinear" in e for e in result.errors)


def test_malicious_expression_rejected():
    spec = make_spec(objective_expression="__import__('os').system('true')")
    result = validate_spec(spec)
    assert not result.is_valid


def test_constant_only_constraint_rejected():
    spec = make_spec(constraints=[
        Constraint(name="c1", expression="5 + 3", sense=">=", rhs=1)])
    result = validate_spec(spec)
    assert not result.is_valid
