from logic4e import (
    is_prop_symbol,
    Expr as Expression,
    expr
)


def pl_true(expression: Expression, model={}):
    """
    Return True if the propositional logic expression is true in the model,
    and False if it is false. If the model does not specify the value for
    every proposition, this may return None to indicate 'not obvious';
    this may happen even when the expression is tautological.
    >>> pl_true(P, {}) is None
    True
    """
    if expression in (True, False):
        return expression
    op, args = expression.op, expression.args
    if is_prop_symbol(op):
        return pl_true(model.get(expression), model)
    elif op == '~':
        p = pl_true(args[0], model)
        if p is None:
            return None
        else:
            return not p
    elif op == '|':
        result = False
        for arg in args:
            p = pl_true(arg, model)
            if p is True:
                return True
            if p is None:
                result = None
        return result
    elif op == '&':
        result = True
        for arg in args:
            p = pl_true(arg, model)
            if p is False:
                return False
            if p is None:
                result = None
        return result
    p, q = args
    if op == '==>':
        return pl_true(~p | q, model)
    elif op == '<==':
        return pl_true(p | ~q, model)
    pt = pl_true(p, model)
    if pt is None:
        return None
    qt = pl_true(q, model)
    if qt is None:
        return None
    if op == '<=>':
        return pt == qt
    elif op == '^':  # xor or 'not equivalent'
        return pt != qt
    else:
        raise ValueError("illegal operator in logic expression" + str(expression))


# Now let's give it a try with a knowledgebase and an expression
A, B, C, D, E, F = expr('A, B, C, D, E, F')
kb = {
    A: True,
    B: False,
    C: A|B,
    D: A&B,
    E: C&~D,
    F: ~E|D
}

for expression in kb.keys():
    expression: Expression
    print(f"Expression {expression}: {kb[expression]} evaluates to {pl_true(expression, kb)}")