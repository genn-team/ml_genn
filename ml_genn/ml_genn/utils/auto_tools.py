import sympy

"""
THis function turns ODEs expressed as two lists for sympy variables and matching rhs expressions 
into C code to update the variables with timestep "dt"
"""

def _linear_euler(dx_dt):
    code = [sympy.ccode((sym + expr) * sympy.Symbol("dt"),
                        assign_to=f"const scalar {sym.name}_tmp")
            for sym, expr in dx_dt.items()]
    code += [f"{sym.name} = {sym.name}_tmp;" for sym in dx_dt.keys()]
    return code

    
""" 
This modified from Brian 2's exponential_euler stateupdater
"""

def _get_conditionally_linear_system(dx_dt):
    """
    Convert equations into a linear system using sympy.

    Parameters
    ----------
    eqs : `Equations`
        The model equations.

    Returns
    -------
    coefficients : dict of (sympy expression, sympy expression) tuples
        For every variable x, a tuple (M, B) containing the coefficients M and
        B (as sympy expressions) for M * x + B

    Raises
    ------
    ValueError
        If one of the equations cannot be converted into a M * x + B form.
    """

    coefficients = {}
    for sym, expr in dx_dt.items():
        if expr.has(sym):
            # Factor out the variable
            expr = expr.expand()
            expr = sympy.collect(expr, sym, evaluate=False)
            if len(expr) > 2 or sym not in expr:
                raise ValueError(
                    f"The expression '{expr}', defining the variable "
                    f"'{sym.name}', could not be separated into linear "
                    "components.")
            coefficients[sym] = (expr[sym], expr.get(1, 0))
        else:
            coefficients[sym] = (0, expr)

    return coefficients


def _exponential_euler(dx_dt):
    # Try whether the equations are conditionally linear
    try:
        system = _get_conditionally_linear_system(dx_dt)
    except ValueError:
        raise NotImplementedError(
            "Can only solve conditionally linear systems with this state updater.")
    s_dt = sympy.Symbol("dt")

    code = []
    for sym, (A, B) in system.items():
        if A == 0:
            update_expression = sym + s_dt * B
        elif B != 0:
            BA = B / A
            # Avoid calculating B/A twice
            BA_name = f"BA_{sym.name}_tmp"
            s_BA = sympy.Symbol(BA_name)
            code += [f"const scalar {BA_name} = {sympy.ccode(BA)};"]
            update_expression = (sym + s_BA) * sympy.exp(A * s_dt) - s_BA
        else:
            update_expression = sym * sympy.exp(A * s_dt)
            
        # The actual update step
        code += [f"const scalar {sym.name}_tmp = {sympy.ccode(update_expression)};"]

    # Replace all the variables with their updated value
    for sym in system.keys():
        code += [f"{sym.name} = {sym.name}_tmp;"]

    return code

"""
End of Brian 2 modified code
"""

# **TODO** remove me
def get_symbols(vars, params, w_name=None):
    sym = {v: sympy.Symbol(v) for v in vars}
    sym.update({p: sympy.Symbol(p) for p in params})
    sym.update({f"Lambda{v}": sympy.Symbol(f"Lambda{v}")
                for v in vars})

    if w_name is not None:
        sym[w_name] = sympy.Symbol(w_name)

    return sym

# solde a set of ODEs. They can be passed as a dict of strings
# or dict of sympy expressions
# **TODO** solver enum
def solve_ode(dx_dt, solver):
    if solver == "exponential_euler":
        clines = _exponential_euler(dx_dt)
    elif solver == "linear_euler":
        clines = _linear_euler(dx_dt)
    else:
        raise NotImplementedError(
            f"EventProp compiler doesn't support "
            f"{solver} solver")
    return "\n".join(clines)
