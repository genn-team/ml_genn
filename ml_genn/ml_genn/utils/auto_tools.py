import sympy

"""
THis function turns ODEs expressed as two lists for sympy variables and matching rhs expressions 
into C code to update the variables with timestep "dt"
"""

def _linear_euler(sym, dx_dt, dt):
    code = [sympy.ccode((sym[var] + expr) * dt,
                        assign_to=f"const scalar {str(var)}_tmp")
            for var, expr in dx_dt.items()]
    code += [f"{var} = {var}_tmp;" for var in exprs]
    return code

    
""" 
This modified from Brian 2's exponential_euler stateupdater
"""

def _get_conditionally_linear_system(vars, exprs):
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
    for var, expr in zip(vars, exprs):
        name = str(var)
        if expr.has(var):
            # Factor out the variable
            expr = expr.expand()
            expr = sympy.collect(expr, var, evaluate=False)
            if len(expr) > 2 or var not in expr:
                raise ValueError(
                    f"The expression '{expr}', defining the variable "
                    f"'{name}', could not be separated into linear "
                    "components.")
            coefficients[name] = (expr[var], expr.get(1, 0))
        else:
            coefficients[name] = (0, expr)

    return coefficients


def _exponential_euler(varname, sym, exprs, dt):
    vars = [sym[var] for var in varname]
    the_exprs = [expr for var, expr in exprs.items()]
    # Try whether the equations are conditionally linear
    try:
        system = _get_conditionally_linear_system(vars, the_exprs)
    except ValueError:
        raise NotImplementedError(
            "Can only solve conditionally linear systems with this state updater.")

    code = []
    for var, (A, B) in system.items():
        s_var = sympy.Symbol(var)
        s_dt = sympy.Symbol("dt")
        if A == 0:
            update_expression = s_var + s_dt * B
        elif B != 0:
            BA = B / A
            # Avoid calculating B/A twice
            BA_name = f"BA_{var}_tmp"
            s_BA = sympy.Symbol(BA_name)
            code += [f"const scalar {BA_name} = {sympy.ccode(BA)};"]
            update_expression = (s_var + s_BA) * sympy.exp(A * s_dt) - s_BA
        else:
            update_expression = s_var * sympy.exp(A * s_dt)
            
        # The actual update step
        code += [f"const scalar {var}_tmp = {sympy.ccode(update_expression)};"]

    # Replace all the variables with their updated value
    for var in system:
        code += [f"{var} = {var}_tmp;"]

    return code

"""
End of Brian 2 modified code
"""

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
def solve_ode(vars, sym, dx_dt, dt, solver):
    if solver == "exponential_euler":
        clines = _exponential_euler(vars, sym, dx_dt, dt)
        print(clines)
    elif solver == "linear_euler":
        clines = _linear_euler(sym, dx_dt, dt)
    else:
        raise NotImplementedError(
            f"EventProp compiler doesn't support "
            f"{solver} solver")
    return clines


# one could reduce saved vars by solving the threshold equation for one of the vars and substituting the equation
# **THOMAS** comments here would be nice
def simplify_using_threshold(varname, sym, g, expr):
    if g is None:
        return expr

    try:
        the_var = next(sym[v] for v in varname if g.has(sym[v]))
    except StopIteration:
        return adj_jump, add_to_pre

    sln = sympy.solve(g, the_var)

    if len(sln) != 1:
        return expr

    if isinstance(expr, dict):
        return {var: ex.subs(the_var, sln[0])
                for var, ex in expr.items()}
    else:
        return expr.subs(the_var,sln[0])
