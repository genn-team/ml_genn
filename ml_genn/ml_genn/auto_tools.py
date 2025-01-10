import sympy
from sympy.parsing.sympy_parser import parse_expr


def add(o, expr):
    if o is None:
        return expr
    else:
        return o+expr

def get_symbols(vars, params, w_name=None):
    sym = {}
    for var in vars:
        sym[var] = sympy.Symbol(var)
    for pname in params:
        sym[pname] = sympy.Symbol(pname)
    if w_name is not None:
        sym[w_name] = sympy.Symbol(w_name)
    for var in vars:
        sym[f"Lambda{var}"] = sympy.Symbol(f"Lambda{var}")
    return sym
    
"""
THis function turns ODEs expressed as two lists for sympy variables and matching rhs expressions 
into C code to update the variables with timestep "dt"
"""

def linear_euler(varname, sym, exprs, dt):
    code = []
    for var, expr in exprs.items():
        code.append(sympy.ccode(sym[var]+expr*dt, assign_to= f"const scalar {str(var)}_tmp"))
    for var in exprs:
        code.append(f"{var} = {var}_tmp;")
    return code

    
""" 
This modified from Brian 2's exponential_euler stateupdater
"""

def get_conditionally_linear_system(vars, exprs):
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
    for var, expr in zip(vars,exprs):
        name = str(var)
        if expr.has(var):
            # Factor out the variable
            expr = expr.expand()
            expr = sympy.collect(expr, var, evaluate=False)
            if len(expr) > 2 or var not in expr:
                raise ValueError(
                    f"The expression '{expr}', defining the variable "
                    f"'{name}', could not be separated into linear "
                    "components."
                )
            coefficients[name] = (expr[var], expr.get(1, 0))
        else:
            coefficients[name] = (0, expr)

    return coefficients


def exponential_euler(varname, sym, exprs, dt):
    vars= [ sym[var] for var in varname ]
    the_exprs= [ expr for var, expr in exprs.items() ]
    # Try whether the equations are conditionally linear
    try:
        system = get_conditionally_linear_system(vars,the_exprs)
    except ValueError:
        raise NotImplementedError(
            "Can only solve conditionally linear systems with this state updater."
        )

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

# solde a set of ODEs. They can be passed as a dict of strings
# or dict of sympy expressions
def solve_ode(vars, sym, ode, dt, solver):
    dx_dt = {}
    for var, expr in ode.items():
        if isinstance(expr, str):
            dx_dt[var] = parse_expr(expr,local_dict= sym)
        else:
            dx_dt[var] = expr
    if solver == "exponential_euler":
        clines = exponential_euler(vars, sym, dx_dt, dt)
    elif solver == "linear_euler":
        clines = linear_euler(vars, sym, dx_dt, dt)
    else:
        raise NotImplementedError(
            f"EventProp compiler doesn't support "
            f"{solver} solver")
    return dx_dt, clines


# the values that need to be saved in the forward pass
def saved_vars(varname, sym, adj_ode, adj_jump, add_to_pre):
    saved = set()
    all = [ adj_ode, adj_jump, add_to_pre ]
    for expr_list in all:
        for var in varname:
            for v2, expr in expr_list.items():
                if expr.has(sym[var]):
                    saved.add(var)

    return saved

# one could reduce saved vars by solving the threshold equation for one of the vars and substituting the equation
def simplify_using_threshold(varname, sym, g, expr):
    the_var = None
    for var in varname:
        if g.has(sym[var]):
            the_var = sym[var]
            break

    if the_var is None:
        return adj_jump, add_to_pre

    sln = sympy.solve(g, the_var)

    if len(sln) != 1:
        return expr
    
    new_expr = {}
    for var, ex in expr.items():
        new_expr[var] = ex.subs(the_var,sln[0])
    return new_expr
