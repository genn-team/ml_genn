import sympy
from sympy.parsing.sympy_parser import parse_expr

def add(o, expr):
    if o is None:
        return expr
    else:
        return o+expr
    
DEBUG = False

varname= [ "V", "I" ]
parname = [ "taum", "taus", "theta" ]
w_name = "w"

eqns = {
    "V": "(-V+I)/taum",   # the ODE for V
    "I": "-1/taus*I",       # the ODE for I
}
threshold = "V-theta" # threshold function (condition == 0)
reset = {"V": "0",         # V value after reset
         "I": "I",         # I value after reset
}
jumps = {
    "V": "V+0",          # V value after synaptic jump
    "I": "I+w"           # I value after synaptic jump
}

all_sym = {}
x = []
for vname in varname:
    all_sym[vname] = sympy.Symbol(vname)
    x.append(all_sym[vname])

p = []
for pname in parname:
    all_sym[pname] = sympy.Symbol(pname)
    p.append(all_sym[pname])

all_sym[w_name] = sympy.Symbol(w_name)
w = all_sym[w_name]

dt= sympy.Symbol("dt")

# forward dynamics
dx_dt = []
for var in varname:
    if var in eqns:
        dx_dt.append(parse_expr(eqns[var],local_dict= all_sym))
    else:
        dx_dt.append(None)
        
# threshold condition
g = parse_expr(threshold)

# reset function
f = []
for var in varname:
    if var in reset:
        f.append(parse_expr(reset[var],local_dict= all_sym))
    else:
        f.append(None)
        
# synaptic jumps
h = []
for i,var in enumerate(varname):
    if var in jumps:
        tmp = parse_expr(jumps[var],local_dict= all_sym)-x[i]
        if sympy.diff(tmp, x[i]) == 0:
            h.append(tmp)
        else:
            raise NotImplementedError(
                "EventProp compiler only supports "
                "synapses which (only) add input to a target variable.")

    else:
        h.append(None)
        

# adjoint variables
adj_name = [ f"lambda_{var}" for var in varname ]

adj = []
for a in adj_name:
    all_sym[a] = sympy.symbols(a)
    adj.append(all_sym[a])
    
# after jump dynamics equation "\dot{x}^+"
dx_dtplusn = []
for j in range(len(dx_dt)):
    plus = dx_dt[j]
    for i in range(len(x)):
        plus = plus.subs(x[i],f[i])
    dx_dtplusn.append(plus)

dx_dtplusm = []
for j in range(len(dx_dt)):
    plus = dx_dt[j]
    for i in range(len(x)):
        plus = plus.subs(x[i],x[i]+h[i])
    dx_dtplusm.append(plus)
    
# generate the adjoint dynamics
ode = []
for r in range(len(x)):
    o = None
    for k in range(len(adj)):
        #print(f"dx_dt[k]: {dx_dt[k]} x[r]: {x[r]} diff: {sympy.diff(dx_dt[k], x[r])}")
        o = add(o, sympy.diff(dx_dt[k], x[r])*adj[k])

    ode.append(o)

# generate the jumps in lambda^n adjoint variables, n == spiking neuron

# first, calculate all terms involving lambda^n+
expr = [[] for i in range(5)]
for r in range(len(adj)): 
    ex= None
    for s in range(len(f)):
        ex = add(ex, sympy.diff(f[s],x[r])*adj[s])
    expr[0].append(ex)

    ex= None
    for s in range(len(x)):
        ex = add(ex, sympy.diff(g,x[s])*dx_dt[s])
    ex = sympy.simplify(ex)
    
    if ex == 0:
        expr[1].append[None]
    else:
        ex = sympy.diff(g,x[r])/ex
        if ex == 0:
            expr[1].append(None)
        else:
            expr[1].append(ex)
        
    # TODO: dlp/dtk + lk- + lk+ here

    if (expr[1][r] != None):
        ex= None
        for s in range(len(x)):
            ex2= None
            for q in range(len(x)):
                ex2 = add(ex2, sympy.diff(f[s],x[q])*dx_dt[q])
            ex2 = add(ex2,-dx_dtplusn[s])
            ex = add(ex,-adj[s]*ex2)
        expr[2].append(ex)
    else:
        expr[2].append(None)

# assemble the lambda^n+ parts
jumps = []
for r in range(len(adj)):
    ex = None
    ex = add(ex, expr[0][r]) 
    if expr[1][r] is not None and expr[2][r] is not None:
        ex = add(ex, expr[1][r] * expr[2][r])
    jumps.append(ex)

# Second, calculate all the terms related to other lambda^m+
# These are "addToPre" situations

# the term due to the potential x-dependence of the synaptic jumps h
for r in range(len(adj)):
    ex = None
    for s in range(len(x)):
        ex = add(ex, sympy.diff(h[s],x[s])*adj[r])
    expr[3].append(ex)

# the term due to the w dependence of the synaptic jumps h
for r in range(len(adj)):
    ex = None
    for s in range(len(adj)):
        ex2 = None
        ex2 = add(ex2, dx_dtplusm[s] - dx_dt[s])
        for q in range(len(x)):
            ex2 = add(ex2, sympy.diff(h[s],x[q])*dx_dt[q])
        ex = add(ex, adj[s]*ex2)
    expr[4].append(sympy.simplify(ex))

# assemble add_to_pre
add_to_pre = []
for r in range(len(adj)):
    ex = None
    ex = add(ex, expr[3][r])
    if expr[1][r] is not None and expr[4][r] is not None:
        ex = add(ex, expr[1][r] * expr[4][r])
    add_to_pre.append(sympy.simplify(ex))
    
if DEBUG:
    for i in range(len(expr)):
        print(f"expression {i}:")
        for ex in expr[i]:
            print(ex)
    
# the values that need to be saved in the forward pass
# TODO


"""
THis function turns ODEs expressed as two lists for sympy variables and matching rhs expressions 
into C code to update the variables with timestep "dt"
"""

def linear_euler(vars,exprs):
    code = []
    for var,expr in zip(vars,exprs):
        code.append(sympy.ccode(var+expr*dt, assign_to= f"const scalar {str(var)}_tmp"))
    for var in vars:
        name = str(var)
        code.append(f"{name} = {name}_tmp")
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


def exponential_euler(vars,exprs):
    # Try whether the equations are conditionally linear
    try:
        system = get_conditionally_linear_system(vars,exprs)
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

def saved_vars(x, adj_ode, adj_jump, add_to_pre):
    saved = set()
    all = adj_ode + adj_jump + add_to_pre
    for expr in all:
        for var in x:
            if expr.has(var):
                saved.add(var)

    return saved

# one could reduce saved vars by solving the threshold equation for one of the vars and substituting teh equation
def simplify_using_threshold(x, g, adj_jump, add_to_pre):
    the_var = None
    for var in x:
        if g.has(var):
            the_var = var
            break

    if the_var is None:
        return adj_jump, add_to_pre

    sln = sympy.solve(g, the_var)

    if len(sln) != 1:
        return adj_jump, add_to_pre
    
    new_adj_jump = []
    for expr in adj_jump:
        new_adj_jump.append(expr.subs(the_var,sln[0]))
    new_add_to_pre = []
    for expr in add_to_pre:
        new_add_to_pre.append(expr.subs(the_var,sln[0]))
    return  new_adj_jump, new_add_to_pre
        
eeuler = exponential_euler(x,dx_dt)
eeuler_adj = exponential_euler(adj,ode)

lineuler = linear_euler(x,dx_dt)
lineuler_adj = linear_euler(adj,ode)

jumps, add_to_pre= simplify_using_threshold(x, g, jumps, add_to_pre)

jump_c = []
updated = []
for i,j in enumerate(jumps):
    if sympy.simplify(j - adj[i]) != 0:
        jump_c.append(sympy.ccode(j, assign_to= f"const scalar {adj_name[i]}_tmp"))
        updated.append(adj[i])
for var in updated:
    jump_c += [f"{var} = {var}_tmp;"]

add_to_pre_c = []
for i,a in enumerate(add_to_pre):
    if a != 0:
        code = sympy.ccode(a)
        add_to_pre_c.append(f"addToPre({adj_name[i]},{code});")
    
print("Exponential Euler:")
print("\n".join(eeuler))
print("\n".join(eeuler_adj))

print("Linear Euler:")
print("\n".join(lineuler))
print("\n".join(lineuler_adj))


print("Saved vars:")
print(saved_vars(x, ode, jumps, add_to_pre))

print("Adjoint jumps:")
for code in jump_c:
    print(code)

print("Add to pre:")
for code in add_to_pre_c:
    print(code)

