import sympy
from sympy.parsing.sympy_parser import parse_expr
from auto_tools import *

DEBUG = False

varname= [ "V", "I", "a" ]
parname = [ "taum", "taus", "theta" ]
w_name = "w"

eqns = {
    "V": "(-V+I)/taum",   # the ODE for V
    "I": "-1/taus*I",       # the ODE for I
    "a": "-a/taus"
}
threshold = "V-theta" # threshold function (condition == 0)
reset = {"V": "0",         # V value after reset
         "I": "I",         # I value after reset
         "a": "a"
}
jumps = {
    "V": "V+0",          # V value after synaptic jump
    "I": "I+a*w",           # I value after synaptic jump
    "a": "a+w"
}

sym = {}
for var in varname:
    sym[var] = sympy.Symbol(var)

for pname in parname:
    sym[pname] = sympy.Symbol(pname)

sym[w_name] = sympy.Symbol(w_name)

dt= sympy.Symbol("dt")

# forward dynamics
dx_dt = {}
for var in varname:
    if var in eqns:
        dx_dt[var] = parse_expr(eqns[var],local_dict= sym)
        
# threshold condition
g = parse_expr(threshold)

# reset function
f = {}
for var in varname:
    if var in reset:
        f[var] = parse_expr(reset[var],local_dict= sym)
        
# synaptic jumps
h = {}
for var in varname:
    if var in jumps:
        tmp = parse_expr(jumps[var],local_dict= sym)-sym[var]
        if sympy.diff(tmp, sym[var]) == 0:
            h[var] = tmp
        else:
            raise NotImplementedError(
                "EventProp compiler only supports "
                "synapses which (only) add input to a target variable.")
# adjoint variables
adj_name = {}
for var in varname:
    adj_name[var] = f"lambda_{var}"
    sym[adj_name[var]] = sympy.Symbol(adj_name[var])
    
# after jump dynamics equation "\dot{x}^+"
dx_dtplusn = {}
for var, expr in dx_dt.items():
    plus = expr
    for v2, f2 in f.items():
        plus = plus.subs(sym[v2],f2)
        dx_dtplusn[var] = plus

dx_dtplusm = {}
for var, expr in dx_dt.items():
    plus = expr
    for v2, h2 in h.items():
        plus = plus.subs(sym[v2],sym[v2]+h2)
    dx_dtplusm[var] = plus

# generate the adjoint dynamics
ode = {}

for var in varname:
    o = None
    for v2, expr in dx_dt.items():
        o = add(o, sympy.diff(expr, sym[var])*sym[adj_name[v2]])

    ode[var] = o

# generate the jumps in lambda^n adjoint variables, n == spiking neuron

# first, calculate all terms involving lambda^n+
expr = [{} for i in range(5)]
for var in varname:
    ex= None
    for v2,f2 in f.items():
        ex = add(ex, sympy.diff(f2,sym[var])*sym[adj_name[v2]])
    expr[0][var] = ex

    ex= None
    for v2 in varname:
        ex = add(ex, sympy.diff(g,sym[v2])*dx_dt[v2])
    ex = sympy.simplify(ex)
    
    if ex != 0:
        ex = sympy.diff(g,sym[var])/ex
        if ex != 0:
            expr[1][var] = ex
        
    # TODO: dlp/dtk + lk- + lk+ here

    if var in expr[1]:
        ex= None
        for v2,f2 in f.items():
            ex2= None
            for v3 in varname:
                ex2 = add(ex2, sympy.diff(f2,sym[v3])*dx_dt[v3])
            ex2 = add(ex2,-dx_dtplusn[v2])
            ex = add(ex,-sym[adj_name[v2]]*ex2)
        expr[2][var] = ex

# assemble the lambda^n+ parts
jumps = {}
for var in varname:
    ex = None
    ex = add(ex, expr[0][var])
    if var in expr[1] and var in expr[2]:
        ex = add(ex, expr[1][var] * expr[2][var])
    jumps[var] = ex

# Second, calculate all the terms related to other lambda^m+
# These are "addToPre" situations

# the term due to the potential x-dependence of the synaptic jumps h


for var in varname:
    ex = None
    for v2,h2 in h.items():
        ex = add(ex, sympy.diff(h2,sym[v2])*sym[adj_name[var]])
    if ex is not None:
        expr[3][var] = ex

# the term due to the w dependence of the synaptic jumps h
for var in varname:
    ex = None
    for v2, expr2 in dx_dt.items():
        ex2 = None
        ex2 = add(ex2, dx_dtplusm[v2] - expr2)
        if v2 in h:
            for v3, expr3 in dx_dt.items():
                ex2 = add(ex2, sympy.diff(h[v2],sym[v3])*expr3)
        ex = add(ex, sym[adj_name[v2]]*ex2)
    expr[4][var] = sympy.simplify(ex)

# assemble add_to_pre
add_to_pre = {}
for var in varname:
    ex = None
    if var in expr[3]:
        ex = add(ex, expr[3][var])
    if var in expr[1] and var in expr[4]:
        ex = add(ex, expr[1][var] * expr[4][var])
    if ex is not None:
        add_to_pre[var] = sympy.simplify(ex)

if DEBUG:
    for i in range(len(expr)):
        print(f"expression {i}:")
        for ex in expr[i]:
            print(ex)
        
eeuler = exponential_euler(varname, sym, dx_dt, dt)
adj = [ adj_name[var] for var in varname ]
eeuler_adj = exponential_euler(adj, sym, ode, dt)

lineuler = linear_euler(varname, sym, dx_dt, dt)
lineuler_adj = linear_euler(adj, sym, ode, dt)

jumps, add_to_pre = simplify_using_threshold(varname, sym, g, jumps, add_to_pre)

grad_update = None
for var in varname:
        grad_update = add(grad_update, -sym[adj_name[var]]*sympy.diff(h[var],sym[w_name]))

print(grad_update)

jump_c = []
updated = []
for var in varname:
    if var in jumps:
        if sympy.simplify(jumps[var] - sym[adj_name[var]]) != 0:
            jump_c.append(sympy.ccode(jumps[var], assign_to= f"const scalar {adj_name[var]}_tmp"))
            updated.append(adj_name[var])
for adj in updated:
    jump_c += [f"{adj} = {adj}_tmp;"]

add_to_pre_c = []
for var in varname:
    if var in add_to_pre and add_to_pre[var] != 0:
        code = sympy.ccode(add_to_pre[var])
        add_to_pre_c.append(f"addToPre({adj_name[var]},{code});")

grad_update_c = sympy.ccode(grad_update)

print("Exponential Euler:")
print("\n".join(eeuler))
print("\n".join(eeuler_adj))

print("Linear Euler:")
print("\n".join(lineuler))
print("\n".join(lineuler_adj))


print("Saved vars:")
print(saved_vars(varname, sym, ode, jumps, add_to_pre))

print("Adjoint jumps:")
for code in jump_c:
    print(code)

print("Add to pre:")
for code in add_to_pre_c:
    print(code)

print("Gradient update:")
print(grad_update_c)
