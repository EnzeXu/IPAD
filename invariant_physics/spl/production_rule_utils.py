import ast
from copy import deepcopy
import itertools

from sympy import expand, simplify


def simplify_eq(eq):
    return str(expand(simplify(eq)))

def simplify_eqs(eqs):
    simplified_eqs = []
    for eq in eqs:
        try:
            simplified_eqs.append(str(expand(simplify(eq))))
        except Exception as e:
            print(e)
            print(f"error in eq: {eq}")
            simplified_eqs.append("")
    return simplified_eqs

def prune_poly_c(eq):
    '''
    if polynomial of C appear in eq, reduce to C for computational efficiency. 
    '''
    eq = simplify_eq(eq)
    if 'C**' in eq:
        c_poly = ['C**'+str(i) for i in range(10)]
        for c in c_poly:
            if c in eq: eq = eq.replace(c, 'C')
    return simplify_eq(eq)

def get_nonterminal_symbols(rulemap):
    return set([r[:r.find("->")] for r in rulemap])

def get_nonterminal_rules(rulemap):
    nonterminal_rules = []
    nonterminal_symbols = get_nonterminal_symbols(rulemap)
    for r in rulemap:
        for s in nonterminal_symbols:
            if s in r[r.find("->"):]:
                nonterminal_rules.append(r)
                break
    return nonterminal_rules

def get_current_eq_length(prods, rulemap):
    ntr = get_nonterminal_rules(rulemap)
    tr = [r for r in rulemap if r not in ntr]
    
    num_ntn = 1
    current_eq_length = None
    for i, p in enumerate(prods):
        if num_ntn == 0:
            current_eq_length = i
            break
        elif p in tr:
            num_ntn -= 1
        elif "f" in p:
            num_ntn += p.count("A") - 1
        else:
            num_ntn += p.count("A") - 2
    if num_ntn == 0 and current_eq_length == None:
        current_eq_length = i+1
    return current_eq_length

def is_symmetric(rule):
    symmetric_ops = ['A+A', 'A*A']
    for o in symmetric_ops:
        if o in rule:
            return True
    return False

def to_eq(prods):
    seq = ['f']
    for prod in prods:
        if str(prod[0]) == 'Nothing':
            break
        for ix, s in enumerate(seq):
            if s == prod[0]:
                seq = seq[:ix] + list(prod[3:]) + seq[ix+1:]
                break
    if len(seq) == 1:
        seq = ['A']
        for prod in prods:
            if str(prod[0]) == 'Nothing':
                break
            for ix, s in enumerate(seq):
                if s == prod[0]:
                    seq = seq[:ix] + list(prod[3:]) + seq[ix+1:]
                    break
        return "".join(seq)
    else:
        return "".join(seq)

def generate_symmetric_prods(prods, rulemap):
    if len(prods) <= 1:
        return [prods]
    next_eq_length = get_current_eq_length(prods[1:], rulemap)
    if not next_eq_length:
        raise Exception(f"input prods do not terminate: {prods}")
    left = generate_symmetric_prods(prods[1:next_eq_length+1], 
                                    rulemap)
    right = generate_symmetric_prods(prods[next_eq_length+1:], rulemap)
    res = []
    for l, r in itertools.product(left, right):
        res.append([prods[0]] + l + r)
        
    if is_symmetric(prods[0]):
        for r, l in itertools.product(right, left):
            res.append([prods[0]] + r + l)
        return res
    else:
        return res
    
def is_symmetric_op(op):
    symmetric_ops = [ast.Mult, ast.Add, '*', '+']
    for o in symmetric_ops:
        if type(o) == type and isinstance(op, o):
            return True
        elif type(o) == str and op == o:
            return True
    return False

def is_simple(exp):
    if isinstance(exp, ast.Name) or (isinstance(exp, ast.Constant) and exp.value < 3 and exp.value > -3):
        return True

def is_equal(exp, exp_rule):
    if isinstance(exp, ast.Constant) and isinstance(exp_rule, ast.Name):
        if exp_rule.id == 'C':
            return True
        return False
    elif isinstance(exp, ast.Name) and isinstance(exp_rule, ast.BinOp):
        # when x = x*1 = x*C = C*x
        if isinstance(exp_rule.left, ast.Name) and exp_rule.left.id == exp.id and\
            isinstance(exp_rule.right, ast.Name) and exp_rule.right.id == 'C':
            return True
        if isinstance(exp_rule.right, ast.Name) and exp_rule.right.id == exp.id and\
            isinstance(exp_rule.left, ast.Name) and exp_rule.left.id == 'C':
            return True
        return False
    elif type(exp) != type(exp_rule):
        return False
    elif isinstance(exp, ast.Expression):
        return is_equal(exp.body, exp_rule.body)
    elif isinstance(exp, ast.BinOp):
        res = is_equal(exp.left, exp_rule.left) and is_equal(exp.right, exp_rule.right)
        if is_symmetric_op(exp.op):
            return res or (is_equal(exp.left, exp_rule.right) and is_equal(exp.right, exp_rule.left))
        return res
    elif isinstance(exp, ast.Name):
        if exp.id == exp_rule.id:
            return True
        return False
    elif isinstance(exp, ast.Call):
        if exp.func.id == exp_rule.func.id:
            return all([is_equal(arg, arg_rule) for arg, arg_rule in zip(exp.args, exp_rule.args)])
        return False
    elif isinstance(exp, ast.Constant):
        if exp.value == exp_rule.value:
            return True
        return False
    else:
        raise Exception(f"Type not supported: {type(exp)} and {type(exp_rule)}")

def preprocess_exp(exp):
    if isinstance(exp, ast.UnaryOp) and isinstance(exp.op, ast.USub):
        if isinstance(exp.operand, ast.Constant):
            return ast.Constant(value=-exp.operand.value)
        elif isinstance(exp.operand, ast.Name):
            raise NotImplementedError
    elif isinstance(exp, ast.Expression):
        return ast.Expression(body=preprocess_exp(exp.body))
    elif isinstance(exp, ast.BinOp):
        return ast.BinOp(left=preprocess_exp(exp.left), 
                         right=preprocess_exp(exp.right),
                         op=exp.op)
    elif isinstance(exp, ast.Call):
        return ast.Call(func=exp.func, args=[preprocess_exp(arg) for arg in exp.args])
    else:
        return exp
        
def to_prod(exp, ntn, terminal_rules, nt_rules): # False if not possible, prods otherwise   
#     print(ast.unparse(exp), ntn)
    if isinstance(exp, ast.Expression):
        return to_prod(exp.body, ntn, terminal_rules, nt_rules)
    
    # if is terminal rule
    for r in terminal_rules:
        if ntn != r[:r.find("->")]:
            continue
        exp_r = ast.parse(r[r.find("->")+2:], "", "eval").body
#         print(ast.unparse(exp), exp_r, r)
        if is_equal(exp, exp_r):
            return [r]
    
    # if is nonterminal rule
    if isinstance(exp, ast.BinOp):
        # check op, find ntrs that fits, 
        exp_temp = deepcopy(exp)
        exp_temp.left = ast.Name(id='A', ctx=ast.Load()) # A just as a placeholder
        exp_temp.right = ast.Name(id='A', ctx=ast.Load())
        exp_temp_string = "".join(ast.unparse(exp_temp).split(" ")) # A[op]A
        op = exp_temp_string[1:-1]
        ntrs = [r for r in nt_rules if op in r[r.find("->")+2:] and ntn == r[:r.find("->")]]
#         print(ntrs, op)
#         raise
        # check if left or right is constant (ntrs restricted to single op)
        # check if left or right is Name
        if isinstance(exp.left, ast.Constant):
            ntr = [r for r in ntrs if f"C{op}" in r]
            if len(ntr) != 1 and is_symmetric_op(op):
                # check for symmetry
                ntr = [r for r in ntrs if f"{op}C" in r]
                if len(ntr) == 1:
                    left_ntn, _ = ntr[0][ntr[0].find("->")+2:].replace('(', '')\
                                                        .replace(')', '').split(op)
                    prods = to_prod(exp.right, left_ntn, terminal_rules, nt_rules)
                    if not prods:
                        pass
                    else:
                        return [ntr[0]] + prods
                else:
                    pass
            elif len(ntr) == 1:
                _, right_ntn = ntr[0][ntr[0].find("->")+2:].replace('(', '')\
                                                        .replace(')', '').split(op)
                prods = to_prod(exp.right, right_ntn, terminal_rules, nt_rules)
                if not prods:
                    pass
                else:
                    return [ntr[0]] + prods 
        elif isinstance(exp.right, ast.Constant):
            ntr = [r for r in ntrs if f"{op}C" in r]
            if len(ntr) != 1 and is_symmetric_op(op):
                # check for symmetry
                ntr = [r for r in ntrs if f"C{op}" in r]
                if len(ntr) == 1:
                    _, right_ntn = ntr[0][ntr[0].find("->")+2:].replace('(', '')\
                                                        .replace(')', '').split(op)
                    prods = to_prod(exp.left, right_ntn, terminal_rules, nt_rules)
                    if not prods:
                        pass
                    else:
                        return [ntr[0]] + prods
                else:
                    pass
            elif len(ntr) == 1:
                left_ntn, _ = ntr[0][ntr[0].find("->")+2:].replace('(', '')\
                                                        .replace(')', '').split(op)
                prods = to_prod(exp.left, left_ntn, terminal_rules, nt_rules)
                if not prods:
                    pass
                else:
                    return [ntr[0]] + prods 
        elif isinstance(exp.left, ast.Name):
            # TODO
            pass
        elif isinstance(exp.right, ast.Name):
            # TODO
            pass
        
        ntrs_ntn_only = []
        ntns = get_nonterminal_symbols(terminal_rules + nt_rules)
        for r in ntrs:
            count = 0
            for n in ntns:
                count += r.count(n)
            if count == 3:
                ntrs_ntn_only.append(r)
            elif count == 2:
                continue
            else:
                raise Exception(f"More/less ntn than expected (2 or 3 expected): {r}")
        
        left, right = None, None
        for ntr in ntrs_ntn_only:
            ntn_two = ntr[ntr.find("->")+2:].replace('(', '')\
                                                        .replace(')', '').split(op)
#             print(ntr, ntn_two)
            left_ntn, right_ntn = ntn_two
            left, right = to_prod(exp.left, left_ntn, terminal_rules, nt_rules), \
                          to_prod(exp.right, right_ntn, terminal_rules, nt_rules)
            if not left or not right or (left.count("-&>") + right.count("-&>") > 1):
                # TODO: if something like x**3 appear, propose adding this
                continue
            else:
                break
        if left and right and left != False and right != False:
            return [ntr] + left + right 
        elif is_simple(exp.left) and is_simple(exp.right):
#             print(f"Recommend adding: {ntn}->{ast.unparse(exp).replace(' ', '')}")
            return [f"{ntn}-&>({ast.unparse(exp).replace(' ', '')})"]
        else:
            pass
    elif isinstance(exp, ast.Call):
        # check id only (restricted to one function / no op allowed in arguments)
#         print(ast.unparse(exp))
        raise NotImplementedError
        # need terminal symbols
#         ntrs = [r for r in nt_rules if exp.func.id in r[r.find("->")+2:] and ntn == r[:r.find("->")]]
#         args = exp.args
#         ntns = get_nonterminal_symbols(terminal_rules + nt_rules)
#         for r in ntrs:
#             # TODO: change the above to this format
# #             print(r[r.find("->")+2:])
#             exp_r = ast.parse(r[r.find("->")+2:]).body
#             if len(exp_r) != 1:
#                 raise Exception("What? Exp_r is", ast.dump(exp_r))
#             else:
#                 exp_r = exp_r[0].value
# #             print(ast.dump(exp_r.value))
#             print([ast.dump(a) for a in exp_r.args])
# #             raise
#             for arg, r_arg in zip(arg, exp_r.args):
#                 if r_arg.id in ntns:
#                     pass
#                 elif r_arg
#                 to_prod(arg, ntn, terminal_rules, nt_rules)
#         if exp.func.id == exp_rule.func.id:
#             return all([is_equal(arg, arg_rule) for arg, arg_rule in zip(exp.args, exp_rule.args)])
#         return False
#     elif isinstance(exp, ast.Name):
#         pass
#     elif isinstance(exp, ast.Constant):
#         pass
    elif isinstance(exp, ast.Constant):
#         print(f"Recommend adding: {ntn}->C")
        return [f"{ntn}-&>C"]
    else:
        print("Unrecognized expression", ast.dump(exp))
        return False
        
#     print("Asdf")
#     ntns = get_nonterminal_symbols(terminal_rules + nt_rules)
#     for n in ntns:
#         if n in ast.unparse(exp):
#             return False
#     print(f"Recommend add {ntn}->{ast.unparse(exp).replace(' ', '')}")
#     return [f"{ntn}-&>({ast.unparse(exp).replace(' ', '')})"]
    return False


def to_common_simplified_skeleton(spl_model, func_score, best_module, data_list):
    # convert best solution to eq for each env
    # choose shortest prods
    # test on train: if >= result, good. If not, try different prods
    # Convert prods to eq and print. Test on this eq.
    nt_rules = get_nonterminal_rules(spl_model.base_grammars)
    t_rules = [r for r in (spl_model.base_grammars + spl_model.added_basic_grammars) if r not in nt_rules]
    train_score_rmse_only, eqs = func_score(spl_model.tree_to_eq(['f->A'] + best_module[0].split(',')), 
                                                       0, data_list)
    simplified_eq = max(simplify_eqs(eqs), key=lambda x: len(x))
#     prods_list = []
#     for eq in simplify_eqs(best_module[2]):
    exp = ast.parse(simplified_eq, "", "eval")
    exp = preprocess_exp(exp)
    prods = to_prod(exp, 'A', t_rules, nt_rules)
    prods = [p.replace('-&>', '->') for p in prods]
    score, eqs = func_score(spl_model.tree_to_eq(['f->A'] + prods), 
                            len(prods)+1, data_list)
    prods = ",".join(prods)
    return prods, score, eqs