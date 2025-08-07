from sympy import simplify, factor, sqrt, Pow, Basic, Integer, Mul, Add, collect, together
from sympy import Eq, ratsimp, expand, Abs, Rational, Number, sympify, symbols, integrate
from sympy import sin, cos, tan, asin, acos, atan, Symbol
from operator import add, sub, mul, truediv
from sympy import sin, cos, exp, log  # add exp, log; light types
# utils/custom_functions.py
from sympy import Pow, sqrt as sym_sqrt, simplify, count_ops
from sympy.simplify.powsimp import powdenest, powsimp
OPS_SIMPLIFY_LIMIT = 15  # try 15–30; lower = safer/faster

# utils/safe_eval.py
import signal
from contextlib import contextmanager

class SympyTimeout(Exception): pass

@contextmanager
def time_limit(seconds: float):
    def _handler(signum, frame): raise SympyTimeout()
    old = signal.signal(signal.SIGALRM, _handler)
    # seconds can be fractional via ITIMER_REAL
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old)


def _cheap_norm(expr):
    # very fast, avoids trig/FU pipeline
    expr = powdenest(expr, force=True)
    expr = powsimp(expr, force=True, combine='exp')
    # ratsimp is relatively cheap and cancels rational garbage
    expr = ratsimp(expr)
    return expr


def custom_sin(expr, term):
    return sin(expr)

def custom_cos(expr, term):
    return cos(expr)

def custom_exp(expr, term):
    return exp(expr)

def custom_log(expr, term):
    return log(expr)

def custom_identity(expr, term):
    return expr

def custom_expand(expr, term):
    return expand(expr)

def custom_simplify(expr, term):
    return simplify(expr)

def custom_factor(expr, term):
    return factor(expr)

def custom_collect(expr, term):
    return collect(expr, term)

def custom_together(expr, term):
    return together(expr)

def custom_ratsimp(expr, term):
    return ratsimp(expr)

def custom_square(expr, term):
    return expr**2


def _cheap_norm(expr):
    # very fast, avoids trig/FU pipeline
    expr = powdenest(expr, force=True)
    expr = powsimp(expr, force=True, combine='exp')
    # ratsimp is relatively cheap and cancels rational garbage
    expr = ratsimp(expr)
    return expr

# def custom_sqrt(expr, term=None):
#     # apply sqrt without global simplify; only do local cheap normalization
#     expr = _cheap_norm(expr)
#     return sqrt(expr)


def custom_sqrt(expr, term=None, *, ops_limit: int = OPS_SIMPLIFY_LIMIT):
    """
    Fast & safe sqrt:
      1) If expr is structurally z**2, return z.
      2) Try powdenest (cheap) to expose hidden squares.
      3) Only if small enough (by count_ops), try simplify to see if it becomes z**2.
      4) Otherwise return sqrt(expr). Never raise.
    """
    try:
        # 1) trivial structural square
        if isinstance(expr, Pow) and expr.exp == 2:
            return expr.base

        # 2) cheap denesting (no trig simpl)
        expr_den = powdenest(expr, force=True)
        if isinstance(expr_den, Pow) and expr_den.exp == 2:
            return expr_den.base

        # 3) guard heavy simplify behind a size check
        if count_ops(expr_den) <= ops_limit:
            simplified = simplify(expr_den)
            if isinstance(simplified, Pow) and simplified.exp == 2:
                return simplified.base

        # 4) default principal branch
        return sym_sqrt(expr)
    except Exception:
        # any symbolic hiccup → no-op to keep the env stable
        return expr


def custom_sqrt_old(expr, term):
    # Check if the expression is a perfect square
    simplified_expr = simplify(expr)

    # Case 1: If it's a square of a single term (like x**2), return the term
    if simplified_expr.is_Pow and simplified_expr.exp == 2:
        base = simplified_expr.base
        return base

    # Case 2: Otherwise, return ±sqrt(expression)
    return sqrt(expr)

def inverse_sin(expr, term):
    if isinstance(expr, (int, float)):
        return asin(expr)
    if expr.has(sin):
        return expr.replace(
            lambda arg: arg.func == sin,
            lambda arg: arg.args[0]
        )
    return asin(expr)

def inverse_cos(expr, term):
    if isinstance(expr, (int, float)):
        return acos(expr)
    if expr.has(cos):
        return expr.replace(
            lambda arg: arg.func == cos,
            lambda arg: arg.args[0]
        )
    return acos(expr)

def inverse_tan(expr, term):
    if isinstance(expr, (int, float)):
        return atan(expr)
    if expr.has(tan):
        return expr.replace(
            lambda arg: arg.func == tan,
            lambda arg: arg.args[0]
        )
    return atan(expr)


operation_names = {
    add: "add",
    sub: "subtract",
    mul: "multiply",
    truediv: "divide",
    custom_expand: "expand",
    custom_simplify: "simplify",
    custom_factor: "factor",
    custom_collect: "collect",
    custom_together: "together",
    custom_ratsimp: "ratsimp",
    custom_square: "square",
    custom_sqrt: "sqrt",
    inverse_sin: 'sin^{-1}',
    inverse_cos: 'cos^{-1}',
    inverse_tan: 'tan^{-1}',
    custom_identity: 'identity',
    custom_sin: "sin",
    custom_cos: "cos",
    inverse_sin: "sin^{-1}",
    inverse_cos: "cos^{-1}",
    inverse_tan: "tan^{-1}",
    custom_exp: "exp",
    custom_log: "log"
}