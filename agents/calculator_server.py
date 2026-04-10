'''
A minimal MCP server that exposes a safe calculator tool.

Run standalone to test:  python calculator_server.py
Used by 03_agent_mcp.py via MCPServerStdio (spawned as a subprocess).
'''
#%% [markdown]
# # MCP Calculator Server
#
# This is a **local MCP server** built with FastMCP.
# It exposes one tool: `calculate(expression)`.
#
# Why a separate server instead of a regular PydanticAI tool?
#
# - MCP servers are **language-agnostic** — any client can call them
# - They can be reused across different agents and frameworks
# - They run in a separate process, so a crash won't affect the agent
# - This shows the Model Context Protocol integration pattern

from mcp.server.fastmcp import FastMCP
import ast
import operator

#%% [markdown]
# ## Server Instance

mcp = FastMCP("calculator")

#%% [markdown]
# ## Safe Math Evaluator
#
# We use Python's `ast` module to parse the expression into a syntax tree
# and evaluate only allowed operations — no `eval()` with arbitrary code.

# Supported binary operators
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}

# Supported unary operators
UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate an AST node using only allowed operations."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPERATORS:
        operand = _safe_eval(node.operand)
        return UNARY_OPERATORS[type(node.op)](operand)
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


#%% [markdown]
# ## The Calculator Tool
#
# FastMCP turns this function into an MCP tool automatically.
# The docstring becomes the tool description that the LLM sees.

@mcp.tool()
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the numeric result.

    Supports: +, -, *, /, **, % (modulo), // (floor division).
    Examples: "2 + 3 * 4", "(10 - 4) ** 2 / 3", "17 % 5"

    Args:
        expression: A math expression as a plain string, e.g. "3 * (2 + 5)"

    Returns:
        The result as a string, or an error message.
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree)
        # Return integer representation when the result is a whole number
        if result == int(result):
            return str(int(result))
        return str(round(result, 10))
    except ZeroDivisionError:
        return "Error: division by zero"
    except ValueError as e:
        return f"Error: {e}"
    except SyntaxError:
        return f"Error: invalid expression '{expression}'"


#%% [markdown]
# ## Run as stdio server (used by MCPServerStdio)

if __name__ == "__main__":
    mcp.run(transport="stdio")
