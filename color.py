RED     = "\x1b[31m"
GREEN   = "\x1b[32m"
YELLOW  = "\x1b[33m"
GREY    = "\x1b[90m"

RESET   = "\x1b[0m"


def red(text: str) -> str:
    return f"{RED}{text}{RESET}"

def green(text: str) -> str:
    return f"{GREEN}{text}{RESET}"

def yellow(text: str) -> str:
    return f"{YELLOW}{text}{RESET}"

def grey(text: str) -> str:
    return f"{GREY}{text}{RESET}"