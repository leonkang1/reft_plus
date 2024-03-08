RED = '\033[31m'  # Red text
GREEN = '\033[32m'  # Green text
YELLOW = '\033[33m'  # Yellow text
BLUE = '\033[34m'  # Blue text
MAGENTA = '\033[35m'  # Magenta text
CYAN = '\033[36m'  # Cyan text
RESET = '\033[0m'  # Reset to default color
BRIGHT_RED = '\033[91m'  # Bright red text
BRIGHT_GREEN = '\033[92m'  # Bright green text
BRIGHT_YELLOW = '\033[93m'  # Bright yellow text
BRIGHT_BLUE = '\033[94m'  # Bright blue text
BRIGHT_MAGENTA = '\033[95m'  # Bright magenta text
BRIGHT_CYAN = '\033[96m'  # Bright cyan text
BRIGHT_WHITE = '\033[97m'  # Bright white text

def rank0Print(rank, msg, color=RESET):
    if rank == 0:
        print(f"{color}{msg}{RESET}")
        
def rankPrint(rank, msg, color=RESET):
    print(f"{color}{rank}: {msg}{RESET}")
    
def nprint(msg, color=RESET):
    print(f"{color}{msg}{RESET}")