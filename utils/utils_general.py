from colorama import Fore, Style, init

def print_header(string, color='magenta'):
    color_map = {
        'magenta': Fore.MAGENTA,
        'cyan': Fore.CYAN,
        'red': Fore.RED,
        'green': Fore.GREEN,
        'yellow': Fore.YELLOW,
        'blue': Fore.BLUE,
        'white': Fore.WHITE
    }
    selected_color = color_map.get(color.lower(), Fore.MAGENTA)
    line = '-' * len(string)
    print('\n\n')
    print(selected_color + line)
    print(selected_color + string)
    print(selected_color + line + Style.RESET_ALL)


def print_parameters(params):
    print(Fore.CYAN + "----------------")
    print(Fore.CYAN + "Parameters")
    print(Fore.CYAN + "----------------" + Style.RESET_ALL)
    for key, value in params.items():
        print(f"{key}: {value}")
    print("\n")