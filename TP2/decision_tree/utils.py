def tab(depth):
    return f"{'|'.join([' ' for _ in range(depth)])}|_"

def cprint(str, c, d):
    colors = {'g': '\033[92m', 'b': '\033[94m', 'y': '\033[93m', 'r': '\033[91m'}
    print(f'{tab(d)}{colors[c]}{str}\033[0m')