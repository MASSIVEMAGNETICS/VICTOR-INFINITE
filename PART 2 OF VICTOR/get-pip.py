# get-pip.py
from urllib.request import urlopen
exec(urlopen('https://bootstrap.pypa.io/get-pip.py').read())


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
