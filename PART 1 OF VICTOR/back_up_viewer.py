import os
import hashlib
from Crypto.Cipher import AES
from xml.etree import ElementTree as ET
from datetime import datetime

# --- CONFIG ---
FILE_PATH = r"E:/docu2025/sms-20240329011048.xml"  # Your file
PASSWORD = ""  # Put your password here if encrypted, else leave blank

def decrypt_aes256(data, password):
    """Decrypts AES-256 CBC encrypted data using password."""
    key = hashlib.sha256(password.encode()).digest()
    iv = data[:16]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted = cipher.decrypt(data[16:])
    pad_len = decrypted[-1]
    if pad_len > AES.block_size:
        raise ValueError("Incorrect password or corrupted file (bad padding).")
    return decrypted[:-pad_len]

def read_sms_file(file_path, password=""):
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return None

    try:
        with open(file_path, "rb") as f:
            data = f.read()
    except Exception as e:
        print(f"[ERROR] Failed to read file: {e}")
        return None

    # Detect if file looks like XML or encrypted
    if data.strip().startswith(b"<"):
        print("[INFO] File is plain XML, no decryption needed.")
        xml_content = data.decode("utf-8", errors="ignore")
    else:
        if not password:
            print("[ERROR] File is encrypted. Please set PASSWORD in script.")
            return None
        print("[INFO] File is encrypted. Attempting AES-256 decryption...")
        try:
            xml_content = decrypt_aes256(data, password).decode("utf-8", errors="ignore")
        except Exception as e:
            print(f"[ERROR] Decryption failed: {e}")
            return None

    return xml_content

def parse_and_display(xml_content):
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        print(f"[ERROR] Failed to parse XML: {e}")
        return

    messages = root.findall("sms")
    print(f"[INFO] Found {len(messages)} messages")
    print("=" * 50)

    for sms in messages[:50]:  # Show first 50 for quick preview
        try:
            date_ms = int(sms.get("date"))
            date_readable = datetime.fromtimestamp(date_ms / 1000).strftime('%Y-%m-%d %I:%M:%S %p')
        except:
            date_readable = sms.get("readable_date", "Unknown Date")

        print(f"[{sms.get('type', '?')}] {sms.get('address', 'Unknown')} @ {date_readable}")
        print(sms.get("body", "").strip())
        print("-" * 50)

if __name__ == "__main__":
    print("[INFO] Starting SMS Reader...")
    xml_data = read_sms_file(FILE_PATH, PASSWORD)
    if xml_data:
        parse_and_display(xml_data)
    else:
        print("[ERROR] No data to display.")
