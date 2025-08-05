===========================================
   SMS Backup Viewer Pro
   by iambandobandz x Victor
===========================================
# ============================================================
# FILE: sms_backup_viewer.py
# VERSION: v1.1.0
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# DESCRIPTION:
#     SMS Backup Viewer Pro
#     - Load SMS Backup XML (plain or encrypted)
#     - AES-256 decrypt support
#     - GUI for viewing messages
#     - Search feature
#     - Export to PDF and TXT
#
# PLATFORM: Windows (Dell Inspiron or similar)
# DEPENDENCIES:
#     pip install pycryptodome lxml fpdf tk
#
# HOW TO RUN:
#     python sms_backup_viewer.py
# HOW TO MAKE EXE:
#     pyinstaller --onefile --noconsole sms_backup_viewer.py
# ============================================================

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from xml.etree import ElementTree as ET
from Crypto.Cipher import AES
import hashlib
from fpdf import FPDF

parsed_messages = []

# ---------------------------
# Decrypt AES Encrypted Backup
# ---------------------------
def decrypt_backup(file_path, password):
    with open(file_path, "rb") as f:
        encrypted_data = f.read()
    key = hashlib.sha256(password.encode()).digest()
    iv = encrypted_data[:16]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted = cipher.decrypt(encrypted_data[16:])
    return decrypted.rstrip(b"\0")

# ---------------------------
# Parse SMS XML
# ---------------------------
def parse_sms(xml_content):
    messages = []
    root = ET.fromstring(xml_content)
    for sms in root.findall("sms"):
        date = sms.get("readable_date") or sms.get("date")
        address = sms.get("address")
        body = sms.get("body")
        msg_type = "Sent" if sms.get("type") == "2" else "Received"
        messages.append(f"[{msg_type}] {date} | {address}: {body}")
    return messages

# ---------------------------
# Export PDF
# ---------------------------
def save_pdf(content, output_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    for line in content.split("\n"):
        pdf.multi_cell(190, 10, line)
    pdf.output(output_path)

# ---------------------------
# GUI Functions
# ---------------------------
def load_file():
    file_path = filedialog.askopenfilename(filetypes=[("XML Files", "*.xml"), ("All Files", "*.*")])
    if file_path:
        entry_file.delete(0, tk.END)
        entry_file.insert(0, file_path)

def process_file():
    global parsed_messages
    file_path = entry_file.get()
    password = entry_password.get().strip()
    try:
        with open(file_path, "rb") as f:
            data = f.read()

        if password:
            data = decrypt_backup(file_path, password)

        xml_text = data.decode("utf-8")
        parsed_messages = parse_sms(xml_text)

        text_area.delete(1.0, tk.END)
        for msg in parsed_messages:
            text_area.insert(tk.END, msg + "\n\n")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def search_messages():
    query = entry_search.get().strip().lower()
    text_area.delete(1.0, tk.END)
    for msg in parsed_messages:
        if query in msg.lower():
            text_area.insert(tk.END, msg + "\n\n")

def export_pdf():
    content = text_area.get(1.0, tk.END).strip()
    if not content:
        messagebox.showwarning("Warning", "No content to save!")
        return
    output_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF Files", "*.pdf")])
    if output_path:
        save_pdf(content, output_path)
        messagebox.showinfo("Success", f"PDF saved to {output_path}")

def export_txt():
    content = text_area.get(1.0, tk.END).strip()
    if not content:
        messagebox.showwarning("Warning", "No content to save!")
        return
    output_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        messagebox.showinfo("Success", f"TXT saved to {output_path}")

# ---------------------------
# MAIN WINDOW
# ---------------------------
root = tk.Tk()
root.title("SMS Backup Viewer Pro by iambandobandz")
root.geometry("750x600")

header_frame = tk.Frame(root, bg="black", height=60)
header_frame.pack(fill=tk.X)
header_label = tk.Label(header_frame, text="📱 SMS Backup Viewer Pro\nby iambandobandz x Victor",
                        font=("Arial", 14, "bold"), fg="white", bg="black")
header_label.pack(pady=10)

tk.Label(root, text="File:").pack()
entry_file = tk.Entry(root, width=70)
entry_file.pack()
tk.Button(root, text="Browse", command=load_file).pack()

tk.Label(root, text="Password (if encrypted):").pack()
entry_password = tk.Entry(root, width=40, show="*")
entry_password.pack()

tk.Button(root, text="Load & View SMS", command=process_file).pack(pady=5)

tk.Label(root, text="Search:").pack()
entry_search = tk.Entry(root, width=40)
entry_search.pack()
tk.Button(root, text="Search Messages", command=search_messages).pack(pady=3)

text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=90, height=20)
text_area.pack(pady=10)

btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)
tk.Button(btn_frame, text="Export as PDF", command=export_pdf).pack(side=tk.LEFT, padx=10)
tk.Button(btn_frame, text="Export as TXT", command=export_txt).pack(side=tk.LEFT, padx=10)

root.mainloop()
