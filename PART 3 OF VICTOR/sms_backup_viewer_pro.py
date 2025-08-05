# ============================================================
# FILE: sms_backup_viewer_pro.py
# VERSION: v2.0.0
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# DESCRIPTION:
#     A professional, robust SMS Backup & Restore viewer with a full GUI.
#     This version is error-proofed, crash-proofed, and enhanced with
#     major new features.
#
#     Features:
#         ✔ Modern GUI with Themed Widgets (TTK)
#         ✔ Load Encrypted or Plain SMS Backup XML
#         ✔ AES-256 Decryption
#         ✔ Conversation View: Groups messages by contact
#         ✔ Non-Blocking UI: Uses threading to prevent freezing on large files
#         ✔ Robust Error Handling (Bad Password, Corrupt File, etc.)
#         ✔ Stateful UI (Buttons disable/enable intelligently)
#         ✔ Real-time Status Bar
#         ✔ Search/Filter within a conversation
#         ✔ Export selected conversation to PDF, TXT, and CSV (for Excel)
#
# PLATFORM: Windows (Dell Inspiron or similar)
# DEPENDENCIES:
#     pip install pycryptodome lxml fpdf
#
# HOW TO MAKE EXE:
#     pyinstaller --onefile --windowed --name "SMS Backup Viewer Pro" sms_backup_viewer_pro.py
# ============================================================

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from xml.etree import ElementTree as ET
from Crypto.Cipher import AES
import hashlib
from fpdf import FPDF
import csv
import threading
from collections import defaultdict
from datetime import datetime

class SmsViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SMS Backup Viewer Pro v2.0")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)

        # Data storage
        self.all_messages = []
        self.conversations = defaultdict(list)

        # --- UI Setup ---
        self.create_widgets()
        self.update_ui_state(False) # Start with action buttons disabled

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Top Controls Frame ---
        top_frame = ttk.LabelFrame(main_frame, text="1. Load File", padding="10")
        top_frame.pack(fill=tk.X, pady=(0, 10))
        top_frame.columnconfigure(1, weight=1)

        ttk.Label(top_frame, text="File:").grid(row=0, column=0, sticky="w", padx=5)
        self.entry_file = ttk.Entry(top_frame, width=70)
        self.entry_file.grid(row=0, column=1, sticky="ew", padx=5)
        self.browse_button = ttk.Button(top_frame, text="Browse...", command=self.load_file)
        self.browse_button.grid(row=0, column=2, padx=5)

        ttk.Label(top_frame, text="Password:").grid(row=1, column=0, sticky="w", padx=5, pady=(5,0))
        self.entry_password = ttk.Entry(top_frame, width=40, show="*")
        self.entry_password.grid(row=1, column=1, sticky="w", padx=5, pady=(5,0))

        self.process_button = ttk.Button(top_frame, text="Load & Process Messages", command=self.start_processing_thread)
        self.process_button.grid(row=2, column=1, pady=10)

        # --- Main Content Paned Window ---
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)

        # --- Left Pane (Conversations List) ---
        conv_frame = ttk.LabelFrame(paned_window, text="2. Conversations", padding="10")
        conv_frame.columnconfigure(0, weight=1)
        conv_frame.rowconfigure(0, weight=1)
        paned_window.add(conv_frame, weight=1)

        self.conv_listbox = tk.Listbox(conv_frame, exportselection=False)
        self.conv_listbox.grid(row=0, column=0, sticky="nsew")
        self.conv_listbox.bind("<<ListboxSelect>>", self.display_conversation)
        
        conv_scrollbar = ttk.Scrollbar(conv_frame, orient=tk.VERTICAL, command=self.conv_listbox.yview)
        conv_scrollbar.grid(row=0, column=1, sticky="ns")
        self.conv_listbox['yscrollcommand'] = conv_scrollbar.set

        # --- Right Pane (Messages View & Export) ---
        msg_frame = ttk.Frame(paned_window, padding="10")
        msg_frame.columnconfigure(0, weight=1)
        msg_frame.rowconfigure(1, weight=1)
        paned_window.add(msg_frame, weight=3)
        
        # Search and Export Controls
        controls_frame = ttk.LabelFrame(msg_frame, text="3. View, Search & Export", padding="10")
        controls_frame.pack(fill=tk.X, expand=False, pady=(0, 10))
        controls_frame.columnconfigure(1, weight=1)

        ttk.Label(controls_frame, text="Search:").grid(row=0, column=0, padx=5, pady=5)
        self.entry_search = ttk.Entry(controls_frame, width=30)
        self.entry_search.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        self.search_button = ttk.Button(controls_frame, text="Search", command=self.search_messages)
        self.search_button.grid(row=0, column=2, padx=5, pady=5)
        
        export_frame = ttk.Frame(controls_frame)
        export_frame.grid(row=0, column=3, padx=10)
        
        self.export_pdf_button = ttk.Button(export_frame, text="Export PDF", command=self.export_pdf)
        self.export_pdf_button.pack(side=tk.LEFT, padx=2)
        self.export_txt_button = ttk.Button(export_frame, text="Export TXT", command=self.export_txt)
        self.export_txt_button.pack(side=tk.LEFT, padx=2)
        self.export_csv_button = ttk.Button(export_frame, text="Export CSV", command=self.export_csv)
        self.export_csv_button.pack(side=tk.LEFT, padx=2)

        # Messages Display Area
        self.text_area = scrolledtext.ScrolledText(msg_frame, wrap=tk.WORD, width=80, height=20, state='disabled')
        self.text_area.pack(fill=tk.BOTH, expand=True)

        # --- Status Bar ---
        self.status_bar = ttk.Label(self.root, text="Ready. Please load an SMS backup file.", relief=tk.SUNKEN, anchor=tk.W, padding=5)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Clear Button
        self.clear_button = ttk.Button(top_frame, text="Clear", command=self.clear_all)
        self.clear_button.grid(row=2, column=2, pady=10, padx=10)

    def update_ui_state(self, enabled):
        """Enable or disable UI elements that require data to be loaded."""
        state = tk.NORMAL if enabled else tk.DISABLED
        self.search_button['state'] = state
        self.export_pdf_button['state'] = state
        self.export_txt_button['state'] = state
        self.export_csv_button['state'] = state
        self.conv_listbox['state'] = state
        self.entry_search['state'] = state

    def set_status(self, text):
        self.status_bar.config(text=text)
        self.root.update_idletasks()

    def clear_all(self):
        """Resets the entire UI to its initial state."""
        self.all_messages.clear()
        self.conversations.clear()
        self.conv_listbox.delete(0, tk.END)
        self.text_area.config(state=tk.NORMAL)
        self.text_area.delete(1.0, tk.END)
        self.text_area.config(state=tk.DISABLED)
        self.entry_file.delete(0, tk.END)
        self.entry_password.delete(0, tk.END)
        self.entry_search.delete(0, tk.END)
        self.update_ui_state(False)
        self.set_status("Ready. Please load an SMS backup file.")
        
    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("XML Files", "*.xml"), ("All Files", "*.*")])
        if file_path:
            self.entry_file.delete(0, tk.END)
            self.entry_file.insert(0, file_path)

    def start_processing_thread(self):
        file_path = self.entry_file.get()
        if not file_path:
            messagebox.showwarning("Warning", "Please select a file first.")
            return

        # Disable buttons to prevent multiple clicks while processing
        self.process_button['state'] = tk.DISABLED
        self.browse_button['state'] = tk.DISABLED
        
        self.set_status(f"Processing '{file_path}'...")
        # Run the heavy lifting in a background thread
        thread = threading.Thread(target=self.process_file_worker, args=(file_path,), daemon=True)
        thread.start()

    def process_file_worker(self, file_path):
        """Worker function to be run in a separate thread."""
        password = self.entry_password.get().strip()
        try:
            with open(file_path, "rb") as f:
                data = f.read()

            if password:
                self.set_status("File is encrypted. Decrypting...")
                try:
                    key = hashlib.sha256(password.encode()).digest()
                    iv = data[:16]
                    cipher = AES.new(key, AES.MODE_CBC, iv)
                    decrypted = cipher.decrypt(data[16:])
                    # PKCS7 unpadding. If this fails, password was likely wrong.
                    pad_len = decrypted[-1]
                    if pad_len > AES.block_size:
                        raise ValueError("Incorrect password or corrupted file (padding error).")
                    data = decrypted[:-pad_len]
                except (ValueError, IndexError) as e:
                    self.root.after(0, self.handle_processing_error, "Decryption Failed. Please check your password or the file may be corrupt.")
                    return
            
            self.set_status("Parsing XML content...")
            xml_content = data.decode("utf-8")
            self.parse_sms_data(xml_content)
            self.root.after(0, self.processing_complete)

        except FileNotFoundError:
            self.root.after(0, self.handle_processing_error, "Error: File not found.")
        except ET.ParseError:
            self.root.after(0, self.handle_processing_error, "Error: Failed to parse XML. The file may not be a valid SMS backup.")
        except Exception as e:
            self.root.after(0, self.handle_processing_error, f"An unexpected error occurred: {e}")

    def handle_processing_error(self, error_message):
        messagebox.showerror("Error", error_message)
        self.set_status(f"Error. {error_message}")
        # Re-enable buttons after error
        self.process_button['state'] = tk.NORMAL
        self.browse_button['state'] = tk.NORMAL

    def parse_sms_data(self, xml_content):
        self.all_messages = []
        self.conversations = defaultdict(list)
        root = ET.fromstring(xml_content)
        
        messages = root.findall("sms")
        for sms in messages:
            try:
                date_ms = int(sms.get("date"))
                date_readable = datetime.fromtimestamp(date_ms / 1000).strftime('%Y-%m-%d %I:%M:%S %p')
            except (ValueError, TypeError):
                date_readable = sms.get("readable_date", "Unknown Date")
            
            msg = {
                "address": sms.get("address", "Unknown"),
                "body": sms.get("body", "").replace('\r\n', '\n'), # Normalize newlines
                "type": "Sent" if sms.get("type") == "2" else "Received",
                "date": date_readable
            }
            self.all_messages.append(msg)
            self.conversations[msg["address"]].append(msg)
        
        # Sort messages within each conversation by date
        for address in self.conversations:
            self.conversations[address].sort(key=lambda m: m["date"])

    def processing_complete(self):
        """Called on the main thread after the worker is done."""
        self.conv_listbox.delete(0, tk.END)
        # Populate the conversation list, sorted by number of messages
        sorted_convs = sorted(self.conversations.keys(), key=lambda k: len(self.conversations[k]), reverse=True)
        for address in sorted_convs:
            self.conv_listbox.insert(tk.END, f"{address} ({len(self.conversations[address])} msgs)")

        self.update_ui_state(True)
        self.set_status(f"Processing complete. Found {len(self.all_messages)} messages in {len(self.conversations)} conversations.")
        # Re-enable buttons
        self.process_button['state'] = tk.NORMAL
        self.browse_button['state'] = tk.NORMAL
        if self.conv_listbox.size() > 0:
            self.conv_listbox.select_set(0)
            self.conv_listbox.event_generate("<<ListboxSelect>>")

    def display_conversation(self, event=None):
        if not self.conv_listbox.curselection():
            return

        selection_index = self.conv_listbox.curselection()[0]
        selection_text = self.conv_listbox.get(selection_index)
        address = selection_text.split(" (")[0]
        
        self.display_messages(self.conversations[address])
        self.set_status(f"Displaying conversation with {address}")

    def search_messages(self):
        if not self.conv_listbox.curselection():
             messagebox.showinfo("Info", "Please select a conversation to search within.")
             return
        
        query = self.entry_search.get().strip().lower()
        if not query:
            self.display_conversation() # If search is cleared, show full conversation
            return

        selection_index = self.conv_listbox.curselection()[0]
        selection_text = self.conv_listbox.get(selection_index)
        address = selection_text.split(" (")[0]

        filtered_messages = [msg for msg in self.conversations[address] if query in msg['body'].lower()]
        self.display_messages(filtered_messages)
        self.set_status(f"Found {len(filtered_messages)} messages matching '{query}' with {address}")

    def display_messages(self, messages):
        self.text_area.config(state=tk.NORMAL)
        self.text_area.delete(1.0, tk.END)
        for msg in messages:
            self.text_area.insert(tk.END, f"[{msg['type']}] - {msg['date']}\n", ("header",))
            self.text_area.insert(tk.END, f"{msg['body']}\n\n")
        self.text_area.config(state=tk.DISABLED)
        self.text_area.tag_config("header", font=("Segoe UI", 9, "bold"))
        
    def _get_displayed_content(self):
        content = self.text_area.get(1.0, tk.END).strip()
        if not content:
            messagebox.showwarning("Warning", "There is no content to export.")
            return None
        return content

    def export_pdf(self):
        content = self._get_displayed_content()
        if not content: return
        output_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF Files", "*.pdf")])
        if not output_path: return
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=10)
        # Use multi_cell for better text wrapping
        pdf.multi_cell(0, 5, content.encode('latin-1', 'replace').decode('latin-1'))
        pdf.output(output_path)
        messagebox.showinfo("Success", f"Conversation saved to {output_path}")

    def export_txt(self):
        content = self._get_displayed_content()
        if not content: return
        output_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
        if not output_path: return
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        messagebox.showinfo("Success", f"Conversation saved to {output_path}")

    def export_csv(self):
        if not self.conv_listbox.curselection():
             messagebox.showinfo("Info", "Please select a conversation to export.")
             return
        
        selection_index = self.conv_listbox.curselection()[0]
        selection_text = self.conv_listbox.get(selection_index)
        address = selection_text.split(" (")[0]
        messages_to_export = self.conversations[address]
        
        output_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if not output_path: return
        
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Date", "Type", "Address", "Message"])
            for msg in messages_to_export:
                writer.writerow([msg['date'], msg['type'], msg['address'], msg['body']])
        messagebox.showinfo("Success", f"Conversation saved to {output_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = SmsViewerApp(root)
    root.mainloop()
    