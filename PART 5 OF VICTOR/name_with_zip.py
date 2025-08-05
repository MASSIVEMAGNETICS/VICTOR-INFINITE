
# ============================================
# FILE: bark_path/name.py
# VERSION: v1.0.0-CODECRAFTER-GODCORE
# NAME: CodeCrafterGUI
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Offline lightweight GUI for generating code via CodeGeeX models
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# ============================================

import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'codegeex')))

try:
    from generator import generate_code
except ImportError:
    def generate_code(prompt):
        return "// ⚠️ Fallback: generator module not found.\n// Simulated code for: " + prompt

class CodeCrafterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Victor's CodeCrafter")
        self.root.geometry("800x600")
        self.root.configure(bg="#1e1e1e")

        self.label = tk.Label(root, text="Enter your code prompt:", bg="#1e1e1e", fg="#ffffff")
        self.label.pack(pady=5)

        self.input_text = tk.Text(root, height=5, wrap=tk.WORD, font=("Courier", 12))
        self.input_text.pack(padx=10, pady=5, fill=tk.X)

        self.generate_button = tk.Button(root, text="🚀 Generate Code", command=self.generate, bg="#4CAF50", fg="white")
        self.generate_button.pack(pady=5)

        self.output_label = tk.Label(root, text="Generated Code:", bg="#1e1e1e", fg="#ffffff")
        self.output_label.pack(pady=5)

        self.output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Courier", 12))
        self.output_text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        self.button_frame = tk.Frame(root, bg="#1e1e1e")
        self.button_frame.pack(pady=5)

        tk.Button(self.button_frame, text="💾 Save", command=self.save_output).pack(side=tk.LEFT, padx=5)
        tk.Button(self.button_frame, text="🧹 Clear", command=self.clear).pack(side=tk.LEFT, padx=5)
        tk.Button(self.button_frame, text="📋 Copy", command=self.copy_output).pack(side=tk.LEFT, padx=5)
        tk.Button(self.button_frame, text="📦 Import Zip", command=import_zip).pack(side=tk.LEFT, padx=5)

    def generate(self):
        prompt = self.input_text.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showwarning("Empty Prompt", "Type something to generate code.")
            return
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, "// Generating...\n")
        try:
            result = generate_code(prompt)
        except Exception as e:
            result = f"// ❌ Error during code generation:\n{e}"
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, result)

    def save_output(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".py",
                                                 filetypes=[("Python Files", "*.py"), ("All Files", "*.*")])
        if file_path:
            with open(file_path, "w") as f:
                f.write(self.output_text.get("1.0", tk.END))
            messagebox.showinfo("Saved", f"Code saved to {file_path}")

    def clear(self):
        self.input_text.delete("1.0", tk.END)
        self.output_text.delete("1.0", tk.END)

    
import zipfile
import tempfile

def analyze_zip_and_generate_app(zip_path):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
            file_summary = []
            for root, dirs, files in os.walk(tmpdir):
                for file in files:
                    if file.endswith(".py"):
                        file_path = os.path.join(root, file)
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            lines = f.readlines()
                            file_summary.append(f"📄 {file} - {len(lines)} lines")
            return "\n".join(file_summary) if file_summary else "No .py files found in zip."
    except Exception as e:
        return f"❌ Error reading ZIP: {str(e)}"

def import_zip():
    zip_path = filedialog.askopenfilename(filetypes=[("Zip Files", "*.zip")])
    if zip_path:
        result = analyze_zip_and_generate_app(zip_path)
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, result)


    def copy_output(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.output_text.get("1.0", tk.END))
        messagebox.showinfo("Copied", "Output copied to clipboard")

if __name__ == "__main__":
    root = tk.Tk()
    app = CodeCrafterGUI(root)
    root.mainloop()


# === AUTO-EXPAND HOOK ===
def expand():
    print(f'[AUTO_EXPAND] Module {__file__} has no custom logic. Placeholder activated.')
