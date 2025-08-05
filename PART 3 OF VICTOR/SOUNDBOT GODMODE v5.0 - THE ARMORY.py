# =================================================================================================
# ||                                                                                           ||
# ||    SOUNDBOT GODMODE v5.0 - THE ARMORY                                                     ||
# ||    "You don't just build the weapon. You build the factory."                              ||
# ||                                                                                           ||
# ||    V5.0 FINAL UPGRADES:                                                                   ||
# ||    - PERSONA MANAGEMENT SYSTEM: GUI tab to create, view, and save a persona database.     ||
# ||    - PERSONA-DRIVEN CAMPAIGNS: Assign named personas to tasks for deep narrative building.||
# ||    - WEAPONIZATION-READY: Primed for compilation into a standalone .exe with PyInstaller. ||
# ||                                                                                           ||
# ================================================================================================
import sys
import time
import random
import requests
import logging
import csv
import os
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from threading import Thread

# -- Third-party libraries (pip install -r requirements.txt)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QSpinBox, QTextEdit, QMessageBox,
                             QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox,
                             QTabWidget)
from PyQt5.QtCore import pyqtSignal, QObject
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from faker import Faker

# --- SETUP ---
if not os.path.exists('personas'): os.makedirs('personas')
PERSONAS_DB_FILE = 'personas.json'
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] (%(threadName)-10s) %(message)s',
                    handlers=[logging.FileHandler("soundbot_godmode_v5.log"), logging.StreamHandler()])

# --- STEALTH CONFIGS ---
TIMEZONES = ['America/New_York', 'America/Chicago', 'America/Denver', 'America/Los_Angeles', 'Europe/London']
LANGUAGES = ['en-US,en;q=0.9', 'en-GB,en;q=0.8', 'es-ES,es;q=0.9']
SCREEN_RESOLUTIONS = ['1920,1080', '1600,900', '1366,768']

# =================================================================================================
# || CORE MODULES (v5.0 ARMORY EDITION)                                                        ||
# =================================================================================================

class ProxyManager:
    """ Manages proxies. Hardened for reliability. """
    def __init__(self, proxy_api_url):
        if not proxy_api_url: raise ValueError("Proxy API URL missing.")
        self.api_url = proxy_api_url; self.proxy_pool = []; self.refresh_proxies()
    def refresh_proxies(self):
        logging.info("Refreshing proxy pool...")
        try:
            resp = requests.get(self.api_url, timeout=15); resp.raise_for_status()
            self.proxy_pool = [{'full': p} for p in resp.text.splitlines() if p]
            logging.info(f"Loaded {len(self.proxy_pool)} new proxies.")
        except Exception as e: logging.error(f"Failed to fetch proxies: {e}"); self.proxy_pool = []
    def get_proxy(self):
        return random.choice(self.proxy_pool) if self.proxy_pool else None

class BrowserEngine:
    """ Handles stealth browser creation, cookie I/O, and automation. """
    @staticmethod
    def get_stealth_browser(proxy, headless, persona_name):
        options = uc.ChromeOptions()
        if headless: options.add_argument('--headless=new')
        options.add_argument(f'--user-agent={Faker().user_agent()}')
        options.add_argument(f'--lang={random.choice(LANGUAGES)}')
        options.add_argument(f'--window-size={random.choice(SCREEN_RESOLUTIONS)}')
        options.add_argument(f'--proxy-server={proxy["full"]}')
        options.add_argument("--disable-blink-features=AutomationControlled"); options.add_argument('--no-sandbox')
        
        browser = uc.Chrome(options=options)
        
        cookie_path = f'personas/{persona_name.replace(" ", "_")}.json'
        if os.path.exists(cookie_path):
            try:
                with open(cookie_path, 'r') as f:
                    for cookie in json.load(f):
                        if 'expiry' in cookie: del cookie['expiry']
                        browser.add_cookie(cookie)
                logging.info(f"Loaded cookies for Persona: {persona_name}.")
                browser.refresh()
            except Exception as e: logging.error(f"Could not load cookies for {persona_name}: {e}")

        return browser

    @staticmethod
    def play_song(browser, selector, persona_name):
        time.sleep(random.uniform(5, 10))
        try: browser.find_element(By.CSS_SELECTOR, selector).click()
        except Exception: logging.warning("Could not click play button.")
        
        time.sleep(random.uniform(45, 120))
        
        cookie_path = f'personas/{persona_name.replace(" ", "_")}.json'
        try:
            with open(cookie_path, 'w') as f: json.dump(browser.get_cookies(), f)
            logging.info(f"Saved cookies for Persona: {persona_name}.")
        except Exception as e: logging.error(f"Could not save cookies for {persona_name}: {e}")
        browser.quit()

class SoundBotSwarm(QObject):
    """ Main bot engine, driven by the persona-based campaign. """
    log_updated = pyqtSignal(str)
    swarm_finished = pyqtSignal()
    
    def __init__(self, config):
        super().__init__(); self.config = config; self.is_running = False
        self.proxy_mgr = ProxyManager(config['proxy_api_url'])

    def _play_task(self, url, persona_name):
        if not self.is_running: return
        proxy = self.proxy_mgr.get_proxy()
        if not proxy: self.log_updated.emit("[ERROR] No proxy available."); return

        try:
            self.log_updated.emit(f"[INFO] Activating Persona '{persona_name}'...")
            browser = BrowserEngine.get_stealth_browser(proxy, self.config['headless'], persona_name)
            browser.get(url) # Navigate after setting up cookies
            BrowserEngine.play_song(browser, self.config['selector'], persona_name)
            self.log_updated.emit(f"[SUCCESS] Persona '{persona_name}' completed task on {url[-35:]}")
        except Exception as e:
            self.log_updated.emit(f"[ERROR] Persona '{persona_name}' failed: {e}")

    def start_swarm(self):
        self.is_running = True
        with ThreadPoolExecutor(max_workers=self.config['rate'], thread_name_prefix='Persona') as executor:
            tasks = self.config['tasks']; random.shuffle(tasks)
            for task in tasks:
                if not self.is_running: break
                executor.submit(self._play_task, task['url'], task['persona_name'])
                time.sleep(60.0 / self.config['rate'])
        if self.is_running: self.swarm_finished.emit()
            
    def stop_swarm(self): self.is_running = False

# =================================================================================================
# || GUI (v5.0 ARMORY INTERFACE)                                                               ||
# =================================================================================================

class SoundBotGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SOUNDBOT GODMODE v5.0 - The Armory")
        self.setGeometry(100, 100, 900, 800)
        self.personas = self.load_personas()
        self._setup_ui()

    def load_personas(self):
        if os.path.exists(PERSONAS_DB_FILE):
            with open(PERSONAS_DB_FILE, 'r') as f: return json.load(f)
        return []

    def save_personas(self):
        with open(PERSONAS_DB_FILE, 'w') as f: json.dump(self.personas, f, indent=4)
        self._log("[SYSTEM] Persona database saved.")

    def _setup_ui(self):
        main_widget = QWidget(); self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        tabs = QTabWidget(); layout.addWidget(tabs)
        
        # --- TABS ---
        self._create_campaign_tab(tabs)
        self._create_persona_manager_tab(tabs)
        self._create_config_tab(tabs)
        self._create_log_tab(tabs)

    def _create_campaign_tab(self, tabs):
        tab = QWidget(); tabs.addTab(tab, "Campaign Manager")
        layout = QVBoxLayout(tab)
        
        layout.addWidget(QLabel("🎯 Campaign Tasks (Assign plays to personas):"))
        self.campaign_table = QTableWidget(); self.campaign_table.setColumnCount(3)
        self.campaign_table.setHorizontalHeaderLabels(["Song URL", "Plays", "Persona Name"])
        self.campaign_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        layout.addWidget(self.campaign_table)

        table_btns = QHBoxLayout()
        add_row_btn = QPushButton("Add Task"); add_row_btn.clicked.connect(self._add_campaign_row)
        remove_row_btn = QPushButton("Remove Task"); remove_row_btn.clicked.connect(self._remove_campaign_row)
        table_btns.addWidget(add_row_btn); table_btns.addWidget(remove_row_btn)
        layout.addLayout(table_btns)

        # Action Buttons
        action_layout = QHBoxLayout()
        self.start_btn = QPushButton("🚀 LAUNCH CAMPAIGN"); self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.stop_btn = QPushButton("🛑 STOP CAMPAIGN"); self.stop_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 10px;"); self.stop_btn.setEnabled(False)
        action_layout.addWidget(self.start_btn); action_layout.addWidget(self.stop_btn)
        layout.addLayout(action_layout)
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)

    def _create_persona_manager_tab(self, tabs):
        tab = QWidget(); tabs.addTab(tab, "Persona Manager")
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("👤 Persona Database:"))
        
        self.persona_table = QTableWidget(); self.persona_table.setColumnCount(3)
        self.persona_table.setHorizontalHeaderLabels(["Name", "Email", "Creation Date"])
        self.persona_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.persona_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        layout.addWidget(self.persona_table)
        self._refresh_persona_table()

        persona_btns = QHBoxLayout()
        create_persona_btn = QPushButton("Create New Persona"); create_persona_btn.clicked.connect(self._create_persona)
        save_personas_btn = QPushButton("Save Database"); save_personas_btn.clicked.connect(self.save_personas)
        persona_btns.addWidget(create_persona_btn); persona_btns.addWidget(save_personas_btn)
        layout.addLayout(persona_btns)

    def _create_config_tab(self, tabs):
        tab = QWidget(); tabs.addTab(tab, "Configuration")
        layout = QVBoxLayout(tab)
        
        layout.addWidget(QLabel("🔑 Proxy API URL:"))
        self.proxy_api_input = QLineEdit()
        layout.addWidget(self.proxy_api_input)
        
        layout.addWidget(QLabel("🖱️ Play Button CSS Selector:"))
        self.selector_input = QLineEdit(); self.selector_input.setText(".player-button.play")
        layout.addWidget(self.selector_input)
        
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Rate (per min):")); self.rate_input = QSpinBox(); self.rate_input.setRange(1, 100); self.rate_input.setValue(10); controls_layout.addWidget(self.rate_input)
        self.headless_checkbox = QCheckBox("Headless Mode"); self.headless_checkbox.setChecked(True); controls_layout.addWidget(self.headless_checkbox)
        layout.addLayout(controls_layout)

    def _create_log_tab(self, tabs):
        tab = QWidget(); tabs.addTab(tab, "Live Log")
        layout = QVBoxLayout(tab)
        self.log_area = QTextEdit(); self.log_area.setReadOnly(True); self.log_area.setStyleSheet("background-color: #1E1E1E; color: #00FF00; font-family: 'Courier New';")
        layout.addWidget(self.log_area)

    def _log(self, msg): self.log_area.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"); self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum())
    def _add_campaign_row(self): row = self.campaign_table.rowCount(); self.campaign_table.insertRow(row); self.campaign_table.setItem(row, 1, QTableWidgetItem("10"))
    def _remove_campaign_row(self):
        if self.campaign_table.currentRow() >= 0: self.campaign_table.removeRow(self.campaign_table.currentRow())
    def _create_persona(self):
        fake = Faker(); name = fake.name(); self.personas.append({'name': name, 'email': fake.email(), 'created': datetime.now().strftime('%Y-%m-%d')})
        self._refresh_persona_table()
    def _refresh_persona_table(self):
        self.persona_table.setRowCount(0)
        for p in self.personas:
            row = self.persona_table.rowCount(); self.persona_table.insertRow(row)
            self.persona_table.setItem(row, 0, QTableWidgetItem(p['name']))
            self.persona_table.setItem(row, 1, QTableWidgetItem(p['email']))
            self.persona_table.setItem(row, 2, QTableWidgetItem(p['created']))

    def _on_start(self):
        tasks = []
        for row in range(self.campaign_table.rowCount()):
            try:
                url = self.campaign_table.item(row, 0).text()
                plays = int(self.campaign_table.item(row, 1).text())
                persona_name = self.campaign_table.item(row, 2).text()
                if not all([url, persona_name, plays > 0]): continue
                for _ in range(plays): tasks.append({'url': url, 'persona_name': persona_name})
            except (AttributeError, ValueError): QMessageBox.critical(self, "Error", f"Invalid data in campaign row {row + 1}."); return
        
        if not tasks or not self.proxy_api_input.text(): QMessageBox.critical(self, "Error", "Campaign tasks and Proxy API are required."); return

        config = {'tasks': tasks, 'proxy_api_url': self.proxy_api_input.text(), 'selector': self.selector_input.text(),
                  'rate': self.rate_input.value(), 'headless': self.headless_checkbox.isChecked()}
        
        self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        self._log("🔥 ARMORY ENGAGED. LAUNCHING CAMPAIGN...")
        
        self.swarm = SoundBotSwarm(config)
        self.swarm.log_updated.connect(self._log); self.swarm.swarm_finished.connect(self._on_finish)
        self.thread = Thread(target=self.swarm.start_swarm, daemon=True); self.thread.start()

    def _on_stop(self):
        if self.swarm: self.swarm.stop_swarm(); self.stop_btn.setEnabled(False)
    def _on_finish(self):
        self._log("✅ CAMPAIGN COMPLETE."); self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)

# =================================================================================================
# || MAIN ENTRY POINT                                                                          ||
# =================================================================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = SoundBotGUI()
    win.show()
    sys.exit(app.exec_())