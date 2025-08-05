# =================================================================================================
# ||                                                                                           ||
# ||    SOUNDBOT GODMODE v4.0 - PHANTOM CELL                                                   ||
# ||    "A ghost is a memory. A phantom is a life."                                            ||
# ||                                                                                           ||
# ||    V4.0 UPGRADES:                                                                         ||
# ||    - CAMPAIGN MANAGER GUI: Table-based input for per-track smart quotas.                  ||
# ||    - PERSONA PERSISTENCE (COOKIE JAR): Bots save/load cookies to simulate returning users.||
# ||    - HEADLESS-RESISTANT MODE: GUI checkbox to toggle browser visibility.                  ||
# ||    - STRUCTURED PERSONA DIRECTORY: Creates a './personas' folder for cookie storage.      ||
# ||                                                                                           ||
# =================================================================================================

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
from threading import Thread, Lock

# -- Third-party libraries (pip install -r requirements.txt)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QSpinBox, QTextEdit, QMessageBox,
                             QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox)
from PyQt5.QtCore import pyqtSignal, QObject
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from faker import Faker

# --- LOGGING & DIRECTORY SETUP ---
if not os.path.exists('personas'):
    os.makedirs('personas')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] (%(threadName)-10s) %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler("soundbot_godmode_v4.log"),
                              logging.StreamHandler()])

# --- V3.0 STEALTH CONFIGS (Retained) ---
TIMEZONES = ['America/New_York', 'America/Chicago', 'America/Denver', 'America/Los_Angeles', 'Europe/London', 'Europe/Berlin', 'Europe/Moscow', 'Asia/Tokyo', 'Australia/Sydney']
LANGUAGES = ['en-US,en;q=0.9', 'en-GB,en;q=0.8', 'es-ES,es;q=0.9', 'fr-FR,fr;q=0.9']
SCREEN_RESOLUTIONS = ['1920,1080', '1600,900', '1366,768']

# =================================================================================================
# || CORE MODULES (v4.0 PHANTOM MUTATIONS)                                                     ||
# =================================================================================================

class ProxyManager:
    """ Manages proxies. No changes in v4.0. """
    def __init__(self, proxy_api_url):
        self.api_url = proxy_api_url; self.proxy_pool = []; self.proxy_lock = Lock()
        if not self.api_url: raise ValueError("Proxy API URL missing.")
        self.refresh_proxies()
    def refresh_proxies(self):
        logging.info("Refreshing proxy pool...")
        try:
            resp = requests.get(self.api_url, timeout=15); resp.raise_for_status()
            with self.proxy_lock: self.proxy_pool = [{'full': p} for p in resp.text.splitlines() if p]
            logging.info(f"Loaded {len(self.proxy_pool)} new proxies.")
        except Exception as e: logging.error(f"Failed to fetch proxies: {e}"); self.proxy_pool = []
    def get_proxy(self):
        with self.proxy_lock:
            if not self.proxy_pool: self.refresh_proxies()
            return random.choice(self.proxy_pool) if self.proxy_pool else None
    def ban_proxy(self, proxy):
        with self.proxy_lock:
            try: self.proxy_pool.remove(proxy); logging.warning(f"Banned proxy: {proxy['full']}")
            except ValueError: pass

class GeoUtils:
    """ GeoIP lookup. No changes in v4.0. """
    @staticmethod
    def get_geo_from_ip(ip):
        try:
            resp = requests.get(f"http://ip-api.com/json/{ip}?fields=status,message,country,city,lat,lon", timeout=5)
            data = resp.json()
            if data.get("status") == "success": return data
        except Exception: pass
        return {"country": "N/A", "city": "N/A", "lat": 0, "lon": 0}

class BrowserEngine:
    """ Handles stealth browser creation, now with Cookie Jar and Headless Toggle. """
    @staticmethod
    def get_stealth_browser(proxy, headless_mode, persona_id):
        options = uc.ChromeOptions()
        # --- V4.0 HEADLESS-RESISTANT MODE ---
        if headless_mode: options.add_argument('--headless=new')
        
        options.add_argument(f'--user-agent={Faker().user_agent()}')
        options.add_argument(f'--lang={random.choice(LANGUAGES)}')
        options.add_argument(f'--window-size={random.choice(SCREEN_RESOLUTIONS)}')
        options.add_argument(f'--proxy-server={proxy["full"]}')
        options.add_argument("--disable-blink-features=AutomationControlled"); options.add_argument('--no-sandbox')
        
        browser = uc.Chrome(options=options)
        
        # --- V4.0 PERSONA PERSISTENCE (COOKIE JAR - LOAD) ---
        cookie_path = f'personas/{persona_id}.json'
        if os.path.exists(cookie_path):
            try:
                with open(cookie_path, 'r') as f:
                    cookies = json.load(f)
                for cookie in cookies:
                    if 'expiry' in cookie: del cookie['expiry'] # Let the browser handle expiry
                    browser.add_cookie(cookie)
                logging.info(f"Loaded {len(cookies)} cookies for Persona {persona_id}.")
                browser.refresh() # Refresh page to apply cookies
            except Exception as e: logging.error(f"Could not load cookies for Persona {persona_id}: {e}")

        # JS Stealth Suite (Timezone, WebGL, Canvas, etc.)
        try:
            js_stealth_suite = f"""
                Object.defineProperty(navigator, 'webdriver', {{get: () => undefined}});
                // Other stealth patches from v3.0 can be included here
            """
            browser.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": js_stealth_suite})
        except Exception: logging.warning("Could not inject JS Stealth Suite.")
            
        return browser

    @staticmethod
    def play_song(browser, url, selector, persona_id):
        browser.get(url)
        time.sleep(random.uniform(5, 10))
        try:
            browser.find_element(By.CSS_SELECTOR, selector).click()
            logging.info(f"Play button clicked for Persona {persona_id}.")
        except Exception: logging.warning("Could not click play button. Assuming autoplay.")
        
        time.sleep(random.uniform(60, 150))
        
        # --- V4.0 PERSONA PERSISTENCE (COOKIE JAR - SAVE) ---
        cookie_path = f'personas/{persona_id}.json'
        try:
            with open(cookie_path, 'w') as f:
                json.dump(browser.get_cookies(), f)
            logging.info(f"Saved cookies for Persona {persona_id}.")
        except Exception as e: logging.error(f"Could not save cookies for Persona {persona_id}: {e}")
            
        browser.quit()

class SoundBotSwarm(QObject):
    """ Main bot engine. Now driven by the Campaign Manager. """
    log_updated = pyqtSignal(str)
    play_logged = pyqtSignal(dict)
    swarm_finished = pyqtSignal()
    
    def __init__(self, config):
        super().__init__(); self.config = config; self.is_running = False
        self.proxy_mgr = ProxyManager(config['proxy_api_url'])

    def _send_webhook(self, message, success=True):
        if not self.config['webhook_url']: return
        try: requests.post(self.config['webhook_url'], json={"embeds": [{"description": message, "color": 0x00ff00 if success else 0xff0000}]}, timeout=5)
        except Exception: self.log_updated.emit("[ERROR] Webhook failed.")

    def _play_task(self, url, persona_id):
        if not self.is_running: return
        proxy = self.proxy_mgr.get_proxy()
        if not proxy: self.log_updated.emit("[ERROR] No proxy. Task aborted."); return

        geo = {}
        try:
            ip = proxy['full'].split(':')[0]; geo = GeoUtils.get_geo_from_ip(ip)
            self.log_updated.emit(f"[INFO] Activating Phantom {persona_id} from {geo.get('city', 'N/A')}")
            
            browser = BrowserEngine.get_stealth_browser(proxy, self.config['headless'], persona_id)
            BrowserEngine.play_song(browser, url, self.config['selector'], persona_id)
            
            log_entry = {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'url': url, 'persona_id': persona_id,
                         'ip': ip, 'city': geo.get('city', 'N/A'), 'country': geo.get('country', 'N/A'), 'status': 'Success'}
            self.play_logged.emit(log_entry); msg = f"✅ SUCCESS: Phantom {persona_id} played {url[-30:]}"
            self.log_updated.emit(f"[INFO] {msg}"); self._send_webhook(msg)
            
        except Exception as e:
            log_entry = {'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'url': url, 'persona_id': persona_id,
                         'ip': proxy.get('full', 'N/A'), 'city': geo.get('city', 'N/A'), 'country': geo.get('country', 'N/A'), 'status': f'Failure: {e}'}
            self.play_logged.emit(log_entry); error_msg = f"❌ ERROR: Phantom {persona_id} failed. Banning proxy."
            self.log_updated.emit(f"[ERROR] {error_msg}"); self._send_webhook(error_msg, success=False)
            self.proxy_mgr.ban_proxy(proxy)

    def start_swarm(self):
        self.is_running = True
        self._send_webhook(f"🚀 PHANTOM CELL ENGAGED! Starting {len(self.config['tasks'])} total plays.", success=True)
        
        with ThreadPoolExecutor(max_workers=self.config['rate'], thread_name_prefix='Phantom') as executor:
            tasks = self.config['tasks']; random.shuffle(tasks) # Randomize play order
            for task in tasks:
                if not self.is_running: self.log_updated.emit("[INFO] Swarm manually stopped."); break
                executor.submit(self._play_task, task['url'], task['persona_id'])
                time.sleep(60.0 / self.config['rate'])

        executor.shutdown(wait=True)
        if self.is_running: self.swarm_finished.emit()
            
    def stop_swarm(self):
        if self.is_running: self.is_running = False; self._send_webhook("🛑 Phantom Cell disengaged.", success=False)

# =================================================================================================
# || GUI (v4.0 PHANTOM CELL INTERFACE)                                                         ||
# =================================================================================================

class SoundBotGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SOUNDBOT GODMODE v4.0 - Phantom Cell")
        self.setGeometry(100, 100, 850, 800)
        self.play_log_data = []
        self._setup_ui()

    def _setup_ui(self):
        main_widget = QWidget(); self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # --- V4.0 CAMPAIGN MANAGER ---
        main_layout.addWidget(QLabel("🎯 Campaign Manager (Right-click to add/remove rows):"))
        self.campaign_table = QTableWidget(); self.campaign_table.setColumnCount(3)
        self.campaign_table.setHorizontalHeaderLabels(["Song URL", "Plays", "Persona ID"])
        self.campaign_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        main_layout.addWidget(self.campaign_table)

        table_btns = QHBoxLayout()
        add_row_btn = QPushButton("Add Row"); add_row_btn.clicked.connect(self._add_row)
        remove_row_btn = QPushButton("Remove Selected Row"); remove_row_btn.clicked.connect(self._remove_row)
        table_btns.addWidget(add_row_btn); table_btns.addWidget(remove_row_btn)
        main_layout.addLayout(table_btns)
        
        config_layout = QVBoxLayout()
        # --- Configs (Proxy, Selector, Webhook) ---
        config_layout.addWidget(QLabel("🔑 Proxy API URL:"))
        self.proxy_api_input = QLineEdit()
        config_layout.addWidget(self.proxy_api_input)
        config_layout.addWidget(QLabel("🖱️ Play Button CSS Selector:"))
        self.selector_input = QLineEdit(); self.selector_input.setText(".player-button.play")
        config_layout.addWidget(self.selector_input)
        config_layout.addWidget(QLabel("🔔 Discord Webhook URL (Optional):"))
        self.webhook_input = QLineEdit()
        config_layout.addWidget(self.webhook_input)
        main_layout.addLayout(config_layout)

        # --- Controls (Rate, Headless Mode) ---
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Rate (per min):")); self.rate_input = QSpinBox(); self.rate_input.setRange(1, 100); self.rate_input.setValue(10); controls_layout.addWidget(self.rate_input)
        self.headless_checkbox = QCheckBox("Headless Mode (Faster, more detectable)"); self.headless_checkbox.setChecked(True); controls_layout.addWidget(self.headless_checkbox)
        main_layout.addLayout(controls_layout)

        # --- Action Buttons ---
        self.start_btn = QPushButton("🚀 Engage Phantom Cell"); self.start_btn.setStyleSheet("background-color: #4CAF50; ...")
        self.stop_btn = QPushButton("🛑 Disengage"); self.stop_btn.setStyleSheet("background-color: #f44336; ..."); self.stop_btn.setEnabled(False)
        self.export_btn = QPushButton("📊 Export Log to CSV"); self.export_btn.setStyleSheet("background-color: #008CBA; ...")
        action_layout = QHBoxLayout(); action_layout.addWidget(self.start_btn); action_layout.addWidget(self.stop_btn);
        main_layout.addLayout(action_layout); main_layout.addWidget(self.export_btn)
        
        main_layout.addWidget(QLabel("Live Intel Feed:"))
        self.log_area = QTextEdit(); self.log_area.setReadOnly(True); self.log_area.setStyleSheet("background-color: #1E1E1E; ...")
        main_layout.addWidget(self.log_area)

        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        self.export_btn.clicked.connect(self._on_export)

    def _add_row(self):
        row_count = self.campaign_table.rowCount(); self.campaign_table.insertRow(row_count)
        self.campaign_table.setItem(row_count, 1, QTableWidgetItem("10")) # Default 10 plays
        self.campaign_table.setItem(row_count, 2, QTableWidgetItem(f"p_{int(time.time())}_{row_count}")) # Unique persona ID

    def _remove_row(self):
        if self.campaign_table.currentRow() >= 0: self.campaign_table.removeRow(self.campaign_table.currentRow())

    def _log(self, msg): self.log_area.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"); self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum())
    def _add_to_log_data(self, entry): self.play_log_data.append(entry)
    def _on_export(self):
        if not self.play_log_data: QMessageBox.information(self, "Export Log", "No data to export."); return
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", f"soundbot_log_{datetime.now().strftime('%Y%m%d')}.csv", "CSV (*.csv)")
        if path:
            try:
                with open(path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.play_log_data[0].keys()); writer.writeheader(); writer.writerows(self.play_log_data)
                self._log("[INFO] Log exported to CSV.")
            except Exception as e: QMessageBox.critical(self, "Export Error", f"Failed to save: {e}")

    def _on_start(self):
        tasks = []
        for row in range(self.campaign_table.rowCount()):
            try:
                url = self.campaign_table.item(row, 0).text()
                plays = int(self.campaign_table.item(row, 1).text())
                persona_id = self.campaign_table.item(row, 2).text()
                if not url or not persona_id: continue
                for _ in range(plays): tasks.append({'url': url, 'persona_id': persona_id})
            except (AttributeError, ValueError):
                QMessageBox.critical(self, "Error", f"Invalid data in campaign table row {row + 1}."); return
        
        if not tasks or not self.proxy_api_input.text() or not self.selector_input.text():
            QMessageBox.critical(self, "Error", "Campaign tasks, Proxy API, and Selector are required."); return

        self.play_log_data.clear(); self._log("[INFO] Log cache cleared.")
        config = {'tasks': tasks, 'proxy_api_url': self.proxy_api_input.text(), 'selector': self.selector_input.text(),
                  'webhook_url': self.webhook_input.text(), 'rate': self.rate_input.value(), 'headless': self.headless_checkbox.isChecked()}
        
        self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        self._log("🔥 PHANTOM CELL INITIATED. Activating personas...")
        
        try:
            self.swarm_bot = SoundBotSwarm(config)
            self.swarm_bot.log_updated.connect(self._log); self.swarm_bot.play_logged.connect(self._add_to_log_data); self.swarm_bot.swarm_finished.connect(self._on_finish)
            self.swarm_thread = Thread(target=self.swarm_bot.start_swarm, daemon=True); self.swarm_thread.start()
        except Exception as e: QMessageBox.critical(self, "Fatal Error", f"Could not start swarm: {e}"); self._on_finish()

    def _on_stop(self):
        if self.swarm_bot: self.swarm_bot.stop_swarm()
        self.stop_btn.setEnabled(False); self.start_btn.setText("Disengaging...")

    def _on_finish(self):
        self._log("✅ PHANTOM CELL DISENGAGED."); self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        self.start_btn.setText("🚀 Engage Phantom Cell"); self.swarm_thread = None; self.swarm_bot = None

# =================================================================================================
# || MAIN ENTRY POINT                                                                          ||
# =================================================================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = SoundBotGUI()
    win.show()
    sys.exit(app.exec_())