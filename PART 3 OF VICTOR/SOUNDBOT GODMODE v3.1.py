# =================================================================================================
# ||                                                                                           ||
# ||    SOUNDBOT GODMODE v3.0 - GHOST PROTOCOL                                                 ||
# ||    "The best operator is the one they never knew was there."                              ||
# ||                                                                                           ||
# ||    V3.0 UPGRADES:                                                                         ||
# ||    - FULL STEALTH PACK: Randomized Timezone, Language, Screen Res, Canvas/WebGL noise.    ||
# ||    - GEO-MIX REPORTING: Export the entire play log to a CSV file for intel/proof.         ||
# ||    - FAKER INTEGRATION: Foundation for generating realistic user personas.                ||
# ||    - HARDENED CORE: All v2.0 features are retained and optimized.                         ||
# ||                                                                                           ||
# =================================================================================================


import sys
import time
import random
import requests
import logging
import csv
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from threading import Thread, Lock

# -- Third-party libraries
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QSpinBox, QTextEdit,
                             QMessageBox, QFileDialog)
from PyQt5.QtCore import pyqtSignal, QObject
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from faker import Faker  # pip install Faker

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] (%(threadName)-10s) %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler("soundbot_godmode_v3.log"),
                              logging.StreamHandler()])

TIMEZONES = ['America/New_York', 'America/Chicago', 'America/Denver', 'America/Los_Angeles', 'Europe/London', 'Europe/Berlin', 'Europe/Moscow', 'Asia/Tokyo', 'Australia/Sydney']
LANGUAGES = ['en-US,en;q=0.9', 'en-GB,en;q=0.8', 'es-ES,es;q=0.9', 'fr-FR,fr;q=0.9', 'de-DE,de;q=0.9', 'ja-JP,ja;q=0.9']
SCREEN_RESOLUTIONS = ['1920,1080', '1600,900', '1366,768', '2560,1440']

class ProxyManager:
    """ Manages proxies. """
    def __init__(self, proxy_api_url):
        self.api_url = proxy_api_url
        self.proxy_pool = []
        self.proxy_lock = Lock()
        if not self.api_url: raise ValueError("Proxy API URL cannot be empty.")
        self.refresh_proxies()
    def refresh_proxies(self):
        logging.info("Refreshing proxy pool...")
        try:
            resp = requests.get(self.api_url, timeout=15); resp.raise_for_status()
            new_proxies = resp.text.splitlines()
            with self.proxy_lock: self.proxy_pool = [{'full': p} for p in new_proxies if p]
            logging.info(f"Loaded {len(self.proxy_pool)} new proxies.")
        except requests.RequestException as e:
            logging.error(f"Failed to fetch proxies: {e}"); self.proxy_pool = []
    def get_proxy(self):
        with self.proxy_lock:
            if not self.proxy_pool: self.refresh_proxies()
            if not self.proxy_pool: return None
            return random.choice(self.proxy_pool)
    def ban_proxy(self, proxy):
        with self.proxy_lock:
            try: self.proxy_pool.remove(proxy); logging.warning(f"Banned proxy: {proxy['full']}")
            except ValueError: pass

class GeoUtils:
    """ GeoIP lookup. """
    @staticmethod
    def get_geo_from_ip(ip_address):
        try:
            resp = requests.get(f"http://ip-api.com/json/{ip_address}?fields=status,message,country,city,lat,lon", timeout=5)
            data = resp.json()
            if data.get("status") == "success":
                return {"lat": data.get("lat"), "lon": data.get("lon"), "city": data.get("city"), "country": data.get("country")}
        except requests.RequestException: pass
        return {"lat": 34.0522, "lon": -118.2437, "city": "Los Angeles (Fallback)", "country": "United States (Fallback)"}

class BrowserEngine:
    """ Handles stealth browser creation and automation. """
    @staticmethod
    def get_stealth_browser(proxy, geo):
        fake = Faker()
        options = uc.ChromeOptions()
        user_agent = fake.user_agent()
        lang = random.choice(LANGUAGES)
        res = random.choice(SCREEN_RESOLUTIONS)
        options.add_argument(f'--user-agent={user_agent}')
        options.add_argument(f'--lang={lang}')
        options.add_argument(f'--window-size={res}')
        options.add_argument(f'--proxy-server={proxy["full"]}')
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument('--no-sandbox')
        options.add_argument("--disable-dev-shm-usage")
        browser = uc.Chrome(options=options)
        js_stealth_suite = f"""
            Object.defineProperty(navigator, 'webdriver', {{get: () => undefined}});
            Object.defineProperty(Intl.DateTimeFormat.prototype, 'resolvedOptions', {{
                value: () => ({{ timeZone: '{random.choice(TIMEZONES)}' }})
            }});
            Object.defineProperty(navigator, 'languages', {{get: () => ['{lang.split(",")[0]}', '{lang.split(",")[1].split(";")[0]}']}});
            const toDataURL = HTMLCanvasElement.prototype.toDataURL;
            HTMLCanvasElement.prototype.toDataURL = function() {{
                const context = this.getContext('2d');
                const str = 'abcdefghijklmnopqrstuvwxyz0123456789';
                const random_text = Array(10).join().split(',').map(() => str.charAt(Math.floor(Math.random() * str.length))).join('');
                context.font = '16px monospace';
                context.fillStyle = 'rgb({Math.floor(Math.random()*255)}, {Math.floor(Math.random()*255)}, {Math.floor(Math.random()*255)})';
                context.fillText(random_text, Math.random()*10, Math.random()*10);
                return toDataURL.apply(this, arguments);
            }};
            const getParameter = WebGLRenderingContext.prototype.getParameter;
            WebGLRenderingContext.prototype.getParameter = function(parameter) {{
                if (parameter === 37445) return 'Intel Open Source Technology Center';
                if (parameter === 37446) return 'Mesa DRI Intel(R) Ivybridge Mobile';
                return getParameter.apply(this, arguments);
            }};
        """
        try:
            browser.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": js_stealth_suite})
            logging.info("Full Stealth Pack (JS Suite) injected.")
        except Exception as e:
            logging.warning(f"Could not inject JS Stealth Suite: {e}")

        try:
            browser.execute_cdp_cmd("Emulation.setGeolocationOverride", {"latitude": geo['lat'], "longitude": geo['lon'], "accuracy": 100})
        except Exception as e:
            logging.error(f"Failed to spoof geolocation: {e}")
        return browser

    @staticmethod
    def play_song(browser, url, play_button_selector):
        browser.get(url)
        time.sleep(random.uniform(5, 10))
        try:
            browser.execute_script(f"window.scrollTo(0, document.body.scrollHeight * {random.uniform(0.2, 0.5)});")
            time.sleep(random.uniform(1, 3))
            play_btn = browser.find_element(By.CSS_SELECTOR, play_button_selector)
            play_btn.click()
            logging.info(f"Play button clicked on ...{url[-25:]}")
        except Exception:
            logging.warning("Could not click play button. Assuming autoplay.")
        listen_time = random.uniform(60, 150)
        logging.info(f"Simulating listen for {listen_time:.0f}s.")
        time.sleep(listen_time)
        browser.quit()

class SoundBotSwarm(QObject):
    log_updated = pyqtSignal(str)
    play_logged = pyqtSignal(dict)
    swarm_finished = pyqtSignal()
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.proxy_mgr = ProxyManager(config['proxy_api_url'])
        self.is_running = False

    def _send_webhook(self, message, success=True):
        if not self.config['webhook_url']: return
        try:
            color = 0x00ff00 if success else 0xff0000
            requests.post(self.config['webhook_url'], json={"embeds": [{"description": message, "color": color, "footer": {"text": f"SOUNDBOT GODMODE v3.0"}}]}, timeout=5)
        except requests.RequestException: self.log_updated.emit("[ERROR] Failed to send Discord webhook.")

    def _play_task(self):
        if not self.is_running: return
        proxy = self.proxy_mgr.get_proxy()
        if not proxy: self.log_updated.emit("[ERROR] No proxy available. Task aborted."); return

        geo = {}
        try:
            ip = proxy['full'].split(':')[0]
            geo = GeoUtils.get_geo_from_ip(ip)
            target_url = random.choice(self.config['urls'])
            self.log_updated.emit(f"[INFO] Spawning ghost browser for {geo.get('city', 'N/A')}, {geo.get('country', 'N/A')}")
            
            browser = BrowserEngine.get_stealth_browser(proxy, geo)
            BrowserEngine.play_song(browser, target_url, self.config['selector'])
            
            log_entry = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'url': target_url, 'ip': ip, 'city': geo.get('city', 'N/A'),
                'country': geo.get('country', 'N/A'), 'status': 'Success'
            }
            self.play_logged.emit(log_entry)
            msg = f"✅ SUCCESS: Play from {geo.get('city', 'N/A')}, {geo.get('country', 'N/A')}"
            self.log_updated.emit(f"[INFO] {msg}")
            self._send_webhook(msg, success=True)
            
        except Exception as e:
            log_entry = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'url': 'N/A', 'ip': proxy.get('full', 'N/A'),
                'city': geo.get('city', 'N/A'), 'country': geo.get('country', 'N/A'), 'status': f'Failure: {e}'
            }
            self.play_logged.emit(log_entry)
            error_msg = f"❌ ERROR: Task failed. Banning proxy {proxy['full']}"
            self.log_updated.emit(f"[ERROR] {error_msg}")
            self._send_webhook(error_msg, success=False)
            self.proxy_mgr.ban_proxy(proxy)

    def start_swarm(self):
        self.is_running = True
        self._send_webhook(f"🚀 GHOST PROTOCOL ENGAGED! Target plays: {self.config['plays']}", success=True)
        with ThreadPoolExecutor(max_workers=self.config['rate'], thread_name_prefix='GhostWorker') as executor:
            for _ in range(self.config['plays']):
                if not self.is_running: self.log_updated.emit("[INFO] Swarm manually stopped."); break
                executor.submit(self._play_task)
                time.sleep(60.0 / self.config['rate'])
        executor.shutdown(wait=True)
        if self.is_running: self.swarm_finished.emit()
            
    def stop_swarm(self):
        if self.is_running: self.is_running = False; self._send_webhook("🛑 Swarm manually disengaged.", success=False)

class SoundBotGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SOUNDBOT GODMODE v3.0 - Ghost Protocol")
        self.setGeometry(100, 100, 800, 750)
        self.swarm_thread, self.swarm_bot = None, None
        self.play_log_data = []
        self._setup_ui()

    def _setup_ui(self):
        main_widget = QWidget(); self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        main_layout.addWidget(QLabel("🎯 Target URLs (Batch Mode: One per line):"))
        self.url_input = QTextEdit(); self.url_input.setPlaceholderText("https://www.soundclick.com/artist/your-song-1\n...")
        main_layout.addWidget(self.url_input)

        config_layout = QVBoxLayout()
        config_layout.addWidget(QLabel("🔑 Proxy API URL:"))
        self.proxy_api_input = QLineEdit()
        config_layout.addWidget(self.proxy_api_input)
        
        config_layout.addWidget(QLabel("🖱️ Play Button CSS Selector:"))
        self.selector_input = QLineEdit(); self.selector_input.setText(".player-button.play")
        config_layout.addWidget(self.selector_input)

        config_layout.addWidget(QLabel("🔔 Discord Webhook URL (Optional):"))
        self.webhook_input = QLineEdit()
        config_layout.addWidget(self.webhook_input)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Plays:")); self.plays_input = QSpinBox(); self.plays_input.setRange(1, 100000); self.plays_input.setValue(10); controls_layout.addWidget(self.plays_input)
        controls_layout.addWidget(QLabel("Rate (per min):")); self.rate_input = QSpinBox(); self.rate_input.setRange(1, 100); self.rate_input.setValue(5); controls_layout.addWidget(self.rate_input)
        
        self.start_btn = QPushButton("🚀 Engage Ghost Protocol"); self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.stop_btn = QPushButton("🛑 Disengage"); self.stop_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 10px;"); self.stop_btn.setEnabled(False)
        self.export_btn = QPushButton("📊 Export Log to CSV"); self.export_btn.setStyleSheet("background-color: #008CBA; color: white; font-weight: bold; padding: 10px;");
        
        main_layout.addLayout(config_layout)
        main_layout.addLayout(controls_layout)
        action_layout = QHBoxLayout(); action_layout.addWidget(self.start_btn); action_layout.addWidget(self.stop_btn);
        main_layout.addLayout(action_layout)
        main_layout.addWidget(self.export_btn)
        main_layout.addWidget(QLabel("Live Intel Feed:"))
        self.log_area = QTextEdit(); self.log_area.setReadOnly(True); self.log_area.setStyleSheet("background-color: #1E1E1E; color: #00FF00; font-family: 'Courier New';")
        main_layout.addWidget(self.log_area)

        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        self.export_btn.clicked.connect(self._on_export)

    def _log(self, msg):
        self.log_area.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum())

    def _add_to_log_data(self, entry): self.play_log_data.append(entry)

    def _on_export(self):
        if not self.play_log_data: QMessageBox.information(self, "Export Log", "No log data to export."); return
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV Log", f"soundbot_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "CSV Files (*.csv)")
        if path:
            try:
                with open(path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.play_log_data[0].keys())
                    writer.writeheader(); writer.writerows(self.play_log_data)
                self._log("[INFO] Successfully exported log to CSV.")
            except Exception as e: QMessageBox.critical(self, "Export Error", f"Failed to save CSV file: {e}")

    def _on_start(self):
        urls = [url.strip() for url in self.url_input.toPlainText().splitlines() if url.strip()]
        if not urls or not self.proxy_api_input.text() or not self.selector_input.text():
            QMessageBox.critical(self, "Error", "Target URLs, Proxy API, and Selector are required."); return
        
        self.play_log_data.clear(); self._log("[INFO] Log cache cleared for new run.")
        config = {'urls': urls, 'proxy_api_url': self.proxy_api_input.text(), 'selector': self.selector_input.text(),
                  'webhook_url': self.webhook_input.text(), 'plays': self.plays_input.value(), 'rate': self.rate_input.value()}

        self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        self._log("🔥 GHOST PROTOCOL INITIATED. STAND BY.")
        
        try:
            self.swarm_bot = SoundBotSwarm(config)
            self.swarm_bot.log_updated.connect(self._log)
            self.swarm_bot.play_logged.connect(self._add_to_log_data)
            self.swarm_bot.swarm_finished.connect(self._on_finish)
            self.swarm_thread = Thread(target=self.swarm_bot.start_swarm, daemon=True)
            self.swarm_thread.start()
        except Exception as e: QMessageBox.critical(self, "Fatal Error", f"Could not start swarm: {e}"); self._on_finish()

    def _on_stop(self):
        if self.swarm_bot: self.swarm_bot.stop_swarm()
        self.stop_btn.setEnabled(False); self.start_btn.setText("Disengaging...")

    def _on_finish(self):
        self._log("✅ GHOST PROTOCOL DISENGAGED. Mission complete or aborted.")
        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        self.start_btn.setText("🚀 Engage Ghost Protocol"); self.swarm_thread = None; self.swarm_bot = None
        
    def closeEvent(self, event):
        if self.swarm_thread and self.swarm_thread.is_alive():
            if QMessageBox.question(self, 'Confirm Exit', "A swarm is active. Disengage?", QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
                self._on_stop(); event.accept()
            else: event.ignore()
        else: event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = SoundBotGUI()
    win.show()
    sys.exit(app.exec_())