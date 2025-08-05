# =================================================================================================
# ||                                                                                           ||
# ||    SOUNDBOT GODMODE v1.0                                                                  ||
# ||    "If you get caught, you don't know me."                                                ||
# ||                                                                                           ||
# ||    CORE ARCHITECTURE:                                                                     ||
# ||    - GUI: PyQt5 (Cross-platform, clean)                                                   ||
# ||    - BOT ENGINE: Multi-threaded (Concurrent.futures ThreadPoolExecutor)                   ||
# ||    - PROXY MGMT: API-integrated, auto-rotate, dead-proxy handling                         ||
# ||    - BROWSER: Stealth Selenium + undetected-chromedriver                                  ||
# ||    - ANTI-DETECT: User-agent rotation, JS fingerprint patching, geo-spoofing              ||
# ||    - LOGGING: Timestamped, GUI display, file output                                       ||
# ||                                                                                           ||
# =================================================================================================

import sys
import time
import random
import requests
import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Thread, Lock

# -- Third-party libraries (pip install -r requirements.txt)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QSpinBox, QComboBox, QTextEdit,
                             QMessageBox)
from PyQt5.QtCore import pyqtSignal, QObject
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By

# --- CONFIGURATION ---
# List of robust user agents to cycle through. The more, the better.
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5.1 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36'
]

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] (%(threadName)-10s) %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler("soundbot_godmode.log"),
                              logging.StreamHandler()])

# =================================================================================================
# || CORE MODULES                                                                              ||
# =================================================================================================

class ProxyManager:
    """
    Manages fetching, validating, and rotating proxies from an API endpoint.
    OPSEC: Residential proxies are key. BrightData, Smartproxy, etc., are the standard.
    """
    def __init__(self, proxy_api_url):
        self.api_url = proxy_api_url
        self.proxy_pool = []
        self.proxy_lock = Lock()
        if not self.api_url:
            raise ValueError("Proxy API URL cannot be empty.")
        self.refresh_proxies()

    def refresh_proxies(self):
        logging.info("Refreshing proxy pool...")
        try:
            # Assumes API returns a list of "ip:port" strings. Adjust if your provider's format differs.
            # Example JSON format: {"proxies": ["1.2.3.4:8080", "5.6.7.8:8080"]}
            resp = requests.get(self.api_url, timeout=10)
            resp.raise_for_status()
            # This logic needs to be adapted to your proxy provider's API response format
            new_proxies = resp.text.splitlines() # Simple example for a list of ip:port
            if not new_proxies:
                logging.error("Proxy API returned an empty list.")
                return
            with self.proxy_lock:
                self.proxy_pool = [{'full': p} for p in new_proxies]
            logging.info(f"Successfully loaded {len(self.proxy_pool)} new proxies.")
        except requests.RequestException as e:
            logging.error(f"Failed to fetch proxies from API: {e}")
            self.proxy_pool = []

    def get_proxy(self):
        with self.proxy_lock:
            if not self.proxy_pool:
                logging.warning("Proxy pool is empty. Attempting to refresh.")
                self.refresh_proxies()
                if not self.proxy_pool:
                    logging.critical("Could not get a proxy. Aborting task.")
                    return None
            return random.choice(self.proxy_pool)

    def ban_proxy(self, proxy):
        with self.proxy_lock:
            try:
                self.proxy_pool.remove(proxy)
                logging.warning(f"Banned and removed dead proxy: {proxy['full']}")
            except ValueError:
                pass # Proxy might have been already removed by another thread

class GeoUtils:
    """
    Utilities for looking up geolocation data from an IP address.
    Uses a free, public API. For serious use, consider a paid, more reliable service.
    """
    @staticmethod
    def get_geo_from_ip(ip_address):
        logging.info(f"Fetching GEO data for IP: {ip_address}")
        try:
            # ip-api.com is a good free option with rate limits.
            url = f"http://ip-api.com/json/{ip_address}?fields=status,message,countryCode,city,lat,lon"
            resp = requests.get(url, timeout=5)
            data = resp.json()
            if data.get("status") == "success":
                return {
                    "lat": data.get("lat", 34.0522),
                    "lon": data.get("lon", -118.2437),
                    "city": data.get("city", "Unknown"),
                    "country": data.get("countryCode", "US")
                }
        except requests.RequestException as e:
            logging.error(f"Geo lookup failed for {ip_address}: {e}")
        # Fallback to a random US location
        return {"lat": 34.0522, "lon": -118.2437, "city": "Los Angeles (Fallback)", "country": "US"}

class BrowserEngine:
    """
    Handles the creation of a stealth browser instance and the play action.
    This is where the anti-detection magic happens.
    """
    @staticmethod
    def get_stealth_browser(proxy, geo):
        options = uc.ChromeOptions()
        # Proxy setup
        options.add_argument(f'--proxy-server={proxy["full"]}')
        # Anti-detection flags
        options.add_argument(f'--user-agent={random.choice(USER_AGENTS)}')
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-infobars")
        options.add_argument("--window-size=1280,720")
        
        # --- ADVANCED FINGERPRINT SPOOFING (PLACEHOLDER) ---
        # For GODMODE, you'd inject JS here to override:
        # - navigator.plugins
        # - screen.resolution
        # - canvas fingerprint
        # - webgl fingerprint
        # This requires complex JavaScript and is highly site-specific.

        browser = uc.Chrome(options=options, version_main=114) # Pin version for stability
        
        # Geolocation spoofing via Chrome DevTools Protocol
        try:
            browser.execute_cdp_cmd("Emulation.setGeolocationOverride", {
                "latitude": geo['lat'],
                "longitude": geo['lon'],
                "accuracy": random.uniform(50, 150)
            })
            logging.info(f"Geolocation spoofed to {geo['city']}, {geo['country']}")
        except Exception as e:
            logging.error(f"Failed to spoof geolocation: {e}")
            
        return browser

    @staticmethod
    def play_song(browser, url):
        browser.get(url)
        # Random delay to simulate human page loading inspection
        time.sleep(random.uniform(3, 7))
        
        # --- DYNAMIC PLAY BUTTON SELECTOR ---
        # This is CRITICAL. The selector for the play button WILL change.
        # You need to find a reliable selector for the SoundClick play button.
        # Inspect the page and find the button's class, ID, or XPath.
        # EXAMPLE: '.play_button_class' or '#playButtonId'
        play_button_selector = ".player-button.play" # THIS IS AN EXAMPLE, UPDATE IT!
        
        try:
            # Human-like scroll
            scroll_height = browser.execute_script("return document.body.scrollHeight")
            browser.execute_script(f"window.scrollTo(0, {scroll_height * random.uniform(0.1, 0.4)});")
            time.sleep(random.uniform(1, 3))

            play_btn = browser.find_element(By.CSS_SELECTOR, play_button_selector)
            play_btn.click()
            logging.info("Play button clicked.")
        except Exception as e:
            logging.warning(f"Could not find or click play button using selector '{play_button_selector}'. Assuming autoplay. Error: {e}")

        # Simulate listen time
        listen_time = random.uniform(35, 120) # 35 to 120 seconds
        logging.info(f"Simulating listen for {listen_time:.2f} seconds.")
        time.sleep(listen_time)
        
        browser.quit()

class SoundBotSwarm(QObject):
    """
    The main bot engine. Manages the thread pool and orchestrates play tasks.
    Communicates with the GUI via signals.
    """
    log_updated = pyqtSignal(str)
    swarm_finished = pyqtSignal()
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.proxy_mgr = ProxyManager(config['proxy_api_url'])
        self.is_running = False

    def _play_task(self):
        if not self.is_running:
            return

        proxy = self.proxy_mgr.get_proxy()
        if not proxy:
            self.log_updated.emit("FATAL: Could not acquire a proxy. Skipping task.")
            return

        try:
            # Extract IP from "ip:port"
            ip = proxy['full'].split(':')[0]
            geo = GeoUtils.get_geo_from_ip(ip)

            self.log_updated.emit(f"Spawning browser for {geo['city']}, {geo['country']} via {proxy['full']}")
            browser = BrowserEngine.get_stealth_browser(proxy, geo)
            BrowserEngine.play_song(browser, self.config['url'])
            self.log_updated.emit(f"SUCCESS: Play from {geo['city']}, {geo['country']} completed.")
        except Exception as e:
            self.log_updated.emit(f"ERROR: Task failed with proxy {proxy['full']}. Banning proxy. Reason: {e}")
            self.proxy_mgr.ban_proxy(proxy)

    def start_swarm(self):
        self.is_running = True
        executor = ThreadPoolExecutor(max_workers=self.config['rate'], thread_name_prefix='BotWorker')
        
        for i in range(self.config['plays']):
            if not self.is_running:
                self.log_updated.emit("Swarm manually stopped.")
                break
                
            executor.submit(self._play_task)
            # Control the rate of new tasks
            sleep_duration = 60.0 / self.config['rate']
            time.sleep(sleep_duration)

        executor.shutdown(wait=True)
        if self.is_running: # Only emit finished if it wasn't stopped manually
            self.swarm_finished.emit()
            
    def stop_swarm(self):
        self.is_running = False
        self.log_updated.emit("Stopping swarm... waiting for active tasks to finish.")


# =================================================================================================
# || GUI (PyQt5)                                                                               ||
# =================================================================================================

class SoundBotGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SOUNDBOT GODMODE v1.0")
        self.setGeometry(100, 100, 700, 500)
        self.swarm_thread = None
        self.swarm_bot = None
        self._setup_ui()

    def _setup_ui(self):
        # --- Main Layout ---
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # --- Inputs ---
        input_layout = QVBoxLayout()
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("https://www.soundclick.com/artist/your-song-here")
        self.proxy_api_input = QLineEdit()
        self.proxy_api_input.setPlaceholderText("Your residential proxy provider API URL")

        input_layout.addWidget(QLabel("🎯 Song URL:"))
        input_layout.addWidget(self.url_input)
        input_layout.addWidget(QLabel("🔑 Proxy API URL:"))
        input_layout.addWidget(self.proxy_api_input)

        # --- Controls ---
        controls_layout = QHBoxLayout()
        self.plays_input = QSpinBox()
        self.plays_input.setRange(1, 100000)
        self.plays_input.setValue(10)
        self.rate_input = QSpinBox()
        self.rate_input.setRange(1, 100)
        self.rate_input.setValue(5)
        
        controls_layout.addWidget(QLabel("Plays:"))
        controls_layout.addWidget(self.plays_input)
        controls_layout.addWidget(QLabel("Rate (per min):"))
        controls_layout.addWidget(self.rate_input)
        
        # --- Action Buttons ---
        action_layout = QHBoxLayout()
        self.start_btn = QPushButton("🚀 Start Swarm")
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.stop_btn = QPushButton("🛑 Stop Swarm")
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        self.stop_btn.setEnabled(False)
        action_layout.addWidget(self.start_btn)
        action_layout.addWidget(self.stop_btn)

        # --- Log Area ---
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet("background-color: #1E1E1E; color: #00FF00; font-family: 'Courier New';")

        # --- Assembly ---
        main_layout.addLayout(input_layout)
        main_layout.addLayout(controls_layout)
        main_layout.addLayout(action_layout)
        main_layout.addWidget(QLabel("Live Log:"))
        main_layout.addWidget(self.log_area)

        # --- Connections ---
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)

    def _log(self, msg):
        logging.info(msg)
        self.log_area.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum())

    def _on_start(self):
        config = {
            'url': self.url_input.text(),
            'proxy_api_url': self.proxy_api_input.text(),
            'plays': self.plays_input.value(),
            'rate': self.rate_input.value()
        }

        if not config['url'] or not config['proxy_api_url']:
            QMessageBox.critical(self, "Error", "Song URL and Proxy API URL cannot be empty.")
            return

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._log("🔥 Swarm initiated. Building proxy pool and preparing for launch...")
        
        try:
            self.swarm_bot = SoundBotSwarm(config)
            self.swarm_bot.log_updated.connect(self._log)
            self.swarm_bot.swarm_finished.connect(self._on_finish)
            
            self.swarm_thread = Thread(target=self.swarm_bot.start_swarm, daemon=True)
            self.swarm_thread.start()

        except Exception as e:
            QMessageBox.critical(self, "Fatal Error", f"Could not start the swarm: {e}")
            self._on_finish()


    def _on_stop(self):
        if self.swarm_bot:
            self.swarm_bot.stop_swarm()
        self.stop_btn.setEnabled(False)
        self.start_btn.setText("Stopping...")

    def _on_finish(self):
        self._log("✅ Swarm has completed its run or has been stopped.")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.start_btn.setText("🚀 Start Swarm")
        self.swarm_thread = None
        self.swarm_bot = None
        
    def closeEvent(self, event):
        if self.is_running():
            reply = QMessageBox.question(self, 'Confirm Exit', 
                                         "A swarm is currently running. Are you sure you want to exit?", 
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self._on_stop()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def is_running(self):
        return self.swarm_thread is not None and self.swarm_thread.is_alive()


# =================================================================================================
# || MAIN ENTRY POINT                                                                          ||
# =================================================================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # You might want to create a `requirements.txt` file with:
    # PyQt5
    # undetected-chromedriver
    # selenium
    # requests
    win = SoundBotGUI()
    win.show()
    sys.exit(app.exec_())