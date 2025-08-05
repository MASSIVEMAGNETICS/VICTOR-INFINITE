# FILE: soundclick_bot.py
# VERSION: v1.0.1-SOUNDBOT-GODMODE-GODCORE
# NAME: SoundClickBot
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Monolithic SoundClick bot with threaded GUI, proxy pool, multi-selector scraping, play automation, and self-healing.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network

import sys, time, random, csv, requests, logging
from datetime import datetime
from threading import Thread, Lock

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QTextEdit, QListWidget, QListWidgetItem,
    QSpinBox, QPushButton, QMenuBar, QAction, QFileDialog, QMessageBox
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

###############
# Proxy Manager
###############
class ProxyPoolManager:
    def __init__(self):
        self.lock = Lock()
        self.proxies = []
        self.idx = 0

    def load_from_file(self, path):
        with open(path, 'r') as f:
            lst = [l.strip() for l in f if l.strip()]
        with self.lock:
            self.proxies = lst
            self.idx = 0
        return len(lst)

    def load_from_api(self, api_url):
        resp = requests.get(api_url, timeout=10)
        resp.raise_for_status()
        lst = [l.strip() for l in resp.text.splitlines() if l.strip()]
        with self.lock:
            self.proxies = lst
            self.idx = 0
        return len(lst)

    def get_next(self):
        with self.lock:
            if not self.proxies: return None
            p = self.proxies[self.idx % len(self.proxies)]
            self.idx += 1
        return p

####################
# Core SoundClickBot
####################
class SoundClickBot:
    def __init__(self, proxy_mgr: ProxyPoolManager):
        self.proxy_mgr = proxy_mgr
        # defaults — GUI will override these if needed
        self.song_selector = 'div.songListItem p.songTitle a'
        self.play_selector = '.player-button.play'

    def _init_driver(self):
        opts = uc.ChromeOptions()
        opts.add_argument('--disable-blink-features=AutomationControlled')
        proxy = self.proxy_mgr.get_next()
        if proxy:
            opts.add_argument(f'--proxy-server={proxy}')
            logging.info(f"Using proxy {proxy}")
        try:
            drv = uc.Chrome(options=opts)
            drv.set_page_load_timeout(60)
            return drv
        except Exception as e:
            raise RuntimeError(f"Chrome driver init failed: {e}")

    def extract_songs(self, driver, url, selector=None):
        sel = selector or self.song_selector
        try:
            driver.get(url)
        except Exception as e:
            raise RuntimeError(f"Failed to load artist page: {e}")
        time.sleep(3)

        # scroll with cutoff
        last_h = driver.execute_script("return document.body.scrollHeight")
        for _ in range(8):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.5)
            new_h = driver.execute_script("return document.body.scrollHeight")
            if new_h == last_h: break
            last_h = new_h

        # try primary selector
        elems = []
        try:
            elems = driver.find_elements(By.CSS_SELECTOR, sel)
        except:
            elems = []

        # fallback to default if custom failed
        if not elems and sel != self.song_selector:
            logging.warning(f"No elems with '{sel}', falling back to default '{self.song_selector}'")
            try:
                elems = driver.find_elements(By.CSS_SELECTOR, self.song_selector)
            except:
                elems = []

        # last-ditch fallback
        if not elems:
            try:
                elems = driver.find_elements(By.CSS_SELECTOR, 'table.SongTable a')
            except:
                elems = []

        urls = []
        for e in elems:
            href = e.get_attribute('href')
            if href: urls.append(href)
        # dedupe
        return list(dict.fromkeys(urls))

    def play_track(self, driver, track_url, selector=None, play_time=30):
        sel = selector or self.play_selector
        try:
            driver.get(track_url)
        except Exception as e:
            logging.error(f"Load track failed: {e}")
            return False
        time.sleep(3)
        try:
            btn = driver.find_element(By.CSS_SELECTOR, sel)
            btn.click()
            time.sleep(play_time + random.uniform(0,5))
            return True
        except Exception as e:
            logging.error(f"Play click failed: {e}")
            return False

    def run_artist(self, artist_url, plays, play_time, song_sel, play_sel, proxy_api, gui_callback=None):
        # dynamic proxy API load
        if proxy_api:
            try:
                cnt = self.proxy_mgr.load_from_api(proxy_api)
                logging.info(f"Loaded {cnt} proxies from API")
            except Exception as e:
                logging.warning(f"Proxy API load failed: {e}")

        driver = None
        try:
            driver = self._init_driver()
            # scrape songs
            songs = self.extract_songs(driver, artist_url, song_sel)
            if gui_callback:
                gui_callback(f"[Found {len(songs)} songs]")
            for track in songs:
                for _ in range(plays):
                    if not self.play_track(driver, track, play_sel, play_time):
                        break
                    # log
                    with open("plays_log.csv","a",newline="") as f:
                        csv.writer(f).writerow([track, datetime.now().isoformat()])
                    time.sleep(60/plays)  # simple rate-limit
        except Exception as e:
            logging.error(f"Error in artist run: {e}")
            if gui_callback:
                gui_callback(f"[ERROR] {e}")
        finally:
            if driver:
                try: driver.quit()
                except: pass

################
# PyQt5 Frontend
################
class SoundClickGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SOUNDBOT GODMODE v3.1 - Ghost Protocol")
        self.proxy_mgr = ProxyPoolManager()
        self.bot = SoundClickBot(self.proxy_mgr)
        self._stop_flag = False
        self._build_menu()
        self._build_ui()

    def _build_menu(self):
        mb = self.menuBar()
        fm = mb.addMenu("File")
        lp = QAction("Load Proxies...", self)
        lp.triggered.connect(self._on_load_proxies)
        fm.addAction(lp)
        fm.addSeparator()
        ex = QAction("Exit", self)
        ex.triggered.connect(self.close)
        fm.addAction(ex)

    def _build_ui(self):
        w = QWidget(); L = QVBoxLayout()

        # Artist URL + Load Songs
        row = QHBoxLayout()
        row.addWidget(QLabel("SoundClick Artist URL:"))
        self.urlIn = QLineEdit()
        row.addWidget(self.urlIn)
        self.loadBtn = QPushButton("Load Songs")
        self.loadBtn.clicked.connect(self._spawn_load_songs)
        row.addWidget(self.loadBtn)
        L.addLayout(row)

        # Song Link CSS Selector
        L.addWidget(QLabel("Song Link CSS Selector:"))
        self.songSelIn = QLineEdit(self.bot.song_selector)
        L.addWidget(self.songSelIn)

        # Scraped songs list
        L.addWidget(QLabel("🎵 Select songs to target:"))
        self.songList = QListWidget()
        self.songList.setSelectionMode(QListWidget.MultiSelection)
        L.addWidget(self.songList)

        # Manual URLs fallback
        L.addWidget(QLabel("Or paste Target URLs manually (one per line):"))
        self.manualIn = QTextEdit()
        L.addWidget(self.manualIn)

        # Proxy API URL
        L.addWidget(QLabel("Proxy API URL:"))
        self.proxyApiIn = QLineEdit()
        L.addWidget(self.proxyApiIn)

        # Play Button CSS Selector
        L.addWidget(QLabel("Play Button CSS Selector:"))
        self.playSelIn = QLineEdit(self.bot.play_selector)
        L.addWidget(self.playSelIn)

        # Discord Webhook (optional)
        L.addWidget(QLabel("Discord Webhook URL (Optional):"))
        self.webhookIn = QLineEdit()
        L.addWidget(self.webhookIn)

        # Plays & Rate
        hr = QHBoxLayout()
        hr.addWidget(QLabel("Plays:"))
        self.playsSpin = QSpinBox(); self.playsSpin.setMinimum(1); self.playsSpin.setValue(1)
        hr.addWidget(self.playsSpin)
        hr.addWidget(QLabel("Rate (per min):"))
        self.rateSpin = QSpinBox(); self.rateSpin.setMinimum(1); self.rateSpin.setValue(5)
        hr.addWidget(self.rateSpin)
        L.addLayout(hr)

        # Start/Stop buttons
        row2 = QHBoxLayout()
        self.startBtn = QPushButton("🚀 Engage Ghost Protocol")
        self.startBtn.clicked.connect(self._spawn_start)
        row2.addWidget(self.startBtn)
        self.stopBtn = QPushButton("⛔ Disengage")
        self.stopBtn.clicked.connect(self._on_stop)
        row2.addWidget(self.stopBtn)
        L.addLayout(row2)

        # Export log
        self.exportBtn = QPushButton("Export Log to CSV")
        self.exportBtn.clicked.connect(self._on_export)
        L.addWidget(self.exportBtn)

        w.setLayout(L)
        self.setCentralWidget(w)

    # ---- threading wrappers ----
    def _spawn_load_songs(self):
        self.loadBtn.setEnabled(False)
        Thread(target=self._load_songs, daemon=True).start()

    def _spawn_start(self):
        self._stop_flag = False
        self.startBtn.setEnabled(False)
        Thread(target=self._run_bot, daemon=True).start()

    # ---- GUI Actions ----
    def _on_load_proxies(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select proxy list")
        if not p: return
        try:
            cnt = self.proxy_mgr.load_from_file(p)
            QMessageBox.information(self, "Proxies", f"Loaded {cnt} proxies.")
        except Exception as e:
            QMessageBox.critical(self, "Proxy Load Error", str(e))

    def _on_stop(self):
        self._stop_flag = True
        QMessageBox.information(self, "Disengage", "Bots will stop after current track.")

    def _on_export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save log", filter="CSV Files (*.csv)")
        if path:
            try:
                import shutil
                shutil.copy("plays_log.csv", path)
                QMessageBox.information(self, "Exported", f"plays_log.csv → {path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))

    # ---- worker functions ----
    def _load_songs(self):
        try:
            url = self.urlIn.text().strip()
            if not url:
                raise ValueError("Artist URL is empty")
            selector = self.songSelIn.text().strip() or None
            drv = self.bot._init_driver()
            songs = self.bot.extract_songs(drv, url, selector)
            drv.quit()
            self.songList.clear()
            for s in songs:
                item = QListWidgetItem(s)
                item.setSelected(True)
                self.songList.addItem(item)
            if not songs:
                QMessageBox.warning(self, "No Songs", "No URLs found – check selector or page.")
        except Exception as e:
            QMessageBox.critical(self, "Load Songs Error", str(e))
        finally:
            self.loadBtn.setEnabled(True)

    def _run_bot(self):
        artists = [self.urlIn.text().strip()]
        plays = self.playsSpin.value()
        rate = self.rateSpin.value()
        play_time = int(60/rate)
        song_sel = self.songSelIn.text().strip() or None
        play_sel = self.playSelIn.text().strip() or None
        proxy_api = self.proxyApiIn.text().strip() or None

        # collect targets
        targets = [item.text() for item in self.songList.selectedItems()]
        if not targets:
            manual = [l.strip() for l in self.manualIn.toPlainText().splitlines() if l.strip()]
            targets = manual
        if not targets:
            QMessageBox.warning(self, "No Targets", "Select or paste at least one URL"); return

        # run each target URL as its own 'artist' for simplicity
        for t in targets:
            if self._stop_flag: break
            self.bot.run_artist(t, plays, play_time, song_sel, play_sel, proxy_api,
                                gui_callback=lambda msg: logging.info(msg))

        QMessageBox.information(self, "Done", "All tasks complete.")
        self.startBtn.setEnabled(True)

def main():
    app = QApplication(sys.argv)
    gui = SoundClickGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
