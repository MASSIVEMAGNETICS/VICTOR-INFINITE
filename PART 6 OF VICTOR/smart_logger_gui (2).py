# FILE: smart_logger_gui.py
# VERSION: v1.2.0-SMARTLOGGER-GODCORE
# NAME: SmartLoggerGUI
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Standalone universal event-logging smart GUI-based logger with process & memory attachment and process explorer
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network

import sys
import logging
import threading
import subprocess
import time
import os

import psutil  # for process listing
try:
    import frida  # for dynamic instrumentation
except ImportError:
    frida = None

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QFileDialog, QInputDialog
)
from PyQt5.QtCore import pyqtSignal, QObject

# Setup core logger
log_filename = os.path.join(os.getcwd(), 'smart_logger.log')
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('SmartLogger')

class EventEmitter(QObject):
    new_event = pyqtSignal(str, str)

class ProcessWatcher(threading.Thread):
    def __init__(self, cmd, emitter):
        super().__init__(daemon=True)
        self.cmd = cmd
        self.emitter = emitter

    def run(self):
        logger.info(f"Starting process: {' '.join(self.cmd)}")
        try:
            proc = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            for line in proc.stdout:
                self.emitter.new_event.emit('INFO', line.strip())
                logger.info(line.strip())
            for err in proc.stderr:
                self.emitter.new_event.emit('ERROR', err.strip())
                logger.error(err.strip())
            proc.wait()
            exit_msg = f"Process exited with code {proc.returncode}"
            logger.info(exit_msg)
            self.emitter.new_event.emit('INFO', exit_msg)
        except Exception as e:
            logger.exception("Failed to watch process")
            self.emitter.new_event.emit('ERROR', str(e))

class MemoryWatcher(threading.Thread):
    """
    Attaches to a running process and hooks memory/API calls via Frida.
    """
    def __init__(self, pid, emitter):
        super().__init__(daemon=True)
        self.pid = pid
        self.emitter = emitter

    def run(self):
        if not frida:
            err = 'frida module not installed. run `pip install frida`'
            logger.error(err)
            self.emitter.new_event.emit('ERROR', err)
            return
        try:
            session = frida.attach(self.pid)
            # Example Frida script: hook malloc calls
            script_src = """
            var malloc = Module.getExportByName(null, 'malloc');
            Interceptor.attach(malloc, {
                onEnter: function(args) {
                    send('malloc called size=' + args[0].toInt32());
                }
            });
            """
            script = session.create_script(script_src)
            script.on('message', lambda msg, data: self.emitter.new_event.emit('INFO', msg['payload']))
            script.load()
            info = f"[MemoryWatcher] Attached to PID {self.pid}"
            logger.info(info)
            self.emitter.new_event.emit('INFO', info)
            while True:
                time.sleep(1)
        except Exception as e:
            logger.exception("Memory watcher error")
            self.emitter.new_event.emit('ERROR', str(e))

class FileWatcher(threading.Thread):
    def __init__(self, file_path, emitter):
        super().__init__(daemon=True)
        self.file_path = file_path
        self.emitter = emitter

    def run(self):
        logger.info(f"Attaching to log file: {self.file_path}")
        try:
            with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(0, os.SEEK_END)
                while True:
                    line = f.readline()
                    if line:
                        self.emitter.new_event.emit('INFO', line.strip())
                        logger.info(line.strip())
                    else:
                        time.sleep(0.2)
        except Exception as e:
            logger.exception("File watcher error")
            self.emitter.new_event.emit('ERROR', str(e))

class SmartLoggerGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Smart Logger GUI')
        self.resize(950, 600)

        self.emitter = EventEmitter()
        self.emitter.new_event.connect(self.add_event)

        main_layout = QVBoxLayout()
        btn_layout = QHBoxLayout()

        # Action buttons
        self.launch_btn = QPushButton('Launch Process')
        self.launch_btn.clicked.connect(self.launch_process)

        self.attach_btn = QPushButton('Attach Log File')
        self.attach_btn.clicked.connect(self.attach_file)

        self.process_btn = QPushButton('Select & Attach Process')
        self.process_btn.clicked.connect(self.select_and_attach_process)

        self.manual_btn = QPushButton('Add Manual Event')
        self.manual_btn.clicked.connect(self.add_manual_event)

        btn_layout.addWidget(self.launch_btn)
        btn_layout.addWidget(self.attach_btn)
        btn_layout.addWidget(self.process_btn)
        btn_layout.addWidget(self.manual_btn)

        # Event table
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(['Timestamp', 'Level', 'Message'])

        main_layout.addLayout(btn_layout)
        main_layout.addWidget(self.table)
        self.setLayout(main_layout)

    def launch_process(self):
        exe_path, _ = QFileDialog.getOpenFileName(self, 'Select Executable or Script')
        if not exe_path:
            return
        args_text, ok = QInputDialog.getText(self, 'Process Arguments', 'Enter arguments (space-separated):')
        if not ok:
            return
        cmd = [exe_path] + args_text.split()
        watcher = ProcessWatcher(cmd, self.emitter)
        watcher.start()

    def attach_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Select Log File to Tail')
        if not file_path:
            return
        watcher = FileWatcher(file_path, self.emitter)
        watcher.start()

    def select_and_attach_process(self):
        procs = [f"{p.pid}: {p.name()}" for p in psutil.process_iter(['name'])]
        proc_item, ok = QInputDialog.getItem(self, 'Select Process', 'Process:', procs, 0, False)
        if not ok or not proc_item:
            return
        pid = int(proc_item.split(':')[0])
        watcher = MemoryWatcher(pid, self.emitter)
        watcher.start()

    def add_manual_event(self):
        text, ok = QInputDialog.getText(self, 'Manual Event', 'Enter event message:')
        if ok and text:
            self.emitter.new_event.emit('INFO', text)
            logger.info(text)

    def add_event(self, level, message):
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(ts))
        self.table.setItem(row, 1, QTableWidgetItem(level))
        self.table.setItem(row, 2, QTableWidgetItem(message))
        self.table.scrollToBottom()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = SmartLoggerGUI()
    win.show()
    sys.exit(app.exec_())
