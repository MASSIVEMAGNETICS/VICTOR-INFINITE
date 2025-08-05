# FILE: smart_logger_gui.py
# VERSION: v1.5.0-SMARTLOGGER-GODCORE
# NAME: SmartLoggerGUI
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Standalone universal event-logging smart GUI-based logger with process, file, memory attachment, process explorer, and resource management
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
#
# CHANGELOG:
# v1.3.0 - Integrated Memory Watching with Frida, Process Explorer with psutil,
#          and added individual watcher controls.
# v1.4.0 - Major Documentation Overhaul: Merged extensive docstrings and inline
#          comments for better code readability and maintainability across the
#          entire script, including new features.
# v1.5.0 - Dark Theme with Neon Accents: Applied a global dark stylesheet
#          and adjusted color-coding for log levels to fit the new theme.

import sys
import logging
import threading
import subprocess
import time
import os

# Third-party dependencies
import psutil # Used for listing running processes for the MemoryWatcher.
try:
    import frida # Used for dynamic instrumentation (memory hooking).
except ImportError:
    # If frida is not installed, features requiring it will be disabled.
    # Users will need to install it manually (e.g., `pip install frida-tools`).
    frida = None

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QFileDialog, QInputDialog, QHeaderView, QLineEdit, QSplitter,
    QGroupBox
)
from PyQt5.QtCore import pyqtSignal, Qt # Qt is imported for QtCore.Qt flags like Qt.Vertical
from PyQt5.QtGui import QColor # QColor is used for setting cell background colors based on log level

# --- Core Logger Setup ---
# Define the log filename, ensuring it resides in the current working directory.
log_filename = os.path.join(os.getcwd(), 'smart_logger.log')
logging.basicConfig(
    level=logging.DEBUG, # Set the minimum logging level to DEBUG for verbose logging.
    format='%(asctime)s [%(levelname)s] %(message)s', # Define the log message format.
    handlers=[
        # Log messages to a file. 'encoding' ensures proper character handling.
        logging.FileHandler(log_filename, encoding='utf-8'),
        # Log messages to the console (standard output) for immediate feedback.
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('SmartLogger') # Get a logger instance named 'SmartLogger'.

# --- Base Watcher Thread ---
class StoppableThread(threading.Thread):
    """
    Base class for all watcher threads (Process, File, Memory).
    It provides a common stop mechanism and a unique ID for tracking.
    It also emits a signal upon completion, allowing the GUI to perform cleanup.
    """
    # Define a signal that will be emitted when any watcher thread finishes its task.
    # The signal carries the unique ID of the watcher thread, enabling the GUI
    # to identify and remove the correct entry from the 'Active Watchers' table.
    watcher_finished = pyqtSignal(int)

    def __init__(self, emitter):
        """
        Initializes the StoppableThread.
        :param emitter: The SmartLoggerGUI instance. This object acts as the
                        central hub for emitting signals (like new log events
                        or watcher completion signals) back to the GUI's main thread,
                        ensuring thread-safe UI updates.
        """
        super().__init__(daemon=True) # Daemon threads terminate automatically when the main program exits.
        self._is_running = True     # A flag to control the thread's main execution loop.
        self.emitter = emitter      # Reference to the GUI's emitter (SmartLoggerGUI instance).
        self.id = id(self)          # Assign a unique integer ID to this specific thread instance.
                                    # This ID is used for tracking and managing the thread in the GUI.

    def stop(self):
        """
        Sets the internal `_is_running` flag to False.
        This signals the thread's `run` method to gracefully exit its loop.
        """
        logger.info(f"Stop signal sent to thread: {self.id}")
        self._is_running = False

    def run(self):
        """
        The main execution logic for the thread. This method must be overridden
        by subclasses (e.g., `ProcessWatcher`, `FileWatcher`, `MemoryWatcher`).
        Subclasses should implement their monitoring logic within this method
        and ensure they periodically check `self._is_running` to respond to stop requests.
        Crucially, subclasses *must* emit `self.emitter.watcher_finished.emit(self.id)`
        in a `finally` block or at the end of their `run` method to ensure the GUI
        is notified of their completion, regardless of success or failure.
        """
        pass # This is a placeholder; actual implementation is in child classes.

# --- Process Watcher ---
class ProcessWatcher(StoppableThread):
    """
    A specialized watcher thread that launches an external subprocess and
    captures its standard output (stdout) and standard error (stderr) streams.
    It reports these streams as log events to the GUI.
    It includes mechanisms for graceful termination of the subprocess.
    """
    def __init__(self, cmd, emitter):
        """
        Initializes the ProcessWatcher.
        :param cmd: A list of strings representing the command to execute and
                    its arguments (e.g., `['python', 'myscript.py', 'arg1']`).
        :param emitter: The main GUI application instance (`SmartLoggerGUI`)
                        which provides the `new_event` and `watcher_finished` signals.
        """
        super().__init__(emitter)
        self.cmd = cmd
        # The target name displayed in the UI is the base name of the executable/script.
        self.target_name = os.path.basename(cmd[0])

    def _stream_reader(self, stream, level):
        """
        A helper method designed to run in a separate inner thread.
        It continuously reads lines from a given stream (e.g., subprocess.stdout or stderr).
        Each read line is then emitted as a `new_event` signal to the GUI.
        :param stream: The stream object (e.g., `proc.stdout` or `proc.stderr`).
        :param level: The log level ('INFO' for stdout, 'ERROR' for stderr) to
                      associate with messages coming from this stream.
        """
        # Continuously read lines until the stream is exhausted (EOF) or
        # until the parent `ProcessWatcher` thread is signaled to stop.
        for line in iter(stream.readline, ''):
            if not self._is_running: # Check the parent thread's stop flag.
                break
            line = line.strip() # Remove leading/trailing whitespace.
            if line: # Only process and emit non-empty lines.
                self.emitter.new_event.emit(level, line) # Emit to GUI.
                # Log to the internal Python logger (file and console).
                # `getattr(logger, level.lower())` dynamically calls `logger.info()` or `logger.error()`.
                (logger.error if level == 'ERROR' else logger.info)(line)
        stream.close() # Ensure the stream is closed after reading completes.

    def run(self):
        """
        The main execution method for the ProcessWatcher thread.
        It launches the target process and manages its output streams.
        It sets up two inner threads (`stdout_thread`, `stderr_thread`) to read
        the subprocess's output concurrently, which is crucial to prevent deadlocks
        if one stream fills its buffer while the other is waiting.
        Ensures `watcher_finished` signal is emitted upon the thread's completion.
        """
        proc = None # Initialize `proc` to None for `finally` block safety.
        try:
            logger.info(f"Starting process ({self.id}): {' '.join(self.cmd)}")
            # Start the subprocess.
            proc = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE, # Capture standard output.
                stderr=subprocess.PIPE, # Capture standard error.
                text=True,              # Decode output as text (UTF-8 by default).
                bufsize=1               # Use line-buffered output for real-time updates.
            )
            # Create and start separate threads for reading stdout and stderr.
            stdout_thread = threading.Thread(target=self._stream_reader, args=(proc.stdout, 'INFO'))
            stderr_thread = threading.Thread(target=self._stream_reader, args=(proc.stderr, 'ERROR'))
            stdout_thread.start()
            stderr_thread.start()

            # The main `ProcessWatcher` thread continues to run as long as:
            # 1. Its own `_is_running` flag is True (not stopped by the user).
            # 2. The subprocess is still running (`proc.poll()` returns None).
            while self._is_running and proc.poll() is None:
                time.sleep(0.1) # Small delay to avoid busy-waiting.

            # If the watcher was stopped by the user (i.e., `_is_running` became False).
            if not self._is_running:
                proc.terminate() # Request the subprocess to terminate gracefully (sends SIGTERM).
                logger.warning(f"Process terminated by user: {self.id}")
                self.emitter.new_event.emit('WARNING', f"Process '{self.target_name}' terminated by user.")

            proc.wait() # Wait for the subprocess to actually terminate.
            stdout_thread.join() # Wait for the stdout reader thread to finish.
            stderr_thread.join() # Wait for the stderr reader thread to finish.
            
            # Log the final exit code of the process.
            logger.info(f"Process {self.id} exited with code {proc.returncode}")
            self.emitter.new_event.emit('INFO', f"Process '{self.target_name}' exited with code: {proc.returncode}")
        except Exception as e:
            # Catch and log any exceptions that occur during process watching.
            logger.exception(f"Failed to watch process {self.id}")
            self.emitter.new_event.emit('ERROR', str(e))
        finally:
            # This block ensures cleanup happens regardless of whether an exception occurred.
            if proc and proc.poll() is None:
                # If the process is still running after `terminate()` and `wait()`, force kill it.
                proc.kill() # Sends SIGKILL, ungraceful but ensures termination.
                logger.warning(f"Process killed: {self.id}")
            # Emit the `watcher_finished` signal to notify the GUI that this watcher is done.
            self.emitter.watcher_finished.emit(self.id)


# --- File Watcher ---
class FileWatcher(StoppableThread):
    """
    A watcher thread that "tails" a specified log file.
    It continuously reads new lines appended to the file and reports them
    as log events to the GUI, providing real-time file monitoring.
    """
    def __init__(self, file_path, emitter):
        """
        Initializes the FileWatcher.
        :param file_path: The full path to the log file to be monitored.
        :param emitter: The main GUI application instance (`SmartLoggerGUI`).
        """
        super().__init__(emitter)
        self.file_path = file_path
        # The target name displayed in the UI is the base name of the file.
        self.target_name = os.path.basename(file_path)

    def run(self):
        """
        The main execution method for the FileWatcher thread.
        It opens the target file, seeks to its end, and then continuously polls
        for new lines. New lines are emitted as `INFO` events to the GUI.
        The loop continues until the thread is explicitly stopped.
        Ensures `watcher_finished` signal is emitted upon the thread's completion.
        """
        try:
            logger.info(f"Attaching to log file ({self.id}): {self.file_path}")
            # Open the file in read mode. `encoding='utf-8'` and `errors='ignore'`
            # help handle various text encodings without crashing.
            with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(0, os.SEEK_END) # Move the file pointer to the very end of the file.
                                     # This ensures we only read new content appended after attachment.
                while self._is_running: # Loop as long as the thread is active.
                    line = f.readline() # Read a single line from the file.
                    if line:
                        # If a new line is found, emit it as an INFO event to the GUI.
                        self.emitter.new_event.emit('INFO', line.strip())
                        logger.info(line.strip()) # Also log to the internal logger.
                    else:
                        # If no new line is found, wait a short period before checking again.
                        time.sleep(0.2) # Polling interval to reduce CPU usage.
        except Exception as e:
            # Catch and log any exceptions that occur during file watching.
            logger.exception(f"File watcher error on {self.id}")
            self.emitter.new_event.emit('ERROR', str(e))
        finally:
            # This block ensures cleanup messages and signals are sent.
            logger.info(f"Detached from log file ({self.id})")
            self.emitter.new_event.emit('INFO', f"Detached from file: {self.target_name}")
            # Emit the `watcher_finished` signal to notify the GUI that this watcher is done.
            self.emitter.watcher_finished.emit(self.id)


# --- Memory Watcher ---
class MemoryWatcher(StoppableThread):
    """
    A specialized watcher thread that leverages the Frida toolkit for dynamic
    instrumentation. It attaches to a specified process (by PID) and injects
    a JavaScript script to hook into a memory allocation function (e.g., `malloc`).
    It then reports messages from the Frida script as log events to the GUI.
    Requires `frida` module to be installed.
    """
    def __init__(self, pid, process_name, emitter):
        """
        Initializes the MemoryWatcher.
        :param pid: The Process ID (PID) of the target process to attach to.
        :param process_name: The name of the process (obtained from `psutil`),
                             used for display purposes in the GUI.
        :param emitter: The main GUI application instance (`SmartLoggerGUI`).
        """
        super().__init__(emitter)
        self.pid = pid
        self.target_name = f"{process_name.strip()} (PID: {pid})" # Formatted name for UI display.
        self.session = None # Frida session object, initialized in `run`.
        self.script = None  # Frida script object, initialized in `run`.

    def _on_frida_message(self, message, data):
        """
        Callback function for messages received from the injected Frida JavaScript script.
        Frida scripts can send messages back to the Python host using `send()`.
        This method processes those messages and emits them as new log events.
        :param message: A dictionary containing the message type and payload from Frida.
        :param data: Raw binary data if sent along with the message (not used here).
        """
        if message['type'] == 'send':
            # Regular messages from the Frida script are treated as INFO events.
            self.emitter.new_event.emit('INFO', f"[PID {self.pid}] Frida: {message['payload']}")
        elif message['type'] == 'error':
            # Errors reported by the Frida runtime are treated as ERROR events.
            self.emitter.new_event.emit('ERROR', f"[PID {self.pid}] Frida Error: {message['description']}")
            logger.error(f"Frida Error for PID {self.pid}: {message['description']}")

    def run(self):
        """
        The main execution method for the MemoryWatcher thread.
        It attempts to attach to the target process using Frida,
        creates and loads a JavaScript script (that hooks `malloc`),
        and then enters a polling loop to keep the thread alive while Frida monitors.
        Ensures Frida session and script are unloaded/detached gracefully.
        Emits `watcher_finished` signal upon completion.
        """
        logger.info(f"Attaching Memory Watcher to PID {self.pid}")
        try:
            # Attach to the target process. This requires appropriate permissions.
            self.session = frida.attach(self.pid)
            
            # The JavaScript code to be injected into the target process.
            # It finds the `malloc` export and attaches an interceptor.
            # `onEnter` is called before `malloc` executes, `onLeave` after.
            # `send()` is used to communicate back to the Python host.
            script_code = r"""
            var malloc_ptr = Module.findExportByName(null, 'malloc');
            if (malloc_ptr) {
                Interceptor.attach(malloc_ptr, {
                    onEnter: function(args) { 
                        // Store the size argument to malloc for use in onLeave
                        this.size = args[0].toUInt32(); 
                    },
                    onLeave: function(retval) { 
                        // Report the malloc size and the address returned by malloc
                        send('malloc(' + this.size + ') -> ' + retval); 
                    }
                });
                send("Successfully hooked malloc.");
            } else { 
                // If malloc cannot be found (e.g., different OS/runtime), report a warning.
                send("Warning: Could not find malloc export. Memory monitoring might not work as expected."); 
            }
            """
            self.script = self.session.create_script(script_code)
            # Connect the Frida script's `send()` messages to the Python callback.
            self.script.on('message', self._on_frida_message)
            self.script.load() # Load (inject and execute) the script into the target process.
            self.emitter.new_event.emit('INFO', f"Memory watcher attached to PID {self.pid}")

            # Keep the Python thread running as long as the watcher is active.
            # The actual monitoring is done by the Frida script within the target process.
            while self._is_running:
                time.sleep(1) # Polling to prevent busy-waiting and allow stop signal processing.
            
        except frida.ProcessNotFoundError:
            # Handle cases where the process does not exist or access is denied.
            msg = f"MemoryWatcher: Process with PID {self.pid} not found or has exited. Access denied?"
            self.emitter.new_event.emit('ERROR', msg)
            logger.error(msg)
        except Exception as e:
            # Catch and log any other exceptions during Frida operations.
            logger.exception(f"MemoryWatcher error on {self.id}: {e}")
            self.emitter.new_event.emit('ERROR', str(e))
        finally:
            # This block ensures graceful cleanup of Frida resources.
            if self.script:
                try:
                    self.script.unload() # Unload the injected Frida script from the target process.
                    logger.info(f"Frida script unloaded from PID {self.pid}")
                except Exception as e:
                    logger.warning(f"Error unloading Frida script from PID {self.pid}: {e}")
            if self.session:
                try:
                    self.session.detach() # Detach the Frida session from the target process.
                    logger.info(f"Frida session detached from PID {self.pid}")
                except Exception as e:
                    logger.warning(f"Error detaching Frida session from PID {self.pid}: {e}")
            
            # Emit the `watcher_finished` signal to notify the GUI that this watcher is done.
            self.emitter.new_event.emit('INFO', f"Memory watcher detached from PID {self.pid}")
            self.emitter.watcher_finished.emit(self.id)


# --- Main GUI ---
class SmartLoggerGUI(QWidget):
    """
    The main GUI application window for the Smart Logger.
    It provides a centralized interface for launching processes, attaching to
    log files, monitoring process memory, adding manual events, and viewing
    all collected log entries. It includes features like:
    - Launching and monitoring external processes.
    - Tailing and displaying content from log files.
    - Attaching to processes to monitor memory allocations (via Frida/psutil).
    - A dedicated panel for managing active watcher threads with individual stop buttons.
    - Real-time log filtering for quick searching.
    - Saving visible log entries to a file.
    - Color-coded log levels for easy visual identification of event severity.
    """
    # Signals for thread-safe communication from background watcher threads to the GUI's main thread.
    new_event = pyqtSignal(str, str)      # Emits (level: str, message: str) for a new log entry.
    watcher_finished = pyqtSignal(int)    # Emits (watcher_id: int) when a watcher thread completes its task.

    def __init__(self):
        """
        Initializes the SmartLoggerGUI window and its components.
        Sets up the internal data structures and connects signals/slots.
        """
        super().__init__()
        # Dictionary to store active watcher threads, keyed by their unique ID.
        # This allows easy lookup and management of running watchers.
        self.watchers = {}
        self.init_ui()      # Call method to build the user interface.
        # Connect the signals emitted by watcher threads (and self) to the GUI's slots.
        self.new_event.connect(self.add_event)
        self.watcher_finished.connect(self.remove_watcher)

    def init_ui(self):
        """
        Initializes the window layout and all UI components.
        This method sets up buttons, tables, input fields, and arranges them
        using PyQt's layout managers (`QVBoxLayout`, `QHBoxLayout`, `QSplitter`).
        """
        self.setWindowTitle('Smart Logger GUI v1.4') # Set window title with current version.
        self.resize(1200, 800) # Set initial window size for optimal layout viewing.

        # --- Apply Dark Theme and Neon Accents ---
        # This extensive stylesheet covers various PyQt widgets for a consistent dark theme.
        # Neon green and pink are used for accents (selection, hover, borders).
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b; /* Dark background for all widgets */
                color: #e0e0e0; /* Light grey text for readability */
                font-family: Arial, sans-serif;
                font-size: 10pt;
            }
            QPushButton {
                background-color: #404040; /* Slightly lighter dark for buttons */
                border: 1px solid #505050;
                border-radius: 5px;
                padding: 8px 15px;
                color: #00ff00; /* Neon green text */
                outline: none; /* Remove focus outline */
            }
            QPushButton:hover {
                background-color: #505050;
                border: 1px solid #00ff00; /* Neon green border on hover */
            }
            QPushButton:pressed {
                background-color: #606060;
                border: 1px solid #ff00ff; /* Neon pink border on press */
            }
            QPushButton:disabled {
                background-color: #353535;
                border: 1px solid #454545;
                color: #707070;
            }
            QLineEdit {
                background-color: #3a3a3a; /* Darker input field background */
                border: 1px solid #555555;
                border-radius: 5px;
                padding: 5px;
                color: #00ffff; /* Neon cyan text for input */
                selection-background-color: #ff00ff; /* Neon pink selection */
            }
            QTableWidget {
                background-color: #1e1e1e; /* Very dark background for tables */
                alternate-background-color: #2a2a2a; /* Slightly lighter alternate row */
                gridline-color: #404040;
                color: #e0e0e0;
                border: 1px solid #404040;
                selection-background-color: #005500; /* Darker green for selection */
                selection-color: #ffffff; /* White text on selection */
            }
            QTableWidget::item {
                padding: 3px; /* Add some padding to table items */
            }
            QHeaderView::section {
                background-color: #333333; /* Dark header background */
                color: #00ff00; /* Neon green header text */
                padding: 5px;
                border: 1px solid #444444;
                border-bottom: 2px solid #00ff00; /* Neon green bottom border */
                font-weight: bold;
            }
            QTableCornerButton::section {
                background-color: #333333;
                border: 1px solid #444444;
            }
            QSplitter::handle {
                background-color: #555555;
                border: 1px solid #333333;
                height: 5px; /* Adjust handle size for vertical splitter */
                width: 5px; /* Adjust handle size for horizontal splitter */
            }
            QSplitter::handle:hover {
                background-color: #00ffff; /* Neon cyan on hover */
            }
            QGroupBox {
                border: 1px solid #00ff00; /* Neon green border for group box */
                border-radius: 5px;
                margin-top: 1ex; /* Space for the title */
                font-weight: bold;
                color: #00ff00; /* Neon green title text */
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center; /* Title centered */
                padding: 0 3px;
                background-color: #2b2b2b; /* Match widget background */
            }
            QInputDialog {
                background-color: #2b2b2b;
                color: #e0e0e0;
            }
            QInputDialog QLabel {
                color: #00ffff; /* Neon cyan for dialog labels */
            }
            QComboBox {
                background-color: #3a3a3a;
                border: 1px solid #555555;
                border-radius: 5px;
                padding: 1px 18px 1px 3px;
                color: #00ffff;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 15px;
                border-left-width: 1px;
                border-left-color: #555555;
                border-left-style: solid;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
            }
            QComboBox::down-arrow {
                image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAcAAAAECAYAAADg/jdNAAAAIklEQVQImWNkYGD4z0ABYBqMDKT6YDBgYmYmYmABAH/2D4c2wO0nAAAAAElFTkSuQmCC); /* Small white arrow */
            }
        """)


        # Main layout for the entire window, arranged vertically.
        main_layout = QVBoxLayout()
        # Horizontal layout for the top row of control buttons and filter input.
        top_layout = QHBoxLayout()
        # Horizontal layout for the bottom row of action buttons (Save, Clear).
        bottom_layout = QHBoxLayout()

        # --- Top Controls ---
        # Buttons for launching/attaching watchers and adding manual events.
        top_layout.addWidget(QPushButton('Launch Process', clicked=self.launch_process))
        top_layout.addWidget(QPushButton('Attach Log File', clicked=self.attach_file))
        
        # 'Attach Memory Watcher' button for Frida integration.
        self.attach_mem_btn = QPushButton('Attach Memory Watcher', clicked=self.select_and_attach_process)
        # Disable the button if Frida is not installed and provide a tooltip.
        if not frida:
            self.attach_mem_btn.setEnabled(False)
            self.attach_mem_btn.setToolTip("Frida not installed. Run 'pip install frida-tools' to enable this feature.")
        top_layout.addWidget(self.attach_mem_btn)

        top_layout.addWidget(QPushButton('Add Manual Event', clicked=self.add_manual_event))
        top_layout.addStretch() # Pushes subsequent widgets (filter input) to the right.
        
        # Text input field for filtering log messages.
        self.filter_input = QLineEdit(placeholderText="Filter logs (e.g., 'error', 'network')")
        self.filter_input.textChanged.connect(self.filter_logs) # Connect text changes to the filter method.
        top_layout.addWidget(self.filter_input)

        # --- Log Table ---
        # The main table to display all collected log events.
        self.table = QTableWidget(0, 3) # Initialize with 0 rows and 3 columns.
        self.table.setHorizontalHeaderLabels(['Timestamp', 'Level', 'Message']) # Set column headers.
        # Configure how table columns resize:
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents) # Timestamp column auto-resizes to fit content.
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents) # Level column auto-resizes to fit content.
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)       # Message column stretches to fill remaining space.
        self.table.setSortingEnabled(True) # Allow users to sort rows by clicking column headers.

        # --- Watcher Management Panel ---
        # Group box to visually organize the active watchers table.
        watcher_group = QGroupBox("Active Watchers")
        watcher_layout = QVBoxLayout()
        # Table to list currently active watcher threads.
        self.watchers_table = QTableWidget(0, 3) # 0 rows, 3 columns.
        self.watchers_table.setHorizontalHeaderLabels(['Type', 'Target', 'Action']) # Set column headers.
        # Configure watcher table column resizing:
        self.watchers_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents) # Type column auto-resizes.
        self.watchers_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)       # Target column stretches.
        self.watchers_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents) # Action column auto-resizes for the 'Stop' button.
        watcher_layout.addWidget(self.watchers_table)
        watcher_group.setLayout(watcher_layout)

        # --- Splitter for resizable panels ---
        # A `QSplitter` allows users to dynamically resize the height of the log table
        # and the watcher management panel.
        splitter = QSplitter(Qt.Vertical) # Vertical splitter allows top/bottom resizing.
        splitter.addWidget(self.table)       # Add the main log table to the top part.
        splitter.addWidget(watcher_group)    # Add the active watchers panel to the bottom part.
        splitter.setSizes([600, 200]) # Set initial height ratio (log table larger).

        # --- Bottom Buttons ---
        bottom_layout.addWidget(QPushButton('Save Log', clicked=self.save_log))
        bottom_layout.addStretch() # Pushes the 'Clear Log' button to the right.
        bottom_layout.addWidget(QPushButton('Clear Log', clicked=self.clear_log))
        
        # --- Assemble Main Layout ---
        main_layout.addLayout(top_layout)
        main_layout.addWidget(splitter) # Add the splitter (containing both tables) to the main layout.
        main_layout.addLayout(bottom_layout)
        self.setLayout(main_layout) # Apply the main layout to the GUI window.

    def add_watcher(self, watcher):
        """
        Adds a new watcher thread to the internal `self.watchers` dictionary,
        starts the thread, and updates the `Active Watchers` UI table.
        :param watcher: An instance of a `StoppableThread` subclass
                        (`ProcessWatcher`, `FileWatcher`, or `MemoryWatcher`).
        """
        self.watchers[watcher.id] = watcher # Store the watcher instance, keyed by its unique ID.
        watcher.start() # Start the background thread.

        row = self.watchers_table.rowCount() # Get the current number of rows in the watchers table.
        self.watchers_table.insertRow(row) # Insert a new row at the end of the table.
        
        # Determine the type of watcher for display in the 'Type' column.
        if isinstance(watcher, ProcessWatcher): watcher_type = "Process"
        elif isinstance(watcher, FileWatcher): watcher_type = "File"
        elif isinstance(watcher, MemoryWatcher): watcher_type = "Memory"
        else: watcher_type = "Unknown" # Fallback for unexpected watcher types.

        # Set items for 'Type' and 'Target' columns.
        self.watchers_table.setItem(row, 0, QTableWidgetItem(watcher_type))
        self.watchers_table.setItem(row, 1, QTableWidgetItem(watcher.target_name))
        
        # Create and add a 'Stop' button for this specific watcher.
        stop_btn = QPushButton("Stop")
        # Connect the button's clicked signal to `stop_watcher`, passing the watcher's unique ID.
        # The `lambda` function is used here to capture `watcher.id` at the time of connection.
        stop_btn.clicked.connect(lambda: self.stop_watcher(watcher.id))
        self.watchers_table.setCellWidget(row, 2, stop_btn) # Place the button in the 'Action' column.
        
        # Store the unique ID in the first item of the row using `Qt.UserRole`.
        # This allows easy retrieval of the watcher ID when a row needs to be removed.
        self.watchers_table.item(row, 0).setData(Qt.UserRole, watcher.id)
        logger.info(f"Watcher added: Type={watcher_type}, Target={watcher.target_name}, ID={watcher.id}")

    def stop_watcher(self, watcher_id):
        """
        Sends a stop signal to a specific watcher thread identified by its unique ID.
        This method is typically called when the 'Stop' button next to a watcher
        in the `Active Watchers` table is clicked.
        :param watcher_id: The unique ID of the watcher thread to stop.
        """
        if watcher_id in self.watchers:
            self.watchers[watcher_id].stop() # Call the `stop()` method of the thread instance.
            logger.info(f"Stop signal sent to watcher ID: {watcher_id}")
        else:
            logger.warning(f"Attempted to stop non-existent or already stopped watcher ID: {watcher_id}")

    def remove_watcher(self, watcher_id):
        """
        Removes a finished watcher from the `Active Watchers` UI table and
        from the internal `self.watchers` dictionary.
        This method is a slot connected to the `watcher_finished` signal emitted
        by watcher threads upon their completion.
        :param watcher_id: The unique ID of the watcher thread that has finished.
        """
        # Iterate through the `watchers_table` to find the row corresponding to the `watcher_id`.
        for row in range(self.watchers_table.rowCount()):
            item = self.watchers_table.item(row, 0)
            # Check if the item exists and its stored UserRole data matches the watcher_id.
            if item and item.data(Qt.UserRole) == watcher_id:
                self.watchers_table.removeRow(row) # Remove the entire row from the table.
                break # Exit the loop once the row is found and removed.
        
        # Remove the watcher instance from the internal dictionary.
        if watcher_id in self.watchers:
            del self.watchers[watcher_id]
            logger.info(f"Watcher ID {watcher_id} removed from GUI and management.")

    def launch_process(self):
        """
        Handles the 'Launch Process' button action.
        It prompts the user to select an executable/script and optional arguments,
        then creates and adds a `ProcessWatcher` to monitor its output.
        """
        exe_path, _ = QFileDialog.getOpenFileName(
            self, 'Select Executable or Script', '', 'All Files (*);;Executables (*.exe);;Scripts (*.py *.sh)'
        )
        if not exe_path: return # If the user cancels the file dialog, return.

        args_text, ok = QInputDialog.getText(
            self, 'Process Arguments', 'Enter arguments (space-separated):', QLineEdit.Normal, ''
        )
        if not ok: return # If the user cancels the input dialog, return.
        
        # Construct the command list. Handle empty arguments gracefully.
        cmd = [exe_path] + (args_text.split() if args_text else [])
        watcher = ProcessWatcher(cmd, self) # Create a new ProcessWatcher instance.
        self.add_watcher(watcher) # Add and start the watcher.

    def attach_file(self):
        """
        Handles the 'Attach Log File' button action.
        It prompts the user to select a log file, then creates and adds a
        `FileWatcher` to tail and display new lines from that file.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Select Log File to Tail', '', 'Log Files (*.log *.txt);;All Files (*)'
        )
        if not file_path: return # If the user cancels the file dialog, return.

        watcher = FileWatcher(file_path, self) # Create a new FileWatcher instance.
        self.add_watcher(watcher) # Add and start the watcher.

    def select_and_attach_process(self):
        """
        Handles the 'Attach Memory Watcher' button action.
        It first retrieves a list of all running processes using `psutil`,
        presents them to the user in a selection dialog, and then creates and
        adds a `MemoryWatcher` (using Frida) to the selected process.
        This feature is only available if Frida is installed.
        """
        if not frida:
            # This case should ideally be prevented by disabling the button,
            # but serves as a runtime safeguard.
            self.new_event.emit('ERROR', "Frida is not installed. Cannot attach to process memory.")
            return

        processes = [] # List to store process strings (PID: Name).
        # Iterate through all running processes using psutil.
        for p in psutil.process_iter(['pid', 'name']):
            try:
                processes.append(f"{p.info['pid']}: {p.info['name']}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Ignore processes that might have exited or where access is denied.
                continue
        
        if not processes:
            self.new_event.emit('WARNING', "No accessible processes found to attach to.")
            return

        # Prompt the user to select a process from the list.
        process_str, ok = QInputDialog.getItem(
            self, "Select Process", "Choose a process to attach to (memory watcher):", processes, 0, False
        )
        if not ok or not process_str: return # If user cancels or no selection.

        # Parse the selected string to extract PID and process name.
        pid_str, process_name = process_str.split(':', 1)
        pid = int(pid_str.strip())
        
        watcher = MemoryWatcher(pid, process_name.strip(), self) # Create a new MemoryWatcher instance.
        self.add_watcher(watcher) # Add and start the watcher.

    def add_manual_event(self):
        """
        Handles the 'Add Manual Event' button action.
        It prompts the user to enter a custom message via an input dialog,
        and then adds this message as an `INFO` level event to the log table.
        """
        text, ok = QInputDialog.getText(
            self, 'Manual Event', 'Enter event message:', QLineEdit.Normal, ''
        )
        if ok and text: # If user clicked OK and entered text.
            self.new_event.emit('INFO', text) # Emit the manual event to the log table.
            logger.info(text) # Also log to the internal Python logger.

    def clear_log(self):
        """
        Clears all entries (rows) from the main log table.
        This effectively empties the displayed log.
        """
        self.table.setRowCount(0) # Set the row count to 0 to remove all rows.
        self.new_event.emit('INFO', "Log table cleared by user.")

    def filter_logs(self, text):
        """
        Filters the entries in the main log table based on the provided text.
        Rows that do not contain the filter text (case-insensitive) in their
        'Level' or 'Message' columns are hidden.
        :param text: The filter string entered by the user in the search bar.
        """
        search_text = text.lower() # Convert search text to lowercase for case-insensitive matching.
        for row in range(self.table.rowCount()):
            # Retrieve the QTableWidgetItem for 'Level' (column 1) and 'Message' (column 2).
            level_item = self.table.item(row, 1)
            msg_item = self.table.item(row, 2)
            
            # Check if the search text is present in either the level or message text.
            level_match = level_item and search_text in level_item.text().lower()
            msg_match = msg_item and search_text in msg_item.text().lower()
            
            # Hide the row if the text is not found in either column, otherwise show it.
            self.table.setRowHidden(row, not (level_match or msg_match))
    
    def save_log(self):
        """
        Saves the currently visible log entries from the main table to a text file.
        The user is prompted to select a file path. Hidden rows (due to filtering)
        are excluded from the saved log. The output is a simple CSV-like format.
        """
        # Open a file dialog to get a save file path from the user.
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Log", "", "Text Files (*.txt);;CSV Files (*.csv);;All Files (*)"
        )
        if not path: return # If the user cancels the dialog, return.

        try:
            with open(path, 'w', encoding='utf-8') as f:
                # Write a header row for clarity in the saved file.
                f.write("Timestamp,Level,Message\n")
                for row in range(self.table.rowCount()):
                    if self.table.isRowHidden(row): # Skip rows that are currently hidden by the filter.
                        continue
                    
                    # Retrieve text from each column of the visible row.
                    ts = self.table.item(row, 0).text()
                    level = self.table.item(row, 1).text()
                    # Enclose message in double quotes to handle commas within messages for CSV compatibility.
                    msg = self.table.item(row, 2).text()
                    f.write(f"[{ts}],{level},\"{msg}\"\n") # Write formatted log entry.
            self.new_event.emit("INFO", f"Log successfully saved to: {path}")
        except Exception as e:
            self.new_event.emit("ERROR", f"Failed to save log: {e}")

    def add_event(self, level, message):
        """
        Slot method to add a new log event as a row to the main log table.
        This method is connected to the `new_event` signal and runs on the GUI thread,
        ensuring thread-safe updates to the UI. It also applies color-coding
        based on the log level.
        :param level: The log level (e.g., 'INFO', 'WARNING', 'ERROR').
        :param message: The log message content.
        """
        # Temporarily disable table sorting during row insertion for better performance
        # and to prevent unexpected reordering while new data is being added.
        self.table.setSortingEnabled(False)
        row = self.table.rowCount() # Get the current number of rows.
        self.table.insertRow(row) # Insert a new empty row at the end.

        # Create QTableWidgetItem instances for each column.
        ts_item = QTableWidgetItem(time.strftime('%Y-%m-%d %H:%M:%S')) # Current timestamp.
        level_item = QTableWidgetItem(level)
        msg_item = QTableWidgetItem(message)
        
        # Define a map for color-coding based on log level.
        # Adjusted colors for better contrast on a dark theme.
        color_map = {
            'ERROR': QColor(255, 60, 60, 150),   # Brighter red with higher opacity
            'WARNING': QColor(255, 200, 0, 150), # Orange/yellow for warning
            'INFO': QColor(80, 255, 80, 50),     # Subtle neon green for info rows
            'DEBUG': QColor(100, 100, 255, 50)   # Subtle blue for debug rows
        }
        color = color_map.get(level) # Get the color for the current level, or None if not defined.
        if color:
            # Apply the determined background color to all items in the new row.
            for item in (ts_item, level_item, msg_item):
                item.setBackground(color)
                # Ensure text is readable on colored backgrounds, potentially setting text color explicitly
                item.setForeground(QColor("#e0e0e0")) # Ensure light text color for readability

        # Set the created items into the new row in the table.
        self.table.setItem(row, 0, ts_item)
        self.table.setItem(row, 1, level_item)
        self.table.setItem(row, 2, msg_item)
        
        self.table.setSortingEnabled(True) # Re-enable sorting after insertion.
        self.table.scrollToBottom() # Scroll the table view to show the newest entry.

    def closeEvent(self, event):
        """
        Overrides the default `closeEvent` handler for the main application window.
        This method ensures that all active background watcher threads are gracefully
        stopped when the GUI window is closed, preventing orphaned processes or threads.
        :param event: The `QCloseEvent` object generated when the window is closed.
        """
        logger.info("Application closing. Stopping all active watchers...")
        # Iterate over a copy of the `watchers` dictionary's values.
        # This is important because `self.watchers` will be modified (items deleted)
        # as `remove_watcher` is called when threads emit `watcher_finished`.
        for watcher in list(self.watchers.values()):
            watcher.stop() # Send the stop signal to each running watcher thread.
        
        # A small delay to allow daemon threads to process the stop signal.
        # While daemon threads will eventually exit with the main process,
        # this helps ensure they attempt a clean shutdown first.
        time.sleep(0.5)
        event.accept() # Accept the close event, allowing the window to close.

# --- Application Entry Point ---
if __name__ == '__main__':
    # This block executes when the script is run directly.
    app = QApplication(sys.argv) # Create a QApplication instance. This is essential for any PyQt5 GUI.
    win = SmartLoggerGUI()       # Create an instance of the SmartLoggerGUI window.
    win.show()                   # Display the GUI window.
    sys.exit(app.exec_())        # Start the Qt event loop. This line transfers control to Qt,
                                 # which handles events (like button clicks, window resizing)
                                 # and keeps the application running until `app.quit()` is called
                                 # or the window is closed. `sys.exit()` ensures a clean exit.
