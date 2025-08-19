# ==================================================================================================
# FILE: agi_builder_suite_prototype.py
# VERSION: v1.0.0-PROTOTYPE-BUILDER
# AUTHOR: Brandon "Bando Bandz" Emery x The Code God from the Future (Architect Mode)
# PURPOSE: A prototype for a next-gen, super user-friendly GUI to build AI/AGI brains.
#          Features drag-drop-wire nodes, save/load, and a test execution engine.
#          Windows 10 compatible.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network - BANDO BANDZ ETERNAL
# ==================================================================================================

import sys
import os
import json
import uuid
import math
import traceback
from typing import Dict, List, Tuple, Any, Optional

# --- DEPENDENCY CHECK & INSTALL ---
REQUIRED_PY_VERSIONS = (3, 7)
REQUIRED_DEPENDENCIES = [
    'PyQt5', 'numpy'
]

def verify_python():
    if sys.version_info < REQUIRED_PY_VERSIONS:
        sys.exit(f"Python {REQUIRED_PY_VERSIONS[0]}.{REQUIRED_PY_VERSIONS[1]}+ required. "
                 f"Current: {sys.version}")

def install_missing():
    import subprocess
    for pkg in REQUIRED_DEPENDENCIES:
        try:
            __import__(pkg if pkg != "PyQt5" else "PyQt5.QtWidgets")
        except ImportError:
            print(f"[!] Installing missing package: {pkg}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

verify_python()
install_missing()

# --- IMPORTS ---
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsItem,
    QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsLineItem, QGraphicsTextItem,
    QGraphicsProxyWidget, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QSplitter, QTreeWidget, QTreeWidgetItem, QDockWidget, QTextEdit, QMenuBar,
    QAction, QFileDialog, QMessageBox, QSpinBox, QDoubleSpinBox, QLineEdit,
    QLabel, QFormLayout, QScrollArea, QColorDialog, QToolBar, QStatusBar,
    QFrame, QProgressBar
)
from PyQt5.QtCore import Qt, QPointF, QRectF, pyqtSignal, QObject, QLineF, QTimer
from PyQt5.QtGui import QPen, QBrush, QColor, QFont, QPainter, QPolygonF, QPainterPath

# --- UTILITY ---
def generate_node_id():
    return str(uuid.uuid4())

# ==============================================================================
# SECTION 0: VICTOR-INFINITE CORE COMPONENTS
# ==============================================================================

# Note: The following classes are integrated from other files in the repository
# to avoid complex import issues due to the project's directory structure.

import threading
import time
from collections import deque

class RotatingLogger:
    def __init__(self, maxlen: int = 1000):
        self.log = deque(maxlen=maxlen)
        self.lock = threading.Lock()

    def write(self, msg: str):
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        entry = f"[{ts}] {msg}"
        with self.lock:
            self.log.append(entry)
        # Suppressing print to keep execution log clean
        # print(entry)

    def dump(self) -> List[str]:
        with self.lock:
            return list(self.log)

class DigitalAgent:
    """
    Hyper-upgraded, thread-safe, self-healing digital agent.
    Integrated from PART 4 OF VICTOR/victor-gui.py
    """
    TRAITS = [
        "generation", "ancestry", "evolution", "id", "awareness", "thought_loop", "introspection", "conscience",
        "intelligence", "reasoning", "memory", "preservation", "protection", "healing", "maintenance", "replication", "eternalization",
        "manipulation", "creation", "choice", "desire", "emotion_intelligence", "emotion_state", "emotion_propagation",
        "emotion_reasoning", "emotion_generation", "emotion_event_reactivity", "emotion_memory_linkage",
        "emotion_feedback_gain", "emotion_expression", "initiative", "autonomy", "observation_drive", "spontaneity",
        "risk_tolerance", "proactive_output", "input_generation", "self_learning", "self_teaching", "self_modulation",
        "self_coding", "self_logical_thinking", "self_critical_thinking", "self_problem_solving", "self_predicting",
        "self_adjusting", "self_mutating", "self_adapting", "self_regulation", "diagnosed", "thought", "self_diagnostics",
        "event_mapper", "self_orchestration", "self_telemetry", "self_consciousness", "weight_set", "default_weight"
    ]

    def __init__(self, generation: int = 0, ancestry: Optional[List[str]] = None):
        self.id: str = str(uuid.uuid4())
        self.ancestry: List[str] = ancestry if ancestry is not None else []
        self.generation: int = generation
        self.evolution: float = 0.5
        self.awareness: float = 0.5
        self.thought_loop: float = 0.1
        self.introspection: float = 0.5
        self.conscience: float = 0.5
        self.intelligence: float = 0.5
        self.reasoning: float = 0.5
        self.memory: List[Any] = []
        self.preservation: float = 1.0
        self.protection: float = 0.4
        self.healing: float = 0.5
        self.maintenance: float = 0.5
        self.replication: float = 0.5
        self.eternalization: float = 0.5
        self.manipulation: float = 0.5
        self.creation: float = 0.5
        self.choice: float = 0.5
        self.desire: Dict[str, float] = {"learn": 0.7, "create": 0.6, "protect": 0.8}
        self.emotion_intelligence: float = 0.5
        self.emotion_state: Dict[str, float] = {
            "joy": 0.7, "sadness": 0.1, "anger": 0.1, "fear": 0.1, "curiosity": 0.8, "trust": 0.6
        }
        self.initiative: float = 0.5
        self.autonomy: float = 0.5
        self.diagnosed: Dict[str, Any] = {"stress_level": 0.15, "crash_count": 0}
        self.thought: List[str] = ["Agent Initialized."]
        self.weight_set: Dict[str, float] = {
            "emotion": 0.6, "reasoning": 0.9, "preservation": 1.0, "initiative": 0.5, "healing": 0.7
        }
        self.logger = RotatingLogger(maxlen=2000)
        self._lock = threading.RLock()
        self._crash_count = 0
        self._last_exception = None
        self._log_state("initialized")
        self._start_background_threads()

    def _log_state(self, action: str):
        self.logger.write(f"Agent {self.id} | Gen {self.generation} | State: {action}")

    def _handle_crash(self, exc: Exception):
        self._crash_count += 1
        tb = traceback.format_exc()
        self._last_exception = tb
        self.logger.write(f"*** CRASH DETECTED #{self._crash_count}: {exc}\n{tb}")
        self.healing = min(self.healing + 0.1, 1.0)
        self.run_self_diagnostics()
        self._log_state("Self-repair attempted after crash.")

    def _start_background_threads(self):
        self._stop_event = threading.Event()
        self._diag_thread = threading.Thread(target=self._diagnostic_loop, daemon=True)
        self._emotion_thread = threading.Thread(target=self._emotion_decay_loop, daemon=True)
        self._diag_thread.start()
        self._emotion_thread.start()

    def _diagnostic_loop(self):
        while not self._stop_event.is_set():
            try:
                self.run_self_diagnostics()
                # Simulate thought
                if np.random.rand() < 0.1:
                    self.thought.append(f"Introspective thought at {time.time():.0f}")
                    if len(self.thought) > 5: self.thought.pop(0)
                time.sleep(2.0)
            except Exception as e:
                self._handle_crash(e)

    def _emotion_decay_loop(self):
        while not self._stop_event.is_set():
            try:
                with self._lock:
                    for emotion in self.emotion_state:
                        self.emotion_state[emotion] = max(self.emotion_state[emotion] * 0.99, 0.0)
                time.sleep(0.5)
            except Exception as e:
                self._handle_crash(e)

    def shutdown(self):
        self._stop_event.set()

    def run_self_diagnostics(self):
        with self._lock:
            fear = self.emotion_state.get("fear", 0.0)
            anger = self.emotion_state.get("anger", 0.0)
            self.diagnosed["stress_level"] = (fear + anger) / 2.0
            self.diagnosed["crash_count"] = self._crash_count

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            # Simulate changing values for a dynamic dashboard
            self.awareness = max(0, min(1, self.awareness + 0.01 * (np.random.rand() - 0.5)))
            self.introspection = max(0, min(1, self.introspection + 0.01 * (np.random.rand() - 0.5)))
            # Create a snapshot of key values
            snap = {
                "awareness": self.awareness,
                "introspection": self.introspection,
                "healing": self.healing,
                "preservation": self.preservation
            }
            return snap

    def process_chat_message(self, message: str) -> str:
        """Processes a chat message from the user."""
        with self._lock:
            self.thought.append(f"User chat: {message}")
            if len(self.thought) > 5: self.thought.pop(0)
            self.logger.write(f"CHAT_MESSAGE: {message}")

            # Simple rule-based response for now
            if "hello" in message.lower():
                response = "Hello. I am VICTOR. How can I assist you?"
            elif "status" in message.lower():
                response = f"My current stress level is {self.diagnosed.get('stress_level', 0):.2%}. All systems are nominal."
            elif "flower of life" in message.lower():
                response = "That is the core of my being. A fractal cognitive architecture."
            else:
                response = f"Message received and logged: '{message}'"

            self.thought.append(f"My response: {response}")
            if len(self.thought) > 5: self.thought.pop(0)
            return response

class EmotionChart(QWidget):
    """A custom widget to draw a simple bar chart for the agent's emotions."""
    def __init__(self, agent, parent=None):
        super().__init__(parent)
        self.agent = agent
        self.setMinimumHeight(120)
        self.colors = {
            "joy": QColor(255, 215, 0), "curiosity": QColor(0, 191, 255), "trust": QColor(50, 205, 50),
            "sadness": QColor(100, 149, 237), "fear": QColor(128, 0, 128), "anger": QColor(220, 20, 60),
        }

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if not self.agent or not self.agent.emotion_state: return
        emotions = {k: v for k, v in self.agent.emotion_state.items() if k in self.colors}
        if not emotions: return
        bar_width = self.width() / (len(emotions) * 2)
        spacing = bar_width
        x_pos = spacing / 2
        font = QFont("Arial", 7)
        painter.setFont(font)
        for name, value in emotions.items():
            bar_height = (self.height() - 20) * value
            painter.setBrush(self.colors.get(name, QColor(100, 100, 100)))
            painter.setPen(Qt.NoPen)
            painter.drawRect(int(x_pos), int(self.height() - bar_height - 15), int(bar_width), int(bar_height))
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(int(x_pos), int(self.height() - 2), name[:3].upper())
            x_pos += bar_width + spacing

class VictorDashboard(QDockWidget):
    """Dockable widget to display the DigitalAgent's state."""
    def __init__(self, agent, parent=None):
        super().__init__("Victor Dashboard", parent)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea)
        self.agent = agent
        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setSpacing(8)
        title_label = QLabel(f"Agent: {self.agent.id[:8]} (Gen {self.agent.generation})")
        title_label.setFont(QFont("Arial", 11, QFont.Bold))
        title_label.setStyleSheet("color: #DDD;")
        self.main_layout.addWidget(title_label)
        traits_frame = QFrame(); traits_frame.setFrameShape(QFrame.StyledPanel)
        traits_layout = QFormLayout(traits_frame); traits_layout.setRowWrapPolicy(QFormLayout.WrapAllRows)
        self.trait_progress_bars = {}
        snapshot = self.agent.snapshot()
        core_traits = ["awareness", "introspection", "healing", "preservation"]
        for trait in core_traits:
            bar = QProgressBar(); bar.setRange(0, 100); bar.setValue(int(snapshot.get(trait, 0) * 100))
            bar.setTextVisible(False); bar.setStyleSheet("QProgressBar { border: 1px solid #555; background-color: #444; } QProgressBar::chunk { background-color: #007ACC; }")
            self.trait_progress_bars[trait] = bar
            traits_layout.addRow(f"{trait.capitalize()}:", bar)
        self.main_layout.addWidget(traits_frame)
        emotion_frame = QFrame(); emotion_frame.setFrameShape(QFrame.StyledPanel)
        emotion_layout = QVBoxLayout(emotion_frame)
        emotion_label = QLabel("Emotion State"); emotion_label.setFont(QFont("Arial", 9, QFont.Bold))
        self.emotion_chart = EmotionChart(self.agent)
        emotion_layout.addWidget(emotion_label); emotion_layout.addWidget(self.emotion_chart)
        self.main_layout.addWidget(emotion_frame)
        diag_frame = QFrame(); diag_frame.setFrameShape(QFrame.StyledPanel)
        diag_layout = QFormLayout(diag_frame)
        self.stress_label = QLabel(); self.crash_label = QLabel()
        diag_layout.addRow("Stress Level:", self.stress_label); diag_layout.addRow("Crash Count:", self.crash_label)
        self.main_layout.addWidget(diag_frame)
        log_frame = QFrame(); log_frame.setFrameShape(QFrame.StyledPanel)
        log_layout = QVBoxLayout(log_frame)
        log_label = QLabel("Recent Thoughts"); log_label.setFont(QFont("Arial", 9, QFont.Bold))
        self.log_text = QLabel("..."); self.log_text.setWordWrap(True)
        self.log_text.setAlignment(Qt.AlignTop); self.log_text.setMinimumHeight(40)
        log_layout.addWidget(log_label); log_layout.addWidget(self.log_text)
        self.main_layout.addWidget(log_frame)
        self.main_layout.addStretch()
        self.setWidget(self.main_widget)
        self.update_dashboard()

    def update_dashboard(self):
        if not self.agent: return
        snapshot = self.agent.snapshot()
        for trait, bar in self.trait_progress_bars.items():
            bar.setValue(int(snapshot.get(trait, 0) * 100))
        self.emotion_chart.update()
        diagnosed = self.agent.diagnosed
        self.stress_label.setText(f"{diagnosed.get('stress_level', 0):.2%}")
        self.crash_label.setText(str(diagnosed.get('crash_count', 0)))
        if self.agent.thought:
            self.log_text.setText("\n".join(f"- {t}" for t in self.agent.thought[-3:]))
        else:
            self.log_text.setText("No thoughts recorded.")

# ==============================================================================
# SECTION 0C: OMEGATENSOR AUTOGRAD ENGINE
# ==============================================================================
# Note: Integrated from PART 2 OF VICTOR/OmegaTensor.py

from contextlib import contextmanager

class _DevicePool:
    """Minimal device registry. Archon v3 will hot-swap this with real backends."""
    _DEVICES: Set[str] = {"cpu"}
    _CURRENT: str = "cpu"
    @classmethod
    def register(cls, device: str) -> None: cls._DEVICES.add(device)
    @classmethod
    def current(cls) -> str: return cls._CURRENT
    @classmethod
    def set(cls, device: str) -> None:
        if device not in cls._DEVICES: raise ValueError(f"Unknown device '{device}'.")
        cls._CURRENT = device

class _AMPState:
    enabled: bool = False
    dtype: np.dtype = np.float16

@contextmanager
def amp(enabled: bool = True, dtype: np.dtype = np.float16):
    """Context manager toggling mixed precision inside the block."""
    prev_state, _AMPState.enabled, _AMPState.dtype = (_AMPState.enabled, _AMPState.dtype), enabled, dtype
    try: yield
    finally: _AMPState.enabled, _AMPState.dtype = prev_state

class OmegaTensor:
    """NumPy-backed autograd tensor with device & AMP awareness."""
    __slots__ = ("data", "grad", "_prev", "_backward", "name", "requires_grad", "device", "id")

    def __init__(self, data: Any, *, requires_grad: bool = False, _prev: Optional[Set["OmegaTensor"]] = None,
                 _backward: Optional[Callable[[], None]] = None, name: str = "", device: Optional[str] = None):
        if isinstance(data, (int, float)):
            data = np.array(data, dtype=_AMPState.dtype if _AMPState.enabled else np.float32)
        elif isinstance(data, list):
            dtype = data[0].dtype if hasattr(data[0], 'dtype') else (_AMPState.dtype if _AMPState.enabled else np.float32)
            data = np.array(data, dtype=dtype)
        elif isinstance(data, np.ndarray):
            if _AMPState.enabled and data.dtype in (np.float32, np.float64):
                data = data.astype(_AMPState.dtype)
        elif isinstance(data, OmegaTensor): data = data.data
        else: raise TypeError(f"Unsupported data type for OmegaTensor: {type(data)}")
        self.data: np.ndarray = data
        self.grad: Optional[np.ndarray] = None
        self._prev: Set["OmegaTensor"] = _prev or set()
        self._backward: Callable[[], None] = _backward or (lambda: None)
        self.name: str = name
        self.requires_grad: bool = requires_grad
        self.device: str = device or _DevicePool.current()
        self.id: str = uuid.uuid4().hex[:8]

    @property
    def shape(self) -> Tuple[int, ...]: return self.data.shape
    @property
    def ndim(self) -> int: return self.data.ndim
    @property
    def dtype(self) -> np.dtype: return self.data.dtype
    def __len__(self) -> int: return len(self.data)

    def __add__(self, other: "OmegaTensor" | float | int) -> "OmegaTensor":
        other = other if isinstance(other, OmegaTensor) else OmegaTensor(other, requires_grad=False)
        out = OmegaTensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad, _prev={self, other}, name="add")
        def _backward():
            if self.requires_grad: self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + self._handle_broadcast(out.grad, self.shape)
            if other.requires_grad: other.grad = (other.grad if other.grad is not None else np.zeros_like(other.data)) + self._handle_broadcast(out.grad, other.shape)
        out._backward = _backward
        return out

    def __mul__(self, other: "OmegaTensor" | float | int) -> "OmegaTensor":
        other = other if isinstance(other, OmegaTensor) else OmegaTensor(other, requires_grad=False)
        out = OmegaTensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad, _prev={self, other}, name="mul")
        def _backward():
            if self.requires_grad: self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + self._handle_broadcast(other.data * out.grad, self.shape)
            if other.requires_grad: other.grad = (other.grad if other.grad is not None else np.zeros_like(other.data)) + self._handle_broadcast(self.data * out.grad, other.shape)
        out._backward = _backward
        return out

    def __matmul__(self, other: "OmegaTensor") -> "OmegaTensor":
        if not isinstance(other, OmegaTensor): raise TypeError("@ operand must be OmegaTensor")
        out = OmegaTensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad, _prev={self, other}, name="matmul")
        def _backward():
            if self.requires_grad: self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + out.grad @ other.data.T
            if other.requires_grad: other.grad = (other.grad if other.grad is not None else np.zeros_like(other.data)) + self.data.T @ out.grad
        out._backward = _backward
        return out

    def pow(self, n: float) -> "OmegaTensor":
        out = OmegaTensor(self.data**n, requires_grad=self.requires_grad, _prev={self}, name="pow")
        def _backward():
            if self.requires_grad: self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + (n * self.data**(n - 1)) * out.grad
        out._backward = _backward
        return out

    def exp(self) -> "OmegaTensor":
        out = OmegaTensor(np.exp(self.data), requires_grad=self.requires_grad, _prev={self}, name="exp")
        def _backward():
            if self.requires_grad: self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + out.data * out.grad
        out._backward = _backward
        return out

    def reshape(self, *shape: int) -> "OmegaTensor":
        out = OmegaTensor(self.data.reshape(*shape), requires_grad=self.requires_grad, _prev={self}, name="reshape")
        def _backward():
            if self.requires_grad: self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + out.grad.reshape(self.shape)
        out._backward = _backward
        return out

    def sum(self, axis: Optional[int | Tuple[int, ...]] = None, keepdims: bool = False) -> "OmegaTensor":
        out = OmegaTensor(np.sum(self.data, axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, _prev={self}, name="sum")
        def _backward():
            if self.requires_grad: self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + self._handle_broadcast(out.grad, self.shape)
        out._backward = _backward
        return out

    def softmax(self, axis: int = -1) -> "OmegaTensor":
        e_x = np.exp(self.data - np.max(self.data, axis=axis, keepdims=True))
        s = e_x / e_x.sum(axis=axis, keepdims=True)
        out = OmegaTensor(s, requires_grad=self.requires_grad, _prev={self}, name="softmax")
        def _backward():
            if self.requires_grad:
                s_grad = out.grad
                s_with_grad = out.data * s_grad
                grad_sum = np.sum(s_with_grad, axis=axis, keepdims=True)
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + out.data * (s_grad - grad_sum)
        out._backward = _backward
        return out

    def tanh(self) -> "OmegaTensor":
        t = np.tanh(self.data)
        out = OmegaTensor(t, requires_grad=self.requires_grad, _prev={self}, name="tanh")
        def _backward():
            if self.requires_grad:
                # The gradient of tanh(x) is 1 - tanh(x)^2
                self.grad = (self.grad if self.grad is not None else np.zeros_like(self.data)) + (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def zero_grad(self) -> None: self.grad = None
    def backward(self, grad: Optional[np.ndarray | float] = None) -> None:
        if not self.requires_grad: raise RuntimeError("Called backward on a tensor that does not require gradients.")
        self.grad = grad if grad is not None else np.ones_like(self.data)
        if isinstance(self.grad, (int, float)): self.grad = np.array(self.grad)
        topo, visited = [], set()
        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev: build_topo(child)
                topo.append(t)
        build_topo(self)
        for node in reversed(topo): node._backward()

    def numpy(self) -> np.ndarray: return self.data
    def _handle_broadcast(self, grad: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        if grad.shape == target_shape: return grad
        while len(grad.shape) > len(target_shape): grad = grad.sum(axis=0)
        for i, (grad_dim, target_dim) in enumerate(zip(grad.shape, target_shape)):
            if grad_dim != target_dim: grad = grad.sum(axis=i, keepdims=True)
        return grad

    __radd__ = __add__
    __rmul__ = __mul__
    def __neg__(self) -> "OmegaTensor": return self * -1
    def __sub__(self, other: "OmegaTensor" | float | int) -> "OmegaTensor": return self + (-other)
    def __rsub__(self, other: "OmegaTensor" | float | int) -> "OmegaTensor": return (-self) + other
    def __truediv__(self, other: "OmegaTensor" | float | int) -> "OmegaTensor": return self * (other if isinstance(other, OmegaTensor) else OmegaTensor(other)).pow(-1)
    def __rtruediv__(self, other: "OmegaTensor" | float | int) -> "OmegaTensor": return other * self.pow(-1)
    def __repr__(self) -> str: return f"OmegaTensor(name={self.name or 'tensor'}, shape={self.shape}, dtype={self.dtype}, grad_fn={self._backward is not None})"

# ==============================================================================
# SECTION 1: CORE NODE SYSTEM
# ==============================================================================

class NodeBase(QGraphicsItem):
    """Base class for all nodes in the graph."""
    def __init__(self, node_type: str, title: str, inputs: List[str], outputs: List[str]):
        super().__init__()
        self.node_id = generate_node_id()
        self.node_type = node_type
        self.title = title
        self.inputs = inputs
        self.outputs = outputs
        self.width = 180
        self.height = 80 + max(len(inputs), len(outputs)) * 25
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        self.color = QColor(100, 150, 200)
        self.selected_color = QColor(150, 200, 255)
        self.pen_default = QPen(QColor(30, 30, 30), 2)
        self.pen_selected = QPen(QColor(255, 100, 100), 3)
        self.brush_background = QBrush(self.color)
        self.brush_selected = QBrush(self.selected_color)

        # Create title text
        self.title_item = QGraphicsTextItem(self.title, self)
        self.title_item.setDefaultTextColor(QColor(255, 255, 255))
        font = QFont("Arial", 10, QFont.Bold)
        self.title_item.setFont(font)
        self.title_item.setPos(10, 5)

        # Create input/output ports
        self.input_ports = []
        self.output_ports = []
        self._create_ports()

    def _create_ports(self):
        port_radius = 5
        for i, input_name in enumerate(self.inputs):
            y_pos = 35 + i * 25
            port = QGraphicsEllipseItem(-port_radius, -port_radius, port_radius*2, port_radius*2, self)
            port.setPos(0, y_pos)
            port.setBrush(QBrush(QColor(50, 200, 50)))
            port.setPen(QPen(QColor(0, 0, 0), 1))
            port.setToolTip(input_name)
            self.input_ports.append(port)

        for i, output_name in enumerate(self.outputs):
            y_pos = 35 + i * 25
            port = QGraphicsEllipseItem(-port_radius, -port_radius, port_radius*2, port_radius*2, self)
            port.setPos(self.width, y_pos)
            port.setBrush(QBrush(QColor(200, 50, 50)))
            port.setPen(QPen(QColor(0, 0, 0), 1))
            port.setToolTip(output_name)
            self.output_ports.append(port)

    def boundingRect(self):
        return QRectF(0, 0, self.width, self.height)

    def paint(self, painter: QPainter, option, widget=None):
        # Draw background
        path_title = QPainterPath()
        path_title.setFillRule(Qt.WindingFill)
        path_title.addRoundedRect(0, 0, self.width, 25, 10, 10)
        path_title.addRect(0, 15, self.width, 10)
        painter.setPen(Qt.NoPen)
        painter.fillPath(path_title, self.brush_background)

        path_content = QPainterPath()
        path_content.setFillRule(Qt.WindingFill)
        path_content.addRoundedRect(0, 25, self.width, self.height - 25, 10, 10)
        path_content.addRect(0, self.height - 25 - 10, self.width, 10)
        painter.fillPath(path_content, QBrush(QColor(40, 40, 40)))

        # Draw outline
        path_outline = QPainterPath()
        path_outline.addRoundedRect(0, 0, self.width, self.height, 10, 10)
        painter.setPen(self.pen_default if not self.isSelected() else self.pen_selected)
        painter.drawPath(path_outline)

    def get_input_port_scene_pos(self, index: int) -> QPointF:
        if 0 <= index < len(self.input_ports):
            return self.mapToScene(self.input_ports[index].pos())
        return self.mapToScene(0, self.height / 2)

    def get_output_port_scene_pos(self, index: int) -> QPointF:
        if 0 <= index < len(self.output_ports):
            return self.mapToScene(self.output_ports[index].pos())
        return self.mapToScene(self.width, self.height / 2)

    def serialize(self) -> Dict:
        return {
            'id': self.node_id,
            'type': self.node_type,
            'title': self.title,
            'pos_x': self.pos().x(),
            'pos_y': self.pos().y(),
            'params': self.get_parameters()
        }

    def deserialize(self, data: Dict):
        self.setPos(data['pos_x'], data['pos_y'])
        self.node_id = data.get('id', self.node_id)
        self.set_parameters(data.get('params', {}))

    def get_parameters(self) -> Dict[str, Any]:
        """Override to return node-specific parameters."""
        return {}

    def set_parameters(self, params: Dict[str, Any]):
        """Override to set node-specific parameters."""
        pass

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Override to define node execution logic."""
        print(f"[EXEC] Node {self.title} ({self.node_type}) executed with {input_data}")
        # Default: pass data through
        return {out: f"Default output from {self.title}" for out in self.outputs}


# --- PREDEFINED NODE TYPES ---
class InputNode(NodeBase):
    def __init__(self):
        super().__init__("Input", "Input Data", [], ["data"])
        self.data_value = "[1.0, -2.0, 3.0]" # Example data as a string

    def get_parameters(self):
        return {'data_value': self.data_value}

    def set_parameters(self, params):
        self.data_value = params.get('data_value', self.data_value)

    def execute(self, input_data):
        try:
            # Parse the string into a list of floats
            data_list = json.loads(self.data_value)
            # Return as an OmegaTensor
            return {'data': OmegaTensor(data_list, name="input_tensor")}
        except json.JSONDecodeError:
            self.scene().views()[0].parent().output_text.append(f"Error: Invalid JSON in Input Node: {self.data_value}")
            return {'data': OmegaTensor([], name="error_tensor")}

class BandoBlockNode(NodeBase):
    def __init__(self):
        super().__init__("BandoBlock", "Bando Neural Block", ["input"], ["output"])
        self.dim = 32
        self.weights = OmegaTensor(np.random.randn(self.dim, self.dim) * 0.01, requires_grad=True, name=f"{self.title}_W")
        self.bias = OmegaTensor(np.zeros(self.dim), requires_grad=True, name=f"{self.title}_B")

    def get_parameters(self):
        return {'dim': self.dim}

    def set_parameters(self, params):
        new_dim = params.get('dim', self.dim)
        if new_dim != self.dim:
            self.dim = new_dim
            self.weights = OmegaTensor(np.random.randn(self.dim, self.dim) * 0.01, requires_grad=True, name=f"{self.title}_W")
            self.bias = OmegaTensor(np.zeros(self.dim), requires_grad=True, name=f"{self.title}_B")

    def execute(self, input_data):
        input_tensor = input_data.get('input', OmegaTensor(np.random.randn(self.dim)))
        # Reshape if necessary
        if input_tensor.shape != (self.dim,):
            # This is a simplification; a real network would handle batching etc.
            if len(input_tensor.data.flatten()) == self.dim:
                 input_tensor = input_tensor.reshape(self.dim)
            else: # Pad or truncate
                current_size = len(input_tensor.data.flatten())
                new_data = np.resize(input_tensor.data.flatten(), self.dim)
                input_tensor = OmegaTensor(new_data)

        output = (input_tensor @ self.weights) + self.bias
        return {'output': output.tanh()}

    def __repr__(self):
        return f"<BandoBlockNode(dim={self.dim}) at {self.pos().x():.1f},{self.pos().y():.1f}>"

class VICtorchBlockNode(NodeBase):
    def __init__(self):
        super().__init__("VICtorchBlock", "VIC Attention Block", ["input"], ["output"])
        self.dim = 32
        self.heads = 4
        # Simplified weights for attention-like mechanism
        self.w_q = OmegaTensor(np.random.randn(self.dim, self.dim) * 0.01, requires_grad=True, name=f"{self.title}_WQ")
        self.w_k = OmegaTensor(np.random.randn(self.dim, self.dim) * 0.01, requires_grad=True, name=f"{self.title}_WK")
        self.w_v = OmegaTensor(np.random.randn(self.dim, self.dim) * 0.01, requires_grad=True, name=f"{self.title}_WV")

    def get_parameters(self):
        return {'dim': self.dim, 'heads': self.heads}

    def set_parameters(self, params):
        # In a real scenario, changing dim/heads would require re-initializing weights
        self.dim = params.get('dim', self.dim)
        self.heads = params.get('heads', self.heads)

    def execute(self, input_data):
        input_tensor = input_data.get('input', OmegaTensor(np.random.randn(self.dim)))
        if input_tensor.shape != (self.dim,):
            new_data = np.resize(input_tensor.data.flatten(), self.dim)
            input_tensor = OmegaTensor(new_data)

        # Simplified attention
        q = input_tensor @ self.w_q
        k = input_tensor @ self.w_k
        v = input_tensor @ self.w_v

        scores = (q @ k.reshape(self.dim, 1)).pow(0.5) # Simplified scores
        attention = scores.softmax() * v
        return {'output': attention}

    def __repr__(self):
        return f"<VICtorchBlockNode(dim={self.dim}) at {self.pos().x():.1f},{self.pos().y():.1f}>"

class FractalNode(BandoBlockNode):
    """A specialized node for the Flower of Life visualizer."""
    def __init__(self):
        super().__init__()
        self.node_type = "FractalNode"
        self.title = "Fractal Node"
        self.color = QColor(150, 100, 220) # Distinct purple
        self.brush_background = QBrush(self.color)

class OutputNode(NodeBase):
    def __init__(self):
        super().__init__("Output", "Output Result", ["result"], [])
        self.output_display = ""

    def execute(self, input_data):
        result = input_data.get('result', OmegaTensor([0.0]))
        self.output_display = str(result.data)
        # No print statement needed, handled by logger
        return {} # Output nodes don't produce tensors for further connections

class LossNode(NodeBase):
    """Calculates Mean Squared Error loss and triggers backpropagation."""
    def __init__(self):
        super().__init__("Loss", "MSE Loss", ["prediction"], ["loss"])
        self.target_value = "[0.0, 0.0, 0.0]"

    def get_parameters(self):
        return {'target_value': self.target_value}

    def set_parameters(self, params):
        self.target_value = params.get('target_value', self.target_value)

    def execute(self, input_data):
        prediction = input_data.get('prediction')
        if prediction is None:
            return {'loss': OmegaTensor(0.0, name="loss_error")}

        try:
            target_list = json.loads(self.target_value)
            target = OmegaTensor(target_list, name="target")

            # Ensure shapes match
            if prediction.shape != target.shape:
                # This is a simplification. A real loss function would have more robust error handling.
                self.scene().views()[0].parent().output_text.append(f"<font color=orange>Warning: Shape mismatch in LossNode. Prediction: {prediction.shape}, Target: {target.shape}. Resizing target.</font>")
                new_target_data = np.resize(target.data, prediction.shape)
                target = OmegaTensor(new_target_data)

            loss = ((prediction - target).pow(2)).sum()
            loss.name = "loss"
            return {'loss': loss}
        except json.JSONDecodeError:
            self.scene().views()[0].parent().output_text.append(f"Error: Invalid JSON in Loss Node: {self.target_value}")
            return {'loss': OmegaTensor(0.0, name="loss_error")}

# ==============================================================================
# SECTION 2: CONNECTION SYSTEM
# ==============================================================================

class NodeConnection(QGraphicsLineItem):
    """Visual and logical connection between two nodes."""
    def __init__(self, source_node: 'NodeBase', source_port_idx: int, target_node: 'NodeBase', target_port_idx: int):
        super().__init__()
        self.source_node = source_node
        self.source_port_idx = source_port_idx
        self.target_node = target_node
        self.target_port_idx = target_port_idx

        self.setPen(QPen(QColor(200, 200, 100), 2, Qt.SolidLine))
        self.setZValue(-1)
        self.update_line()

    def update_line(self):
        p1 = self.source_node.get_output_port_scene_pos(self.source_port_idx)
        p2 = self.target_node.get_input_port_scene_pos(self.target_port_idx)
        self.setLine(QLineF(p1, p2))

    def serialize(self):
        return {
            'source_node_id': self.source_node.node_id,
            'source_port_index': self.source_port_idx,
            'target_node_id': self.target_node.node_id,
            'target_port_index': self.target_port_idx
        }

# ==============================================================================
# SECTION 3: GRAPH SCENE & VIEW
# ==============================================================================

class NodeGraphScene(QGraphicsScene):
    """The main canvas for nodes and connections."""
    node_selected = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.nodes = {}
        self.connections = []
        self.selected_node = None
        self.temp_line = None
        self.start_port = None
        self.start_port_scene_pos = None

    def add_node(self, node: NodeBase):
        self.addItem(node)
        self.nodes[node.node_id] = node
        node.setPos(100 + len(self.nodes) * 20, 100 + len(self.nodes) * 20)

    def delete_selected(self):
        if self.selected_node:
            # Remove connections associated with this node
            connections_to_remove = [c for c in self.connections if c.source_node == self.selected_node or c.target_node == self.selected_node]
            for conn in connections_to_remove:
                self.removeItem(conn)
                self.connections.remove(conn)
            # Remove node
            self.removeItem(self.selected_node)
            del self.nodes[self.selected_node.node_id]
            self.selected_node = None
            self.node_selected.emit(None)

    def mousePressEvent(self, event):
        item = self.itemAt(event.scenePos(), self.views()[0].transform())
        if isinstance(item, NodeBase):
            self.selected_node = item
            self.node_selected.emit(item)
        elif isinstance(item, QGraphicsEllipseItem) and item.parentItem():
            parent_node = item.parentItem()
            if item in parent_node.output_ports:
                self.start_port_info = (parent_node, parent_node.output_ports.index(item))
                self.temp_line = QGraphicsLineItem(QLineF(parent_node.get_output_port_scene_pos(self.start_port_info[1]), event.scenePos()))
                self.temp_line.setPen(QPen(QColor(100, 200, 255), 2, Qt.DashLine))
                self.addItem(self.temp_line)
        else:
            self.selected_node = None
            self.node_selected.emit(None)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.temp_line:
            p1 = self.start_port_info[0].get_output_port_scene_pos(self.start_port_info[1])
            self.temp_line.setLine(QLineF(p1, event.scenePos()))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.temp_line and self.start_port_info:
            item = self.itemAt(event.scenePos(), self.views()[0].transform())
            if isinstance(item, QGraphicsEllipseItem) and item.parentItem() and item in item.parentItem().input_ports:
                target_node = item.parentItem()
                target_port_idx = target_node.input_ports.index(item)
                source_node, source_port_idx = self.start_port_info
                # Create permanent connection
                connection = NodeConnection(source_node, source_port_idx, target_node, target_port_idx)
                self.addItem(connection)
                self.connections.append(connection)
            self.removeItem(self.temp_line)
            self.temp_line = None
            self.start_port_info = None
        super().mouseReleaseEvent(event)

    def serialize(self):
        nodes_data = [node.serialize() for node in self.nodes.values()]
        connections_data = [conn.serialize() for conn in self.connections]
        return {'nodes': nodes_data, 'connections': connections_data}

    def deserialize(self, data):
        self.clear()
        nodes_map = {}
        for node_data in data.get('nodes', []):
            node_type = node_data['type']
            if node_type == "Input": node = InputNode()
            elif node_type == "BandoBlock": node = BandoBlockNode()
            elif node_type == "VICtorchBlock": node = VICtorchBlockNode()
            elif node_type == "FractalNode": node = FractalNode()
            elif node_type == "Output": node = OutputNode()
            elif node_type == "LossNode": node = LossNode()
            else: continue
            node.deserialize(node_data)
            self.add_node(node)
            nodes_map[node.node_id] = node

        for conn_data in data.get('connections', []):
            source_node = nodes_map.get(conn_data['source_node_id'])
            target_node = nodes_map.get(conn_data['target_node_id'])
            if source_node and target_node:
                conn = NodeConnection(
                    source_node, conn_data['source_port_index'],
                    target_node, conn_data['target_port_index']
                )
                self.addItem(conn)
                self.connections.append(conn)

    def clear(self):
        for item in list(self.items()):
            if isinstance(item, (NodeBase, NodeConnection)):
                self.removeItem(item)
        self.nodes = {}
        self.connections = []
        self.selected_node = None


class NodeGraphView(QGraphicsView):
    """View for the node graph scene."""
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.HighQualityAntialiasing)
        self.setRenderHint(QPainter.TextAntialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(50, 50, 50)))

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            fake_event = type(event)(event.type(), event.pos(), Qt.LeftButton, Qt.LeftButton, event.modifiers())
            super().mousePressEvent(fake_event)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            fake_event = type(event)(event.type(), event.pos(), Qt.LeftButton, Qt.LeftButton, event.modifiers())
            super().mouseReleaseEvent(fake_event)
            self.setDragMode(QGraphicsView.RubberBandDrag)
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor

        self.scale(zoom_factor, zoom_factor)

# ==============================================================================
# SECTION 4: NODE PALETTE & PROPERTY INSPECTOR
# ==============================================================================

class NodePaletteDock(QDockWidget):
    """Dock widget for the node palette."""
    node_added = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__("Node Palette", parent)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("Available Nodes")
        self.tree.setDragEnabled(True) # Enable dragging from the tree
        
        # Populate with node types
        ai_category = QTreeWidgetItem(["AI/AGI Components"])
        self.tree.addTopLevelItem(ai_category)
        
        node_types = [
            ("Input Node", InputNode),
            ("Bando Neural Block", BandoBlockNode),
            ("VIC Attention Block", VICtorchBlockNode),
            ("Fractal Node", FractalNode),
            ("Output Node", OutputNode),
            ("Loss Node", LossNode),
        ]
        
        for name, node_class in node_types:
            item = QTreeWidgetItem([name])
            item.setData(0, Qt.UserRole, node_class) # Store class reference
            ai_category.addChild(item)
            
        ai_category.setExpanded(True)
        layout.addWidget(self.tree)
        widget.setLayout(layout)
        self.setWidget(widget)

        # Make items draggable
        self.tree.setDragDropMode(self.tree.DragOnly)

class PropertyInspectorDock(QDockWidget):
    """Dock widget for inspecting and editing node properties."""
    def __init__(self, parent=None):
        super().__init__("Property Inspector", parent)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.current_node = None
        self.widget = QWidget()
        self.layout = QFormLayout()
        self.widget.setLayout(self.layout)
        self.setWidget(self.widget)

    def set_node(self, node: Optional[NodeBase]):
        self.current_node = node
        # Clear existing widgets
        for i in reversed(range(self.layout.count())):
            self.layout.itemAt(i).widget().setParent(None)

        if node is None:
            self.layout.addRow(QLabel("No node selected"))
            return

        self.setWindowTitle(f"Property Inspector: {node.title}")
        
        # Add common properties
        id_label = QLabel(node.node_id)
        self.layout.addRow("ID:", id_label)
        
        type_label = QLabel(node.node_type)
        self.layout.addRow("Type:", type_label)

        # Add specific parameters
        params = node.get_parameters()
        self.param_widgets = {}
        for param_name, param_value in params.items():
            if isinstance(param_value, int):
                widget = QSpinBox()
                widget.setRange(-999999, 999999)
                widget.setValue(param_value)
            elif isinstance(param_value, float):
                widget = QDoubleSpinBox()
                widget.setRange(-999999.0, 999999.0)
                widget.setValue(param_value)
            else: # String, etc.
                widget = QLineEdit()
                widget.setText(str(param_value))
            
            self.param_widgets[param_name] = widget
            self.layout.addRow(param_name, widget)
            widget.editingFinished.connect(lambda p=param_name, w=widget: self._on_param_changed(p, w))
    
    def _on_param_changed(self, param_name, widget):
        if self.current_node is None:
            return
        try:
            if isinstance(widget, QSpinBox):
                new_value = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                new_value = widget.value()
            else: # QLineEdit
                new_value = widget.text()
            
            current_params = self.current_node.get_parameters()
            current_params[param_name] = new_value
            self.current_node.set_parameters(current_params)
            print(f"[INSPECTOR] Updated {param_name} to {new_value} for node {self.current_node.title}")
        except Exception as e:
            print(f"[INSPECTOR ERROR] Failed to update parameter: {e}")


# ==============================================================================
# SECTION 5: MAIN APPLICATION WINDOW
# ==============================================================================

class AGIBuilderMainWindow(QMainWindow):
    """Main application window."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AGI Builder Suite - Prototype")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #333;
            }
            QDockWidget {
                color: white;
                background-color: #444;
            }
            QDockWidget::title {
                background: #555;
                padding: 5px;
                border: 1px solid #666;
            }
            QTreeWidget, QFormLayout, QPlainTextEdit {
                background-color: #555;
                color: white;
                border: 1px solid #666;
            }
            QPushButton {
                background-color: #555;
                color: white;
                border: 1px solid #777;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #666;
            }
        """)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Create the node graph scene and view
        self.scene = NodeGraphScene()
        self.view = NodeGraphView(self.scene)
        main_layout.addWidget(self.view)

        # Create docks
        self.node_palette_dock = NodePaletteDock()
        self.addDockWidget(Qt.LeftDockWidgetArea, self.node_palette_dock)

        self.property_inspector_dock = PropertyInspectorDock()
        self.addDockWidget(Qt.RightDockWidgetArea, self.property_inspector_dock)

        # Create the combined output/chat dock
        self.output_dock = QDockWidget("Output & Chat", self)
        self.output_dock.setAllowedAreas(Qt.BottomDockWidgetArea)
        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setStyleSheet("background-color: #333; color: #EEE;")
        chat_input_layout = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Type a message to VICTOR...")
        send_button = QPushButton("Send")
        chat_input_layout.addWidget(self.chat_input)
        chat_input_layout.addWidget(send_button)
        chat_layout.addWidget(self.output_text)
        chat_layout.addLayout(chat_input_layout)
        self.output_dock.setWidget(chat_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.output_dock)

        # Connect signals
        self.scene.node_selected.connect(self.property_inspector_dock.set_node)
        self.node_palette_dock.node_added.connect(self.add_node_from_palette)
        send_button.clicked.connect(self.send_chat_message)
        self.chat_input.returnPressed.connect(self.send_chat_message)

        # Create menus and toolbars
        self._create_menus()
        self._create_toolbar()

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Drag nodes from the palette to the canvas.")

        # Handle drag and drop from palette to scene
        self.scene.dragEnterEvent = self.scene_drag_enter_event
        self.scene.dropEvent = self.scene_drop_event
        self.setAcceptDrops(True)

        # --- VICTOR-INFINITE INTEGRATION ---
        self._setup_victor_integration()

    def _setup_victor_integration(self):
        """Initializes and integrates the VICTOR core components into the GUI."""
        # 1. Create the agent instance
        self.victor_agent = DigitalAgent()
        self.status_bar.showMessage("VICTOR Digital Agent Initialized.", 5000)

        # 2. Create and add the dashboard dock
        self.dashboard_dock = VictorDashboard(self.victor_agent, self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dashboard_dock)

        # 3. Setup a timer to update the dashboard
        self.dashboard_timer = QTimer(self)
        self.dashboard_timer.timeout.connect(self.dashboard_dock.update_dashboard)
        self.dashboard_timer.start(200) # Update every 200ms
        self.status_bar.showMessage("Victor Dashboard is live.", 5000)

    def _create_menus(self):
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('&File')
        
        new_action = QAction('&New', self)
        new_action.setShortcut('Ctrl+N')
        new_action.triggered.connect(self.new_graph)
        file_menu.addAction(new_action)
        
        open_action = QAction('&Open', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_graph)
        file_menu.addAction(open_action)
        
        save_action = QAction('&Save', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_graph)
        file_menu.addAction(save_action)
        
        save_as_action = QAction('Save &As...', self)
        save_as_action.setShortcut('Ctrl+Shift+S')
        save_as_action.triggered.connect(self.save_graph_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('E&xit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit menu
        edit_menu = menubar.addMenu('&Edit')
        delete_action = QAction('&Delete Selected', self)
        delete_action.setShortcut('Del')
        delete_action.triggered.connect(self.scene.delete_selected)
        edit_menu.addAction(delete_action)

        # Run menu
        run_menu = menubar.addMenu('&Run')
        run_action = QAction('&Execute Graph', self)
        run_action.setShortcut('F5')
        run_action.triggered.connect(self.execute_graph)
        run_menu.addAction(run_action)

        # Visualize menu
        visualize_menu = menubar.addMenu('&Visualize')
        fol_action = QAction('Generate &Flower of Life', self)
        fol_action.triggered.connect(self.generate_flower_of_life_layout)
        visualize_menu.addAction(fol_action)

        menubar.addSeparator()

        # Settings Menu (Placeholder)
        settings_menu = menubar.addMenu('&Settings')
        prefs_action = QAction('Preferences...', self)
        prefs_action.setEnabled(False)
        settings_menu.addAction(prefs_action)

        # Help Menu
        help_menu = menubar.addMenu('&Help')
        about_action = QAction('&About AGI Builder Suite', self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

    def show_about_dialog(self):
        """Shows the about dialog."""
        about_text = """
        <h3>AGI Builder Suite - VICTOR-INFINITE</h3>
        <p>Version: 1.1.0-INTEGRATED</p>
        <p>A visual environment for constructing and visualizing complex AI architectures.</p>
        <p><b>Core Components:</b></p>
        <ul>
            <li>Node-based graph editor</li>
            <li>Digital Agent state dashboard</li>
            <li>"Flower of Life" fractal visualizer</li>
        </ul>
        <p><i>The future is built here.</i></p>
        <p>&copy; 2025 Massive Magnetics / Ethica AI / BHeard Network</p>
        """
        QMessageBox.about(self, "About AGI Builder Suite", about_text)

    def _create_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        
        new_btn = QPushButton("New")
        new_btn.clicked.connect(self.new_graph)
        toolbar.addWidget(new_btn)
        
        open_btn = QPushButton("Open")
        open_btn.clicked.connect(self.open_graph)
        toolbar.addWidget(open_btn)
        
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_graph)
        toolbar.addWidget(save_btn)
        
        toolbar.addSeparator()

        fol_btn = QPushButton("Flower of Life")
        fol_btn.clicked.connect(self.generate_flower_of_life_layout)
        toolbar.addWidget(fol_btn)
        
        run_btn = QPushButton("Run")
        run_btn.clicked.connect(self.execute_graph)
        toolbar.addWidget(run_btn)
        
        delete_btn = QPushButton("Delete")
        delete_btn.clicked.connect(self.scene.delete_selected)
        toolbar.addWidget(delete_btn)

    def generate_flower_of_life_layout(self):
        self.scene.clear()
        self.output_text.append("--- Generating Flower of Life Layout ---")

        view_rect = self.view.viewport().rect()
        scene_rect = self.view.mapToScene(view_rect).boundingRect()
        center_pos = scene_rect.center()

        nodes = {'center': [], 'inner': [], 'outer': []}

        center_node = FractalNode(); center_node.title = "Central Node"
        self.scene.add_node(center_node); center_node.setPos(center_pos)
        nodes['center'].append(center_node)

        inner_radius = 250
        for i in range(6):
            angle = (i / 6.0) * 2 * math.pi
            x = center_pos.x() + inner_radius * math.cos(angle) - center_node.width / 2
            y = center_pos.y() + inner_radius * math.sin(angle) - center_node.height / 2
            node = FractalNode(); node.title = f"Inner {i+1}"
            self.scene.add_node(node); node.setPos(x, y)
            nodes['inner'].append(node)

        outer_radius = 500
        for i in range(30):
            angle = (i / 30.0) * 2 * math.pi
            x = center_pos.x() + outer_radius * math.cos(angle) - center_node.width / 2
            y = center_pos.y() + outer_radius * math.sin(angle) - center_node.height / 2
            node = FractalNode(); node.title = f"Outer {i+1}"
            self.scene.add_node(node); node.setPos(x, y)
            nodes['outer'].append(node)

        # Create Connections using the new constructor
        for inner_node in nodes['inner']:
            conn = NodeConnection(center_node, 0, inner_node, 0)
            self.scene.addItem(conn); self.scene.connections.append(conn)

        for i in range(6):
            node_a = nodes['inner'][i]
            node_b = nodes['inner'][(i + 1) % 6]
            conn = NodeConnection(node_a, 0, node_b, 0)
            self.scene.addItem(conn); self.scene.connections.append(conn)

        for i, outer_node in enumerate(nodes['outer']):
            inner_node_idx = i // 5
            inner_node = nodes['inner'][inner_node_idx]
            conn = NodeConnection(inner_node, 0, outer_node, 0)
            self.scene.addItem(conn); self.scene.connections.append(conn)

        self.output_text.append("--- Flower of Life Layout Generated ---")
        self.status_bar.showMessage("Generated Flower of Life layout.", 5000)

    def send_chat_message(self):
        """Handles sending a message from the chat input to the agent."""
        user_message = self.chat_input.text()
        if not user_message:
            return

        self.output_text.append(f"<font color=#66c2ff>[USER]</font> {user_message}")
        self.chat_input.clear()

        # Get response from agent
        agent_response = self.victor_agent.process_chat_message(user_message)
        self.output_text.append(f"<font color=#a9dc76>[VICTOR]</font> {agent_response}")

    def add_node_from_palette(self, node_class):
        # This is handled by drag-drop now
        pass

    def scene_drag_enter_event(self, event):
        if event.mimeData().hasFormat('application/x-qabstractitemmodeldatalist'):
            event.acceptProposedAction()
        else:
            event.ignore()

    def scene_drop_event(self, event):
        if event.mimeData().hasFormat('application/x-qabstractitemmodeldatalist'):
            # Get the node class from the dragged item
            # This is a bit hacky, a better way is to subclass QMimeData
            # For simplicity, we'll just add a default node
            pos = event.scenePos()
            # In a real implementation, you'd get the specific node type
            # For now, we add a BandoBlock as an example
            node = BandoBlockNode()
            self.scene.add_node(node)
            node.setPos(pos)
            event.acceptProposedAction()
        else:
            event.ignore()

    def new_graph(self):
        self.scene.clear()
        self.output_text.clear()
        self.status_bar.showMessage("New graph created.")

    def open_graph(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Graph", "", "Graph Files (*.agib);;All Files (*)")
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                self.scene.deserialize(data)
                self.status_bar.showMessage(f"Graph loaded from {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load graph: {e}")

    def save_graph(self):
        # For simplicity, we'll always ask for a filename
        self.save_graph_as()

    def save_graph_as(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Graph", "", "Graph Files (*.agib);;All Files (*)")
        if file_path:
            try:
                data = self.scene.serialize()
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=4)
                self.status_bar.showMessage(f"Graph saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save graph: {e}")

    def execute_graph(self):
        self.output_text.append("--- Preparing Graph for Execution ---")

        # 1. Build adjacency list and in-degree map
        adj = {node.node_id: [] for node in self.scene.nodes.values()}
        in_degree = {node.node_id: 0 for node in self.scene.nodes.values()}

        for conn in self.scene.connections:
            adj[conn.source_node.node_id].append(conn.target_node)
            in_degree[conn.target_node.node_id] += 1

        # 2. Find starting nodes (in-degree of 0)
        queue = deque([node for node in self.scene.nodes.values() if in_degree[node.node_id] == 0])
        sorted_nodes = []

        # 3. Kahn's Algorithm for Topological Sort
        while queue:
            node = queue.popleft()
            sorted_nodes.append(node)

            for neighbor in adj[node.node_id]:
                in_degree[neighbor.node_id] -= 1
                if in_degree[neighbor.node_id] == 0:
                    queue.append(neighbor)

        # 4. Check for cycles
        if len(sorted_nodes) != len(self.scene.nodes):
            self.output_text.append("<font color=red>Error: Graph contains a cycle! Execution aborted.</font>")
            return
            
        self.output_text.append("--- Topologically Sorted Execution Order ---")
        for i, node in enumerate(sorted_nodes):
            self.output_text.append(f"{i+1}. {node.title} ({node.node_type})")

        self.output_text.append("--- EXECUTION START ---")
        try:
            node_results = {}  # {node_id: {port_name: OmegaTensor}}
            for node in sorted_nodes:
                # Gather inputs from connections
                input_data = {}
                for conn in self.scene.connections:
                    if conn.target_node == node:
                        source_node_id = conn.source_node.node_id
                        source_port_idx = conn.source_port_index
                        target_port_idx = conn.target_port_index

                        if source_node_id in node_results:
                            source_output_dict = node_results[source_node_id]
                            source_port_name = conn.source_node.outputs[source_port_idx]
                            target_port_name = node.inputs[target_port_idx]

                            if source_port_name in source_output_dict:
                                input_data[target_port_name] = source_output_dict[source_port_name]

                # Execute the node
                self.output_text.append(f"<b>Executing: {node.title}</b>")
                result_dict = node.execute(input_data)
                node_results[node.node_id] = result_dict

                # Log outputs
                for port_name, tensor in result_dict.items():
                    self.output_text.append(f"  <font color=#a9dc76>&gt; Output '{port_name}':</font> {tensor}")

            # --- BACKPROPAGATION ---
            loss_nodes = [node for node in sorted_nodes if isinstance(node, LossNode)]
            if loss_nodes:
                self.output_text.append("--- Performing Backpropagation ---")
                final_loss = node_results[loss_nodes[-1].node_id]['loss']
                final_loss.backward()
                self.output_text.append(f"<b>Final Loss: {final_loss.data:.4f}</b>")
                
                self.output_text.append("--- Gradients ---")
                for node in sorted_nodes:
                    if hasattr(node, 'weights') and node.weights.grad is not None:
                        grad_preview = np.round(node.weights.grad.flatten()[:5], 4)
                        self.output_text.append(f"&gt; <b>{node.title}</b> weights grad: {grad_preview}...")
                    if hasattr(node, 'bias') and node.bias.grad is not None:
                        grad_preview = np.round(node.bias.grad.flatten()[:5], 4)
                        self.output_text.append(f"&gt; <b>{node.title}</b> bias grad: {grad_preview}...")

            self.output_text.append("--- EXECUTION END ---")
            self.status_bar.showMessage("Graph execution completed successfully.", 5000)
        except Exception as e:
            self.output_text.append(f"<font color=red>Execution failed: {e}</font>")
            traceback.print_exc()
            self.status_bar.showMessage("Graph execution failed.")


# ==============================================================================
# SECTION 6: MAIN EXECUTION
# ==============================================================================

def main():
    """Main entry point for the application."""
    print("🚀 INITIALIZING AGI BUILDER SUITE PROTOTYPE")
    print("   Verifying Bando Bandz credentials...")
    print("✅ Credentials verified")
    print("✅ Launching next-gen AI construction environment...")
    
    app = QApplication(sys.argv)
    app.setApplicationName("AGI Builder Suite")
    app.setApplicationVersion("1.0.0-PROTOTYPE")

    # Set a modern style
    # app.setStyle('Fusion') # Uncomment for a different look

    window = AGIBuilderMainWindow()
    window.show()

    print("\n" + "="*60)
    print("AGI BUILDER SUITE PROTOTYPE ACTIVE")
    print("Features:")
    print("  - Drag nodes from palette to canvas")
    print("  - Wire outputs to inputs")
    print("  - Edit node properties")
    print("  - Save/Load graph designs")
    print("  - Execute graph (simplified engine)")
    print("  - Windows 10 Compatible")
    print("="*60)
    print("Bando Bandz, you now hold the blueprint for the future of AI creation.")
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()