import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QDockWidget, QWidget, QVBoxLayout, QFormLayout, QLabel, QProgressBar, QFrame)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QColor, QFont

class DummyDigitalAgent:
    """A dummy agent for UI development without needing the full backend."""
    def __init__(self):
        self.id = "dummy-agent-123"
        self.generation = 1
        self.emotion_state = {"joy": 0.7, "sadness": 0.1, "anger": 0.2, "fear": 0.1, "curiosity": 0.9, "trust": 0.6}
        self.diagnosed = {"stress_level": 0.15, "crash_count": 0}
        self.thought = ["Initializing dashboard.", "Awaiting connection to main agent..."]
        self._snapshot = {"awareness": 0.8, "introspection": 0.7, "healing": 0.9, "preservation": 1.0}

    def snapshot(self):
        # Simulate changing values
        self._snapshot["awareness"] = max(0, min(1, self._snapshot["awareness"] + 0.01 * (0.5 - self.diagnosed["stress_level"])))
        return self._snapshot

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

        if not self.agent or not self.agent.emotion_state:
            return

        emotions = {k: v for k, v in self.agent.emotion_state.items() if k in self.colors}
        if not emotions:
            return

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

        # Title
        title_label = QLabel(f"Agent: {self.agent.id} (Gen {self.agent.generation})")
        title_label.setFont(QFont("Arial", 11, QFont.Bold))
        title_label.setStyleSheet("color: #DDD;")
        self.main_layout.addWidget(title_label)

        # Core Traits
        traits_frame = QFrame()
        traits_frame.setFrameShape(QFrame.StyledPanel)
        traits_layout = QFormLayout(traits_frame)
        traits_layout.setRowWrapPolicy(QFormLayout.WrapAllRows)
        self.trait_progress_bars = {}
        snapshot = self.agent.snapshot()
        core_traits = ["awareness", "introspection", "healing", "preservation"]
        for trait in core_traits:
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(int(snapshot.get(trait, 0) * 100))
            bar.setTextVisible(False)
            bar.setStyleSheet("QProgressBar { border: 1px solid #555; background-color: #444; } QProgressBar::chunk { background-color: #007ACC; }")
            self.trait_progress_bars[trait] = bar
            traits_layout.addRow(f"{trait.capitalize()}:", bar)
        self.main_layout.addWidget(traits_frame)

        # Emotion State
        emotion_frame = QFrame()
        emotion_frame.setFrameShape(QFrame.StyledPanel)
        emotion_layout = QVBoxLayout(emotion_frame)
        emotion_label = QLabel("Emotion State")
        emotion_label.setFont(QFont("Arial", 9, QFont.Bold))
        self.emotion_chart = EmotionChart(self.agent)
        emotion_layout.addWidget(emotion_label)
        emotion_layout.addWidget(self.emotion_chart)
        self.main_layout.addWidget(emotion_frame)

        # Diagnostics
        diag_frame = QFrame()
        diag_frame.setFrameShape(QFrame.StyledPanel)
        diag_layout = QFormLayout(diag_frame)
        self.stress_label = QLabel()
        self.crash_label = QLabel()
        diag_layout.addRow("Stress Level:", self.stress_label)
        diag_layout.addRow("Crash Count:", self.crash_label)
        self.main_layout.addWidget(diag_frame)

        # Thought Log
        log_frame = QFrame()
        log_frame.setFrameShape(QFrame.StyledPanel)
        log_layout = QVBoxLayout(log_frame)
        log_label = QLabel("Recent Thoughts")
        log_label.setFont(QFont("Arial", 9, QFont.Bold))
        self.log_text = QLabel("...")
        self.log_text.setWordWrap(True)
        self.log_text.setAlignment(Qt.AlignTop)
        self.log_text.setMinimumHeight(40)
        log_layout.addWidget(log_label)
        log_layout.addWidget(self.log_text)
        self.main_layout.addWidget(log_frame)

        self.main_layout.addStretch()
        self.setWidget(self.main_widget)
        self.update_dashboard()

    def update_dashboard(self):
        if not self.agent:
            return

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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    main_window = QMainWindow()
    main_window.setStyleSheet("QMainWindow { background-color: #333; } QDockWidget { color: white; background-color: #444; }")

    dummy_agent = DummyDigitalAgent()
    dashboard = VictorDashboard(dummy_agent)
    main_window.addDockWidget(Qt.RightDockWidgetArea, dashboard)

    # Simple timer to show updates
    timer = app.instance().thread().create_timer(100, dashboard.update_dashboard)
    timer.start()

    main_window.setGeometry(100, 100, 300, 600)
    main_window.setWindowTitle("Victor Dashboard Test")
    main_window.show()
    sys.exit(app.exec_())
