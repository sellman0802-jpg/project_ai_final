#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
๐ฟ Plant Vision โ€” Coral Micro Object Detection (+ MQTT)
เนเธญเธเธ•เธฃเธงเธเธเธฑเธเธงเธฑเธ•เธ–เธธเนเธเธเน€เธฃเธตเธขเธฅเนเธ—เธกเน เธเธตเธกเธ”เธณ-เน€เธเธตเธขเธง (Dark Green Theme)

เธ”เธฑเธ”เนเธเธฅเธเธเธฒเธเธชเธเธฃเธดเธเธ•เน OpenCV เน€เธ”เธดเธก เนเธซเนเน€เธเนเธ UI เธ—เธตเนเธชเธงเธขเธเธฒเธกเธ”เนเธงเธข PyQt5
- เน€เธเธทเนเธญเธกเธ•เนเธญ Coral Micro เธเนเธฒเธ HTTP (เธ เธฒเธ + bounding boxes) เธ—เธตเน http://10.10.10.1
- เธชเนเธเธเธฅเธ•เธฃเธงเธเธเธฑเธเธเธถเนเธ MQTT broker (เธชเธ–เธฒเธเธฐ normal/wither, เธเธณเธเธงเธ, alert)
- เธ—เธณเธเธฒเธเธเธเน€เธเธฃเธ”เนเธขเธ เนเธกเนเธ—เธณเนเธซเนเธซเธเนเธฒเธ•เนเธฒเธเธเนเธฒเธ
- เธงเธฒเธ”เธเธฃเธญเธเธ”เนเธงเธข QPainter (เธกเธธเธกเน€เธเนเธ, เธเนเธฒเธขเธเธทเนเธญเนเธเธเธเธดเธ) เธ”เธนเธฃเธตเน€เธกเธตเธขเธก
- เนเธเธเธชเธ–เธดเธ•เธด: FPS, เธเธณเธเธงเธเธงเธฑเธ•เธ–เธธ, เธชเธ–เธฒเธเธฐเธเธฒเธฃเน€เธเธทเนเธญเธกเธ•เนเธญ (เธเธฅเนเธญเธ + MQTT)
- เธเธฃเธฑเธ IP เนเธฅเธฐเธเนเธฒเธเธงเธฒเธกเธกเธฑเนเธเนเธ (confidence) เนเธ”เน

เธ•เธดเธ”เธ•เธฑเนเธเนเธฅเธเธฃเธฒเธฃเธต:
    pip install PyQt5 opencv-python requests numpy paho-mqtt

เธงเธดเธเธตเธฃเธฑเธ:
    python plant_detector.py
"""
import matplotlib.pyplot as plt
import sys
import time
import json

import cv2
import numpy as np
import requests

# paho-mqtt เน€เธเนเธ optional: เธ–เนเธฒเนเธกเนเธกเธตเธเนเธขเธฑเธเธฃเธฑเธเนเธญเธเนเธ”เน (เธเธธเนเธก MQTT เธเธฐเธ–เธนเธเธเธดเธ”)
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

from PyQt5.QtCore import Qt, QThread, QObject, pyqtSignal, QRectF, QTimer
from PyQt5.QtGui import (
    QImage, QPainter, QColor, QPen, QBrush, QFont, QFontMetrics, QPainterPath
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QLineEdit,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QSlider, QScrollArea,
    QSizePolicy, QGraphicsDropShadowEffect
)

# ----------------------------------------------------------------------------
# เธเนเธฒเธ•เธฑเนเธเธ•เนเธ (Configuration)
# ----------------------------------------------------------------------------
DEFAULT_IP = "10.10.10.1"           # Coral dev board (เธเธฅเนเธญเธ + bboxes)

# --- MQTT ---
DEFAULT_MQTT_HOST     = "10.0.90.119"  # broker (เนเธเนเนเธ”เนเนเธเธซเธเนเธฒเธเธญ)
DEFAULT_MQTT_PORT     = 1883
MQTT_BASE_TOPIC       = "plant_vision"
MQTT_PUBLISH_INTERVAL = 1.0           # เธงเธดเธเธฒเธ—เธต: เธชเนเธเธเธฅเธ—เธธเธ เน เธเธตเนเธงเธดเธเธฒเธ—เธต (เธเธฑเธเธขเธดเธเธ–เธตเนเน€เธเธดเธเนเธ)
WITHER_CLASS_ID       = 2             # id เธเธญเธเธเธฅเธฒเธช "wither" เธชเธณเธซเธฃเธฑเธเธขเธดเธ alert



# client.publish("plant_vision/count", "3")
# เนเธกเธเธ—เธตเนเธเธทเนเธญเธเธฅเธฒเธช -> เธเธทเนเธญเนเธซเนเธ•เธฃเธเธเธฑเธเนเธกเน€เธ”เธฅเธเธญเธเธเธธเธ“เนเธ”เนเน€เธฅเธข
# (เธ–เนเธฒ id เนเธซเธเนเธกเนเธกเธตเนเธเธเธตเน เธเธฐเนเธชเธ”เธเน€เธเนเธ "Object N")
CLASS_NAMES = {
    0: "เธ•เนเธ (Plant)",
    1: "normal",
    2: "wither",
}

# ----------------------------------------------------------------------------
# เธเธธเธ”เธชเธต เธ”เธณ-เน€เธเธตเธขเธง (Dark Green Palette)
# ----------------------------------------------------------------------------
BG_DARK     = "#0a0e0a"
BG_PANEL    = "#101810"
BG_CARD     = "#162116"
BORDER      = "#1f2e20"
ACCENT      = "#2ecc71"
ACCENT_DIM  = "#1f8f4e"
ACCENT_GLOW = "#4eff9f"
TEXT        = "#e8f5e9"
TEXT_MUTED  = "#7a9a7e"
DANGER      = "#ff5252"


# ============================================================================
# เน€เธเธฃเธ”เธ—เธณเธเธฒเธเน€เธเธทเนเธญเธกเธ•เนเธญเธซเธฅเธฑเธ: เธ”เธถเธเธ เธฒเธ + เธเธฃเธญเธ เธเธฒเธ Coral Micro
# ============================================================================
class DetectionWorker(QThread):
    frame_ready   = pyqtSignal(object, list)   # (frame BGR, detections)
    status_changed = pyqtSignal(str, bool)     # (เธเนเธญเธเธงเธฒเธก, เน€เธเธทเนเธญเธกเธ•เนเธญเธชเธณเน€เธฃเนเธ?)

    def __init__(self, coral_ip):
        super().__init__()
        self.coral_ip = coral_ip
        self._running = False

    def run(self):
        self._running = True
        url_bboxes = f"http://{self.coral_ip}/bboxes"
        url_camera = f"http://{self.coral_ip}/camera_stream"
        was_connected = False

        while self._running:
            try:
                res_bboxes = requests.get(url_bboxes, timeout=2.0)
                res_camera = requests.get(url_camera, timeout=2.0)

                img_array = np.frombuffer(res_camera.content, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if frame is None:
                    continue

                height, width = frame.shape[:2]
                detections = []

                try:
                    bbox_data = res_bboxes.json()
                    for box in bbox_data.get("bboxes", []):
                        detections.append({
                            "xmin": int(box["xmin"] * width),
                            "ymin": int(box["ymin"] * height),
                            "xmax": int(box["xmax"] * width),
                            "ymax": int(box["ymax"] * height),
                            "score": float(box.get("score", 0.0)),
                            "id": int(box.get("id", -1)),
                        })
                except ValueError:
                    # JSON เน€เธชเธตเธขเธซเธฒเธข -> เธเนเธฒเธกเน€เธเธฃเธกเธเธตเน (เธขเธฑเธเนเธชเธ”เธเธ เธฒเธเนเธ”เน)
                    pass

                if not was_connected:
                    self.status_changed.emit("เน€เธเธทเนเธญเธกเธ•เนเธญเนเธฅเนเธง", True)
                    was_connected = True

                self.frame_ready.emit(frame, detections)

            except requests.exceptions.RequestException:
                if was_connected:
                    was_connected = False
                self.status_changed.emit("เธเธณเธฅเธฑเธเน€เธเธทเนเธญเธกเธ•เนเธญโ€ฆ", False)
                time.sleep(1.0)

    def stop(self):
        self._running = False
        self.wait(2500)


# ============================================================================
# เธ•เธฑเธงเธเธฑเธ”เธเธฒเธฃ MQTT (publish เธเธฅเธ•เธฃเธงเธเธเธฑเธเธเธถเนเธ broker)
# ============================================================================
class MqttPublisher(QObject):
    """
    เธซเนเธญเธซเธธเนเธก paho-mqtt เนเธงเนเนเธ QObject เน€เธเธทเนเธญเธชเนเธเธชเธฑเธเธเธฒเธ“เธชเธ–เธฒเธเธฐเธเธฅเธฑเธเธกเธฒเธ—เธตเน UI
    paho เธเธฐเธฃเธฑเธ network loop เธเธเน€เธเธฃเธ”เธเธญเธเธ•เธฑเธงเน€เธญเธ (loop_start) เธเธถเธเนเธกเนเธเธฅเนเธญเธ GUI

    เธซเธฑเธงเธเนเธญ (topic) เธ—เธตเนเนเธเน:
        {base}/status      -> JSON เธชเธฃเธธเธเธเธฅเธ—เธฑเนเธเธซเธกเธ” (timestamp, fps, total, counts, detections)
        {base}/count       -> เธเธณเธเธงเธเธงเธฑเธ•เธ–เธธเธ—เธตเนเธ•เธฃเธงเธเธเธ (เธ•เธฑเธงเน€เธฅเธเธฅเนเธงเธ)
        {base}/alert       -> เธชเนเธเน€เธกเธทเนเธญเธเธ "wither" (เธชเธ–เธฒเธเธฐเธ•เนเธเน€เธซเธตเนเธขเธง)
    """
    connection_changed = pyqtSignal(bool, str)   # (เน€เธเธทเนเธญเธกเธ•เนเธญ?, เธเนเธญเธเธงเธฒเธก)

    def __init__(self, base_topic=MQTT_BASE_TOPIC):
        super().__init__()
        self.client = None
        self.base_topic = base_topic
        self._connected = False
        self._host = None
        self._port = None

    # ---- เธเธฑเธ”เธเธฒเธฃเธเธฒเธฃเน€เธเธทเนเธญเธกเธ•เนเธญ ----
    def connect_to(self, host, port=DEFAULT_MQTT_PORT, username=None, password=None):
        if not MQTT_AVAILABLE:
            self.connection_changed.emit(False, "เนเธกเนเธเธ paho-mqtt")
            return

        # เธฃเธญเธเธฃเธฑเธเธ—เธฑเนเธ paho v1 เนเธฅเธฐ v2
        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
        except (AttributeError, TypeError):
            self.client = mqtt.Client()

        if username:
            self.client.username_pw_set(username, password or None)

        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

        self._host, self._port = host, port
        # Last Will: เนเธเนเธ broker เธงเนเธฒเธญเธญเธเนเธฅเธเน เธ–เนเธฒเธซเธฅเธธเธ”เธเธฐเธ—เธฑเธเธซเธฑเธ
        self.client.will_set(f"{self.base_topic}/online", payload="0",
                             qos=1, retain=True)

        try:
            self.connection_changed.emit(False, "เธเธณเธฅเธฑเธเน€เธเธทเนเธญเธกเธ•เนเธญโ€ฆ")
            self.client.connect_async(host, int(port), keepalive=30)
            self.client.loop_start()
        except Exception as e:
            self.connection_changed.emit(False, f"เธเธดเธ”เธเธฅเธฒเธ”: {e}")

    def disconnect(self):
        if self.client is not None:
            try:
                # เนเธเนเธเธญเธญเธเนเธฅเธเนเธเนเธญเธเธเธดเธ”
                self.client.publish(f"{self.base_topic}/online", "0",
                                    qos=1, retain=True)
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:
                pass
        self._connected = False
        self.client = None
        self.connection_changed.emit(False, "เธ•เธฑเธ”เธเธฒเธฃเน€เธเธทเนเธญเธกเธ•เนเธญ")

    @property
    def is_connected(self):
        return self._connected

    # ---- callbacks เธเธญเธ paho (เธฃเธฑเธเธเธเน€เธเธฃเธ” network เธเธญเธ paho) ----
    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            client.publish(f"{self.base_topic}/online", "1", qos=1, retain=True)
            self.connection_changed.emit(True, "MQTT เธเธฃเนเธญเธก")
        else:
            self._connected = False
            self.connection_changed.emit(False, f"เธเธเธดเน€เธชเธ (rc={rc})")

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        self.connection_changed.emit(False, "MQTT เธซเธฅเธธเธ”")

    # ---- เธชเนเธเธเนเธญเธกเธนเธฅ ----
    def publish_detections(self, detections, fps=0.0):
        """เธชเนเธเธเธฅเธ•เธฃเธงเธเธเธฑเธเธเธถเนเธ broker (เน€เธฃเธตเธขเธเนเธเธ throttle เธเธฒเธ MainWindow)"""
        if not (self.client and self._connected):
            return

        # เธเธฑเธเธเธณเธเธงเธเนเธขเธเธ•เธฒเธกเธเธฅเธฒเธช
        counts = {}
        for d in detections:
            name = CLASS_NAMES.get(d["id"], f"Object {d['id']}")
            counts[name] = counts.get(name, 0) + 1

        payload = {
            "timestamp": time.time(),
            "fps": round(fps, 1),
            "total": len(detections),
            "counts": counts,
            "detections": [
                {
                    "id": d["id"],
                    "name": CLASS_NAMES.get(d["id"], f"Object {d['id']}"),
                    "score": round(d["score"], 3),
                    "bbox": [d["xmin"], d["ymin"], d["xmax"], d["ymax"]],
                }
                for d in detections
            ],
        }

        try:
            self.client.publish(f"{self.base_topic}/status",
                                json.dumps(payload, ensure_ascii=False), qos=0)
            self.client.publish(f"{self.base_topic}/count",
                                str(len(detections)), qos=0)

            # เนเธเนเธเน€เธ•เธทเธญเธเน€เธกเธทเนเธญเธเธเธ•เนเธเน€เธซเธตเนเธขเธง
            withers = [d for d in detections if d["id"] == WITHER_CLASS_ID]
            if withers:
                top = max(withers, key=lambda d: d["score"])
                alert = {
                    "timestamp": time.time(),
                    "status": "wither",
                    "count": len(withers),
                    "max_score": round(top["score"], 3),
                }
                self.client.publish(f"{self.base_topic}/alert",
                                    json.dumps(alert, ensure_ascii=False), qos=1)
        except Exception:
            pass


# ============================================================================
# เธงเธดเธ”เน€เธเนเธ•เนเธชเธ”เธเธงเธดเธ”เธตเนเธญ + เธงเธฒเธ”เธเธฃเธญเธเธ”เนเธงเธข QPainter (เธชเธงเธขเธเธงเนเธฒ cv2.rectangle)
# ============================================================================
class VideoWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(720, 540)
        self._image = None
        self._detections = []
        self._threshold = 0.30
        self.setStyleSheet("background-color: #000000; border-radius: 14px;")

    def set_threshold(self, value):
        self._threshold = value

    def update_frame(self, frame_bgr, detections):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        self._src_w, self._src_h = w, h
        # เน€เธเนเธ buffer เนเธงเนเธเธฑเธ GC เน€เธเนเธเธเนเธญเธเธกเธนเธฅเธ เธฒเธเธงเธฒเธ”
        self._buffer = np.ascontiguousarray(rgb)
        self._image = QImage(self._buffer.data, w, h, 3 * w, QImage.Format_RGB888)
        self._detections = detections
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)

        rect = self.rect()
        # เธเธทเนเธเธซเธฅเธฑเธเธงเนเธฒเธ
        p.fillRect(rect, QColor("#000000"))

        if self._image is None:
            p.setPen(QColor(TEXT_MUTED))
            p.setFont(QFont("Segoe UI", 13))
            p.drawText(rect, Qt.AlignCenter,
                       "๐ฟ  เธฃเธญเธชเธฑเธเธเธฒเธ“เธ เธฒเธเธเธฒเธเธเธฅเนเธญเธโ€ฆ\nPress Connect to start")
            return

        # เธเธณเธเธงเธ“เธเธฒเธฃเธเธญเธ”เธตเธ เธฒเธเนเธเธ letterbox (เธฃเธฑเธเธฉเธฒเธชเธฑเธ”เธชเนเธงเธ)
        iw, ih = self._src_w, self._src_h
        scale = min(rect.width() / iw, rect.height() / ih)
        dw, dh = iw * scale, ih * scale
        ox = (rect.width() - dw) / 2
        oy = (rect.height() - dh) / 2
        target = QRectF(ox, oy, dw, dh)
        p.drawImage(target, self._image)

        # เธงเธฒเธ”เธเธฃเธญเธเธ•เธฃเธงเธเธเธฑเธ
        font = QFont("Segoe UI Semibold", 10)
        p.setFont(font)
        fm = QFontMetrics(font)

        for det in self._detections:
            if det["score"] < self._threshold:
                continue

            x1 = ox + det["xmin"] * scale
            y1 = oy + det["ymin"] * scale
            x2 = ox + det["xmax"] * scale
            y2 = oy + det["ymax"] * scale
            box = QRectF(x1, y1, x2 - x1, y2 - y1)

            # เนเธเนเธชเธตเนเธ”เธเธชเธณเธซเธฃเธฑเธเธ•เนเธเน€เธซเธตเนเธขเธง เน€เธเธทเนเธญเนเธซเนเน€เธซเนเธเธเธฑเธ”
            base_color = DANGER if det["id"] == WITHER_CLASS_ID else ACCENT

            # เน€เธฃเธทเธญเธเนเธชเธเธฃเธญเธเธเธฃเธญเธ
            glow = QColor(ACCENT_GLOW if det["id"] != WITHER_CLASS_ID else DANGER)
            glow.setAlpha(60)
            p.setPen(QPen(glow, 6))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(box, 8, 8)

            # เธเธฃเธญเธเธซเธฅเธฑเธ
            p.setPen(QPen(QColor(base_color), 2.2))
            p.drawRoundedRect(box, 8, 8)

            # เธกเธธเธกเน€เธเนเธ (corner accents)
            self._draw_corners(p, box, color=base_color)

            # เธเนเธฒเธขเธเธทเนเธญ
            name = CLASS_NAMES.get(det["id"], f"Object {det['id']}")
            label = f"{name}  {det['score']*100:.0f}%"
            tw = fm.width(label) + 16
            th = fm.height() + 6
            ly = max(oy, y1 - th - 2)
            chip = QRectF(x1, ly, tw, th)

            path = QPainterPath()
            path.addRoundedRect(chip, 6, 6)
            chip_bg = QColor(base_color)
            chip_bg.setAlpha(235)
            p.fillPath(path, chip_bg)

            p.setPen(QColor("#06160c"))
            p.drawText(chip.adjusted(8, 0, 0, 0),
                       Qt.AlignVCenter | Qt.AlignLeft, label)
            print(CLASS_NAMES.get(det["id"], f"Object {det['id']}"))
    def _draw_corners(self, p, box, length=14, color=ACCENT_GLOW):
        p.setPen(QPen(QColor(color), 3))
        x, y, w, h = box.x(), box.y(), box.width(), box.height()
        # เธเธเธเนเธฒเธข
        p.drawLine(int(x), int(y), int(x + length), int(y))
        p.drawLine(int(x), int(y), int(x), int(y + length))
        # เธเธเธเธงเธฒ
        p.drawLine(int(x + w), int(y), int(x + w - length), int(y))
        p.drawLine(int(x + w), int(y), int(x + w), int(y + length))
        # เธฅเนเธฒเธเธเนเธฒเธข
        p.drawLine(int(x), int(y + h), int(x + length), int(y + h))
        p.drawLine(int(x), int(y + h), int(x), int(y + h - length))
        # เธฅเนเธฒเธเธเธงเธฒ
        p.drawLine(int(x + w), int(y + h), int(x + w - length), int(y + h))
        p.drawLine(int(x + w), int(y + h), int(x + w), int(y + h - length))


# ============================================================================
# เธเธฒเธฃเนเธ”เธชเธ–เธดเธ•เธดเน€เธฅเนเธ เน
# ============================================================================
class StatCard(QFrame):
    def __init__(self, title, value="โ€”"):
        super().__init__()
        self.setObjectName("statCard")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(2)

        self.title_lbl = QLabel(title)
        self.title_lbl.setObjectName("statTitle")
        self.value_lbl = QLabel(value)
        self.value_lbl.setObjectName("statValue")

        lay.addWidget(self.title_lbl)
        lay.addWidget(self.value_lbl)

    def set_value(self, value):
        self.value_lbl.setText(str(value))


# ============================================================================
# เธซเธเนเธฒเธ•เนเธฒเธเธซเธฅเธฑเธ
# ============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("๐ฟ Plant Vision โ€” Coral Micro Detection")
        self.resize(1180, 820)

        self.worker = None
        self._frame_count = 0
        self._fps = 0.0
        self._last_fps_time = time.time()

        # MQTT
        self.mqtt = MqttPublisher(MQTT_BASE_TOPIC)
        self.mqtt.connection_changed.connect(self._on_mqtt_status)
        self._mqtt_online = False
        self._last_mqtt_publish = 0.0

        self._build_ui()
        self._apply_styles()

        # เธ•เธฑเธงเธเธฑเธ FPS
        self._fps_timer = QTimer(self)
        self._fps_timer.timeout.connect(self._tick_fps)
        self._fps_timer.start(1000)

    # ---- เนเธเธฃเธเธชเธฃเนเธฒเธ UI ----
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(14)

        # === Header ===
        header = QHBoxLayout()
        title = QLabel("๐ฟ  PLANT VISION")
        title.setObjectName("appTitle")
        subtitle = QLabel("เธฃเธฐเธเธเธ•เธฃเธงเธเธเธฑเธเธงเธฑเธ•เธ–เธธเนเธเธเน€เธฃเธตเธขเธฅเนเธ—เธกเน ยท Coral Micro ยท MQTT")
        subtitle.setObjectName("appSubtitle")

        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        # เธชเธ–เธฒเธเธฐเธเธฅเนเธญเธ + เธชเธ–เธฒเธเธฐ MQTT (เธชเธญเธเธเนเธฒเธข)
        self.status_pill = QLabel("โ— เธญเธญเธเนเธฅเธเน")
        self.status_pill.setObjectName("statusOffline")
        self.status_pill.setAlignment(Qt.AlignCenter)

        self.mqtt_pill = QLabel("โ— MQTT: เธเธดเธ”")
        self.mqtt_pill.setObjectName("statusOffline")
        self.mqtt_pill.setAlignment(Qt.AlignCenter)

        header.addLayout(title_box)
        header.addStretch()
        header.addWidget(self.mqtt_pill)
        header.addWidget(self.status_pill)
        root.addLayout(header)

        # === Body: เธงเธดเธ”เธตเนเธญ (เธเนเธฒเธข) + เนเธเธเธเนเธฒเธ (เธเธงเธฒ) ===
        body = QHBoxLayout()
        body.setSpacing(14)

        self.video = VideoWidget()
        video_wrap = QFrame()
        video_wrap.setObjectName("videoWrap")
        vw_lay = QVBoxLayout(video_wrap)
        vw_lay.setContentsMargins(6, 6, 6, 6)
        vw_lay.addWidget(self.video)
        body.addWidget(video_wrap, 3)

        # เนเธเธเธเนเธฒเธ
        side = QVBoxLayout()
        side.setSpacing(12)

        stats_row = QHBoxLayout()
        self.card_fps = StatCard("FPS", "0")
        self.card_obj = StatCard("เธงเธฑเธ•เธ–เธธเธ—เธตเนเธเธ", "0")
        stats_row.addWidget(self.card_fps)
        stats_row.addWidget(self.card_obj)
        side.addLayout(stats_row)

        det_label = QLabel("เธฃเธฒเธขเธเธฒเธฃเธ•เธฃเธงเธเธเธฑเธ")
        det_label.setObjectName("sectionLabel")
        side.addWidget(det_label)

        self.det_scroll = QScrollArea()
        self.det_scroll.setObjectName("detScroll")
        self.det_scroll.setWidgetResizable(True)
        self.det_container = QWidget()
        self.det_layout = QVBoxLayout(self.det_container)
        self.det_layout.setContentsMargins(0, 0, 0, 0)
        self.det_layout.setSpacing(8)
        self.det_layout.addStretch()
        self.det_scroll.setWidget(self.det_container)
        side.addWidget(self.det_scroll, 1)

        side_wrap = QFrame()
        side_wrap.setObjectName("sidePanel")
        side_wrap.setLayout(side)
        side_wrap.setFixedWidth(330)
        body.addWidget(side_wrap)

        root.addLayout(body, 1)

        # === Control Bar (เธเธฅเนเธญเธ) ===
        controls = QFrame()
        controls.setObjectName("controlBar")
        c_lay = QHBoxLayout(controls)
        c_lay.setContentsMargins(16, 12, 16, 12)
        c_lay.setSpacing(12)

        ip_label = QLabel("Coral IP")
        ip_label.setObjectName("ctrlLabel")
        self.ip_input = QLineEdit(DEFAULT_IP)
        self.ip_input.setObjectName("ipInput")
        self.ip_input.setFixedWidth(140)

        self.connect_btn = QPushButton("โ–ถ  เน€เธเธทเนเธญเธกเธ•เนเธญ")
        self.connect_btn.setObjectName("connectBtn")
        self.connect_btn.setCursor(Qt.PointingHandCursor)
        self.connect_btn.clicked.connect(self._toggle_connection)

        conf_label = QLabel("เธเธงเธฒเธกเธกเธฑเนเธเนเธ โฅ")
        conf_label.setObjectName("ctrlLabel")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(30)
        self.conf_slider.setFixedWidth(160)
        self.conf_slider.valueChanged.connect(self._on_threshold)
        self.conf_value = QLabel("30%")
        self.conf_value.setObjectName("ctrlValue")

        c_lay.addWidget(ip_label)
        c_lay.addWidget(self.ip_input)
        c_lay.addWidget(self.connect_btn)
        c_lay.addStretch()
        c_lay.addWidget(conf_label)
        c_lay.addWidget(self.conf_slider)
        c_lay.addWidget(self.conf_value)

        root.addWidget(controls)

        # === MQTT Bar ===
        mqtt_bar = QFrame()
        mqtt_bar.setObjectName("controlBar")
        m_lay = QHBoxLayout(mqtt_bar)
        m_lay.setContentsMargins(16, 12, 16, 12)
        m_lay.setSpacing(12)

        mq_label = QLabel("MQTT Broker")
        mq_label.setObjectName("ctrlLabel")
        self.mqtt_host_input = QLineEdit(DEFAULT_MQTT_HOST)
        self.mqtt_host_input.setObjectName("ipInput")
        self.mqtt_host_input.setFixedWidth(140)

        port_label = QLabel("Port")
        port_label.setObjectName("ctrlLabel")
        self.mqtt_port_input = QLineEdit(str(DEFAULT_MQTT_PORT))
        self.mqtt_port_input.setObjectName("ipInput")
        self.mqtt_port_input.setFixedWidth(70)

        topic_label = QLabel("Topic")
        topic_label.setObjectName("ctrlLabel")
        self.mqtt_topic_input = QLineEdit(MQTT_BASE_TOPIC)
        self.mqtt_topic_input.setObjectName("ipInput")
        self.mqtt_topic_input.setFixedWidth(130)

        self.mqtt_btn = QPushButton("โ—  เน€เธเธทเนเธญเธก MQTT")
        self.mqtt_btn.setObjectName("connectBtn")
        self.mqtt_btn.setCursor(Qt.PointingHandCursor)
        self.mqtt_btn.clicked.connect(self._toggle_mqtt)

        m_lay.addWidget(mq_label)
        m_lay.addWidget(self.mqtt_host_input)
        m_lay.addWidget(port_label)
        m_lay.addWidget(self.mqtt_port_input)
        m_lay.addWidget(topic_label)
        m_lay.addWidget(self.mqtt_topic_input)
        m_lay.addStretch()
        m_lay.addWidget(self.mqtt_btn)

        root.addWidget(mqtt_bar)

        # เธ–เนเธฒเนเธกเนเธกเธต paho-mqtt เธเธดเธ”เธเธฒเธฃเนเธเนเธเธฒเธเธเธธเนเธก MQTT
        if not MQTT_AVAILABLE:
            self.mqtt_btn.setEnabled(False)
            self.mqtt_btn.setText("โ—  เนเธกเนเธเธ paho-mqtt")
            self.mqtt_host_input.setEnabled(False)
            self.mqtt_port_input.setEnabled(False)
            self.mqtt_topic_input.setEnabled(False)

    # ---- เธชเนเธ•เธฅเนเธเธตเธ• ----
    def _apply_styles(self):
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {BG_DARK}; }}
            QWidget {{ color: {TEXT}; font-family: 'Segoe UI', 'Tahoma', sans-serif; }}

            #appTitle {{ font-size: 22px; font-weight: 800; color: {ACCENT}; letter-spacing: 1px; }}
            #appSubtitle {{ font-size: 12px; color: {TEXT_MUTED}; }}

            #statusOffline {{ background-color: #2a1818; color: {DANGER};
                padding: 6px 16px; border-radius: 14px; font-weight: 600;
                border: 1px solid #4a2222; }}
            #statusOnline {{ background-color: #10301c; color: {ACCENT_GLOW};
                padding: 6px 16px; border-radius: 14px; font-weight: 600;
                border: 1px solid {ACCENT_DIM}; }}

            #videoWrap {{ background-color: {BG_PANEL}; border: 1px solid {BORDER};
                border-radius: 16px; }}
            #sidePanel {{ background-color: {BG_PANEL}; border: 1px solid {BORDER};
                border-radius: 16px; padding: 14px; }}

            #statCard {{ background-color: {BG_CARD}; border: 1px solid {BORDER};
                border-radius: 12px; }}
            #statTitle {{ color: {TEXT_MUTED}; font-size: 11px; }}
            #statValue {{ color: {ACCENT}; font-size: 26px; font-weight: 800; }}

            #sectionLabel {{ color: {TEXT_MUTED}; font-size: 12px; font-weight: 700;
                letter-spacing: 1px; padding-top: 4px; }}

            #detScroll {{ background: transparent; border: none; }}
            QScrollBar:vertical {{ background: {BG_DARK}; width: 8px; border-radius: 4px; }}
            QScrollBar::handle:vertical {{ background: {ACCENT_DIM}; border-radius: 4px; min-height: 24px; }}
            QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}

            #detItem {{ background-color: {BG_CARD}; border: 1px solid {BORDER};
                border-left: 3px solid {ACCENT}; border-radius: 8px; }}
            #detName {{ font-size: 13px; font-weight: 600; color: {TEXT}; }}
            #detScore {{ font-size: 12px; color: {ACCENT_GLOW}; font-weight: 700; }}

            #controlBar {{ background-color: {BG_PANEL}; border: 1px solid {BORDER};
                border-radius: 14px; }}
            #ctrlLabel {{ color: {TEXT_MUTED}; font-size: 13px; }}
            #ctrlValue {{ color: {ACCENT}; font-size: 13px; font-weight: 700; min-width: 40px; }}

            #ipInput {{ background-color: {BG_CARD}; border: 1px solid {BORDER};
                border-radius: 8px; padding: 8px 12px; color: {TEXT}; font-size: 13px; }}
            #ipInput:focus {{ border: 1px solid {ACCENT}; }}

            #connectBtn {{ background-color: {ACCENT}; color: #06160c; border: none;
                border-radius: 8px; padding: 9px 22px; font-weight: 800; font-size: 13px; }}
            #connectBtn:hover {{ background-color: {ACCENT_GLOW}; }}
            #connectBtn:pressed {{ background-color: {ACCENT_DIM}; }}
            #connectBtn:disabled {{ background-color: {BG_CARD}; color: {TEXT_MUTED}; }}

            QSlider::groove:horizontal {{ height: 5px; background: {BG_CARD};
                border-radius: 3px; }}
            QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 3px; }}
            QSlider::handle:horizontal {{ background: {ACCENT_GLOW}; width: 16px;
                margin: -6px 0; border-radius: 8px; }}
        """)

        # เน€เธเธฒเธเธธเนเธกเนเธซเนเธเธฅเนเธญเธเธงเธดเธ”เธตเนเธญ
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(40)
        shadow.setColor(QColor(0, 0, 0, 160))
        shadow.setOffset(0, 8)
        self.centralWidget().findChild(QFrame, "videoWrap").setGraphicsEffect(shadow)

    # ---- เธเธธเนเธก / เธ•เธฑเธงเธเธงเธเธเธธเธก (เธเธฅเนเธญเธ) ----
    def _toggle_connection(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker = None
            self.connect_btn.setText("โ–ถ  เน€เธเธทเนเธญเธกเธ•เนเธญ")
            self._set_status("เธญเธญเธเนเธฅเธเน", False)
        else:
            ip = self.ip_input.text().strip() or DEFAULT_IP
            self.worker = DetectionWorker(ip)
            self.worker.frame_ready.connect(self._on_frame)
            self.worker.status_changed.connect(self._set_status)
            self.worker.start()
            self.connect_btn.setText("โ–    เธซเธขเธธเธ”")
            self._set_status("เธเธณเธฅเธฑเธเน€เธเธทเนเธญเธกเธ•เนเธญโ€ฆ", False)

    # ---- เธเธธเนเธก / เธ•เธฑเธงเธเธงเธเธเธธเธก (MQTT) ----
    def _toggle_mqtt(self):
        if self._mqtt_online or (self.mqtt.client is not None):
            self.mqtt.disconnect()
            self.mqtt_btn.setText("โ—  เน€เธเธทเนเธญเธก MQTT")
        else:
            host = self.mqtt_host_input.text().strip() or DEFAULT_MQTT_HOST
            try:
                port = int(self.mqtt_port_input.text().strip())
            except ValueError:
                port = DEFAULT_MQTT_PORT
                self.mqtt_port_input.setText(str(port))
            topic = self.mqtt_topic_input.text().strip() or MQTT_BASE_TOPIC
            self.mqtt.base_topic = topic
            self.mqtt.connect_to(host, port)
            self.mqtt_btn.setText("โ—  เธ•เธฑเธ” MQTT")

    def _on_mqtt_status(self, online, text):
        self._mqtt_online = online
        if online:
            self.mqtt_pill.setText("โ— MQTT: เน€เธเธทเนเธญเธกเธ•เนเธญ")
            self.mqtt_pill.setObjectName("statusOnline")
            self.mqtt_btn.setText("โ—  เธ•เธฑเธ” MQTT")
        else:
            self.mqtt_pill.setText(f"โ— {text}")
            self.mqtt_pill.setObjectName("statusOffline")
        self.mqtt_pill.style().unpolish(self.mqtt_pill)
        self.mqtt_pill.style().polish(self.mqtt_pill)

    def _on_threshold(self, value):
        self.conf_value.setText(f"{value}%")
        self.video.set_threshold(value / 100.0)

    # ---- เธฃเธฑเธเธเนเธญเธกเธนเธฅเธ เธฒเธเน€เธเธฃเธ” ----
    def _on_frame(self, frame, detections):
        self.video.update_frame(frame, detections)
        self._frame_count += 1

        thr = self.conf_slider.value() / 100.0
        visible = [d for d in detections if d["score"] >= thr]
        self.card_obj.set_value(len(visible))
        self._refresh_detection_list(visible)

        # เธชเนเธเธเธถเนเธ MQTT เนเธเธ throttle
        now = time.time()
        if self._mqtt_online and (now - self._last_mqtt_publish) >= MQTT_PUBLISH_INTERVAL:
            self.mqtt.publish_detections(visible, fps=self._fps)
            self._last_mqtt_publish = now

    def _refresh_detection_list(self, detections):
        # เธฅเนเธฒเธเธฃเธฒเธขเธเธฒเธฃเน€เธ”เธดเธก
        while self.det_layout.count() > 1:
            item = self.det_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        for det in sorted(detections, key=lambda d: d["score"], reverse=True):
            name = CLASS_NAMES.get(det["id"], f"Object {det['id']}")
            self.det_layout.insertWidget(
                self.det_layout.count() - 1,
                self._make_det_item(name, det["score"])
            )

    def _make_det_item(self, name, score):
        item = QFrame()
        item.setObjectName("detItem")
        lay = QHBoxLayout(item)
        lay.setContentsMargins(12, 10, 12, 10)

        name_lbl = QLabel(f"๐ฑ  {name}")
        name_lbl.setObjectName("detName")
        score_lbl = QLabel(f"{score*100:.0f}%")
        score_lbl.setObjectName("detScore")

        lay.addWidget(name_lbl)
        lay.addStretch()
        lay.addWidget(score_lbl)
        return item

    def _set_status(self, text, online):
        if online:
            self.status_pill.setText("โ— เธญเธญเธเนเธฅเธเน")
            self.status_pill.setObjectName("statusOnline")
        else:
            self.status_pill.setText(f"โ— {text}")
            self.status_pill.setObjectName("statusOffline")
        # เธเธฑเธเธเธฑเธเนเธซเนเธชเนเธ•เธฅเนเนเธซเธกเนเธกเธตเธเธฅ
        self.status_pill.style().unpolish(self.status_pill)
        self.status_pill.style().polish(self.status_pill)

    def _tick_fps(self):
        now = time.time()
        elapsed = now - self._last_fps_time
        if elapsed > 0:
            self._fps = self._frame_count / elapsed
        self.card_fps.set_value(f"{self._fps:.0f}")
        self._frame_count = 0
        self._last_fps_time = now

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
        if self.mqtt:
            self.mqtt.disconnect()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
