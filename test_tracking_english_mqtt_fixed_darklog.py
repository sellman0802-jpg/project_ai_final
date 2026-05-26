#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plant Vision - Coral Micro Object Detection (+ MQTT)
Real-time PyQt5 dashboard for Coral Micro camera streaming, bounding boxes,
MQTT publishing, and detection tracking/hold to reduce flickering boxes.

Install:
    pip install PyQt5 opencv-python requests numpy paho-mqtt

Run:
    python test_tracking.py
"""
import matplotlib.pyplot as plt
import sys
import time
import json

import cv2
import numpy as np
import requests

# paho-mqtt is optional. If unavailable, the MQTT controls are disabled.
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
# Default configuration
# ----------------------------------------------------------------------------
DEFAULT_IP = "10.10.10.1"           # Coral dev board camera and bbox endpoint

# --- MQTT ---
DEFAULT_MQTT_HOST     = "10.0.90.119"  # MQTT broker address
DEFAULT_MQTT_PORT     = 1883
MQTT_BASE_TOPIC       = "plant_vision"
MQTT_PUBLISH_INTERVAL = 1.0           # Seconds between MQTT publish events
WITHER_CLASS_ID       = 2             # Class id used as the wither alert class

# --- Detection tracking / hold bounding boxes ---
# Used when AI confidence is low or bounding boxes flicker.
# The last valid box is held briefly so the UI and MQTT can read it reliably.
TRACKING_ENABLED       = True
TRACK_HOLD_SECONDS     = 2.0     # Seconds to keep a box after it disappears
TRACK_IOU_THRESHOLD    = 0.25    # Minimum IoU needed to match a new box to an old track
TRACK_MIN_RAW_SCORE    = 0.05    # Minimum raw detection score accepted by the tracker
TRACK_MAX_CENTER_SHIFT = 0.45    # Maximum normalized center shift allowed for matching




# client.publish("plant_vision/count", "3")
# Map model class ids to display names.
# Unknown ids are displayed as "Object N".
CLASS_NAMES = {
    0: "Plant",
    1: "normal",
    2: "wither",
}

# ----------------------------------------------------------------------------
# Dark green color palette
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
# Worker thread: fetch camera frames and bounding boxes from Coral Micro
# ============================================================================
class DetectionWorker(QThread):
    frame_ready   = pyqtSignal(object, list)   # (frame BGR, detections)
    status_changed = pyqtSignal(str, bool)     # (message, connected?)

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
                    # Invalid JSON: skip bboxes for this frame but still show the camera image.
                    pass

                if not was_connected:
                    self.status_changed.emit("Connected", True)
                    was_connected = True

                self.frame_ready.emit(frame, detections)

            except requests.exceptions.RequestException:
                if was_connected:
                    was_connected = False
                self.status_changed.emit("Connecting...", False)
                time.sleep(1.0)

    def stop(self):
        self._running = False
        self.wait(2500)



# ============================================================================
# Bounding-box Tracker / Hold
# ============================================================================
def _bbox_iou(a, b):
    """Compute IoU between two bbox dictionaries with xmin, ymin, xmax, ymax."""
    ax1, ay1, ax2, ay2 = a["xmin"], a["ymin"], a["xmax"], a["ymax"]
    bx1, by1, bx2, by2 = b["xmin"], b["ymin"], b["xmax"], b["ymax"]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _center_distance_ratio(track_box, det_box):
    """Center distance divided by previous bbox size; prevents wrong cross-object matches."""
    tcx = (track_box["xmin"] + track_box["xmax"]) / 2.0
    tcy = (track_box["ymin"] + track_box["ymax"]) / 2.0
    dcx = (det_box["xmin"] + det_box["xmax"]) / 2.0
    dcy = (det_box["ymin"] + det_box["ymax"]) / 2.0
    tw = max(1.0, track_box["xmax"] - track_box["xmin"])
    th = max(1.0, track_box["ymax"] - track_box["ymin"])
    scale = max(tw, th)
    return (((tcx - dcx) ** 2 + (tcy - dcy) ** 2) ** 0.5) / scale


class DetectionTracker:
    """
    Lightweight tracker that temporarily holds bounding boxes.
    - Matches by class id, IoU, and center distance.
    - If a new frame has no box, the old box is output until TRACK_HOLD_SECONDS expires.
    - Helps MQTT avoid missing short flickers or low-confidence frames.
    """
    def __init__(self, hold_seconds=TRACK_HOLD_SECONDS):
        self.hold_seconds = hold_seconds
        self.tracks = []
        self._next_track_id = 1

    def reset(self):
        self.tracks.clear()
        self._next_track_id = 1

    def update(self, detections):
        now = time.time()

        # Drop extremely low-confidence raw detections before tracking.
        detections = [dict(d) for d in detections if d.get("score", 0.0) >= TRACK_MIN_RAW_SCORE]

        matched_track_indexes = set()
        matched_detection_indexes = set()

        # 1) Match new detections to existing tracks.
        for det_index, det in enumerate(detections):
            best_track_index = None
            best_iou = 0.0

            for track_index, track in enumerate(self.tracks):
                if track_index in matched_track_indexes:
                    continue
                if track["id"] != det["id"]:
                    continue

                iou = _bbox_iou(track, det)
                center_shift = _center_distance_ratio(track, det)

                if iou >= TRACK_IOU_THRESHOLD and center_shift <= TRACK_MAX_CENTER_SHIFT:
                    if iou > best_iou:
                        best_iou = iou
                        best_track_index = track_index

            if best_track_index is not None:
                track = self.tracks[best_track_index]

                # Smooth bbox coordinates to reduce small visual jitter.
                alpha = 0.65
                for key in ("xmin", "ymin", "xmax", "ymax"):
                    track[key] = int(alpha * det[key] + (1.0 - alpha) * track[key])

                # Use the latest score but decay the previous score so tracks do not drop too easily.
                track["score"] = max(float(det.get("score", 0.0)), float(track.get("score", 0.0)) * 0.92)
                track["last_seen"] = now
                track["missed"] = 0
                track["held"] = False

                matched_track_indexes.add(best_track_index)
                matched_detection_indexes.add(det_index)

        # 2) Create new tracks for unmatched detections.
        for det_index, det in enumerate(detections):
            if det_index in matched_detection_indexes:
                continue
            det = dict(det)
            det["track_id"] = self._next_track_id
            det["last_seen"] = now
            det["created_at"] = now
            det["missed"] = 0
            det["held"] = False
            self._next_track_id += 1
            self.tracks.append(det)

        # 3) Mark tracks that were not updated as held.
        for track_index, track in enumerate(self.tracks):
            if track_index not in matched_track_indexes:
                # Do not mark a track as missed if it was just created this frame.
                if track.get("last_seen", now) < now:
                    track["missed"] = int(track.get("missed", 0)) + 1
                    track["held"] = True

        # 4) Remove tracks that have been missing for longer than the hold time.
        alive = []
        for track in self.tracks:
            age_since_seen = now - float(track.get("last_seen", now))
            if age_since_seen <= self.hold_seconds:
                alive.append(track)
        self.tracks = alive

        # 5) Return the detection list used by the UI and MQTT.
        output = []
        for track in self.tracks:
            out = {
                "xmin": int(track["xmin"]),
                "ymin": int(track["ymin"]),
                "xmax": int(track["xmax"]),
                "ymax": int(track["ymax"]),
                "score": float(track.get("score", 0.0)),
                "id": int(track.get("id", -1)),
                "track_id": int(track.get("track_id", -1)),
                "held": bool(track.get("held", False)),
            }
            output.append(out)

        return output


# ============================================================================
# MQTT publisher for detection results
# ============================================================================
class MqttPublisher(QObject):
    """
    MQTT wrapper around paho-mqtt for sending connection-state signals back to the UI.
    paho runs its own network loop with loop_start(), so it does not block the GUI.

    Topics used:
        {base}/status      -> JSON summary: timestamp, fps, total, counts, detections
        {base}/count       -> Plain number of visible detections
        {base}/alert       -> Published when the wither class is detected
    """
    connection_changed = pyqtSignal(bool, str)   # (connected?, message)

    def __init__(self, base_topic=MQTT_BASE_TOPIC):
        super().__init__()
        self.client = None
        self.base_topic = base_topic
        self._connected = False
        self._host = None
        self._port = None

    # ---- Connection management ----
    def connect_to(self, host, port=DEFAULT_MQTT_PORT, username=None, password=None):
        if not MQTT_AVAILABLE:
            self.connection_changed.emit(False, "paho-mqtt not installed")
            return

        # Support both paho-mqtt v1 and v2 callback APIs.
        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
        except (AttributeError, TypeError):
            self.client = mqtt.Client()

        if username:
            self.client.username_pw_set(username, password or None)

        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

        self._host, self._port = host, port
        # Last Will: tells the broker this client is offline if it disconnects unexpectedly.
        self.client.will_set(f"{self.base_topic}/online", payload="0",
                             qos=1, retain=True)

        try:
            # Use synchronous connect here to match the older working behavior more closely.
            # It also reports connection errors immediately instead of silently waiting.
            self.connection_changed.emit(False, "Connecting...")
            self.client.connect(host, int(port), keepalive=30)
            self.client.loop_start()
        except Exception as e:
            self._connected = False
            self.connection_changed.emit(False, f"Error: {e}")

    def disconnect(self):
        if self.client is not None:
            try:
                # Publish offline state before closing.
                self.client.publish(f"{self.base_topic}/online", "0",
                                    qos=1, retain=True)
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:
                pass
        self._connected = False
        self.client = None
        self.connection_changed.emit(False, "Disconnected")

    @property
    def is_connected(self):
        return self._connected

    # ---- paho callbacks; these run on paho network thread ----
    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            client.publish(f"{self.base_topic}/online", "1", qos=1, retain=True)
            self.connection_changed.emit(True, "MQTT ready")
        else:
            self._connected = False
            self.connection_changed.emit(False, f"Rejected (rc={rc})")

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        self.connection_changed.emit(False, "MQTT disconnected")

    # ---- Publish data ----
    def publish_detections(self, detections, fps=0.0):
        """Publish detection results to the broker. Called with throttling from MainWindow."""
        if not (self.client and self._connected):
            return

        # Count detections by class name.
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
                    "track_id": d.get("track_id"),
                    "held": bool(d.get("held", False)),
                }
                for d in detections
            ],
        }

        try:
            self.client.publish(f"{self.base_topic}/status",
                                json.dumps(payload, ensure_ascii=False), qos=0)
            self.client.publish(f"{self.base_topic}/count",
                                str(len(detections)), qos=0)

            # Send an alert when the wither class is detected.
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
# Video widget: display frames and draw bounding boxes with QPainter
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
        # Keep the buffer alive so QImage can safely reference it.
        self._buffer = np.ascontiguousarray(rgb)
        self._image = QImage(self._buffer.data, w, h, 3 * w, QImage.Format_RGB888)
        self._detections = detections
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)

        rect = self.rect()
        # Empty background.
        p.fillRect(rect, QColor("#000000"))

        if self._image is None:
            p.setPen(QColor(TEXT_MUTED))
            p.setFont(QFont("Segoe UI", 13))
            p.drawText(rect, Qt.AlignCenter,
                       "Waiting for camera stream...\nPress Connect to start")
            return

        # Compute letterbox fit while preserving aspect ratio.
        iw, ih = self._src_w, self._src_h
        scale = min(rect.width() / iw, rect.height() / ih)
        dw, dh = iw * scale, ih * scale
        ox = (rect.width() - dw) / 2
        oy = (rect.height() - dh) / 2
        target = QRectF(ox, oy, dw, dh)
        p.drawImage(target, self._image)

        # Draw detection boxes.
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

            # Use red for the wither class for better visibility.
            base_color = DANGER if det["id"] == WITHER_CLASS_ID else ACCENT

            # Outer glow.
            glow = QColor(ACCENT_GLOW if det["id"] != WITHER_CLASS_ID else DANGER)
            glow.setAlpha(60)
            p.setPen(QPen(glow, 6))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(box, 8, 8)

            # Main box.
            p.setPen(QPen(QColor(base_color), 2.2))
            p.drawRoundedRect(box, 8, 8)

            # Corner accents.
            self._draw_corners(p, box, color=base_color)

            # Label chip.
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
        # Top-left corner.
        p.drawLine(int(x), int(y), int(x + length), int(y))
        p.drawLine(int(x), int(y), int(x), int(y + length))
        # Top-right corner.
        p.drawLine(int(x + w), int(y), int(x + w - length), int(y))
        p.drawLine(int(x + w), int(y), int(x + w), int(y + length))
        # Bottom-left corner.
        p.drawLine(int(x), int(y + h), int(x + length), int(y + h))
        p.drawLine(int(x), int(y + h), int(x), int(y + h - length))
        # Bottom-right corner.
        p.drawLine(int(x + w), int(y + h), int(x + w - length), int(y + h))
        p.drawLine(int(x + w), int(y + h), int(x + w), int(y + h - length))


# ============================================================================
# Small statistic card widget
# ============================================================================
class StatCard(QFrame):
    def __init__(self, title, value="-"):
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
# Main application window
# ============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plant Vision - Coral Micro Detection")
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

        # Hold and track bounding boxes to reduce flicker and help MQTT read stable values.
        self.tracker = DetectionTracker(TRACK_HOLD_SECONDS)

        self._build_ui()
        self._apply_styles()

        # FPS timer.
        self._fps_timer = QTimer(self)
        self._fps_timer.timeout.connect(self._tick_fps)
        self._fps_timer.start(1000)

    # ---- Build UI ----
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(14)

        # === Header ===
        header = QHBoxLayout()
        title = QLabel("PLANT VISION")
        title.setObjectName("appTitle")
        subtitle = QLabel("Real-time object detection system - Coral Micro - MQTT")
        subtitle.setObjectName("appSubtitle")

        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        # Camera and MQTT status badges.
        self.status_pill = QLabel("OFFLINE")
        self.status_pill.setObjectName("statusOffline")
        self.status_pill.setAlignment(Qt.AlignCenter)

        self.mqtt_pill = QLabel("MQTT: OFF")
        self.mqtt_pill.setObjectName("statusOffline")
        self.mqtt_pill.setAlignment(Qt.AlignCenter)

        header.addLayout(title_box)
        header.addStretch()
        header.addWidget(self.mqtt_pill)
        header.addWidget(self.status_pill)
        root.addLayout(header)

        # === Body: video on the left, side panel on the right ===
        body = QHBoxLayout()
        body.setSpacing(14)

        self.video = VideoWidget()
        video_wrap = QFrame()
        video_wrap.setObjectName("videoWrap")
        vw_lay = QVBoxLayout(video_wrap)
        vw_lay.setContentsMargins(6, 6, 6, 6)
        vw_lay.addWidget(self.video)
        body.addWidget(video_wrap, 3)

        # Side panel.
        side = QVBoxLayout()
        side.setSpacing(12)

        stats_row = QHBoxLayout()
        self.card_fps = StatCard("FPS", "0")
        self.card_obj = StatCard("Objects", "0")
        stats_row.addWidget(self.card_fps)
        stats_row.addWidget(self.card_obj)
        side.addLayout(stats_row)

        det_label = QLabel("Detections")
        det_label.setObjectName("sectionLabel")
        side.addWidget(det_label)

        self.det_scroll = QScrollArea()
        self.det_scroll.setObjectName("detScroll")
        self.det_scroll.setWidgetResizable(True)
        self.det_scroll.setStyleSheet("""
            QScrollArea {
                background-color: #000000;
                border: none;
            }
            QScrollArea > QWidget > QWidget {
                background-color: #000000;
                color: #e8f5e9;
            }
        """)
        self.det_container = QWidget()
        self.det_container.setObjectName("detContainer")
        self.det_container.setStyleSheet("background-color: #000000; color: #e8f5e9;")
        self.det_scroll.viewport().setStyleSheet("background-color: #000000;")
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

        # === Camera control bar ===
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

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setObjectName("connectBtn")
        self.connect_btn.setCursor(Qt.PointingHandCursor)
        self.connect_btn.clicked.connect(self._toggle_connection)

        conf_label = QLabel("Confidence >=")
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

        self.mqtt_btn = QPushButton("Connect MQTT")
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

        # Disable MQTT controls if paho-mqtt is not installed.
        if not MQTT_AVAILABLE:
            self.mqtt_btn.setEnabled(False)
            self.mqtt_btn.setText("paho-mqtt not installed")
            self.mqtt_host_input.setEnabled(False)
            self.mqtt_port_input.setEnabled(False)
            self.mqtt_topic_input.setEnabled(False)

    # ---- Style sheet ----
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

            #detScroll {{ background-color: #000000; border: none; color: {TEXT}; }}
            #detScroll QWidget {{ background-color: #000000; color: {TEXT}; }}
            #detScroll QViewport {{ background-color: #000000; }}
            #detContainer {{ background-color: #000000; color: {TEXT}; }}
            QScrollArea {{ background-color: #000000; color: {TEXT}; }}
            QScrollArea QWidget {{ background-color: #000000; color: {TEXT}; }}
            QScrollBar:vertical {{ background: #050805; width: 8px; border-radius: 4px; }}
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

        # Soft shadow for the video frame.
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(40)
        shadow.setColor(QColor(0, 0, 0, 160))
        shadow.setOffset(0, 8)
        self.centralWidget().findChild(QFrame, "videoWrap").setGraphicsEffect(shadow)

    # ---- Camera controls ----
    def _toggle_connection(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker = None
            self.connect_btn.setText("Connect")
            self._set_status("Offline", False)
        else:
            # Reset old tracks when starting a new connection session.
            self.tracker.reset()
            ip = self.ip_input.text().strip() or DEFAULT_IP
            self.worker = DetectionWorker(ip)
            self.worker.frame_ready.connect(self._on_frame)
            self.worker.status_changed.connect(self._set_status)
            self.worker.start()
            self.connect_btn.setText("Stop")
            self._set_status("Connecting...", False)

    # ---- MQTT controls ----
    def _toggle_mqtt(self):
        if self._mqtt_online or (self.mqtt.client is not None):
            self.mqtt.disconnect()
            self.mqtt_btn.setText("Connect MQTT")
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
            self.mqtt_btn.setText("Disconnect MQTT")

    def _on_mqtt_status(self, online, text):
        self._mqtt_online = online
        if online:
            self.mqtt_pill.setText("MQTT: Connected")
            self.mqtt_pill.setObjectName("statusOnline")
            self.mqtt_btn.setText("Disconnect MQTT")
        else:
            self.mqtt_pill.setText(f"MQTT: {text}")
            self.mqtt_pill.setObjectName("statusOffline")
        self.mqtt_pill.style().unpolish(self.mqtt_pill)
        self.mqtt_pill.style().polish(self.mqtt_pill)

    def _on_threshold(self, value):
        self.conf_value.setText(f"{value}%")
        self.video.set_threshold(value / 100.0)

    # ---- Frame handler ----
    def _on_frame(self, frame, detections):
        # Keep recent boxes briefly to prevent flickering or disappearing boxes.
        # raw detections = Coral results for this frame.
        # tracked_detections = matched and held results for UI/MQTT.
        if TRACKING_ENABLED:
            detections = self.tracker.update(detections)

        self.video.update_frame(frame, detections)
        self._frame_count += 1

        thr = self.conf_slider.value() / 100.0
        visible = [d for d in detections if d["score"] >= thr]
        self.card_obj.set_value(len(visible))
        self._refresh_detection_list(visible)

        # Publish tracked boxes to MQTT with throttling.
        now = time.time()
        if self._mqtt_online and (now - self._last_mqtt_publish) >= MQTT_PUBLISH_INTERVAL:
            self.mqtt.publish_detections(visible, fps=self._fps)
            self._last_mqtt_publish = now

    def _refresh_detection_list(self, detections):
        # Clear previous list items.
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

        name_lbl = QLabel(f"{name}")
        name_lbl.setObjectName("detName")
        score_lbl = QLabel(f"{score*100:.0f}%")
        score_lbl.setObjectName("detScore")

        lay.addWidget(name_lbl)
        lay.addStretch()
        lay.addWidget(score_lbl)
        return item

    def _set_status(self, text, online):
        if online:
            self.status_pill.setText("ONLINE")
            self.status_pill.setObjectName("statusOnline")
        else:
            self.status_pill.setText(f"{text}")
            self.status_pill.setObjectName("statusOffline")
        # Force Qt to re-apply the style after changing objectName.
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
