from datetime import datetime
from typing import Any, Dict, List, Optional
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.buddy.messages import VALID_EXPRESSIONS, BuddyAdvice, pick_preview_advice
from src.config import save_config
from src.database import DatabaseManager
from src.ui.camera_worker import CameraWorker
from src.ui.cat_canvas import CatBuddyWidget, build_cat_icon
from src.ui.overlay_window import BuddyOverlayWindow
from src.ui.theme import build_stylesheet, get_palette
from src.ui.tray_icon import BuddyTrayIcon


class ScoreTrendChart(QWidget):
    """Grafik garis yang menjawab kapan skor kelelahan melewati ambang Waspada (40) dan Tinggi (70)."""

    def __init__(self, theme_name: str = "light", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._theme = theme_name
        self._series: List[Dict[str, Any]] = []
        self.setMinimumHeight(185)

    def set_theme(self, theme_name: str) -> None:
        self._theme = theme_name
        self.update()

    def set_series(self, series: List[Dict[str, Any]]) -> None:
        self._series = series
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        p = get_palette(self._theme)

        w = float(self.width())
        h = float(self.height())
        pad_l, pad_r, pad_t, pad_b = 46.0, 20.0, 28.0, 28.0
        plot_w = max(10.0, w - pad_l - pad_r)
        plot_h = max(10.0, h - pad_t - pad_b)

        painter.fillRect(QRectF(0, 0, w, h), QColor(p["surface"]))
        painter.setPen(QPen(QColor(p["border"]), 1.2))
        painter.drawRect(QRectF(pad_l, pad_t, plot_w, plot_h))

        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)

        # Draw threshold lines at 40 (Waspada) and 70 (Tinggi)
        for val, color_key, label in (
            (40.0, "waspada", "40 (Waspada)"),
            (70.0, "tinggi", "70 (Tinggi)"),
        ):
            y = pad_t + plot_h * (1.0 - val / 100.0)
            pen_th = QPen(QColor(p[color_key]), 1.4, Qt.DashLine)
            painter.setPen(pen_th)
            painter.drawLine(QPointF(pad_l, y), QPointF(pad_l + plot_w, y))
            painter.setPen(QColor(p["text"]))
            painter.drawText(QRectF(4, y - 10, pad_l - 6, 20), Qt.AlignRight | Qt.AlignVCenter, str(int(val)))
            painter.drawText(QPointF(pad_l + 6, y - 4), label)

        painter.setPen(QColor(p["text"]))
        painter.drawText(QRectF(4, pad_t - 8, pad_l - 6, 18), Qt.AlignRight, "100")
        painter.drawText(QRectF(4, pad_t + plot_h - 10, pad_l - 6, 18), Qt.AlignRight, "0")

        if not self._series:
            painter.end()
            return

        n = len(self._series)
        pts: List[QPointF] = []
        for idx, item in enumerate(self._series):
            score = float(item.get("stress_score", 0.0))
            x = pad_l + (plot_w * idx / max(1, n - 1)) if n > 1 else pad_l + plot_w * 0.5
            y = pad_t + plot_h * (1.0 - max(0.0, min(100.0, score)) / 100.0)
            pts.append(QPointF(x, y))

        line_pen = QPen(QColor(p["accent"]), 2.5)
        painter.setPen(line_pen)
        for i in range(len(pts) - 1):
            painter.drawLine(pts[i], pts[i + 1])

        painter.setBrush(QColor(p["accent"]))
        for pt in pts:
            painter.drawEllipse(pt, 3.2, 3.2)

        first_ts = str(self._series[0].get("timestamp", ""))[-8:]
        last_ts = str(self._series[-1].get("timestamp", ""))[-8:]
        painter.setPen(QColor(p["text_muted"]))
        painter.drawText(QRectF(pad_l, pad_t + plot_h + 4, 120, 20), Qt.AlignLeft, first_ts)
        painter.drawText(QRectF(pad_l + plot_w - 120, pad_t + plot_h + 4, 120, 20), Qt.AlignRight, last_ts)
        painter.end()


class MainWindow(QMainWindow):
    """Jendela utama aplikasi Kucing Teman Kerja."""

    def __init__(
        self,
        config: Dict[str, Any],
        db: DatabaseManager,
        simulate: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.db = db
        self.theme_name = str(config.get("app", {}).get("theme", "light"))
        self._show_camera_preview = True

        self.setWindowTitle("Kucing Teman Kerja | Pemantau Kelelahan & Stres Lokal")
        self.setWindowIcon(build_cat_icon(64))
        self.resize(1020, 720)

        self.overlay = BuddyOverlayWindow(theme_name=self.theme_name)
        self.overlay.action_recorded.connect(self._on_buddy_action_recorded)
        self.overlay.snooze_requested.connect(self._on_snooze_requested)

        self.tray = BuddyTrayIcon(self)
        self.tray.open_dashboard_requested.connect(self._bring_to_front)
        self.tray.toggle_pause_requested.connect(self._toggle_pause)
        self.tray.snooze_requested.connect(self._on_snooze_requested)
        self.tray.preview_buddy_requested.connect(lambda: self.trigger_preview_buddy("istirahat"))
        self.tray.quit_requested.connect(self.close_application)

        self._build_ui()
        self.apply_theme(self.theme_name)

        self.worker = CameraWorker(config=self.config, db=self.db, simulate=simulate, parent=self)
        self.worker.frame_processed.connect(self._on_frame_processed)
        self.worker.calibration_completed.connect(self._on_calibration_completed)
        self.worker.buddy_triggered.connect(self._on_buddy_triggered)
        self.worker.camera_error.connect(self._on_camera_error)
        self.worker.status_changed.connect(self._on_worker_status_changed)
        if not simulate:
            self.worker.open_camera_on_main_thread()
        self.worker.start()

        self.refresh_dashboard()

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Persistent top status and privacy bar
        self.top_bar = QFrame(central)
        self.top_bar.setObjectName("TopStatusBar")
        top_layout = QHBoxLayout(self.top_bar)
        top_layout.setContentsMargins(16, 10, 16, 10)
        top_layout.setSpacing(12)

        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        self.lbl_app_title = QLabel("Kucing Teman Kerja", self.top_bar)
        self.lbl_app_title.setObjectName("HeadingLarge")
        title_col.addWidget(self.lbl_app_title)

        self.lbl_draft_notice = QLabel(
            "Draft without direction (Dial: ENERGY 1 / RHYTHM 1 / MOTION 1) | Pemrosesan Lokal, Video Tidak Disimpan",
            self.top_bar,
        )
        self.lbl_draft_notice.setObjectName("MutedText")
        title_col.addWidget(self.lbl_draft_notice)
        top_layout.addLayout(title_col, stretch=1)

        self.lbl_camera_badge = QLabel("[AKTIF] Kamera Memantau Secara Lokal", self.top_bar)
        self.lbl_camera_badge.setObjectName("HeadingMedium")
        top_layout.addWidget(self.lbl_camera_badge)

        self.btn_pause = QPushButton("Jeda Kamera", self.top_bar)
        self.btn_pause.clicked.connect(self._toggle_pause)
        top_layout.addWidget(self.btn_pause)

        self.btn_recalibrate = QPushButton("Kalibrasi Ulang (10 dtk)", self.top_bar)
        self.btn_recalibrate.clicked.connect(self._request_recalibration)
        top_layout.addWidget(self.btn_recalibrate)

        self.btn_snooze_top = QPushButton("Tunda Buddy 15 Menit", self.top_bar)
        self.btn_snooze_top.clicked.connect(lambda: self._on_snooze_requested(15))
        top_layout.addWidget(self.btn_snooze_top)

        self.btn_theme = QPushButton("Mode Gelap", self.top_bar)
        self.btn_theme.clicked.connect(self.toggle_theme)
        top_layout.addWidget(self.btn_theme)

        root_layout.addWidget(self.top_bar)

        # Main tab widget
        self.tabs = QTabWidget(central)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        root_layout.addWidget(self.tabs, stretch=1)

        self._build_live_tab()
        self._build_dashboard_tab()
        self._build_settings_tab()

    def _build_live_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        upper_row = QHBoxLayout()
        upper_row.setSpacing(16)

        # Left column: Primary focal point (Stress Score & Burnout status)
        score_panel = QFrame(tab)
        score_panel.setObjectName("SurfacePanel")
        score_layout = QVBoxLayout(score_panel)
        score_layout.setContentsMargins(16, 14, 16, 14)
        score_layout.setSpacing(8)

        lbl_focal_header = QLabel("Skor Kelelahan & Stres Saat Ini (Window 60 Detik)", score_panel)
        lbl_focal_header.setObjectName("HeadingMedium")
        score_layout.addWidget(lbl_focal_header)

        self.lbl_score_value = QLabel("0.0 / 100", score_panel)
        self.lbl_score_value.setObjectName("ScoreFocal")
        score_layout.addWidget(self.lbl_score_value)

        self.lbl_stress_level = QLabel("Status: Normal (0 - 39)", score_panel)
        self.lbl_stress_level.setObjectName("HeadingMedium")
        score_layout.addWidget(self.lbl_stress_level)

        self.lbl_burnout_avg = QLabel(
            "Rata-rata Bergerak 15 Menit (Burnout): 0.0 / 100 (Aman)",
            score_panel,
        )
        score_layout.addWidget(self.lbl_burnout_avg)

        self.lbl_dominant_reason = QLabel("Indikator Dominan: Belum ada ketegangan", score_panel)
        self.lbl_dominant_reason.setObjectName("MutedText")
        score_layout.addWidget(self.lbl_dominant_reason)

        self.cal_progress = QProgressBar(score_panel)
        self.cal_progress.setRange(0, 100)
        self.cal_progress.setValue(0)
        self.cal_progress.setFormat("Kalibrasi Baseline Awal: %p%")
        score_layout.addWidget(self.cal_progress)

        self.lbl_baseline_info = QLabel(
            "Baseline Pengguna: EAR 0.280 | Jarak Alis 0.240 | Rasio Postur 0.550",
            score_panel,
        )
        self.lbl_baseline_info.setObjectName("MutedText")
        score_layout.addWidget(self.lbl_baseline_info)

        # Breakdown grid inside the left column
        grid_frame = QFrame(score_panel)
        grid = QGridLayout(grid_frame)
        grid.setContentsMargins(0, 6, 0, 0)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(6)

        self.lbl_ind_ear = QLabel("EAR Mata: 0.280", grid_frame)
        self.lbl_ind_blink = QLabel("Kedipan: 15.0 / menit", grid_frame)
        self.lbl_ind_closed = QLabel("Mata Tertutup (60d): 0%", grid_frame)
        self.lbl_ind_yawn = QLabel("Menguap (60d): 0%", grid_frame)
        self.lbl_ind_brow = QLabel("Kerutan Dahi/Alis (60d): 0%", grid_frame)
        self.lbl_ind_slouch = QLabel("Postur Bungkuk (60d): 0%", grid_frame)
        self.lbl_ind_hand = QLabel("Tangan ke Kepala (60d): 0%", grid_frame)
        self.lbl_ind_jitter = QLabel("Gelisah Kepala: 0%", grid_frame)

        grid.addWidget(self.lbl_ind_ear, 0, 0)
        grid.addWidget(self.lbl_ind_blink, 0, 1)
        grid.addWidget(self.lbl_ind_closed, 1, 0)
        grid.addWidget(self.lbl_ind_yawn, 1, 1)
        grid.addWidget(self.lbl_ind_brow, 2, 0)
        grid.addWidget(self.lbl_ind_slouch, 2, 1)
        grid.addWidget(self.lbl_ind_hand, 3, 0)
        grid.addWidget(self.lbl_ind_jitter, 3, 1)
        score_layout.addWidget(grid_frame)
        score_layout.addStretch(1)

        upper_row.addWidget(score_panel, stretch=5)

        # Right column: Camera preview + performance telemetry
        cam_panel = QFrame(tab)
        cam_panel.setObjectName("SurfacePanel")
        cam_layout = QVBoxLayout(cam_panel)
        cam_layout.setContentsMargins(16, 14, 16, 14)
        cam_layout.setSpacing(8)

        cam_header_row = QHBoxLayout()
        lbl_cam_title = QLabel("Pratinjau Kamera Lokal", cam_panel)
        lbl_cam_title.setObjectName("HeadingMedium")
        cam_header_row.addWidget(lbl_cam_title, stretch=1)

        self.btn_connect_real_cam = QPushButton("Gunakan Kamera Asli", cam_panel)
        self.btn_connect_real_cam.clicked.connect(self._connect_real_camera)
        cam_header_row.addWidget(self.btn_connect_real_cam)

        self.btn_toggle_preview = QPushButton("Sembunyikan Video", cam_panel)
        self.btn_toggle_preview.clicked.connect(self._toggle_camera_preview)
        cam_header_row.addWidget(self.btn_toggle_preview)
        cam_layout.addLayout(cam_header_row)

        self.lbl_video = QLabel("Menyiapkan kamera...", cam_panel)
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setMinimumSize(360, 260)
        self.lbl_video.setObjectName("SurfacePanel")
        cam_layout.addWidget(self.lbl_video, stretch=1)

        self.lbl_perf_info = QLabel(
            "Performa: 0.0 FPS | Latensi: 0.0 ms | Detektor Kustom: Cadangan Landmark",
            cam_panel,
        )
        self.lbl_perf_info.setObjectName("MutedText")
        cam_layout.addWidget(self.lbl_perf_info)

        upper_row.addWidget(cam_panel, stretch=4)
        layout.addLayout(upper_row, stretch=1)

        # Bottom bar: Virtual Buddy interactive preview and status
        buddy_panel = QFrame(tab)
        buddy_panel.setObjectName("SurfacePanel")
        buddy_layout = QHBoxLayout(buddy_panel)
        buddy_layout.setContentsMargins(16, 12, 16, 12)
        buddy_layout.setSpacing(12)

        self.live_cat_widget = CatBuddyWidget(expression="istirahat", size=72, parent=buddy_panel)
        buddy_layout.addWidget(self.live_cat_widget)

        buddy_info_col = QVBoxLayout()
        lbl_buddy_sec = QLabel("Uji Overlay Kucing Virtual Buddy (5 Ekspresi)", buddy_panel)
        lbl_buddy_sec.setObjectName("HeadingMedium")
        buddy_info_col.addWidget(lbl_buddy_sec)

        self.lbl_buddy_status = QLabel(
            "Klik salah satu tombol ekspresi di samping untuk menampilkan teguran di sudut kanan bawah layar.",
            buddy_panel,
        )
        self.lbl_buddy_status.setObjectName("MutedText")
        buddy_info_col.addWidget(self.lbl_buddy_status)
        buddy_layout.addLayout(buddy_info_col, stretch=1)

        for expr in VALID_EXPRESSIONS:
            btn = QPushButton(f"Uji: {expr.capitalize()}", buddy_panel)
            btn.clicked.connect(lambda _checked=False, e=expr: self.trigger_preview_buddy(e))
            buddy_layout.addWidget(btn)

        layout.addWidget(buddy_panel)
        self.tabs.addTab(tab, "Pemantauan Langsung")

    def _build_dashboard_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        header_row = QHBoxLayout()
        lbl_dash_title = QLabel("Evaluasi Harian & Tren Mingguan", tab)
        lbl_dash_title.setObjectName("HeadingMedium")
        header_row.addWidget(lbl_dash_title, stretch=1)

        self.btn_seed_demo = QPushButton("Catat Sampel Evaluasi Sekarang", tab)
        self.btn_seed_demo.clicked.connect(self._record_immediate_sample)
        header_row.addWidget(self.btn_seed_demo)

        self.btn_refresh_dash = QPushButton("Muat Ulang Data", tab)
        self.btn_refresh_dash.setObjectName("PrimaryButton")
        self.btn_refresh_dash.clicked.connect(self.refresh_dashboard)
        header_row.addWidget(self.btn_refresh_dash)
        layout.addLayout(header_row)

        # Stacked widget for Empty, Loading, Error, and Content states (R-27)
        self.dash_stack = QStackedWidget(tab)

        # Page 0: Empty State
        empty_page = QFrame()
        empty_page.setObjectName("SurfacePanel")
        empty_layout = QVBoxLayout(empty_page)
        empty_layout.setContentsMargins(24, 32, 24, 32)
        empty_layout.setSpacing(10)
        lbl_empty_title = QLabel("Belum Ada Riwayat Skor pada Hari Ini", empty_page)
        lbl_empty_title.setObjectName("HeadingMedium")
        empty_layout.addWidget(lbl_empty_title)
        lbl_empty_desc = QLabel(
            "Database lokal belum memiliki log skor untuk tanggal hari ini karena sesi baru saja dimulai. "
            "Biarkan kamera memantau selama 5 detik atau klik tombol di bawah untuk mencatat sampel evaluasi pertama.",
            empty_page,
        )
        lbl_empty_desc.setWordWrap(True)
        empty_layout.addWidget(lbl_empty_desc)
        btn_empty_act = QPushButton("Catat Sampel Evaluasi Pertama", empty_page)
        btn_empty_act.setObjectName("PrimaryButton")
        btn_empty_act.clicked.connect(self._record_immediate_sample)
        empty_layout.addWidget(btn_empty_act, alignment=Qt.AlignLeft)
        empty_layout.addStretch(1)
        self.dash_stack.addWidget(empty_page)

        # Page 1: Loading State
        loading_page = QFrame()
        loading_page.setObjectName("SurfacePanel")
        loading_layout = QVBoxLayout(loading_page)
        loading_layout.setContentsMargins(24, 32, 24, 32)
        lbl_loading = QLabel("Sedang membaca riwayat sesi dan skor dari database SQLite lokal...", loading_page)
        lbl_loading.setObjectName("HeadingMedium")
        loading_layout.addWidget(lbl_loading)
        loading_layout.addStretch(1)
        self.dash_stack.addWidget(loading_page)

        # Page 2: Error State
        error_page = QFrame()
        error_page.setObjectName("SurfacePanel")
        error_layout = QVBoxLayout(error_page)
        error_layout.setContentsMargins(24, 32, 24, 32)
        lbl_err_title = QLabel("Gagal Membaca Database Evaluasi", error_page)
        lbl_err_title.setObjectName("HeadingMedium")
        error_layout.addWidget(lbl_err_title)
        self.lbl_dash_error = QLabel("", error_page)
        self.lbl_dash_error.setWordWrap(True)
        error_layout.addWidget(self.lbl_dash_error)
        btn_retry = QPushButton("Coba Muat Ulang", error_page)
        btn_retry.clicked.connect(self.refresh_dashboard)
        error_layout.addWidget(btn_retry, alignment=Qt.AlignLeft)
        error_layout.addStretch(1)
        self.dash_stack.addWidget(error_page)

        # Page 3: Real Data Content
        content_page = QWidget()
        content_layout = QVBoxLayout(content_page)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(12)

        summary_bar = QFrame(content_page)
        summary_bar.setObjectName("SurfacePanel")
        sum_layout = QHBoxLayout(summary_bar)
        sum_layout.setContentsMargins(14, 10, 14, 10)
        self.lbl_sum_samples = QLabel("Sampel Hari Ini: 0", summary_bar)
        self.lbl_sum_avg = QLabel("Rata-rata Skor: 0.0", summary_bar)
        self.lbl_sum_max = QLabel("Skor Tertinggi: 0.0", summary_bar)
        self.lbl_sum_blink = QLabel("Rata-rata Kedipan: 0.0/mnt", summary_bar)
        self.lbl_sum_buddy = QLabel("Teguran Buddy: 0 kali", summary_bar)
        for w_lbl in (
            self.lbl_sum_samples,
            self.lbl_sum_avg,
            self.lbl_sum_max,
            self.lbl_sum_blink,
            self.lbl_sum_buddy,
        ):
            w_lbl.setObjectName("HeadingMedium")
            sum_layout.addWidget(w_lbl)
        content_layout.addWidget(summary_bar)

        lbl_chart_q = QLabel(
            "Pertanyaan Evaluasi: Kapan skor kelelahan melewati batas Waspada (40) dan Tinggi (70) hari ini?",
            content_page,
        )
        lbl_chart_q.setObjectName("HeadingMedium")
        content_layout.addWidget(lbl_chart_q)

        self.trend_chart = ScoreTrendChart(theme_name=self.theme_name, parent=content_page)
        content_layout.addWidget(self.trend_chart)

        tables_row = QHBoxLayout()
        tables_row.setSpacing(12)

        weekly_col = QVBoxLayout()
        lbl_weekly = QLabel("Ringkasan 7 Hari Terakhir (Perbandingan Harian)", content_page)
        lbl_weekly.setObjectName("HeadingMedium")
        weekly_col.addWidget(lbl_weekly)
        self.tbl_weekly = QTableWidget(0, 4, content_page)
        self.tbl_weekly.setHorizontalHeaderLabels(
            ["Tanggal", "Rata-rata Skor", "Skor Maksimum", "Sampel Tinggi"]
        )
        self.tbl_weekly.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_weekly.setEditTriggers(QTableWidget.NoEditTriggers)
        weekly_col.addWidget(self.tbl_weekly)
        tables_row.addLayout(weekly_col, stretch=1)

        buddy_col = QVBoxLayout()
        lbl_buddy_hist = QLabel("Riwayat Teguran Virtual Buddy Hari Ini", content_page)
        lbl_buddy_hist.setObjectName("HeadingMedium")
        buddy_col.addWidget(lbl_buddy_hist)
        self.tbl_buddy = QTableWidget(0, 4, content_page)
        self.tbl_buddy.setHorizontalHeaderLabels(
            ["Waktu", "Ekspresi", "Pemicu", "Respons Pengguna"]
        )
        self.tbl_buddy.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_buddy.setEditTriggers(QTableWidget.NoEditTriggers)
        buddy_col.addWidget(self.tbl_buddy)
        tables_row.addLayout(buddy_col, stretch=1)

        content_layout.addLayout(tables_row, stretch=1)
        self.dash_stack.addWidget(content_page)

        layout.addWidget(self.dash_stack, stretch=1)
        self.tabs.addTab(tab, "Evaluasi Harian dan Mingguan")

    def _build_settings_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        form_panel = QFrame(tab)
        form_panel.setObjectName("SurfacePanel")
        form_layout = QFormLayout(form_panel)
        form_layout.setContentsMargins(18, 16, 18, 16)
        form_layout.setSpacing(12)

        app_cfg = self.config.get("app", {})
        buddy_cfg = self.config.get("buddy", {})
        models_cfg = self.config.get("models", {})

        self.spin_sensitivity = QDoubleSpinBox(form_panel)
        self.spin_sensitivity.setRange(0.5, 2.0)
        self.spin_sensitivity.setSingleStep(0.1)
        self.spin_sensitivity.setValue(float(app_cfg.get("sensitivity", 1.0)))
        form_layout.addRow("Sensitivitas Skor Stres (0.5x - 2.0x):", self.spin_sensitivity)

        self.spin_cooldown = QSpinBox(form_panel)
        self.spin_cooldown.setRange(15, 1800)
        self.spin_cooldown.setSingleStep(15)
        self.spin_cooldown.setValue(int(buddy_cfg.get("cooldown_seconds", 180)))
        form_layout.addRow("Jeda Cooldown Antar Teguran Buddy (detik):", self.spin_cooldown)

        self.spin_snooze = QSpinBox(form_panel)
        self.spin_snooze.setRange(1, 120)
        self.spin_snooze.setValue(int(buddy_cfg.get("snooze_minutes", 15)))
        form_layout.addRow("Durasi Tunda / Snooze Default (menit):", self.spin_snooze)

        self.chk_active_hours = QCheckBox("Batasi pemantauan hanya pada jam kerja aktif", form_panel)
        self.chk_active_hours.setChecked(bool(app_cfg.get("active_hours_enabled", False)))
        form_layout.addRow("Filter Jam Aktif:", self.chk_active_hours)

        hours_row = QHBoxLayout()
        self.inp_hour_start = QLineEdit(str(app_cfg.get("active_hour_start", "08:00")), form_panel)
        self.inp_hour_start.setPlaceholderText("08:00")
        self.inp_hour_end = QLineEdit(str(app_cfg.get("active_hour_end", "22:00")), form_panel)
        self.inp_hour_end.setPlaceholderText("22:00")
        hours_row.addWidget(QLabel("Mulai (JJ:MM):", form_panel))
        hours_row.addWidget(self.inp_hour_start)
        hours_row.addWidget(QLabel("Selesai (JJ:MM):", form_panel))
        hours_row.addWidget(self.inp_hour_end)
        form_layout.addRow("Rentang Jam Kerja:", hours_row)

        self.chk_two_class = QCheckBox(
            "Gunakan mode 2 kelas (mata_tertutup & menguap) untuk detektor YOLO dan serahkan kerutan_dahi ke jarak alis MediaPipe",
            form_panel,
        )
        self.chk_two_class.setChecked(bool(models_cfg.get("use_two_class_fallback", False)))
        form_layout.addRow("Mode Rencana Cadangan:", self.chk_two_class)

        self.inp_model_path = QLineEdit(
            str(models_cfg.get("face_yolo_path", "models/face_expression_yolov8n.pt")),
            form_panel,
        )
        self.inp_model_path.setPlaceholderText("models/face_expression_yolov8n.pt")
        form_layout.addRow("Lokasi Model YOLOv8 Wajah (.pt / .onnx):", self.inp_model_path)

        btn_save = QPushButton("Simpan Pengaturan", form_panel)
        btn_save.setObjectName("PrimaryButton")
        btn_save.clicked.connect(self._save_settings)
        form_layout.addRow("", btn_save)

        self.lbl_settings_feedback = QLabel("", form_panel)
        self.lbl_settings_feedback.setObjectName("MutedText")
        form_layout.addRow("", self.lbl_settings_feedback)

        layout.addWidget(form_panel)

        # Privacy & Data Deletion Panel
        priv_panel = QFrame(tab)
        priv_panel.setObjectName("SurfacePanel")
        priv_layout = QVBoxLayout(priv_panel)
        priv_layout.setContentsMargins(18, 14, 18, 14)
        priv_layout.setSpacing(8)

        lbl_priv_title = QLabel("Privasi & Penghapusan Data Lokal", priv_panel)
        lbl_priv_title.setObjectName("HeadingMedium")
        priv_layout.addWidget(lbl_priv_title)

        lbl_priv_desc = QLabel(
            "Seluruh pemrosesan webcam dilakukan secara lokal di perangkat ini. Tidak ada gambar atau rekaman video "
            "yang disimpan ke disk. Hanya angka indikator dan skor yang dicatat di data/wellness_buddy.db. "
            "Catatan batasan: skor stres merupakan indikator visual perilaku di depan layar, bukan diagnosis medis.",
            priv_panel,
        )
        lbl_priv_desc.setWordWrap(True)
        priv_layout.addWidget(lbl_priv_desc)

        self.btn_clear_db = QPushButton("Hapus Semua Riwayat Data (SQLite)", priv_panel)
        self.btn_clear_db.setObjectName("DangerButton")
        self.btn_clear_db.clicked.connect(self._clear_all_data)
        priv_layout.addWidget(self.btn_clear_db, alignment=Qt.AlignLeft)

        layout.addWidget(priv_panel)
        layout.addStretch(1)
        self.tabs.addTab(tab, "Pengaturan dan Privasi")

    def apply_theme(self, theme_name: str) -> None:
        self.theme_name = theme_name if theme_name in ("light", "dark") else "light"
        self.setStyleSheet(build_stylesheet(self.theme_name))
        self.overlay.apply_theme(self.theme_name)
        if hasattr(self, "trend_chart"):
            self.trend_chart.set_theme(self.theme_name)
        if hasattr(self, "btn_theme"):
            self.btn_theme.setText("Mode Terang" if self.theme_name == "dark" else "Mode Gelap")

    def toggle_theme(self) -> None:
        next_theme = "dark" if self.theme_name == "light" else "light"
        self.config.setdefault("app", {})["theme"] = next_theme
        save_config(self.config)
        self.apply_theme(next_theme)

    def _toggle_pause(self) -> None:
        paused = not self.worker.is_paused
        self.worker.set_paused(paused)
        self.btn_pause.setText("Lanjutkan Kamera" if paused else "Jeda Kamera")
        self.tray.update_camera_state(paused)

    def _connect_real_camera(self) -> None:
        was_paused = self.worker.is_paused
        self.worker.set_paused(True)
        self.worker._simulate = False
        ok = self.worker.open_camera_on_main_thread()
        if ok:
            self.worker.request_recalibration()
            self.lbl_buddy_status.setText(
                "Kamera fisik berhasil terhubung. Melakukan kalibrasi baseline selama 10 detik."
            )
        self.worker.set_paused(was_paused)

    def _toggle_camera_preview(self) -> None:
        self._show_camera_preview = not self._show_camera_preview
        if self._show_camera_preview:
            self.btn_toggle_preview.setText("Sembunyikan Video")
        else:
            self.btn_toggle_preview.setText("Tampilkan Video")
            self.lbl_video.setPixmap(QPixmap())
            self.lbl_video.setText(
                "Pratinjau video disembunyikan.\nPemantauan indikator tetap berjalan secara lokal."
            )

    def _request_recalibration(self) -> None:
        self.worker.request_recalibration()
        self.lbl_buddy_status.setText(
            "Kalibrasi ulang dimulai. Tatap layar dengan posisi duduk tegak selama 10 detik."
        )

    def _on_snooze_requested(self, minutes: int) -> None:
        self.worker.snooze_buddy(minutes)
        self.lbl_buddy_status.setText(f"Peringatan Virtual Buddy ditunda selama {minutes} menit.")

    def _on_buddy_action_recorded(self, log_id: int, action: str) -> None:
        self.db.update_buddy_action(log_id, action)
        if self.tabs.currentIndex() == 1:
            self.refresh_dashboard()

    def trigger_preview_buddy(self, expression: str) -> None:
        advice = pick_preview_advice(expression)
        self.live_cat_widget.set_expression(advice.expression)
        session_id = self.worker.session_id or 1
        log_id = self.db.log_buddy(
            session_id=session_id,
            expression=advice.expression,
            message=advice.message,
            trigger_reason=advice.trigger_reason,
            user_action="pratinjau_manual",
        )
        dur = int(self.config.get("buddy", {}).get("overlay_duration_seconds", 8))
        snooze_m = int(self.config.get("buddy", {}).get("snooze_minutes", 15))
        self.overlay.show_advice(
            advice,
            duration_seconds=dur,
            snooze_minutes=snooze_m,
            buddy_log_id=log_id,
        )
        self.lbl_buddy_status.setText(
            f"Menampilkan ekspresi '{advice.expression}': {advice.title}"
        )

    def _on_buddy_triggered(self, advice: BuddyAdvice, log_id: int) -> None:
        self.live_cat_widget.set_expression(advice.expression)
        dur = int(self.config.get("buddy", {}).get("overlay_duration_seconds", 8))
        snooze_m = int(self.config.get("buddy", {}).get("snooze_minutes", 15))
        self.overlay.show_advice(
            advice,
            duration_seconds=dur,
            snooze_minutes=snooze_m,
            buddy_log_id=log_id,
        )

    def _on_calibration_completed(self, base: dict) -> None:
        self.lbl_baseline_info.setText(
            f"Baseline Terkalibrasi: EAR {base['ear']:.3f} | "
            f"Jarak Alis {base['brow_ratio']:.3f} | Rasio Postur {base['posture_ratio']:.3f}"
        )

    def _on_camera_error(self, msg: str) -> None:
        self.lbl_buddy_status.setText(msg)

    def _on_worker_status_changed(self, state: str) -> None:
        if state == "paused":
            self.lbl_camera_badge.setText("[DIJEDA] Kamera Nonaktif Sementara")
        elif state == "calibrating":
            self.lbl_camera_badge.setText("[KALIBRASI] Mengukur Baseline Awal Pengguna")
        elif state == "simulation":
            self.lbl_camera_badge.setText("[SIMULASI] Pemantauan Simulasi Lokal Aktif")
        else:
            self.lbl_camera_badge.setText("[AKTIF] Kamera Memantau Secara Lokal")

    def _on_frame_processed(self, qimg: QImage, data: dict) -> None:
        if self._show_camera_preview:
            pix = QPixmap.fromImage(qimg).scaled(
                self.lbl_video.width(),
                self.lbl_video.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.lbl_video.setPixmap(pix)

        score = float(data["stress_score"])
        level = str(data["stress_level"])
        avg_15m = float(data["moving_avg_15m"])
        burnout = bool(data["burnout_detected"])

        self.lbl_score_value.setText(f"{score:.1f} / 100")
        p = get_palette(self.theme_name)
        level_color = (
            p["tinggi"]
            if level == "Tinggi"
            else (p["waspada"] if level == "Waspada" else p["normal"])
        )
        self.lbl_stress_level.setStyleSheet(f"color: {level_color}; font-weight: 700; font-size: 15px;")
        self.lbl_stress_level.setText(f"Level Saat Ini: [{level.upper()}]")

        burnout_label = "PERINGATAN BURNOUT" if burnout else "Aman"
        self.lbl_burnout_avg.setText(
            f"Rata-rata Bergerak 15 Menit (Burnout): {avg_15m:.1f} / 100 ({burnout_label})"
        )
        self.lbl_dominant_reason.setText(
            f"Indikator Dominan: {str(data['dominant_reason']).replace('_', ' ').capitalize()}"
        )

        cal_pct = int(round(float(data["calibration_progress"]) * 100))
        self.cal_progress.setValue(cal_pct)
        if data["calibrating"]:
            self.cal_progress.setFormat("Sedang Kalibrasi Baseline: %p%")
        else:
            self.cal_progress.setFormat("Kalibrasi Baseline Selesai (%p%)")

        self.lbl_ind_ear.setText(f"EAR Mata: {data['ear']:.3f}")
        self.lbl_ind_blink.setText(f"Kedipan: {data['blink_rate']:.1f} / menit")
        self.lbl_ind_closed.setText(f"Mata Tertutup (60d): {data['eye_closed_ratio']*100:.0f}%")
        self.lbl_ind_yawn.setText(f"Menguap (60d): {data['yawn_ratio']*100:.0f}%")
        self.lbl_ind_brow.setText(f"Kerutan Dahi/Alis (60d): {data['brow_furrow_ratio']*100:.0f}%")
        self.lbl_ind_slouch.setText(f"Postur Bungkuk (60d): {data['slouch_ratio']*100:.0f}%")
        self.lbl_ind_hand.setText(f"Tangan ke Kepala (60d): {data['hand_to_head_ratio']*100:.0f}%")
        self.lbl_ind_jitter.setText(f"Gelisah Kepala: {data['head_restlessness']*100:.0f}%")

        yolo_str = "YOLOv8 Kustom Aktif" if data["yolo_custom_loaded"] else "Landmark MediaPipe + YOLOv8-Pose"
        snooze_rem = int(data.get("snooze_remaining", 0))
        snooze_txt = f" | Snooze: {snooze_rem}d" if snooze_rem > 0 else ""
        self.lbl_perf_info.setText(
            f"Performa: {data['fps']:.1f} FPS | Latensi: {data['latency_ms']:.1f} ms | Mode: {yolo_str}{snooze_txt}"
        )

    def _on_tab_changed(self, index: int) -> None:
        if index == 1:
            self.refresh_dashboard()

    def _record_immediate_sample(self) -> None:
        session_id = self.worker.session_id or self.db.start_session(0.28, 0.24, 0.55)
        self.db.log_score(
            session_id=session_id,
            ear_value=0.25,
            blink_rate_per_min=14.0,
            eye_closed_ratio=0.12,
            yawn_ratio=0.0,
            brow_furrow_ratio=0.10,
            slouch_ratio=0.15,
            hand_to_head_ratio=0.0,
            head_restlessness=0.08,
            stress_score=24.5,
            stress_level="Normal",
            moving_avg_15m=24.5,
            burnout_detected=False,
        )
        self.refresh_dashboard()

    def refresh_dashboard(self) -> None:
        self.dash_stack.setCurrentIndex(1)
        QApplication.processEvents()
        try:
            daily = self.db.get_daily_summary()
            weekly = self.db.get_weekly_summary()
        except Exception as exc:
            self.lbl_dash_error.setText(
                f"Terjadi kesalahan saat membaca file database ({exc}). Pastikan file data/wellness_buddy.db tidak terkunci."
            )
            self.dash_stack.setCurrentIndex(2)
            return

        if daily["log_count"] == 0:
            self.dash_stack.setCurrentIndex(0)
            return

        self.lbl_sum_samples.setText(f"Sampel Hari Ini: {daily['log_count']}")
        self.lbl_sum_avg.setText(f"Rata-rata Skor: {daily['avg_score']:.1f}")
        self.lbl_sum_max.setText(f"Skor Tertinggi: {daily['max_score']:.1f}")
        self.lbl_sum_blink.setText(f"Rata-rata Kedipan: {daily['avg_blink']:.1f}/mnt")
        self.lbl_sum_buddy.setText(f"Teguran Buddy: {daily['buddy_triggers']} kali")

        self.trend_chart.set_series(daily["series"])

        self.tbl_weekly.setRowCount(len(weekly))
        for r_idx, row in enumerate(weekly):
            self.tbl_weekly.setItem(r_idx, 0, QTableWidgetItem(str(row["day"])))
            self.tbl_weekly.setItem(r_idx, 1, QTableWidgetItem(f"{row['avg_score']:.1f}"))
            self.tbl_weekly.setItem(r_idx, 2, QTableWidgetItem(f"{row['max_score']:.1f}"))
            self.tbl_weekly.setItem(r_idx, 3, QTableWidgetItem(str(row["high_samples"])))

        buddy_logs = daily["buddy_logs"]
        self.tbl_buddy.setRowCount(len(buddy_logs))
        for r_idx, b_row in enumerate(buddy_logs):
            ts_short = str(b_row["timestamp"])[-8:]
            self.tbl_buddy.setItem(r_idx, 0, QTableWidgetItem(ts_short))
            self.tbl_buddy.setItem(r_idx, 1, QTableWidgetItem(str(b_row["expression"]).capitalize()))
            self.tbl_buddy.setItem(r_idx, 2, QTableWidgetItem(str(b_row["trigger_reason"])))
            self.tbl_buddy.setItem(r_idx, 3, QTableWidgetItem(str(b_row["user_action"])))

        self.dash_stack.setCurrentIndex(3)

    def _save_settings(self) -> None:
        self.config.setdefault("app", {})["sensitivity"] = float(self.spin_sensitivity.value())
        self.config["app"]["active_hours_enabled"] = bool(self.chk_active_hours.isChecked())
        self.config["app"]["active_hour_start"] = self.inp_hour_start.text().strip() or "08:00"
        self.config["app"]["active_hour_end"] = self.inp_hour_end.text().strip() or "22:00"

        self.config.setdefault("buddy", {})["cooldown_seconds"] = int(self.spin_cooldown.value())
        self.config["buddy"]["snooze_minutes"] = int(self.spin_snooze.value())

        self.config.setdefault("models", {})["use_two_class_fallback"] = bool(
            self.chk_two_class.isChecked()
        )
        self.config["models"]["face_yolo_path"] = (
            self.inp_model_path.text().strip() or "models/face_expression_yolov8n.pt"
        )

        save_config(self.config)
        self.worker.update_live_config(self.config)
        now_str = datetime.now().strftime("%H:%M:%S")
        self.lbl_settings_feedback.setText(
            f"Pengaturan berhasil disimpan ke config.yaml dan diterapkan pada {now_str}."
        )

    def _clear_all_data(self) -> None:
        reply = QMessageBox.question(
            self,
            "Konfirmasi Hapus Data",
            "Apakah Anda yakin ingin menghapus seluruh riwayat sesi, log skor, dan riwayat Buddy dari database lokal?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.db.clear_all_data()
            self.refresh_dashboard()
            self.lbl_settings_feedback.setText(
                "Seluruh riwayat data di SQLite telah dihapus."
            )

    def _bring_to_front(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def close_application(self) -> None:
        self.overlay.close()
        self.worker.stop_worker()
        self.tray.tray.hide()
        QApplication.quit()

    def closeEvent(self, event) -> None:
        self.overlay.close()
        self.worker.stop_worker()
        self.tray.tray.hide()
        event.accept()
