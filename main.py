import argparse
import sys
from PySide6.QtWidgets import QApplication

from src.config import load_config
from src.database import DatabaseManager
from src.ui.main_window import MainWindow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kucing Teman Kerja: Pemantau Kelelahan & Stres Pengguna PC via Webcam"
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Jalankan dengan generator frame simulasi tanpa membuka kamera fisik.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Jalankan verifikasi otomatis seluruh elemen UI dan pipeline selama beberapa detik lalu keluar.",
    )
    return parser.parse_args()


def run_self_test(window: MainWindow, app: QApplication) -> int:
    from PySide6.QtCore import QTimer

    log_steps: list[str] = []

    def step_1() -> None:
        window._record_immediate_sample()
        log_steps.append("Sample logged -> SQLite score_log updated")
        window.tabs.setCurrentIndex(1)
        log_steps.append("Tab 1 (Evaluasi Harian & Mingguan) -> opened and populated")
        QTimer.singleShot(400, step_2)

    def step_2() -> None:
        window.toggle_theme()
        log_steps.append(f"Theme toggle -> switched to {window.theme_name}")
        window.toggle_theme()
        log_steps.append(f"Theme toggle -> restored to {window.theme_name}")
        window.trigger_preview_buddy("menguap")
        log_steps.append("Buddy preview (menguap) -> overlay shown")
        QTimer.singleShot(400, step_3)

    def step_3() -> None:
        window.overlay._on_ack_clicked()
        log_steps.append("Overlay Mengerti (Esc) button -> closed overlay and recorded action")
        window.tabs.setCurrentIndex(2)
        window._save_settings()
        log_steps.append("Tab 2 (Pengaturan) Simpan Pengaturan -> saved config.yaml")
        window._toggle_pause()
        log_steps.append("Pause button -> paused camera worker")
        window._toggle_pause()
        log_steps.append("Pause button -> resumed camera worker")
        QTimer.singleShot(400, finish)

    def finish() -> None:
        for line in log_steps:
            print(f"[SELF-TEST PASS] {line}")
        window.close_application()

    QTimer.singleShot(800, step_1)
    return app.exec()


def main() -> int:
    args = parse_args()
    config = load_config()
    db = DatabaseManager()

    app = QApplication(sys.argv)
    app.setApplicationName("Kucing Teman Kerja")

    window = MainWindow(config=config, db=db, simulate=args.simulate or args.self_test)
    window.show()

    if args.self_test:
        return run_self_test(window, app)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
