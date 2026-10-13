"""
Modern Dark Theme Styling for MangaColorizer.
Clean, high-contrast, polished interface with accent highlights.
"""

DARK_THEME_QSS = """
QMainWindow {
    background-color: #0f1117;
    color: #e2e8f0;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
}

QWidget {
    color: #cbd5e1;
    font-size: 13px;
}

/* ScrollBars */
QScrollBar:vertical {
    border: none;
    background: #181b26;
    width: 8px;
    margin: 0px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #334155;
    min-height: 20px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover {
    background: #475569;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    border: none;
    background: #181b26;
    height: 8px;
    margin: 0px;
    border-radius: 4px;
}
QScrollBar::handle:horizontal {
    background: #334155;
    min-width: 20px;
    border-radius: 4px;
}
QScrollBar::handle:horizontal:hover {
    background: #475569;
}

/* Cards & Panels */
QFrame.CardFrame {
    background-color: #171923;
    border: 1px solid #282d3f;
    border-radius: 8px;
    padding: 12px;
}

QGroupBox {
    background-color: #171923;
    border: 1px solid #282d3f;
    border-radius: 8px;
    margin-top: 18px;
    font-weight: 600;
    font-size: 13px;
    color: #f1f5f9;
    padding-top: 14px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 4px;
    color: #a5b4fc;
}

/* Buttons */
QPushButton {
    background-color: #242938;
    color: #f8fafc;
    border: 1px solid #3b4256;
    border-radius: 6px;
    padding: 7px 14px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #2d3347;
    border-color: #6366f1;
}
QPushButton:pressed {
    background-color: #1e2230;
}
QPushButton:disabled {
    background-color: #161822;
    color: #64748b;
    border-color: #222634;
}

/* Primary Action Button */
QPushButton#PrimaryBtn {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4f46e5, stop:1 #6366f1);
    color: #ffffff;
    border: 1px solid #6366f1;
    font-size: 14px;
    font-weight: 600;
    border-radius: 6px;
    padding: 9px 20px;
}
QPushButton#PrimaryBtn:hover {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4338ca, stop:1 #4f46e5);
    border-color: #818cf8;
}
QPushButton#PrimaryBtn:pressed {
    background-color: #3730a3;
}
QPushButton#PrimaryBtn:disabled {
    background-color: #2e3047;
    color: #64748b;
    border-color: #3b3f5c;
}

/* Stop Button */
QPushButton#StopBtn {
    background-color: #3f1d24;
    color: #fca5a5;
    border: 1px solid #7f1d1d;
    font-weight: 600;
}
QPushButton#StopBtn:hover {
    background-color: #991b1b;
    color: #ffffff;
    border-color: #ef4444;
}

/* Secondary Button */
QPushButton#SecondaryBtn {
    background-color: #1e293b;
    border: 1px solid #334155;
    color: #cbd5e1;
}
QPushButton#SecondaryBtn:hover {
    background-color: #334155;
    border-color: #475569;
}

/* Input Fields */
QLineEdit {
    background-color: #13151f;
    border: 1px solid #2d3246;
    border-radius: 6px;
    padding: 6px 10px;
    color: #f8fafc;
}
QLineEdit:focus {
    border: 1px solid #6366f1;
}

/* Combobox */
QComboBox {
    background-color: #13151f;
    border: 1px solid #2d3246;
    border-radius: 6px;
    padding: 6px 10px;
    color: #f8fafc;
}
QComboBox:hover {
    border-color: #4f46e5;
}
QComboBox::drop-down {
    border: none;
    width: 24px;
}
QComboBox QAbstractItemView {
    background-color: #1a1d29;
    border: 1px solid #334155;
    color: #f1f5f9;
    selection-background-color: #4f46e5;
    selection-color: #ffffff;
}

/* Checkbox */
QCheckBox {
    spacing: 8px;
    color: #cbd5e1;
}
QCheckBox::indicator {
    width: 17px;
    height: 17px;
    border-radius: 4px;
    border: 1px solid #3b4256;
    background-color: #13151f;
}
QCheckBox::indicator:hover {
    border-color: #6366f1;
}
QCheckBox::indicator:checked {
    background-color: #4f46e5;
    border-color: #6366f1;
    image: none;
}

/* SpinBox / Slider */
QSpinBox, QSlider {
    background-color: transparent;
}
QSlider::groove:horizontal {
    border: none;
    height: 6px;
    background: #232738;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #6366f1;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #f8fafc;
    border: 2px solid #6366f1;
    width: 16px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 8px;
}
QSlider::handle:horizontal:hover {
    background: #a5b4fc;
}

/* Progress Bar */
QProgressBar {
    background-color: #13151f;
    border: 1px solid #282d3f;
    border-radius: 6px;
    text-align: center;
    color: #f8fafc;
    font-weight: 600;
    font-size: 12px;
    height: 18px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4f46e5, stop:1 #10b981);
    border-radius: 5px;
}

/* Badges */
QLabel.BadgeGpuOn {
    background-color: #064e3b;
    color: #34d399;
    border: 1px solid #059669;
    border-radius: 12px;
    padding: 3px 10px;
    font-weight: 600;
    font-size: 11px;
}
QLabel.BadgeGpuOff {
    background-color: #451a03;
    color: #fbbf24;
    border: 1px solid #d97706;
    border-radius: 12px;
    padding: 3px 10px;
    font-weight: 600;
    font-size: 11px;
}

/* Splitter */
QSplitter::handle {
    background-color: #212638;
}
"""
