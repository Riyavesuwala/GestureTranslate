"""
app_ui.py
─────────
ISL Gujarati Sign Language Translator
Full Tkinter UI with:
  - Live webcam feed (left panel)
  - Prediction info panel (right panel)
  - Recognized text panel (bottom right)
  - Control buttons (bottom)
  - Keyboard shortcuts
"""

import tkinter as tk
from tkinter import font as tkfont
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pandas as pd
from collections import deque, Counter
from PIL import ImageFont, ImageDraw, Image, ImageTk
import os
import threading

# ════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════
MODEL_PATH = "best_cnn_gru_model.h5"
MEAN_PATH  = "feature_mean.csv"
STD_PATH   = "feature_std.csv"

SEQ_LEN       = 30
NUM_LANDMARKS = 21
NUM_FEATURES  = 63

LABEL_NAMES = {
    0: "૦", 1: "૧", 2: "૨", 3: "૩", 4: "૪",
    5: "૫", 6: "૬", 7: "૭", 8: "૮", 9: "૯",
}

CAM_INDEX                = 0
CONFIDENCE_THRESHOLD     = 0.75
PREDICTION_COOLDOWN      = 5
TEMPERATURE              = 1.0
STABILITY_COUNT_REQUIRED = 2
ACCEPT_COOLDOWN          = 5
RECENT_PRED_WINDOW       = 3

# UI Colors
BG_DARK        = "#1a1a2e"    # main background
BG_PANEL       = "#16213e"    # panel background
BG_CARD        = "#0f3460"    # card background
ACCENT_GREEN   = "#00ff88"    # prediction / confirmed
ACCENT_YELLOW  = "#ffd700"    # confidence
ACCENT_CYAN    = "#00d4ff"    # building
ACCENT_RED     = "#ff4757"    # cooldown / no hand
ACCENT_PURPLE  = "#c8a8ff"    # stability
TEXT_WHITE     = "#ffffff"
TEXT_GRAY      = "#a0a0b0"
BTN_BLUE       = "#0066cc"
BTN_RED        = "#cc0000"
BTN_GREEN      = "#006600"
BTN_ORANGE     = "#cc6600"

# Gujarati font path
FONT_PATH = "NotoSansGujarati-Regular.ttf"

# ════════════════════════════════════════════════════
# LOAD MODEL & NORMALIZATION
# ════════════════════════════════════════════════════
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

mean_df     = pd.read_csv(MEAN_PATH, index_col=0)
std_df      = pd.read_csv(STD_PATH,  index_col=0)
mean_values = mean_df.values.flatten()
std_values  = std_df.values.flatten()
print("Model loaded.")

# ════════════════════════════════════════════════════
# MEDIAPIPE
# ════════════════════════════════════════════════════
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ════════════════════════════════════════════════════
# GUJARATI PIL FONT
# ════════════════════════════════════════════════════
def load_pil_font(size):
    if os.path.exists(FONT_PATH):
        return ImageFont.truetype(FONT_PATH, size)
    print(f"WARNING: {FONT_PATH} not found!")
    return ImageFont.load_default()

pil_font_large  = load_pil_font(42)
pil_font_medium = load_pil_font(30)
pil_font_small  = load_pil_font(22)

# ════════════════════════════════════════════════════
# PREDICTION STATE
# ════════════════════════════════════════════════════
frame_buffer        = deque(maxlen=SEQ_LEN)
prediction_cooldown = 0
current_word_digits = []
sentence            = ""
stability_counter   = 0
last_stable_class   = None
accept_cooldown     = 0
recent_preds        = deque(maxlen=RECENT_PRED_WINDOW)
hand_detected       = False
current_prediction  = "---"
current_confidence  = 0.0


# ════════════════════════════════════════════════════
# PREDICTION FUNCTIONS
# ════════════════════════════════════════════════════

def normalize_landmarks(landmarks):
    if not landmarks:
        return np.zeros(NUM_FEATURES)
    coords  = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    wrist   = coords[0]
    rel     = coords - wrist
    max_val = np.max(np.abs(rel))
    if max_val < 1e-6:
        max_val = 1.0
    return (rel / max_val).flatten()


def extract_landmarks(results):
    if results.multi_hand_landmarks:
        return normalize_landmarks(
            results.multi_hand_landmarks[0].landmark
        )
    return np.zeros(NUM_FEATURES)


def normalize_sequence(seq):
    return (seq - mean_values) / std_values


def predict_gesture(buffer):
    if len(buffer) < SEQ_LEN:
        return None, 0.0
    sequence = np.array(buffer)
    sequence = normalize_sequence(sequence)
    sequence = np.expand_dims(sequence, axis=0)
    pred = model.predict(sequence, verbose=0)[0]
    if TEMPERATURE != 1.0:
        pred = np.exp(np.log(np.clip(pred, 1e-8, 1.0)) / TEMPERATURE)
        pred = pred / np.sum(pred)
    return int(np.argmax(pred)), float(np.max(pred))


# ════════════════════════════════════════════════════
# SENTENCE GENERATION
# ════════════════════════════════════════════════════

def get_current_word():
    return "".join(current_word_digits)


def get_display_sentence():
    return sentence + get_current_word()


def commit_word():
    global sentence, current_word_digits
    word = get_current_word()
    if word:
        sentence += word + " "
    current_word_digits = []


def backspace_action():
    global sentence, current_word_digits
    if current_word_digits:
        current_word_digits.pop()
    elif sentence:
        sentence = sentence[:-1]


def clear_action():
    global sentence, current_word_digits
    sentence            = ""
    current_word_digits = []


def accept_digit(digit_str):
    current_word_digits.append(digit_str)


def try_accept_prediction(class_id, confidence):
    global stability_counter, last_stable_class, accept_cooldown
    if accept_cooldown > 0:
        accept_cooldown -= 1
        return False
    if confidence < CONFIDENCE_THRESHOLD or class_id is None:
        stability_counter = 0
        last_stable_class = None
        return False
    if class_id == last_stable_class:
        stability_counter += 1
    else:
        last_stable_class = class_id
        stability_counter = 1
    if stability_counter >= STABILITY_COUNT_REQUIRED:
        digit_str = LABEL_NAMES.get(class_id, "?")
        accept_digit(digit_str)
        stability_counter = 0
        accept_cooldown   = ACCEPT_COOLDOWN
        return True
    return False


def reset_stability():
    global stability_counter, last_stable_class
    stability_counter = 0
    last_stable_class = None


# ════════════════════════════════════════════════════
# GUJARATI TEXT RENDER ON PIL IMAGE
# ════════════════════════════════════════════════════

def render_gujarati_on_pil(pil_img, text, pos,
                            font, color=(255, 255, 255)):
    """Draw Gujarati text on a PIL Image directly."""
    draw = ImageDraw.Draw(pil_img)
    draw.text(pos, text, font=font, fill=color)
    return pil_img


# ════════════════════════════════════════════════════
# MAIN UI CLASS
# ════════════════════════════════════════════════════

class ISLTranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ISL | ગુજરાતી સાંકેતિક ભાષા અનુવાદક")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(False, False)

        # Webcam
        self.cap = cv2.VideoCapture(CAM_INDEX)
        self.running = True

        # Build UI
        self._build_ui()

        # Bind keyboard shortcuts
        self.root.bind("<space>",     lambda e: self._on_space())
        self.root.bind("<BackSpace>", lambda e: self._on_backspace())
        self.root.bind("c",           lambda e: self._on_clear())
        self.root.bind("C",           lambda e: self._on_clear())
        self.root.bind("q",           lambda e: self._on_quit())
        self.root.bind("Q",           lambda e: self._on_quit())

        # Start video loop
        self._update_frame()

    # ── UI BUILDER ────────────────────────────────────

    def _build_ui(self):
        """Build all UI widgets."""

        # ── Title bar ────────────────────────────────
        title_frame = tk.Frame(self.root, bg=BG_CARD, pady=8)
        title_frame.pack(fill=tk.X)

        tk.Label(
            title_frame,
            text="🤟  ISL ગુજરાતી સાંકેતિક ભાષા અનુવાદક",
            font=("Arial", 16, "bold"),
            bg=BG_CARD,
            fg=ACCENT_GREEN
        ).pack()

        tk.Label(
            title_frame,
            text="Indian Sign Language → Gujarati Digit Translator",
            font=("Arial", 10),
            bg=BG_CARD,
            fg=TEXT_GRAY
        ).pack()

        # ── Main content area ─────────────────────────
        content = tk.Frame(self.root, bg=BG_DARK)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        # Left: webcam
        self._build_webcam_panel(content)

        # Right: info panels
        self._build_info_panel(content)

        # Bottom: control buttons
        self._build_button_panel()

        # Status bar
        self._build_status_bar()

    def _build_webcam_panel(self, parent):
        """Left panel: live webcam feed."""
        left = tk.Frame(parent, bg=BG_PANEL,
                        relief=tk.RIDGE, bd=2)
        left.pack(side=tk.LEFT, padx=(0, 8))

        tk.Label(
            left,
            text="📷  Live Camera Feed",
            font=("Arial", 11, "bold"),
            bg=BG_PANEL, fg=ACCENT_CYAN
        ).pack(pady=(6, 2))

        # Webcam canvas
        self.cam_label = tk.Label(
            left, bg="black",
            width=480, height=360
        )
        self.cam_label.pack(padx=6, pady=6)

        # Hand detection indicator below webcam
        self.hand_status_var = tk.StringVar(value="● No Hand Detected")
        self.hand_status_label = tk.Label(
            left,
            textvariable=self.hand_status_var,
            font=("Arial", 11, "bold"),
            bg=BG_PANEL,
            fg=ACCENT_RED
        )
        self.hand_status_label.pack(pady=(0, 6))

    def _build_info_panel(self, parent):
        """Right panel: prediction info + recognized text."""
        right = tk.Frame(parent, bg=BG_DARK)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ── Prediction card ───────────────────────────
        pred_card = tk.Frame(right, bg=BG_PANEL,
                             relief=tk.RIDGE, bd=2)
        pred_card.pack(fill=tk.X, pady=(0, 6))

        tk.Label(
            pred_card,
            text="📊  Prediction Info",
            font=("Arial", 11, "bold"),
            bg=BG_PANEL, fg=ACCENT_CYAN
        ).pack(anchor=tk.W, padx=10, pady=(6, 2))

        # Prediction value (Gujarati)
        pred_row = tk.Frame(pred_card, bg=BG_PANEL)
        pred_row.pack(fill=tk.X, padx=10, pady=2)

        tk.Label(
            pred_row,
            text="Prediction :",
            font=("Arial", 13, "bold"),
            bg=BG_PANEL, fg=TEXT_WHITE
        ).pack(side=tk.LEFT)

        # Gujarati digit shown as image label
        self.pred_img_label = tk.Label(
            pred_row, bg=BG_PANEL
        )
        self.pred_img_label.pack(side=tk.LEFT, padx=(6, 0))

        # Confidence
        conf_row = tk.Frame(pred_card, bg=BG_PANEL)
        conf_row.pack(fill=tk.X, padx=10, pady=2)

        tk.Label(
            conf_row,
            text="Confidence :",
            font=("Arial", 12),
            bg=BG_PANEL, fg=TEXT_WHITE
        ).pack(side=tk.LEFT)

        self.conf_var = tk.StringVar(value="0.00")
        tk.Label(
            conf_row,
            textvariable=self.conf_var,
            font=("Arial", 13, "bold"),
            bg=BG_PANEL, fg=ACCENT_YELLOW
        ).pack(side=tk.LEFT, padx=6)

        # Stability
        stab_row = tk.Frame(pred_card, bg=BG_PANEL)
        stab_row.pack(fill=tk.X, padx=10, pady=2)

        tk.Label(
            stab_row,
            text="Stability  :",
            font=("Arial", 12),
            bg=BG_PANEL, fg=TEXT_WHITE
        ).pack(side=tk.LEFT)

        self.stab_var = tk.StringVar(value="0/2")
        tk.Label(
            stab_row,
            textvariable=self.stab_var,
            font=("Arial", 13, "bold"),
            bg=BG_PANEL, fg=ACCENT_PURPLE
        ).pack(side=tk.LEFT, padx=6)

        # Stability progress bar (canvas)
        self.stab_canvas = tk.Canvas(
            pred_card, height=16,
            bg=BG_PANEL, highlightthickness=0
        )
        self.stab_canvas.pack(fill=tk.X, padx=10, pady=(2, 6))

        # Building
        build_row = tk.Frame(pred_card, bg=BG_PANEL)
        build_row.pack(fill=tk.X, padx=10, pady=(0, 4))

        tk.Label(
            build_row,
            text="Building   :",
            font=("Arial", 12),
            bg=BG_PANEL, fg=TEXT_WHITE
        ).pack(side=tk.LEFT)

        self.build_img_label = tk.Label(
            build_row, bg=BG_PANEL
        )
        self.build_img_label.pack(side=tk.LEFT, padx=6)

        # Cooldown indicator
        self.cooldown_var = tk.StringVar(value="")
        self.cooldown_label = tk.Label(
            pred_card,
            textvariable=self.cooldown_var,
            font=("Arial", 10),
            bg=BG_PANEL, fg=ACCENT_RED
        )
        self.cooldown_label.pack(anchor=tk.W,
                                  padx=10, pady=(0, 6))

        # ── Recognized text card ──────────────────────
        rec_card = tk.Frame(right, bg=BG_PANEL,
                            relief=tk.RIDGE, bd=2)
        rec_card.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            rec_card,
            text="📝  Recognized Text",
            font=("Arial", 11, "bold"),
            bg=BG_PANEL, fg=ACCENT_CYAN
        ).pack(anchor=tk.W, padx=10, pady=(6, 2))

        # Recognized text shown as PIL image (Gujarati)
        self.rec_img_label = tk.Label(
            rec_card, bg=BG_CARD,
            relief=tk.SUNKEN, bd=1
        )
        self.rec_img_label.pack(
            fill=tk.BOTH, expand=True,
            padx=10, pady=(0, 10)
        )

    def _build_button_panel(self):
        """Bottom row of control buttons."""
        btn_frame = tk.Frame(self.root, bg=BG_CARD, pady=8)
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 6))

        tk.Label(
            btn_frame,
            text="Controls:",
            font=("Arial", 10, "bold"),
            bg=BG_CARD, fg=TEXT_GRAY
        ).pack(side=tk.LEFT, padx=(10, 15))

        buttons = [
            ("SPACE\nCommit Word",  self._on_space,     BTN_GREEN,  "⎵"),
            ("BACKSPACE\nDelete",   self._on_backspace,  BTN_ORANGE, "⌫"),
            ("C\nClear All",        self._on_clear,      BTN_RED,    "✗"),
            ("Q\nQuit",             self._on_quit,       "#555555",  "✕"),
        ]

        for label, cmd, color, icon in buttons:
            btn = tk.Button(
                btn_frame,
                text=f"{icon}\n{label}",
                command=cmd,
                font=("Arial", 9, "bold"),
                bg=color,
                fg="white",
                relief=tk.RAISED,
                bd=2,
                width=10,
                height=3,
                cursor="hand2",
                activebackground=color,
                activeforeground="white"
            )
            btn.pack(side=tk.LEFT, padx=6)

    def _build_status_bar(self):
        """Bottom status bar."""
        status = tk.Frame(self.root, bg="#0a0a1a")
        status.pack(fill=tk.X)

        self.status_var = tk.StringVar(
            value="Ready  |  Show your hand gesture to the camera"
        )
        tk.Label(
            status,
            textvariable=self.status_var,
            font=("Arial", 9),
            bg="#0a0a1a",
            fg=TEXT_GRAY,
            anchor=tk.W
        ).pack(side=tk.LEFT, padx=10, pady=3)

        tk.Label(
            status,
            text="ISL Gujarati Translator v1.0",
            font=("Arial", 9),
            bg="#0a0a1a",
            fg=TEXT_GRAY
        ).pack(side=tk.RIGHT, padx=10)

    # ── GUJARATI IMAGE HELPERS ────────────────────────

    def _make_gujarati_image(self, text, font,
                              width, height,
                              bg_color=(15, 52, 96),
                              text_color=(255, 255, 255)):
        """
        Render Gujarati text into a PIL image
        and convert to Tkinter PhotoImage.
        """
        img  = Image.new("RGB", (width, height), bg_color)
        draw = ImageDraw.Draw(img)

        # Center text
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            tw   = bbox[2] - bbox[0]
            th   = bbox[3] - bbox[1]
            x    = max(0, (width  - tw) // 2)
            y    = max(0, (height - th) // 2)
        except Exception:
            x, y = 10, 10

        draw.text((x, y), text, font=font, fill=text_color)
        return ImageTk.PhotoImage(img)

    def _make_gujarati_text_panel(self, text,
                                   width=320, height=150):
        """
        Render multi-line Gujarati sentence into
        a PIL image for the recognized text panel.
        """
        img  = Image.new("RGB", (width, height), (10, 20, 40))
        draw = ImageDraw.Draw(img)

        if not text.strip():
            draw.text((10, 10),
                      "[ Waiting for gestures... ]",
                      font=pil_font_small,
                      fill=(80, 80, 100))
            return ImageTk.PhotoImage(img)

        # Word wrap
        words    = text.strip().split()
        lines    = []
        line     = ""
        max_chars = 12

        for w in words:
            if len(line) + len(w) + 1 <= max_chars:
                line = (line + " " + w).strip()
            else:
                if line:
                    lines.append(line)
                line = w
        if line:
            lines.append(line)

        # Draw each line
        y = 10
        for ln in lines:
            draw.text((10, y), ln,
                      font=pil_font_large,
                      fill=(255, 255, 255))
            try:
                bb = draw.textbbox((10, y), ln, font=pil_font_large)
                y += (bb[3] - bb[1]) + 8
            except Exception:
                y += 48
            if y > height - 10:
                break

        return ImageTk.PhotoImage(img)

    # ── BUTTON CALLBACKS ──────────────────────────────

    def _on_space(self):
        commit_word()
        self.status_var.set(
            f"Word committed → Sentence: '{sentence.strip()}'"
        )
        self._refresh_ui_state()

    def _on_backspace(self):
        backspace_action()
        self.status_var.set("Deleted last character.")
        self._refresh_ui_state()

    def _on_clear(self):
        clear_action()
        self.status_var.set("Sentence cleared.")
        self._refresh_ui_state()

    def _on_quit(self):
        self.running = False
        self.cap.release()
        hands.close()
        self.root.destroy()

    # ── UI STATE REFRESH ──────────────────────────────

    def _refresh_ui_state(self):
        """Update all dynamic UI widgets from current state."""

        # 1. Prediction image
        pred_img = self._make_gujarati_image(
            text       = current_prediction,
            font       = pil_font_large,
            width      = 80,
            height     = 50,
            bg_color   = (15, 52, 96),
            text_color = (0, 255, 136)   # green
        )
        self.pred_img_label.config(image=pred_img)
        self.pred_img_label.image = pred_img   # keep reference

        # 2. Confidence
        self.conf_var.set(f"{current_confidence:.2f}")

        # 3. Stability
        self.stab_var.set(
            f"{stability_counter}/{STABILITY_COUNT_REQUIRED}"
        )

        # 4. Stability progress bar
        self.stab_canvas.delete("all")
        cw = self.stab_canvas.winfo_width() or 280
        filled = int(cw * min(stability_counter,
                               STABILITY_COUNT_REQUIRED)
                     / STABILITY_COUNT_REQUIRED)
        self.stab_canvas.create_rectangle(
            0, 2, cw, 14, fill="#333344", outline=""
        )
        if filled > 0:
            color = ACCENT_GREEN if (stability_counter
                                     >= STABILITY_COUNT_REQUIRED) \
                    else "#00aa55"
            self.stab_canvas.create_rectangle(
                0, 2, filled, 14, fill=color, outline=""
            )

        # 5. Building image
        building_str = get_current_word()
        display_building = f"[ {building_str} ]" \
                           if building_str else "[  ]"
        build_img = self._make_gujarati_image(
            text       = display_building,
            font       = pil_font_medium,
            width      = 180,
            height     = 44,
            bg_color   = (10, 30, 50),
            text_color = (0, 212, 255)   # cyan
        )
        self.build_img_label.config(image=build_img)
        self.build_img_label.image = build_img

        # 6. Cooldown
        if accept_cooldown > 0:
            self.cooldown_var.set(
                f"⏳ Cooldown: {accept_cooldown} cycles"
            )
        else:
            self.cooldown_var.set("")

        # 7. Hand status
        if hand_detected:
            self.hand_status_var.set("● Hand Detected")
            self.hand_status_label.config(fg=ACCENT_GREEN)
        else:
            self.hand_status_var.set("● No Hand Detected")
            self.hand_status_label.config(fg=ACCENT_RED)

        # 8. Recognized text panel
        rec_img = self._make_gujarati_text_panel(
            get_display_sentence(),
            width=320, height=160
        )
        self.rec_img_label.config(image=rec_img)
        self.rec_img_label.image = rec_img

    # ── MAIN VIDEO LOOP ───────────────────────────────

    def _update_frame(self):
        """
        Called every ~33ms via root.after().
        Reads webcam frame, runs prediction,
        updates all UI widgets.
        """
        global prediction_cooldown, current_prediction
        global current_confidence, hand_detected

        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(33, self._update_frame)
            return

        frame        = cv2.flip(frame, 1)
        rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results      = hands.process(rgb)
        hand_detected = results.multi_hand_landmarks is not None

        # Extract features
        feats = extract_landmarks(results)
        frame_buffer.append(feats)

        # Draw hand landmarks on frame
        if hand_detected:
            for hlm in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hlm, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(
                        color=(0, 255, 136), thickness=2
                    ),
                    mp_draw.DrawingSpec(
                        color=(0, 200, 255), thickness=2
                    )
                )

        # Prediction
        if len(frame_buffer) == SEQ_LEN \
                and prediction_cooldown == 0:

            if not hand_detected:
                current_prediction = "---"
                current_confidence = 0.0
                reset_stability()
            else:
                raw_class_id, raw_confidence = \
                    predict_gesture(frame_buffer)

                if (raw_class_id is not None
                        and raw_confidence >= CONFIDENCE_THRESHOLD):
                    recent_preds.append(raw_class_id)
                    majority_class = Counter(
                        recent_preds
                    ).most_common(1)[0][0]
                    current_prediction = LABEL_NAMES.get(
                        majority_class, "?"
                    )
                    current_confidence  = raw_confidence
                    prediction_cooldown = PREDICTION_COOLDOWN
                    try_accept_prediction(
                        majority_class, raw_confidence
                    )
                    self.status_var.set(
                        f"Detected: {current_prediction} "
                        f"(conf: {raw_confidence:.2f})"
                    )
                else:
                    current_prediction = "Low Conf"
                    current_confidence = raw_confidence \
                        if raw_class_id is not None else 0.0
                    reset_stability()
                    self.status_var.set(
                        "Low confidence — adjust hand position"
                    )

        if prediction_cooldown > 0:
            prediction_cooldown -= 1

        # ── Show webcam in UI ─────────────────────────
        # Resize frame for display
        display_frame = cv2.resize(frame, (480, 360))

        # Add minimal overlay on webcam (hand status dot)
        dot_color = (0, 255, 136) if hand_detected \
                    else (0, 0, 255)
        cv2.circle(display_frame, (460, 20), 10,
                   dot_color, -1)

        # Convert BGR → RGB → PIL → ImageTk
        rgb_display = cv2.cvtColor(display_frame,
                                    cv2.COLOR_BGR2RGB)
        pil_display = Image.fromarray(rgb_display)
        tk_img      = ImageTk.PhotoImage(pil_display)

        self.cam_label.config(image=tk_img)
        self.cam_label.image = tk_img

        # ── Refresh all info widgets ──────────────────
        self._refresh_ui_state()

        # Schedule next frame (~30 FPS)
        self.root.after(33, self._update_frame)


# ════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()

    # Window size and position
    root.geometry("900x620+100+50")
    root.minsize(900, 620)

    app = ISLTranslatorApp(root)

    print("\n" + "="*50)
    print("  ISL Gujarati Translator UI Started")
    print("="*50)
    print("  Keyboard Shortcuts:")
    print("    SPACE     → Commit word")
    print("    BACKSPACE → Delete last")
    print("    C         → Clear all")
    print("    Q         → Quit")
    print("="*50 + "\n")

    root.mainloop()