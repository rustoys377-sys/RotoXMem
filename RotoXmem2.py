# RotoXmem2_Fixed.py
# Full GUI app using local XMem repo in-process (inference/inference_core.py).
# Edit CHECKPOINT_PATH and XMEM_BASE_FOLDER if necessary.

import sys
import os
import json
import subprocess
import shutil
import tempfile
import traceback
import re
import math
from pathlib import Path

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import torch
import imageio
from tqdm import tqdm

# ------------------ USER CONFIG (edit as needed) ------------------
CHECKPOINT_PATH = r"C:\Users\chima\Downloads\XMem.pth"     # your checkpoint
XMEM_BASE_FOLDER = r"C:\Users\chima\Downloads\XMem-main"   # folder where model/ and inference/ exist
TMP_DIR = os.path.join(tempfile.gettempdir(), "xmem_roto_tmp")
os.makedirs(TMP_DIR, exist_ok=True)
# ---------------------------------------------------------------

# Ensure repo paths are importable
def setup_xmem_paths():
    """Setup XMem import paths more reliably"""
    candidates = [
        XMEM_BASE_FOLDER,
        os.path.join(XMEM_BASE_FOLDER, "XMem"),
        os.path.join(XMEM_BASE_FOLDER, "xmem"),
        os.path.join(XMEM_BASE_FOLDER, "xmem-main"),
        os.path.join(XMEM_BASE_FOLDER, "XMem-main"),
    ]
    
    for c in candidates:
        if os.path.isdir(c):
            # Check if this directory contains the expected structure
            model_dir = os.path.join(c, "model")
            inference_dir = os.path.join(c, "inference")
            if os.path.isdir(model_dir) and os.path.isdir(inference_dir):
                if c not in sys.path:
                    sys.path.insert(0, c)
                print(f"Added to path: {c}")
                return c
    
    print(f"Warning: Could not find XMem repository structure in {XMEM_BASE_FOLDER}")
    return None

# Setup paths
xmem_path = setup_xmem_paths()

# Try import XMem classes (best-effort)
IN_PROCESS_AVAILABLE = False
InferenceCore = None
MaskMapper = None
XMemModelClass = None

try:
    from model.network import XMem as XMemModelClass
    from inference.inference_core import InferenceCore
    try:
        from inference.data.mask_mapper import MaskMapper
    except (ImportError, ModuleNotFoundError):
        print("Warning: MaskMapper not available, using fallback")
        MaskMapper = None
    IN_PROCESS_AVAILABLE = True
    print("✓ XMem modules loaded successfully")
except Exception as e:
    print(f"✗ Failed to import XMem modules: {e}")
    XMemModelClass = None
    InferenceCore = None
    MaskMapper = None
    IN_PROCESS_AVAILABLE = False

# ---------- helpers ----------
def read_mask_png(path, target_shape=None):
    """Read mask PNG with proper error handling and resizing"""
    if not os.path.exists(path):
        print(f"Warning: Mask file does not exist: {path}")
        return None
        
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        print(f"Warning: Could not read mask: {path}")
        return None
        
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    
    if target_shape and (m.shape[0] != target_shape[0] or m.shape[1] != target_shape[1]):
        m = cv2.resize(m, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    
    return (m > 127).astype(np.uint8) * 255

def prepare_video_frames_temp(video_path, output_dir):
    """Extract video frames to temporary directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    frame_idx = 0
    frame_paths = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_path = os.path.join(output_dir, f"frame_{frame_idx:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        frame_idx += 1
    
    cap.release()
    return frame_paths

# ---------- In-process XMem runner using inference_core API ----------
def run_xmem_inprocess(video_path, init_mask_path, init_frame, checkpoint_path, out_dir, device="cuda"):
    """
    Runs XMem segmentation directly inside Python without using CLI.
    Fixed version with proper error handling and tensor management.
    """
    
    if XMemModelClass is None:
        raise ImportError("XMem model class not available in this Python environment.")
    
    if InferenceCore is None:
        raise ImportError("InferenceCore not available in this XMem source, cannot run in-process.")

    print(f"Starting XMem in-process with device: {device}")
    
    # Ensure device is available
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, falling back to CPU")

    # ---- Create minimal config ----
    config = {
        "mem_every": 5,              # store memory every 5 frames
        "deep_update_every": -1,     # -1 = sync deep update every mem frame
        "enable_long_term": True,    # use long-term memory
        "size": 480,                 # input size (can be adjusted)
        "top_k": 30,
        "min_mid_term_frames": 5,
        "max_mid_term_frames": 10,
        "num_prototypes": 128,
        "enable_long_term_count_usage": True,
        "max_long_term_elements": 10000,
    }

    # ---- Instantiate model ----
    try:
        model = XMemModelClass(config, checkpoint_path, device)
    except TypeError:
        try:
            # Try with config only
            model = XMemModelClass(config)
            model = model.to(device)
        except TypeError:
            # Fallback to no arguments
            model = XMemModelClass()
            model = model.to(device)

    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            sd = checkpoint["state_dict"]
        else:
            sd = checkpoint
        
        # Handle potential key mismatches
        model_dict = model.state_dict()
        filtered_dict = {}
        for k, v in sd.items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    filtered_dict[k] = v
                else:
                    print(f"Skipping {k} due to shape mismatch: {model_dict[k].shape} vs {v.shape}")
            else:
                print(f"Skipping unknown key: {k}")
        
        model.load_state_dict(filtered_dict, strict=False)
        model.eval()
        print("✓ Model loaded successfully")
        
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")

    # ---- Initialize inference core ----
    try:
        propagator = InferenceCore(model, config=config)
    except TypeError:
        try:
            propagator = InferenceCore(model)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize InferenceCore: {e}")

    print("✓ InferenceCore initialized")

    # ---- Load video and get frame info ----
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    # Get the reference frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame)
    ret, ref_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(f"Couldn't read reference frame {init_frame} from video.")
    
    ref_frame_rgb = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)
    h, w = ref_frame_rgb.shape[:2]
    
    print(f"Video info: {total_frames} frames, {fps} fps, {w}x{h}")

    # ---- Load initial mask ----
    mask_img = read_mask_png(init_mask_path, target_shape=(h, w))
    if mask_img is None:
        cap.release()
        raise RuntimeError("Reference mask could not be loaded.")

    print(f"✓ Reference mask loaded: {mask_img.shape}")

    # ---- Prepare tensors ----
    # Convert frame to tensor [1, 3, H, W]
    ref_frame_tensor = torch.from_numpy(ref_frame_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    ref_frame_tensor = ref_frame_tensor.to(device)
    
    # Convert mask to tensor [1, 1, H, W] or [1, H, W] depending on what InferenceCore expects
    if MaskMapper is not None:
        try:
            mapper = MaskMapper()
            ref_mask = mapper.convert_mask(mask_img, 1)
            if isinstance(ref_mask, np.ndarray):
                ref_mask = torch.from_numpy(ref_mask)
            if ref_mask.dim() == 2:
                ref_mask = ref_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            elif ref_mask.dim() == 3 and ref_mask.shape[0] != 1:
                ref_mask = ref_mask.unsqueeze(0)  # [1, H, W] -> [1, 1, H, W]
        except Exception as e:
            print(f"MaskMapper failed, using fallback: {e}")
            ref_mask = torch.from_numpy((mask_img > 0).astype(np.uint8))
            if ref_mask.dim() == 2:
                ref_mask = ref_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    else:
        # Fallback mask preparation
        ref_mask = torch.from_numpy((mask_img > 0).astype(np.uint8))
        if ref_mask.dim() == 2:
            ref_mask = ref_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    ref_mask = ref_mask.to(device)
    print(f"✓ Tensors prepared - Frame: {ref_frame_tensor.shape}, Mask: {ref_mask.shape}")

    # ---- Initialize propagator with first frame ----
    try:
        propagator.step(ref_frame_tensor, ref_mask)
        print("✓ First frame set in propagator")
    except Exception as e:
        cap.release()
        raise RuntimeError(f"Failed to set first frame: {e}")

    # ---- Process all frames ----
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Processing {total_frames} frames...")
    
    for frame_idx in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {frame_idx}")
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        frame_tensor = frame_tensor.to(device)
        
        try:
            with torch.no_grad():
                if frame_idx == init_frame:
                    # For the reference frame, use the input mask
                    pred_mask = ref_mask.squeeze().cpu().numpy()
                else:
                    # Propagate from previous frame
                    pred = propagator.step(frame_tensor)
                    
                    if isinstance(pred, torch.Tensor):
                        pred_mask = pred.squeeze().cpu().numpy()
                    elif isinstance(pred, dict):
                        # Handle dictionary output (common in some XMem versions)
                        if 'masks' in pred:
                            pred_mask = pred['masks'].squeeze().cpu().numpy()
                        elif 'logits' in pred:
                            pred_mask = torch.sigmoid(pred['logits']).squeeze().cpu().numpy()
                        else:
                            pred_mask = list(pred.values())[0].squeeze().cpu().numpy()
                    else:
                        pred_mask = np.array(pred)
                
                # Ensure proper format and save
                if pred_mask.ndim > 2:
                    pred_mask = pred_mask[0] if pred_mask.shape[0] == 1 else pred_mask.max(axis=0)
                
                # Convert to binary mask
                if pred_mask.dtype != np.uint8:
                    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
                else:
                    pred_mask = (pred_mask > 127).astype(np.uint8) * 255
                
                # Resize if necessary
                if pred_mask.shape != (h, w):
                    pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                
                output_path = os.path.join(out_dir, f"frame_{frame_idx:05d}.png")
                cv2.imwrite(output_path, pred_mask)
                
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            # Save empty mask to maintain sequence
            empty_mask = np.zeros((h, w), dtype=np.uint8)
            output_path = os.path.join(out_dir, f"frame_{frame_idx:05d}.png")
            cv2.imwrite(output_path, empty_mask)
            continue

    cap.release()
    print(f"✓ Processing complete, outputs saved to: {out_dir}")
    return 0

# --------- GUI app (full) ----------
class RotoApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RotoXmem — Multi-segment Rotoscoping (XMem in-process) [FIXED]")
        self.resize(1400, 920)

        self.video_path = None
        self.frames = []
        self.fps = 25.0
        self.current_idx = 0

        # segments: id -> {"ref_frame": int, "color":(r,g,b), "masks": {frame_idx: mask}}
        self.segment_order = []
        self.segments = {}
        self.next_seg_id = 1

        # drawing
        self.temp_poly = []

        self._build_ui()
        self.statusBar().showMessage("Ready" + (" (XMem Available)" if IN_PROCESS_AVAILABLE else " (XMem NOT Available)"))

    def _build_ui(self):
        w = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(w)
        grid.setSpacing(8)
        self.setCentralWidget(w)

        # left panel
        left_v = QtWidgets.QVBoxLayout()
        
        # Status indicator
        status_color = "green" if IN_PROCESS_AVAILABLE else "red"
        status_text = "XMem Available" if IN_PROCESS_AVAILABLE else "XMem NOT Available"
        self.status_label = QtWidgets.QLabel(f"● {status_text}")
        self.status_label.setStyleSheet(f"color: {status_color}; font-weight: bold;")
        left_v.addWidget(self.status_label)
        
        self.btn_import = QtWidgets.QPushButton("Import Video")
        self.btn_import.clicked.connect(self.import_video)
        self.btn_save_proj = QtWidgets.QPushButton("Save Project")
        self.btn_save_proj.clicked.connect(self.save_project)
        self.btn_load_proj = QtWidgets.QPushButton("Load Project")
        self.btn_load_proj.clicked.connect(self.load_project)
        
        left_v.addWidget(self.btn_import)
        left_v.addWidget(self.btn_save_proj)
        left_v.addWidget(self.btn_load_proj)
        left_v.addStretch(1)
        
        left_w = QtWidgets.QWidget()
        left_w.setLayout(left_v)
        grid.addWidget(left_w, 0, 0, 3, 1)

        # center
        center_v = QtWidgets.QVBoxLayout()
        self.preview = QtWidgets.QLabel("Preview")
        self.preview.setAlignment(QtCore.Qt.AlignCenter)
        self.preview.setStyleSheet("background:#111; color:#ddd; border:1px solid #333;")
        self.preview.setMinimumSize(820, 480)
        self.preview.installEventFilter(self)
        center_v.addWidget(self.preview, 9)

        # playback controls
        ctrl_h = QtWidgets.QHBoxLayout()
        self.btn_play = QtWidgets.QPushButton("Play")
        self.btn_play.clicked.connect(self.play_pause)
        self.btn_prev = QtWidgets.QPushButton("Prev")
        self.btn_prev.clicked.connect(self.prev_frame)
        self.btn_next = QtWidgets.QPushButton("Next")
        self.btn_next.clicked.connect(self.next_frame)
        ctrl_h.addWidget(self.btn_play)
        ctrl_h.addWidget(self.btn_prev)
        ctrl_h.addWidget(self.btn_next)
        center_v.addLayout(ctrl_h)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderMoved.connect(self.on_slider)
        center_v.addWidget(self.slider)

        bottom_h = QtWidgets.QHBoxLayout()
        self.chk_onion = QtWidgets.QCheckBox("Onion-skin (prev frame)")
        bottom_h.addWidget(self.chk_onion)
        bottom_h.addStretch(1)
        bottom_h.addWidget(QtWidgets.QLabel("Segment ID"))
        self.spin_seg = QtWidgets.QSpinBox()
        self.spin_seg.setMinimum(1)
        self.spin_seg.setValue(1)
        bottom_h.addWidget(self.spin_seg)
        self.btn_pick_color = QtWidgets.QPushButton("Pick Color")
        self.btn_pick_color.clicked.connect(self.pick_color)
        bottom_h.addWidget(self.btn_pick_color)
        center_v.addLayout(bottom_h)

        center_w = QtWidgets.QWidget()
        center_w.setLayout(center_v)
        grid.addWidget(center_w, 0, 1, 3, 1)

        # right
        right_v = QtWidgets.QVBoxLayout()
        right_v.addWidget(QtWidgets.QLabel("Segments"))
        self.list_segments = QtWidgets.QListWidget()
        self.list_segments.itemSelectionChanged.connect(self.on_seg_selected)
        right_v.addWidget(self.list_segments, 8)
        
        self.btn_delete = QtWidgets.QPushButton("Delete Selected")
        self.btn_delete.clicked.connect(self.delete_selected)
        right_v.addWidget(self.btn_delete)
        
        self.btn_track_f = QtWidgets.QPushButton("Track Forward (XMem)")
        self.btn_track_f.clicked.connect(self.track_forward)
        self.btn_track_f.setEnabled(IN_PROCESS_AVAILABLE)
        
        self.btn_track_b = QtWidgets.QPushButton("Track Backward (XMem)")
        self.btn_track_b.clicked.connect(self.track_backward)
        self.btn_track_b.setEnabled(IN_PROCESS_AVAILABLE)
        
        right_v.addWidget(self.btn_track_f)
        right_v.addWidget(self.btn_track_b)
        right_v.addSpacing(6)
        
        right_v.addWidget(QtWidgets.QLabel("Export"))
        self.btn_export_png = QtWidgets.QPushButton("Export Selected (PNG sequence)")
        self.btn_export_png.clicked.connect(self.export_selected_png)
        right_v.addWidget(self.btn_export_png)
        
        self.btn_export_combined = QtWidgets.QPushButton("Export Combined Mask Video (MP4)")
        self.btn_export_combined.clicked.connect(self.export_combined_mp4)
        right_v.addWidget(self.btn_export_combined)
        
        self.btn_export_overlay = QtWidgets.QPushButton("Export Overlay Video (MP4)")
        self.btn_export_overlay.clicked.connect(self.export_overlay)
        right_v.addWidget(self.btn_export_overlay)
        
        self.btn_export_alpha = QtWidgets.QPushButton("Export Alpha (WebM/MOV/MP4 matte)")
        self.btn_export_alpha.clicked.connect(self.export_alpha_dialog)
        right_v.addWidget(self.btn_export_alpha)
        
        right_v.addStretch(1)
        
        right_w = QtWidgets.QWidget()
        right_w.setLayout(right_v)
        grid.addWidget(right_w, 0, 2, 3, 1)

        self.status = QtWidgets.QLabel("")
        self.statusBar().addWidget(self.status, 1)

        # timer
        self.play_timer = QtCore.QTimer(self)
        self.play_timer.timeout.connect(self._play_step)

    # ------------ Video load / nav -----------
    def import_video(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open video", "", "Videos (*.mp4 *.mov *.mkv *.avi *.wmv)"
        )
        if not fn:
            return
        self.load_video(fn)

    def load_video(self, path):
        if not os.path.exists(path):
            QtWidgets.QMessageBox.critical(self, "Error", f"Video file does not exist: {path}")
            return
            
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "Error", "Cannot open video")
            return
            
        frames = []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        
        progress = QtWidgets.QProgressDialog("Loading video frames...", "Cancel", 0, total, self)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.show()
        
        for i in range(total):
            if progress.wasCanceled():
                cap.release()
                return
                
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if i % 10 == 0:  # Update progress every 10 frames
                progress.setValue(i)
                QtWidgets.QApplication.processEvents()
        
        progress.close()
        cap.release()
        
        if not frames:
            QtWidgets.QMessageBox.warning(self, "Warning", "No frames loaded from video.")
            return
            
        self.video_path = path
        self.frames = frames
        self.current_idx = 0
        self.slider.setMaximum(max(0, len(frames)-1))
        self.slider.setEnabled(True)
        self.update_preview()
        self.statusBar().showMessage(f"Loaded: {os.path.basename(path)} ({len(frames)} frames, {self.fps:.1f} fps)")

    def play_pause(self):
        if not self.frames:
            return
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.btn_play.setText("Play")
        else:
            interval = int(1000.0 / max(1, round(self.fps)))
            self.play_timer.start(interval)
            self.btn_play.setText("Pause")

    def _play_step(self):
        if not self.frames:
            return
        self.current_idx = min(len(self.frames)-1, self.current_idx + 1)
        if self.current_idx >= len(self.frames) - 1:
            self.play_timer.stop()
            self.btn_play.setText("Play")
        self.slider.setValue(self.current_idx)
        self.update_preview()

    def prev_frame(self):
        if not self.frames:
            return
        self.current_idx = max(0, self.current_idx - 1)
        self.slider.setValue(self.current_idx)
        self.update_preview()

    def next_frame(self):
        if not self.frames:
            return
        self.current_idx = min(len(self.frames)-1, self.current_idx + 1)
        self.slider.setValue(self.current_idx)
        self.update_preview()

    def on_slider(self, pos):
        if not self.frames:
            return
        self.current_idx = int(pos)
        self.update_preview()

    # ---------- Drawing & Segments (auto-new behavior) ----------
    def eventFilter(self, source, event):
        if source is self.preview:
            if event.type() == QtCore.QEvent.MouseButtonPress:
                if event.button() == QtCore.Qt.LeftButton and self.frames:
                    pix = self.preview.pixmap()
                    if pix is None:
                        return False
                    scaled = pix.size()
                    w_label, h_label = self.preview.width(), self.preview.height()
                    pm_w, pm_h = scaled.width(), scaled.height()
                    offset_x = max(0, (w_label-pm_w)//2)
                    offset_y = max(0, (h_label-pm_h)//2)
                    x = event.x() - offset_x
                    y = event.y() - offset_y
                    if x < 0 or y < 0 or x >= pm_w or y >= pm_h:
                        return False
                    img = self.frames[self.current_idx]
                    img_h, img_w = img.shape[:2]
                    img_x = int(x * img_w / pm_w)
                    img_y = int(y * img_h / pm_h)
                    self.temp_poly.append((img_x, img_y))
                    self.update_preview()
                    return True

                elif event.button() == QtCore.Qt.RightButton and self.temp_poly:
                    # Complete polygon - if selection exists -> add to selected segment, else create new segment
                    sel_items = self.list_segments.selectedItems()
                    if sel_items:
                        sid = int(sel_items[0].text().split()[0])
                        if sid not in self.segments:
                            color = (int(255*np.random.random()), int(255*np.random.random()), int(255*np.random.random()))
                            self.segments[sid] = {"ref_frame": self.current_idx, "color": color, "masks": {}}
                            if sid not in self.segment_order:
                                self.segment_order.append(sid)
                    else:
                        sid = self.next_seg_id
                        color = (int(255*np.random.random()), int(255*np.random.random()), int(255*np.random.random()))
                        self.segments[sid] = {"ref_frame": self.current_idx, "color": color, "masks": {}}
                        self.segment_order.append(sid)
                        self.next_seg_id += 1
                        self.spin_seg.setValue(self.next_seg_id)

                    # rasterize polygon
                    mask = np.zeros(self.frames[self.current_idx].shape[:2], dtype=np.uint8)
                    pts = np.array(self.temp_poly, dtype=np.int32)
                    cv2.fillPoly(mask, [pts], 255)
                    
                    # union with existing
                    existing = self.segments[sid]["masks"].get(self.current_idx)
                    if existing is not None:
                        combined = np.maximum(existing, mask)
                        self.segments[sid]["masks"][self.current_idx] = combined
                    else:
                        self.segments[sid]["masks"][self.current_idx] = mask
                    
                    self.segments[sid]["ref_frame"] = self.current_idx
                    self.temp_poly = []
                    self._refresh_segment_list()
                    self.update_preview()
                    return True
        return super().eventFilter(source, event)

    def _refresh_segment_list(self):
        self.list_segments.clear()
        for sid in self.segment_order:
            seg = self.segments[sid]
            mask_count = len(seg['masks'])
            item = QtWidgets.QListWidgetItem(f"{sid} (ref {seg['ref_frame']}, {mask_count} masks)")
            self.list_segments.addItem(item)
        # set spin maximum
        self.spin_seg.setMaximum(max(self.next_seg_id, 9999))

    def on_seg_selected(self):
        items = self.list_segments.selectedItems()
        if items:
            sid = int(items[0].text().split()[0])
            self.spin_seg.setValue(sid)
            self.statusBar().showMessage(f"Selected segment {sid}")

    def delete_selected(self):
        for it in self.list_segments.selectedItems():
            sid = int(it.text().split()[0])
            if sid in self.segments:
                del self.segments[sid]
            if sid in self.segment_order:
                self.segment_order.remove(sid)
        self._refresh_segment_list()
        self.update_preview()

    def pick_color(self):
        c = QtWidgets.QColorDialog.getColor()
        if not c.isValid():
            return
        sel = self.list_segments.selectedItems()
        if not sel:
            QtWidgets.QMessageBox.information(self, "Pick Color", "Select a segment first")
            return
        sid = int(sel[0].text().split()[0])
        if sid in self.segments:
            self.segments[sid]['color'] = (c.red(), c.green(), c.blue())
            self.update_preview()

    # -------- Preview rendering ----------
    def update_preview(self):
        if not self.frames:
            self.preview.setText("Preview")
            return
            
        img = self.frames[self.current_idx].copy()
        overlay = img.copy()
        
        if self.chk_onion.isChecked() and self.current_idx-1 >= 0:
            prev = self.frames[self.current_idx-1]
            overlay = (overlay * 0.7 + prev * 0.3).astype(np.uint8)
        
        # draw each segment mask at this frame
        for sid in self.segment_order:
            seg = self.segments.get(sid)
            if not seg:
                continue
            color = seg.get('color', (255, 0, 0))
            mask = seg['masks'].get(self.current_idx)
            if mask is not None:
                overlay[mask>0] = (overlay[mask>0]*0.35 + np.array(color)*0.65).astype(np.uint8)
                contours, _ = cv2.findContours((mask>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color, 2)
        
        # draw temp polygon
        if self.temp_poly:
            pts = np.array(self.temp_poly, dtype=np.int32)
            cv2.polylines(overlay, [pts], False, (255,255,255), 2, cv2.LINE_AA)
            for (x,y) in self.temp_poly:
                cv2.circle(overlay, (int(x),int(y)), 4, (255,255,255), -1)
        
        h, w = overlay.shape[:2]
        qimg = QtGui.QImage(overlay.data, w, h, 3*w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        scaled = pix.scaled(self.preview.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.preview.setPixmap(scaled)
        self.statusBar().showMessage(f"Frame {self.current_idx+1}/{len(self.frames)}")

    # ------------ Tracking (forward/backward) using in-process XMem ----------
    def track_forward(self):
        self._track_segment("forward")

    def track_backward(self):
        self._track_segment("backward")

    def _track_segment(self, direction="forward"):
        if not IN_PROCESS_AVAILABLE:
            QtWidgets.QMessageBox.warning(
                self, "XMem Not Available", 
                "XMem modules are not properly imported. Please check:\n"
                "1. XMEM_BASE_FOLDER path is correct\n"
                "2. XMem repository structure is valid\n"
                "3. Required dependencies are installed"
            )
            return
            
        sel = self.list_segments.selectedItems()
        if not sel:
            QtWidgets.QMessageBox.information(self, "Track", "Select a segment to track")
            return
            
        sid = int(sel[0].text().split()[0])
        self._propagate_segment(sid, direction=direction)

    def _propagate_segment(self, sid, direction="forward"):
        seg = self.segments.get(sid)
        if not seg:
            QtWidgets.QMessageBox.warning(self, "Track", "Segment not found")
            return
            
        ref_frame = seg.get('ref_frame')
        if ref_frame is None:
            QtWidgets.QMessageBox.warning(self, "Track", "No reference frame set for segment")
            return
            
        ref_mask = seg['masks'].get(ref_frame)
        if ref_mask is None:
            QtWidgets.QMessageBox.warning(self, "Track", "No reference mask at the reference frame")
            return

        # Prepare output directory
        out_dir = os.path.join(TMP_DIR, f"seg_{sid}_out_{direction}")
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        
        # Save reference mask
        ref_mask_path = os.path.join(TMP_DIR, f"seg_{sid}_ref_{ref_frame:05d}.png")
        cv2.imwrite(ref_mask_path, ref_mask)

        # Show progress dialog
        progress = QtWidgets.QProgressDialog("Running XMem propagation...", "Cancel", 0, 0, self)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.show()
        QtWidgets.QApplication.processEvents()

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.statusBar().showMessage(f"Running XMem in-process on {device}...")
            QtWidgets.QApplication.processEvents()
            
            run_xmem_inprocess(
                self.video_path, 
                ref_mask_path, 
                ref_frame, 
                CHECKPOINT_PATH, 
                out_dir,
                device=device
            )
            
        except Exception as e:
            progress.close()
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(
                self, "XMem Error", 
                f"In-process XMem failed:\n{str(e)[:500]}..."
            )
            self.statusBar().showMessage("XMem failed")
            return
        finally:
            progress.close()

        # Load masks from output directory
        loaded = 0
        mask_files = [f for f in os.listdir(out_dir) if f.lower().endswith(".png")]
        
        for fname in mask_files:
            # Extract frame index from filename
            stem = os.path.splitext(fname)[0]
            m = re.search(r"(\d{1,6})$", stem)
            frame_idx = None
            
            if m:
                frame_idx = int(m.group(1))
            else:
                m2 = re.search(r"(\d{1,6})", stem)
                if m2:
                    frame_idx = int(m2.group(1))
            
            if frame_idx is None or frame_idx >= len(self.frames):
                continue
                
            mask_path = os.path.join(out_dir, fname)
            mask = read_mask_png(mask_path, target_shape=self.frames[0].shape[:2])
            if mask is None:
                continue
                
            # Apply direction filtering if needed
            if direction == "forward" and frame_idx < ref_frame:
                continue
            elif direction == "backward" and frame_idx > ref_frame:
                continue
                
            self.segments[sid]['masks'][frame_idx] = mask
            loaded += 1

        # Fallback: if no masks loaded by frame index, try sequential mapping
        if loaded == 0:
            pngs = sorted([p for p in os.listdir(out_dir) if p.lower().endswith(".png")])
            for i, fname in enumerate(pngs):
                mask = read_mask_png(os.path.join(out_dir, fname), target_shape=self.frames[0].shape[:2])
                if mask is None:
                    continue
                    
                if direction == "forward":
                    guessed = ref_frame + i
                else:
                    guessed = ref_frame - i
                    
                if 0 <= guessed < len(self.frames):
                    self.segments[sid]['masks'][guessed] = mask
                    loaded += 1

        self._refresh_segment_list()
        self.update_preview()
        
        msg = f"Propagation finished: loaded {loaded} masks for segment {sid} ({direction})"
        self.statusBar().showMessage(msg)
        print(msg)

    # ---------------- Exports ----------------
    def export_selected_png(self):
        sel = self.list_segments.selectedItems()
        if not sel:
            QtWidgets.QMessageBox.information(self, "Export", "Select a segment to export")
            return
            
        sid = int(sel[0].text().split()[0])
        seg = self.segments.get(sid)
        if not seg:
            return
            
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output folder for PNG sequence")
        if not out_dir:
            return
            
        exported = 0
        for i in range(len(self.frames)):
            frame = self.frames[i]
            alpha = np.zeros(frame.shape[:2], dtype=np.uint8)
            m = seg['masks'].get(i)
            if m is not None:
                alpha[m > 0] = 255
            rgba = np.dstack([frame, alpha])
            outp = os.path.join(out_dir, f"seg{sid}_frame_{i:05d}.png")
            cv2.imwrite(outp, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
            exported += 1
            
        QtWidgets.QMessageBox.information(self, "Export", f"PNG sequence exported: {exported} frames")

    def export_combined_mp4(self):
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save combined mask video (MP4)", "", "MP4 (*.mp4)"
        )
        if not out_path:
            return
            
        h, w = self.frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(out_path, fourcc, self.fps, (w, h), False)
        
        if not vw.isOpened():
            QtWidgets.QMessageBox.critical(self, "Export", "Failed to open video writer for combined MP4")
            return
            
        for i in range(len(self.frames)):
            combined = np.zeros((h, w), dtype=np.uint8)
            for sid in self.segment_order:
                seg = self.segments.get(sid)
                if not seg:
                    continue
                m = seg['masks'].get(i)
                if m is not None:
                    combined = np.maximum(combined, (m > 0).astype(np.uint8) * 255)
            vw.write(combined)
            
        vw.release()
        QtWidgets.QMessageBox.information(self, "Export", "Combined MP4 exported")

    def export_overlay(self):
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save overlay video (MP4)", "", "MP4 (*.mp4)"
        )
        if not out_path:
            return
            
        h, w = self.frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(out_path, fourcc, self.fps, (w, h), True)
        
        if not vw.isOpened():
            QtWidgets.QMessageBox.critical(self, "Export", "Failed to open video writer for overlay")
            return
            
        for i in range(len(self.frames)):
            frame = self.frames[i].copy()
            for sid in self.segment_order:
                seg = self.segments.get(sid)
                if not seg:
                    continue
                m = seg['masks'].get(i)
                if m is not None:
                    color = seg.get('color', (255, 0, 0))
                    frame[m>0] = (frame[m>0]*0.35 + np.array(color)*0.65).astype(np.uint8)
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            vw.write(bgr)
            
        vw.release()
        QtWidgets.QMessageBox.information(self, "Export", "Overlay exported")

    def export_alpha_dialog(self):
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save alpha video", "", 
            "WebM (*.webm);;MOV (*.mov);;MP4 Matte (*.mp4)"
        )
        if not out_path:
            return
            
        ext = os.path.splitext(out_path)[1].lower()
        frames_rgba = []
        
        for i in range(len(self.frames)):
            rgb = self.frames[i]
            alpha = np.zeros(rgb.shape[:2], dtype=np.uint8)
            
            for sid in self.segment_order:
                m = self.segments.get(sid, {}).get('masks', {}).get(i)
                if m is not None:
                    alpha = np.maximum(alpha, (m > 0).astype(np.uint8)*255)
                    
            rgba = np.dstack([rgb, alpha])
            frames_rgba.append(rgba)
            
        if ext == ".webm":
            try:
                imageio.mimwrite(out_path, frames_rgba, fps=self.fps, codec="libvpx-vp9")
                QtWidgets.QMessageBox.information(self, "Export", "WebM exported (alpha)")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Export", f"Failed to export WebM: {e}")
                
        elif ext == ".mov":
            tmp = os.path.join(TMP_DIR, "alpha_tmp")
            if os.path.exists(tmp):
                shutil.rmtree(tmp)
            os.makedirs(tmp, exist_ok=True)
            
            for i, rgba in enumerate(frames_rgba):
                cv2.imwrite(
                    os.path.join(tmp, f"frame_{i:05d}.png"), 
                    cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
                )
                
            cmd = [
                'ffmpeg', '-y', '-framerate', str(int(self.fps)), 
                '-i', os.path.join(tmp, "frame_%05d.png"),
                '-c:v', 'png', '-pix_fmt', 'rgba', out_path
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                QtWidgets.QMessageBox.information(self, "Export", "MOV (PNG codec) exported")
            except subprocess.CalledProcessError as e:
                QtWidgets.QMessageBox.critical(self, "Export", f"Failed to export MOV:\n{e}")
            finally:
                if os.path.exists(tmp):
                    shutil.rmtree(tmp)
                    
        else:
            # MP4 matte: grayscale alpha saved as video
            h, w = frames_rgba[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(out_path, fourcc, self.fps, (w, h), False)
            
            if not vw.isOpened():
                QtWidgets.QMessageBox.critical(self, "Export", "Failed to open video writer for MP4 matte")
                return
                
            for rgba in frames_rgba:
                matte = rgba[:, :, 3]
                vw.write(matte)
                
            vw.release()
            QtWidgets.QMessageBox.information(self, "Export", "MP4 matte exported")

    # ---------------- Project save/load ----------------
    def save_project(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Project", "", "Roto project (*.json)"
        )
        if not path:
            return
            
        data = {
            "video": self.video_path, 
            "fps": self.fps, 
            "segments": {},
            "next_seg_id": self.next_seg_id,
            "segment_order": self.segment_order
        }
        
        base_dir = os.path.dirname(path)
        
        for sid in self.segment_order:
            seg = self.segments[sid]
            segdir = os.path.join(base_dir, f"seg_{sid}_masks")
            os.makedirs(segdir, exist_ok=True)
            
            segmap = {}
            for fi, m in seg['masks'].items():
                fname = os.path.join(segdir, f"mask_{fi:05d}.png")
                cv2.imwrite(fname, m)
                segmap[str(fi)] = fname
                
            data['segments'][str(sid)] = {
                "ref_frame": seg['ref_frame'], 
                "color": seg['color'], 
                "masks": segmap
            }
            
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
            
        QtWidgets.QMessageBox.information(self, "Save", "Project saved")

    def load_project(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Project", "", "Roto project (*.json)"
        )
        if not path:
            return
            
        try:
            with open(path, "r") as f:
                data = json.load(f)
                
            # Load video
            video_path = data.get('video')
            if video_path and os.path.exists(video_path):
                self.load_video(video_path)
            else:
                QtWidgets.QMessageBox.warning(
                    self, "Load Project", 
                    f"Video file not found: {video_path}\nPlease load video manually."
                )
                
            # Load segments
            self.segments = {}
            self.segment_order = data.get('segment_order', [])
            
            for sid_str, seg_entry in data.get('segments', {}).items():
                sid = int(sid_str)
                if sid not in self.segment_order:
                    self.segment_order.append(sid)
                    
                masks = {}
                for fi_str, p in seg_entry['masks'].items():
                    fi = int(fi_str)
                    if os.path.exists(p):
                        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                        if m is not None:
                            masks[fi] = (m > 127).astype(np.uint8)*255
                    else:
                        print(f"Warning: Mask file not found: {p}")
                        
                self.segments[sid] = {
                    "ref_frame": seg_entry['ref_frame'], 
                    "color": tuple(seg_entry['color']), 
                    "masks": masks
                }
                
            self.next_seg_id = data.get('next_seg_id', max(self.segment_order)+1 if self.segment_order else 1)
            self._refresh_segment_list()
            self.update_preview()
            
            QtWidgets.QMessageBox.information(self, "Load", "Project loaded successfully")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load Error", f"Failed to load project:\n{e}")

# -------------------- Run --------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    
    # Show startup info
    print("=" * 60)
    print("RotoXmem2 - Fixed Version")
    print("=" * 60)
    print(f"XMem Base Folder: {XMEM_BASE_FOLDER}")
    print(f"Checkpoint Path: {CHECKPOINT_PATH}")
    print(f"XMem Available: {IN_PROCESS_AVAILABLE}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print("=" * 60)
    
    win = RotoApp()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()