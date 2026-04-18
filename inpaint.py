"""
Interactive Image Inpainting Pipeline
Stages: Original → Grayscale → Canny (variance) → Dilation → BinaryFillHole → Inpaint
Fix: removed Canvas-based scrollable left panel (breaks on Linux); plain Frame used instead.
Dependencies: pip install opencv-python numpy pillow scipy
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from scipy import ndimage
import math

# ── colours ──────────────────────────────────
BG      = "#0f0f13"
SURFACE = "#1a1a22"
CARD    = "#22222e"
ACCENT  = "#7c6af5"
ACCENT2 = "#5dc9a5"
TEXT    = "#e8e6f0"
MUTED   = "#7a7890"
BORDER  = "#2e2e3e"

PANEL_W, PANEL_H = 520, 420
FH = ("Courier New", 13, "bold")
FB = ("Courier New", 11, "bold")
FS = ("Courier New", 10)
FT = ("Courier New", 9)


# ── SliderField: slider + live entry input ────
class SliderField(tk.Frame):
    def __init__(self, parent, label, lo, hi, default, step,
                 callback, fmt="{:.0f}"):
        super().__init__(parent, bg=CARD)
        self._lo   = lo
        self._hi   = hi
        self._step = step
        self._fmt  = fmt
        self._cb   = callback
        self._busy = False

        # row 1 – label + entry
        r1 = tk.Frame(self, bg=CARD)
        r1.pack(fill="x")
        tk.Label(r1, text=label, font=FT, bg=CARD, fg=MUTED,
                 anchor="w").pack(side="left")
        self._evar = tk.StringVar(value=fmt.format(default))
        e = tk.Entry(r1, textvariable=self._evar, font=FB,
                     bg=SURFACE, fg=ACCENT2, insertbackground=ACCENT2,
                     relief="flat", bd=0, width=9,
                     highlightthickness=1,
                     highlightbackground=BORDER,
                     highlightcolor=ACCENT,
                     justify="right")
        e.pack(side="right", ipady=3)
        e.bind("<Return>",   self._from_entry)
        e.bind("<FocusOut>", self._from_entry)
        e.bind("<Up>",   lambda _: self._nudge(+step))
        e.bind("<Down>", lambda _: self._nudge(-step))

        # row 2 – range hints + slider
        r2 = tk.Frame(self, bg=CARD)
        r2.pack(fill="x", pady=(1, 0))
        tk.Label(r2, text=fmt.format(lo), font=("Courier New", 8),
                 bg=CARD, fg=BORDER).pack(side="left")
        tk.Label(r2, text=fmt.format(hi), font=("Courier New", 8),
                 bg=CARD, fg=BORDER).pack(side="right")

        self._svar = tk.DoubleVar(value=default)
        tk.Scale(r2, variable=self._svar, from_=lo, to=hi,
                 resolution=step, orient="horizontal",
                 bg=CARD, fg=TEXT, troughcolor=SURFACE,
                 highlightthickness=0, bd=0, sliderrelief="flat",
                 activebackground=ACCENT, showvalue=False,
                 command=self._from_slider).pack(fill="x", padx=4)

    def _from_slider(self, v):
        if self._busy: return
        self._busy = True
        self._evar.set(self._fmt.format(float(v)))
        self._busy = False
        self._cb()

    def _from_entry(self, _=None):
        if self._busy: return
        try:
            v = float(self._evar.get())
            v = max(self._lo, min(self._hi, v))
            v = round(v / self._step) * self._step
            self._busy = True
            self._svar.set(v)
            self._evar.set(self._fmt.format(v))
            self._busy = False
            self._cb()
        except ValueError:
            self._evar.set(self._fmt.format(self._svar.get()))

    def _nudge(self, d):
        v = max(self._lo, min(self._hi, self._svar.get() + d))
        self._svar.set(v); self._evar.set(self._fmt.format(v))
        self._cb()

    def get(self):  return self._svar.get()
    def set(self, v):
        self._svar.set(v); self._evar.set(self._fmt.format(v))


# ── Main app ──────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Inpainting Pipeline")
        self.configure(bg=BG)
        self.geometry("1380x900")
        self.resizable(True, True)

        self.orig_bgr  = None
        self.gray      = None
        self.edges     = None
        self.dilated   = None
        self.mask      = None
        self.inpainted = None
        self._timer    = None

        self._build()

    # ── top bar ──────────────────────────────
    def _build(self):
        bar = tk.Frame(self, bg=BG, padx=20, pady=10)
        bar.pack(fill="x")
        tk.Label(bar, text="// INPAINT PIPELINE",
                 font=("Courier New", 15, "bold"),
                 bg=BG, fg=ACCENT).pack(side="left")
        tk.Button(bar, text="[ LOAD IMAGE ]", font=FB,
                  bg=CARD, fg=ACCENT2, relief="flat",
                  padx=14, pady=5, cursor="hand2",
                  activebackground=ACCENT2, activeforeground=BG,
                  command=self._load).pack(side="right")
        self._status = tk.StringVar(value="No image loaded")
        tk.Label(bar, textvariable=self._status, font=FT,
                 bg=BG, fg=MUTED).pack(side="right", padx=20)

        # notebook
        sty = ttk.Style(self)
        sty.theme_use("default")
        sty.configure("P.TNotebook", background=BG, borderwidth=0,
                      tabmargins=[0, 0, 0, 0])
        sty.configure("P.TNotebook.Tab", background=SURFACE,
                      foreground=MUTED, font=FB,
                      padding=[14, 7], borderwidth=0)
        sty.map("P.TNotebook.Tab",
                background=[("selected", CARD)],
                foreground=[("selected", ACCENT)])

        self.nb = ttk.Notebook(self, style="P.TNotebook")
        self.nb.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self._tab_original()
        self._tab_gray()
        self._tab_canny()
        self._tab_dilation()
        self._tab_mask()
        self._tab_inpaint()

        self.nb.bind("<<NotebookTabChanged>>", self._on_tab)

    # ── layout helper: returns (ctrl_frame, img_label, info_label) ──
    def _page(self, tab_title, desc, ctrl_w=300):
        outer = tk.Frame(self.nb, bg=CARD)
        self.nb.add(outer, text=tab_title)

        # ── left panel (plain Frame – no canvas/scrollbar) ──
        left = tk.Frame(outer, bg=CARD, width=ctrl_w)
        left.pack(side="left", fill="y", padx=(14, 6), pady=14)
        left.pack_propagate(False)

        # title
        tk.Label(left, text=tab_title.split("·")[-1].strip(),
                 font=FH, bg=CARD, fg=ACCENT,
                 anchor="w").pack(fill="x")
        tk.Label(left, text=desc, font=FT, bg=CARD, fg=MUTED,
                 anchor="w", wraplength=ctrl_w - 10,
                 justify="left").pack(fill="x", pady=(2, 6))
        tk.Frame(left, bg=BORDER, height=1).pack(fill="x", pady=4)

        ctrl = tk.Frame(left, bg=CARD)
        ctrl.pack(fill="both", expand=True)

        # ── right panel ──
        right = tk.Frame(outer, bg=BG)
        right.pack(side="left", fill="both", expand=True,
                   padx=(0, 14), pady=14)

        img_wrap = tk.Frame(right, bg=SURFACE,
                            highlightthickness=1,
                            highlightbackground=BORDER)
        img_wrap.pack(fill="both", expand=True)

        img_lbl = tk.Label(img_wrap, bg=SURFACE, fg=MUTED,
                           font=FS, text="Awaiting image…")
        img_lbl.pack(fill="both", expand=True)

        info_lbl = tk.Label(right, text="", font=FT,
                            bg=BG, fg=MUTED, anchor="w")
        info_lbl.pack(fill="x", pady=(4, 0))

        return ctrl, img_lbl, info_lbl

    # ── small helpers ─────────────────────────
    def _sep(self, p):
        tk.Frame(p, bg=BORDER, height=1).pack(fill="x", pady=6)

    def _sec(self, p, t):
        tk.Label(p, text=t, font=("Courier New", 9, "bold"),
                 bg=CARD, fg=ACCENT, anchor="w").pack(fill="x", pady=(8, 2))

    def _sf(self, p, lbl, lo, hi, default, step, cb, fmt="{:.0f}"):
        w = SliderField(p, lbl, lo, hi, default, step,
                        lambda: self._debounce(cb), fmt=fmt)
        w.pack(fill="x", pady=3)
        return w

    def _chk(self, p, text, var, cb):
        tk.Checkbutton(p, text=text, variable=var, font=FT,
                       bg=CARD, fg=TEXT, selectcolor=SURFACE,
                       activebackground=CARD, activeforeground=ACCENT,
                       command=lambda: self._debounce(cb)).pack(anchor="w", pady=2)

    def _radio(self, p, opts, var, cb):
        row = tk.Frame(p, bg=CARD)
        row.pack(fill="x", pady=4)
        for txt, val in opts:
            tk.Radiobutton(row, text=txt, value=val, variable=var,
                           font=FT, bg=CARD, fg=TEXT,
                           selectcolor=SURFACE,
                           activebackground=CARD,
                           activeforeground=ACCENT,
                           indicatoron=0, relief="flat",
                           padx=8, pady=3,
                           command=lambda: self._debounce(cb)
                           ).pack(side="left", padx=(0, 4))

    # ── tabs ──────────────────────────────────
    def _tab_original(self):
        ctrl, lbl, info = self._page(
            "01 · ORIGINAL",
            "Load any PNG / JPG. The original is preserved across all stages.")
        tk.Button(ctrl, text="[ LOAD IMAGE ]", font=FB,
                  bg=SURFACE, fg=ACCENT2, relief="flat",
                  padx=12, pady=5, cursor="hand2",
                  command=self._load).pack(anchor="w", pady=8)
        self._lbl_orig = lbl; self._info_orig = info

    def _tab_gray(self):
        ctrl, lbl, info = self._page(
            "02 · GRAYSCALE",
            "Convert to single-channel grayscale. "
            "Optional Gaussian blur reduces sensor noise before edge detection.")
        self._sec(ctrl, "GAUSSIAN PRE-BLUR")
        self._blur_k = self._sf(ctrl, "Kernel size (odd)",
                                1, 31, 1, 2, self._run_gray)
        self._blur_s = self._sf(ctrl, "Sigma",
                                0.0, 10.0, 0.0, 0.1,
                                self._run_gray, fmt="{:.1f}")
        tk.Label(ctrl, text="Set kernel = 1 to skip blur.",
                 font=FT, bg=CARD, fg=MUTED,
                 anchor="w").pack(fill="x", pady=(4, 0))
        self._lbl_gray = lbl; self._info_gray = info

    def _tab_canny(self):
        ctrl, lbl, info = self._page(
            "03 · CANNY EDGE",
            "Variance-based smoothing (matches 3D Slicer's "
            "itk::CannyEdgeDetectionImageFilter). "
            "Variance → σ → Gaussian kernel applied before gradient.",
            ctrl_w=320)

        self._sec(ctrl, "VARIANCE  (3D Slicer parameter)")
        tk.Label(ctrl,
                 text="Variance = σ².  Higher value → stronger\n"
                      "smoothing → fewer noisy/fine edges.",
                 font=FT, bg=CARD, fg=MUTED,
                 justify="left").pack(fill="x", pady=(0, 4))
        self._canny_var    = self._sf(ctrl, "Variance (σ²)",
                                      0.01, 20.0, 1.0, 0.01,
                                      self._run_canny, fmt="{:.2f}")
        self._canny_maxerr = self._sf(ctrl, "Max error (kernel cutoff)",
                                      0.001, 0.5, 0.01, 0.001,
                                      self._run_canny, fmt="{:.3f}")

        # live derived value display
        self._sigma_info = tk.StringVar(value="σ=1.000  kernel=7")
        tk.Label(ctrl, textvariable=self._sigma_info,
                 font=FT, bg=CARD, fg=ACCENT2,
                 anchor="w").pack(fill="x", pady=(2, 0))

        self._sep(ctrl)
        self._sec(ctrl, "THRESHOLD")
        self._canny_lo = self._sf(ctrl, "Lower threshold",
                                  0, 500, 50, 1, self._run_canny)
        self._canny_hi = self._sf(ctrl, "Upper threshold",
                                  0, 500, 150, 1, self._run_canny)

        self._sep(ctrl)
        self._sec(ctrl, "APERTURE  &  GRADIENT")
        self._canny_ap = self._sf(ctrl, "Aperture (3 / 5 / 7)",
                                  3, 7, 3, 2, self._run_canny)
        self._canny_l2 = tk.BooleanVar(value=False)
        self._chk(ctrl, "L2 gradient (more accurate)",
                  self._canny_l2, self._run_canny)

        self._lbl_canny = lbl; self._info_canny = info

    def _tab_dilation(self):
        ctrl, lbl, info = self._page(
            "04 · DILATION",
            "Thickens detected edges to close small gaps before hole-filling.")
        self._sec(ctrl, "KERNEL SHAPE")
        self._dil_shape = tk.StringVar(value="Ellipse")
        self._radio(ctrl,
                    [("Rect", "Rect"),
                     ("Ellipse", "Ellipse"),
                     ("Cross", "Cross")],
                    self._dil_shape, self._run_dilation)
        self._sep(ctrl)
        self._sec(ctrl, "SIZE  &  ITERATIONS")
        self._dil_k = self._sf(ctrl, "Kernel size (odd)",
                               1, 31, 3, 2, self._run_dilation)
        self._dil_i = self._sf(ctrl, "Iterations",
                               1, 20, 1, 1, self._run_dilation)
        self._lbl_dil = lbl; self._info_dil = info

    def _tab_mask(self):
        ctrl, lbl, info = self._page(
            "05 · MASK  –  BINARY FILL HOLE",
            "scipy.ndimage.binary_fill_holes fills every enclosed region "
            "automatically — no seed point needed. "
            "Equivalent to 3D Slicer's BinaryFillHoleImageFilter.",
            ctrl_w=320)

        self._sec(ctrl, "CONNECTIVITY  (structuring element)")
        tk.Label(ctrl,
                 text="Cross = 4-conn  (3D Slicer default)\n"
                      "Square = 8-conn (fully connected)",
                 font=FT, bg=CARD, fg=MUTED,
                 justify="left").pack(fill="x", pady=(0, 4))
        self._fill_conn = tk.StringVar(value="Cross")
        self._radio(ctrl,
                    [("Cross (4-conn)", "Cross"),
                     ("Square (8-conn)", "Square")],
                    self._fill_conn, self._run_mask)

        self._sep(ctrl)
        self._sec(ctrl, "POST-FILL MORPHOLOGY")
        tk.Label(ctrl,
                 text="Erode shrinks mask edges; dilate expands them.",
                 font=FT, bg=CARD, fg=MUTED).pack(fill="x", pady=(0, 4))
        self._fill_erode  = self._sf(ctrl, "Erode iterations",
                                     0, 20, 0, 1, self._run_mask)
        self._fill_dilate = self._sf(ctrl, "Dilate iterations",
                                     0, 20, 0, 1, self._run_mask)

        self._sep(ctrl)
        self._sec(ctrl, "DISPLAY")
        self._mask_overlay = tk.BooleanVar(value=True)
        self._chk(ctrl, "Show edge overlay (green)",
                  self._mask_overlay, self._run_mask)
        self._mask_invert = tk.BooleanVar(value=False)
        self._chk(ctrl, "Invert mask",
                  self._mask_invert, self._run_mask)

        self._lbl_mask = lbl; self._info_mask = info

    def _tab_inpaint(self):
        ctrl, lbl, info = self._page(
            "06 · INPAINT",
            "Fill masked region. Fast Marching works well for textures; "
            "Navier-Stokes better preserves sharp edges.")

        self._sec(ctrl, "METHOD")
        self._inp_method = tk.StringVar(value="TELEA")
        self._radio(ctrl,
                    [("Fast Marching (TELEA)", "TELEA"),
                     ("Navier-Stokes (NS)", "NS")],
                    self._inp_method, self._run_inpaint)

        self._sep(ctrl)
        self._sec(ctrl, "PARAMETERS")
        self._inp_r = self._sf(ctrl, "Inpaint radius (px)",
                               1, 50, 5, 1, self._run_inpaint)

        self._sep(ctrl)
        self._sec(ctrl, "EXPORT")
        for label, cmd in [("[ SAVE RESULT ]", self._save_result),
                           ("[ SAVE MASK ]",   self._save_mask),
                           ("[ SAVE EDGES ]",  self._save_edges)]:
            tk.Button(ctrl, text=label, font=FB,
                      bg=SURFACE, fg=ACCENT2, relief="flat",
                      padx=12, pady=4, cursor="hand2",
                      command=cmd).pack(anchor="w", pady=3)

        self._lbl_inp = lbl; self._info_inp = info

    # ── load ──────────────────────────────────
    def _load(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                       ("All", "*.*")])
        if not path: return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Error", "Could not load image.")
            return
        self.orig_bgr = img
        h, w = img.shape[:2]
        self._status.set(f"{w}×{h}px  ·  {path.split('/')[-1]}")
        self._show(self._lbl_orig, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self._info_orig.config(text=f"Size: {w}×{h}  Channels: 3")
        self._run_gray()

    # ── pipeline stages ───────────────────────
    def _run_gray(self):
        if self.orig_bgr is None: return
        g = cv2.cvtColor(self.orig_bgr, cv2.COLOR_BGR2GRAY)
        k = int(self._blur_k.get())
        k = k if k % 2 == 1 else k + 1
        k = max(1, k)
        if k > 1:
            s = self._blur_s.get()
            g = cv2.GaussianBlur(g, (k, k), s if s > 0 else 0)
        self.gray = g
        self._show(self._lbl_gray, g, gray=True)
        self._info_gray.config(
            text=f"Blur kernel: {k}  σ: {self._blur_s.get():.1f}")
        self._run_canny()

    def _run_canny(self):
        if self.gray is None: return
        variance  = max(1e-4, self._canny_var.get())
        sigma     = math.sqrt(variance)
        max_error = max(1e-6, self._canny_maxerr.get())

        # kernel size derived from σ and max_error
        # (mirrors ITK CannyEdgeDetectionImageFilter)
        half  = sigma * math.sqrt(-2.0 * math.log(max_error))
        ksize = 2 * int(math.ceil(half)) + 1
        ksize = max(3, ksize if ksize % 2 == 1 else ksize + 1)
        ksize = min(ksize, 31)

        smoothed = cv2.GaussianBlur(self.gray, (ksize, ksize), sigma)

        lo = int(self._canny_lo.get())
        hi = int(self._canny_hi.get())
        ap = int(self._canny_ap.get())
        ap = ap if ap % 2 == 1 else ap + 1
        ap = max(3, min(7, ap))
        l2 = self._canny_l2.get()

        self.edges = cv2.Canny(smoothed, lo, hi,
                               apertureSize=ap, L2gradient=l2)
        cnt = int(np.sum(self.edges > 0))
        self._sigma_info.set(f"σ = {sigma:.3f}   kernel = {ksize}")
        self._show(self._lbl_canny, self.edges, gray=True)
        self._info_canny.config(
            text=f"σ={sigma:.3f}  ksize={ksize}  lo={lo}  "
                 f"hi={hi}  ap={ap}  |  edge px: {cnt:,}")
        self._run_dilation()

    def _run_dilation(self):
        if self.edges is None: return
        smap = {"Rect":    cv2.MORPH_RECT,
                "Ellipse": cv2.MORPH_ELLIPSE,
                "Cross":   cv2.MORPH_CROSS}
        shape = smap.get(self._dil_shape.get(), cv2.MORPH_ELLIPSE)
        k = int(self._dil_k.get())
        k = k if k % 2 == 1 else k + 1
        k = max(1, k)
        iters = int(self._dil_i.get())
        kern  = cv2.getStructuringElement(shape, (k, k))
        dil   = cv2.dilate(self.edges, kern, iterations=iters)
        self.dilated = dil
        self._show(self._lbl_dil, dil, gray=True)
        self._info_dil.config(
            text=f"Shape: {self._dil_shape.get()}  k={k}  "
                 f"iters={iters}  edge px: {int(np.sum(dil>0)):,}")
        self._run_mask()

    def _run_mask(self):
        if self.dilated is None: return
        binary = self.dilated > 0
        struct = (np.ones((3, 3), dtype=bool)
                  if self._fill_conn.get() == "Square"
                  else ndimage.generate_binary_structure(2, 1))

        filled = (ndimage.binary_fill_holes(binary, structure=struct)
                  .astype(np.uint8) * 255)

        en = int(self._fill_erode.get())
        dn = int(self._fill_dilate.get())
        if en > 0:
            ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            filled = cv2.erode(filled, ke, iterations=en)
        if dn > 0:
            kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            filled = cv2.dilate(filled, kd, iterations=dn)

        if self._mask_invert.get():
            filled = cv2.bitwise_not(filled)

        self.mask = filled
        h, w  = filled.shape
        area  = int(np.sum(filled > 0))

        if self._mask_overlay.get():
            disp = cv2.cvtColor(filled, cv2.COLOR_GRAY2RGB)
            disp[self.dilated > 0] = [80, 220, 130]
            self._show(self._lbl_mask, disp)
        else:
            self._show(self._lbl_mask, filled, gray=True)

        conn = ("8-conn" if self._fill_conn.get() == "Square"
                else "4-conn")
        self._info_mask.config(
            text=f"BinaryFillHole  {conn}  |  "
                 f"area: {area:,} px  ({100*area/(w*h):.1f}%)")
        self._run_inpaint()

    def _run_inpaint(self):
        if self.orig_bgr is None or self.mask is None: return
        r   = int(self._inp_r.get())
        m   = (cv2.INPAINT_NS
               if self._inp_method.get() == "NS"
               else cv2.INPAINT_TELEA)
        res = cv2.inpaint(self.orig_bgr, self.mask, r, m)
        self.inpainted = res
        self._show(self._lbl_inp, cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
        mname = ("Navier-Stokes" if self._inp_method.get() == "NS"
                 else "Fast Marching")
        h, w  = res.shape[:2]
        self._info_inp.config(
            text=f"Method: {mname}  radius: {r}  output: {w}×{h}")

    # ── debounce ──────────────────────────────
    def _debounce(self, fn):
        if self._timer: self.after_cancel(self._timer)
        self._timer = self.after(120, fn)

    def _on_tab(self, _=None):
        fns = [None,
               self._run_gray,
               self._run_canny,
               self._run_dilation,
               self._run_mask,
               self._run_inpaint]
        idx = self.nb.index("current")
        if idx < len(fns) and fns[idx]:
            fns[idx]()

    # ── display ───────────────────────────────
    def _show(self, label, img, gray=False):
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if gray else img
        h, w = rgb.shape[:2]
        scale = min(PANEL_W / w, PANEL_H / h, 1.0)
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        r  = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        tk_img = ImageTk.PhotoImage(Image.fromarray(r))
        label.config(image=tk_img, text="")
        label._img = tk_img       # prevent GC

    # ── save ──────────────────────────────────
    def _save_result(self):
        if self.inpainted is None:
            messagebox.showinfo("Nothing to save", "Run pipeline first.")
            return
        p = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if p:
            cv2.imwrite(p, self.inpainted)
            messagebox.showinfo("Saved", f"Result saved:\n{p}")

    def _save_mask(self):
        if self.mask is None:
            messagebox.showinfo("Nothing to save", "Run pipeline first.")
            return
        p = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png")])
        if p:
            cv2.imwrite(p, self.mask)
            messagebox.showinfo("Saved", f"Mask saved:\n{p}")

    def _save_edges(self):
        if self.edges is None:
            messagebox.showinfo("Nothing to save", "Run pipeline first.")
            return
        p = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png")])
        if p:
            cv2.imwrite(p, self.edges)
            messagebox.showinfo("Saved", f"Edges saved:\n{p}")


if __name__ == "__main__":
    app = App()
    app.mainloop()
