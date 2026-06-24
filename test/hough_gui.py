#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os

class HoughApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hough Line Detection Tool")
        self.root.geometry("1150x780")
        self.root.minsize(900, 600)
        
        # State variables
        self.original_cv_image = None  # Full-resolution original image (BGR)
        self.working_image = None      # Resized image used for processing (BGR)
        self.display_scale = 1.0       # Scale factor applied to fit screen
        
        # Setup UI styles
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Define clean, modern color scheme for Tkinter widgets
        self.style.configure('.', font=('Helvetica', 10))
        self.style.configure('TFrame', background='#f5f5f7')
        self.style.configure('TLabel', background='#f5f5f7', foreground='#1d1d1f')
        self.style.configure('Header.TLabel', font=('Helvetica', 12, 'bold'), foreground='#1d1d1f')
        self.style.configure('Status.TLabel', font=('Helvetica', 10, 'italic'), foreground='#515154')
        self.style.configure('TButton', font=('Helvetica', 10, 'bold'), borderwidth=1, background='#007aff', foreground='#ffffff')
        self.style.map('TButton',
                       foreground=[('pressed', '#ffffff'), ('active', '#ffffff')],
                       background=[('pressed', '#0062cc'), ('active', '#0056b3')])
        
        # Main notebook container for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.tab1 = ttk.Frame(self.notebook, style='TFrame')
        self.tab2 = ttk.Frame(self.notebook, style='TFrame')
        
        self.notebook.add(self.tab1, text="  Original Image  ")
        self.notebook.add(self.tab2, text="  Hough Line Detection  ")
        
        # Build tabs
        self.setup_tab1()
        self.setup_tab2()
        
        # Bind tab change event to trigger update when moving to tab 2
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

    def setup_tab1(self):
        # Top toolbar frame
        toolbar = ttk.Frame(self.tab1, padding=10, style='TFrame')
        toolbar.pack(side=tk.TOP, fill=tk.X)
        
        load_btn = ttk.Button(toolbar, text="Load Image", command=self.load_image)
        load_btn.pack(side=tk.LEFT, padx=5)
        
        test_btn = ttk.Button(toolbar, text="Generate Test Image", command=self.generate_test_image)
        test_btn.pack(side=tk.LEFT, padx=5)
        
        self.file_path_lbl = ttk.Label(toolbar, text="No image loaded.", font=('Helvetica', 10, 'italic'))
        self.file_path_lbl.pack(side=tk.LEFT, padx=15)
        
        # Image view area
        self.img1_container = ttk.Frame(self.tab1, padding=10, relief=tk.SUNKEN, borderwidth=1)
        self.img1_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.img1_lbl = ttk.Label(self.img1_container, text="Please load an image to begin.", font=('Helvetica', 12))
        self.img1_lbl.pack(expand=True)

    def setup_tab2(self):
        # Two-column layout: Left for parameters/controls, Right for processed image
        self.tab2.columnconfigure(0, weight=1, minsize=350)
        self.tab2.columnconfigure(1, weight=3)
        self.tab2.rowconfigure(0, weight=1)
        
        # Left Panel (Controls)
        ctrl_panel = ttk.Frame(self.tab2, padding=15, relief=tk.SOLID, borderwidth=1, style='TFrame')
        ctrl_panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Title
        title_lbl = ttk.Label(ctrl_panel, text="Hough Parameters", style='Header.TLabel')
        title_lbl.pack(anchor=tk.W, pady=(0, 15))
        
        # Scrollable area for sliders if needed, but a regular frame fits perfectly here
        sliders_frame = ttk.Frame(ctrl_panel, style='TFrame')
        sliders_frame.pack(fill=tk.BOTH, expand=True)
        
        # Helper to create a slider with label and real-time value display
        def create_slider(parent, label_text, from_val, to_val, default_val, resolution=1):
            frame = ttk.Frame(parent, style='TFrame')
            frame.pack(fill=tk.X, pady=8)
            
            lbl_row = ttk.Frame(frame, style='TFrame')
            lbl_row.pack(fill=tk.X)
            
            name_lbl = ttk.Label(lbl_row, text=label_text, font=('Helvetica', 10, 'bold'))
            name_lbl.pack(side=tk.LEFT)
            
            val_lbl = ttk.Label(lbl_row, text=str(default_val), font=('Helvetica', 10))
            val_lbl.pack(side=tk.RIGHT)
            
            slider = ttk.Scale(frame, from_=from_val, to=to_val, value=default_val, 
                               orient=tk.HORIZONTAL, command=lambda v: self.on_slider_change(slider, val_lbl, resolution))
            slider.pack(fill=tk.X, pady=(2, 0))
            return slider, val_lbl

        # Sliders creation
        self.canny_low_slider, self.canny_low_val_lbl = create_slider(sliders_frame, "Canny Lower Threshold", 0, 255, 50)
        self.canny_high_slider, self.canny_high_val_lbl = create_slider(sliders_frame, "Canny Upper Threshold", 0, 255, 150)
        self.rho_slider, self.rho_val_lbl = create_slider(sliders_frame, "Rho (Distance Res in px)", 1, 10, 1)
        self.theta_slider, self.theta_val_lbl = create_slider(sliders_frame, "Theta (Angle Res in deg)", 1, 90, 1)
        self.threshold_slider, self.threshold_val_lbl = create_slider(sliders_frame, "Hough Threshold (Votes)", 1, 300, 50)
        self.min_line_len_slider, self.min_line_len_val_lbl = create_slider(sliders_frame, "Min Line Length (px)", 0, 300, 50)
        self.max_line_gap_slider, self.max_line_gap_val_lbl = create_slider(sliders_frame, "Max Line Gap (px)", 0, 150, 10)
        
        # Display Mode combobox
        mode_frame = ttk.Frame(sliders_frame, style='TFrame')
        mode_frame.pack(fill=tk.X, pady=12)
        mode_lbl = ttk.Label(mode_frame, text="Display Mode:", font=('Helvetica', 10, 'bold'))
        mode_lbl.pack(anchor=tk.W)
        
        self.display_mode_combobox = ttk.Combobox(mode_frame, values=[
            "Show Lines on Original Image",
            "Show Lines on Canny Edges",
            "Show Canny Edges Only"
        ], state="readonly")
        self.display_mode_combobox.set("Show Lines on Original Image")
        self.display_mode_combobox.pack(fill=tk.X, pady=(4, 0))
        self.display_mode_combobox.bind("<<ComboboxSelected>>", lambda e: self.update_hough())
        
        # Line width option
        width_frame = ttk.Frame(sliders_frame, style='TFrame')
        width_frame.pack(fill=tk.X, pady=8)
        width_lbl = ttk.Label(width_frame, text="Line Thickness:", font=('Helvetica', 10, 'bold'))
        width_lbl.pack(side=tk.LEFT)
        self.line_thickness_spin = ttk.Spinbox(width_frame, from_=1, to=10, width=5, command=self.update_hough)
        self.line_thickness_spin.set(2)
        self.line_thickness_spin.pack(side=tk.RIGHT)
        self.line_thickness_spin.bind("<KeyRelease>", lambda e: self.update_hough())
        
        # Status & Line Count Info
        self.status_lbl = ttk.Label(ctrl_panel, text="Lines detected: 0", style='Status.TLabel', font=('Helvetica', 11, 'bold'))
        self.status_lbl.pack(side=tk.BOTTOM, fill=tk.X, pady=(15, 0))
        
        # Right Panel (Processed Image Display)
        self.img2_container = ttk.Frame(self.tab2, padding=10, relief=tk.SUNKEN, borderwidth=1)
        self.img2_container.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        self.img2_lbl = ttk.Label(self.img2_container, text="Please load an image in the first tab.", font=('Helvetica', 12))
        self.img2_lbl.pack(expand=True)

    def on_slider_change(self, slider, val_lbl, resolution):
        # Round the slider value to match resolution
        val = slider.get()
        if resolution == 1:
            val = int(round(val))
            slider.set(val)
            val_lbl.config(text=str(val))
        else:
            val = round(val, 2)
            slider.set(val)
            val_lbl.config(text=str(val))
        
        # Run Hough update
        self.update_hough()

    def on_tab_changed(self, event):
        selected_tab = self.notebook.tab(self.notebook.select(), "text").strip()
        if selected_tab == "Hough Line Detection":
            self.update_hough()

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.webp *.tiff *.tif")]
        )
        if not file_path:
            return
            
        cv_img = cv2.imread(file_path)
        if cv_img is None:
            messagebox.showerror("Error", "Could not read the selected image file.")
            return
            
        self.process_and_store_image(cv_img, file_path)

    def generate_test_image(self):
        # Create a synthetic image with distinct lines for testing
        w, h = 600, 450
        img = np.ones((h, w, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Draw some lines and shapes
        cv2.line(img, (50, 50), (550, 50), (0, 0, 0), 2)     # Horizontal top
        cv2.line(img, (50, 400), (550, 400), (0, 0, 0), 2)   # Horizontal bottom
        cv2.line(img, (100, 50), (100, 400), (0, 0, 0), 2)   # Vertical left
        cv2.line(img, (500, 50), (500, 400), (0, 0, 0), 2)   # Vertical right
        
        # Diagonals
        cv2.line(img, (100, 100), (500, 350), (50, 50, 50), 2)
        cv2.line(img, (100, 350), (500, 100), (50, 50, 50), 2)
        
        # Add some random noise and circles to make Canny/Hough tweaking interesting
        cv2.circle(img, (300, 225), 60, (100, 100, 255), 2)  # Circle (should not be detected as lines)
        cv2.putText(img, "Hough Transform Test", (180, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 0), 2)
        
        self.process_and_store_image(img, "Synthetic Test Image")

    def process_and_store_image(self, cv_img, file_name):
        self.original_cv_image = cv_img.copy()
        
        # Scale image to fit inside 700x500 window for responsive rendering and processing
        max_w, max_h = 700, 500
        h, w = self.original_cv_image.shape[:2]
        scale = min(max_w / w, max_h / h)
        
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            self.working_image = cv2.resize(self.original_cv_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            self.display_scale = scale
        else:
            self.working_image = self.original_cv_image.copy()
            self.display_scale = 1.0
            
        # Update Tab 1 original image view
        self.display_original_image()
        
        # Update file name label
        self.file_path_lbl.config(text=f"Loaded: {os.path.basename(file_name)}")
        
        # Switch display message on Tab 2 if it was empty
        self.img2_lbl.config(text="Processing image...")
        
        # Run Hough update
        self.update_hough()

    def display_original_image(self):
        if self.working_image is None:
            return
            
        # Convert BGR (OpenCV) to RGB (Tkinter/PIL)
        rgb_img = cv2.cvtColor(self.working_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        tk_img = ImageTk.PhotoImage(image=pil_img)
        
        self.img1_lbl.config(image=tk_img, text="")
        self.img1_lbl.image = tk_img  # Store reference

    def display_processed_image(self, cv_img):
        # Convert BGR (OpenCV) to RGB (Tkinter/PIL)
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        tk_img = ImageTk.PhotoImage(image=pil_img)
        
        self.img2_lbl.config(image=tk_img, text="")
        self.img2_lbl.image = tk_img  # Store reference

    def update_hough(self):
        if self.working_image is None:
            self.status_lbl.config(text="Lines detected: 0 (No image loaded)")
            return
            
        # Get parameter values
        canny_low = int(self.canny_low_slider.get())
        canny_high = int(self.canny_high_slider.get())
        rho = int(self.rho_slider.get())
        theta_deg = int(self.theta_slider.get())
        threshold = int(self.threshold_slider.get())
        min_line_len = int(self.min_line_len_slider.get())
        max_line_gap = int(self.max_line_gap_slider.get())
        
        try:
            line_thickness = int(self.line_thickness_spin.get())
        except ValueError:
            line_thickness = 2
            
        # Convert theta to radians
        theta = theta_deg * np.pi / 180.0
        
        # Grayscale and blur
        gray = cv2.cvtColor(self.working_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, canny_low, canny_high)
        
        # Hough Lines P
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, 
                                minLineLength=min_line_len, 
                                maxLineGap=max_line_gap)
        
        display_mode = self.display_mode_combobox.get()
        num_lines = 0 if lines is None else len(lines)
        
        # Prepare output image
        if display_mode == "Show Canny Edges Only":
            output_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        elif display_mode == "Show Lines on Canny Edges":
            output_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(output_img, (x1, y1), (x2, y2), (0, 0, 255), line_thickness)
        else:  # "Show Lines on Original Image"
            output_img = self.working_image.copy()
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(output_img, (x1, y1), (x2, y2), (0, 0, 255), line_thickness)
                    
        self.status_lbl.config(text=f"Lines detected: {num_lines}")
        self.display_processed_image(output_img)

if __name__ == "__main__":
    root = tk.Tk()
    app = HoughApp(root)
    root.mainloop()
