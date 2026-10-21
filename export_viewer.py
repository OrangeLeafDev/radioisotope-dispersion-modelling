import os
import re
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

class PredictiveMapBrowserApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Santa Cruz River Basin — Predictive Map & Time Browser")
        self.geometry("1100x800")

        self.output_dir = Path("outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Control Panel Frame
        control_frame = ttk.LabelFrame(self, text="Interactive Map & Time Selector (Use Left/Right Arrows to Cycle)", padding=12)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=12, pady=12)

        # 1. Visualization Type Selector
        ttk.Label(control_frame, text="Visualization Type:").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.map_mode_var = tk.StringVar(value="2D Contour Map")
        modes = [
            "2D Contour Map", 
            "3D Terrain & Pseudo-3D Layers"
        ]
        self.mode_dropdown = ttk.Combobox(control_frame, textvariable=self.map_mode_var, values=modes, width=34, state="readonly")
        self.mode_dropdown.grid(row=0, column=1, padx=5, sticky=tk.W)
        self.mode_dropdown.bind("<<ComboboxSelected>>", lambda e: self.load_predictive_image())
        
        # Bind Left/Right arrow keys for instant cycling without dropping down
        self.mode_dropdown.bind("<Left>", lambda e: self.cycle_combobox(self.mode_dropdown, -1))
        self.mode_dropdown.bind("<Right>", lambda e: self.cycle_combobox(self.mode_dropdown, 1))

        # 2. Available Time Selection Dropdown (Auto-scanned and sorted by duration)
        ttk.Label(control_frame, text="Available Time Intervals:").grid(row=1, column=0, padx=5, pady=8, sticky=tk.W)
        self.time_combo_var = tk.StringVar()
        self.time_dropdown = ttk.Combobox(control_frame, textvariable=self.time_combo_var, width=34, state="readonly")
        self.time_dropdown.grid(row=1, column=1, padx=5, pady=8, sticky=tk.W)
        self.time_dropdown.bind("<<ComboboxSelected>>", lambda e: self.load_predictive_image())
        
        # Bind Left/Right arrow keys for time interval cycling
        self.time_dropdown.bind("<Left>", lambda e: self.cycle_combobox(self.time_dropdown, -1))
        self.time_dropdown.bind("<Right>", lambda e: self.cycle_combobox(self.time_dropdown, 1))

        # Refresh button
        refresh_btn = ttk.Button(control_frame, text="Scan outputs/ Folder", command=self.scan_and_populate)
        refresh_btn.grid(row=0, column=2, rowspan=2, padx=15, sticky=tk.NS)

        # Image Display Area
        self.display_frame = ttk.LabelFrame(self, text="Visual Output Preview", padding=10)
        self.display_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=12, pady=5)

        self.lbl_image = ttk.Label(self.display_frame, text="Scanning outputs/ for exported maps...", anchor=tk.CENTER)
        self.lbl_image.pack(fill=tk.BOTH, expand=True)

        # Status Bar
        self.status_var = tk.StringVar(value="Ready. (Tip: Use Left/Right Arrow keys on focused dropdowns to cycle instantly!)")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=5)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.scan_and_populate()

    def cycle_combobox(self, combo, direction):
        """Cycles through combobox values using Left/Right arrow keys without opening the dropdown list."""
        vals = combo["values"]
        if not vals:
            return "break"
        current_val = combo.get()
        if current_val in vals:
            idx = vals.index(current_val)
            new_idx = max(0, min(len(vals) - 1, idx + direction))
            combo.set(vals[new_idx])
        else:
            combo.set(vals[0])
        
        # Trigger image reload
        self.load_predictive_image()
        return "break"  # Prevent default dropdown popup behavior

    def parse_to_seconds(self, time_str):
        """Helper to convert a time string like '90 minutes' or '14 days' into total seconds for sorting."""
        parts = time_str.strip().lower().split()
        if len(parts) < 2:
            return 0.0
        try:
            val = float(parts[0])
            unit = parts[1]
            if unit in ["day", "days", "d"]:
                return val * 86400.0
            elif unit in ["hour", "hours", "h", "hr", "hrs"]:
                return val * 3600.0
            elif unit in ["minute", "minutes", "m", "min", "mins"]:
                return val * 60.0
            elif unit in ["second", "seconds", "s", "sec", "secs"]:
                return val
        except ValueError:
            pass
        return 0.0

    def scan_and_populate(self):
        if not self.output_dir.exists():
            self.status_var.set("Outputs folder does not exist yet. Run simulation scripts first.")
            self.time_dropdown["values"] = []
            return

        files = list(self.output_dir.glob("predictive_*.png"))
        time_options = set()

        for f in files:
            match = re.search(r"predictive_(?:2d_map|3d_and_pseudo3d)_([^_]+)_([^_.]+)\.png", f.name)
            if match:
                val, unit = match.groups()
                time_options.add(f"{val} {unit}")

        # Sort available time intervals precisely by total duration in seconds (shortest to longest)
        sorted_times = sorted(list(time_options), key=self.parse_to_seconds)
        
        if not sorted_times:
            sorted_times = ["90 minutes", "48 hours", "14 days"]

        self.time_dropdown["values"] = sorted_times
        if sorted_times:
            self.time_dropdown.current(0)
            self.load_predictive_image()

        self.status_var.set(f"Scanned outputs/. Found {len(files)} image files across {len(sorted_times)} time intervals (sorted by duration).")

    def load_predictive_image(self):
        selected_time = self.time_combo_var.get()
        mode = self.map_mode_var.get()

        if not selected_time:
            return

        parts = selected_time.split()
        if len(parts) >= 2:
            val, unit = parts[0], parts[1]
        else:
            return

        if "2D" in mode:
            filename = f"predictive_2d_map_{val}_{unit}.png"
        else:
            filename = f"predictive_3d_and_pseudo3d_{val}_{unit}.png"

        img_path = self.output_dir / filename

        if not img_path.exists():
            matches = list(self.output_dir.glob(f"*{val}*{unit}*.png"))
            if matches:
                sub_matches = [m for m in matches if ("2d" in m.name and "2D" in mode) or ("3d" in m.name and "3D" in mode)]
                img_path = sub_matches[0] if sub_matches else matches[0]
            else:
                self.lbl_image.config(image="", text=f"Image not found:\n{filename}\n\nRun simulation for {selected_time} to generate it.")
                self.status_var.set(f"Missing: {filename}")
                return

        try:
            pil_img = Image.open(img_path)
            w_max, h_max = 980, 620
            pil_img.thumbnail((w_max, h_max), Image.Resampling.LANCZOS)
            
            self.tk_img = ImageTk.PhotoImage(pil_img)
            self.lbl_image.config(image=self.tk_img, text="")
            self.status_var.set(f"Displaying: {img_path.name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")

if __name__ == "__main__":
    app = PredictiveMapBrowserApp()
    app.mainloop()