# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 13:53:42 2025

@author: mjpreciado
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import csv
import os

class RainwaterHarvestingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Rainwater Harvesting Analysis System")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f8ff')
        
        # Initial parameters
        self.dt = 1  # minutes
        self.num_barras = 10
        self.intensidades = np.linspace(0, 10, self.num_barras)
        self.vol_total = 1.7671
        
        # Projected areas
        self.area_copa = np.pi * 3**2
        self.area_ramas = np.pi * (1.5**2 - 0.5**2)
        self.area_tronco = np.pi * 0.5**2
        
        # Interception coefficients
        self.coef_copa = 0.6
        self.coef_ramas = 0.3
        self.coef_tronco = 0.1
        
        # Calculation history
        self.calculation_history = []
        
        self.setup_ui()
        self.calculate_results()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, 
                               text="Advanced Rainwater Harvesting Analysis System", 
                               font=('Arial', 12, 'bold'), 
                               foreground='#2E86AB')
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Control frame
        control_frame = ttk.LabelFrame(main_frame, text="Input Parameters", padding="15")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 15))
        
        # Input controls
        inputs = [
            ("Total rain volume (m³):", "vol_entry", str(self.vol_total)),
            ("Crown radius (m):", "radio_copa_entry", "3"),
            ("Cone radius (m):", "radio_cono_entry", "2.5"),
            ("Crown interception (%):", "coef_copa_entry", "60"),
            ("Branch interception (%):", "coef_ramas_entry", "30"),
            ("Trunk interception (%):", "coef_tronco_entry", "10"),
            ("Number of intervals:", "num_barras_entry", "10"),
            ("Maximum intensity (mm/min):", "max_intensity_entry", "10")
        ]
        
        for i, (label, attr_name, default) in enumerate(inputs):
            ttk.Label(control_frame, text=label, font=('Arial', 8)).grid(
                row=i, column=0, sticky=tk.W, pady=5)
            entry = ttk.Entry(control_frame, width=15, font=('Arial', 8))
            entry.insert(0, default)
            entry.grid(row=i, column=1, pady=5, padx=(10, 0))
            setattr(self, attr_name, entry)
        
        # Hyetogram type selection
        ttk.Label(control_frame, text="Hyetogram Type:", font=('Arial', 8)).grid(
            row=len(inputs), column=0, sticky=tk.W, pady=5)
        self.hyetogram_type = tk.StringVar(value="linear")
        hyetogram_combo = ttk.Combobox(control_frame, textvariable=self.hyetogram_type, 
                                      values=["linear", "triangular"], width=13, state="readonly")
        hyetogram_combo.grid(row=len(inputs), column=1, pady=5, padx=(10, 0))
        
        # Button frame
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=len(inputs)+1, column=0, columnspan=2, pady=15)
        
        ttk.Button(button_frame, text="Calculate", 
                  command=self.recalculate, style='Accent.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Export Results", 
                  command=self.export_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset", 
                  command=self.reset_values).pack(side=tk.LEFT, padx=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Analysis Results", padding="15")
        results_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Results text with scrollbar
        results_container = ttk.Frame(results_frame)
        results_container.pack(fill=tk.BOTH, expand=True)
        
        self.results_text = tk.Text(results_container, height=12, width=55, 
                                   font=('Courier New', 8), wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(results_container, orient="vertical", 
                                 command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Graph frame
        graph_frame = ttk.Frame(main_frame)
        graph_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=15)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(graph_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.efficiency_frame = ttk.Frame(self.notebook)
        self.visualization_frame = ttk.Frame(self.notebook)
        self.visualization_3d_frame = ttk.Frame(self.notebook)
        self.hyetogram_frame = ttk.Frame(self.notebook)
        self.flow_frame = ttk.Frame(self.notebook)
        self.analysis_frame = ttk.Frame(self.notebook)
        
        self.notebook.add(self.efficiency_frame, text="Efficiency Analysis")
        self.notebook.add(self.visualization_frame, text="2D Visualization")
        self.notebook.add(self.visualization_3d_frame, text="3D Visualization")
        self.notebook.add(self.hyetogram_frame, text="Rain Hyetogram")
        self.notebook.add(self.flow_frame, text="Water Distribution")
        self.notebook.add(self.analysis_frame, text="Advanced Analysis")
        
        # Configure weights for responsive
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Create custom style for accent button
        style = ttk.Style()
        style.configure('Accent.TButton', background='#2E86AB', foreground='white')
        
    def calculate_results(self):
        try:
            # Get input values with validation
            self.vol_total = float(self.vol_entry.get())
            radio_copa = float(self.radio_copa_entry.get())
            radio_cono = float(self.radio_cono_entry.get())
            
            # Update rain parameters
            self.num_barras = int(self.num_barras_entry.get())
            max_intensity = float(self.max_intensity_entry.get())
            
            # Generate hyetogram based on selected type
            if self.hyetogram_type.get() == "triangular":
                # Triangular hyetogram (grows and decreases)
                peak_time = self.num_barras // 2
                self.intensidades = np.zeros(self.num_barras)
                for i in range(self.num_barras):
                    if i <= peak_time:
                        self.intensidades[i] = (max_intensity / peak_time) * i
                    else:
                        self.intensidades[i] = max_intensity - (max_intensity / (self.num_barras - peak_time - 1)) * (i - peak_time)
            else:
                # Linear growing hyetogram
                self.intensidades = np.linspace(0, max_intensity, self.num_barras)
            
            # Update coefficients
            self.coef_copa = float(self.coef_copa_entry.get()) / 100
            self.coef_ramas = float(self.coef_ramas_entry.get()) / 100
            self.coef_tronco = float(self.coef_tronco_entry.get()) / 100
            
            # Update areas
            self.area_copa = np.pi * radio_copa**2
            self.area_ramas = np.pi * (1.5**2 - 0.5**2)
            self.area_tronco = np.pi * 0.5**2
            
            # Calculate intercepted volumes
            self.vol_copa = self.vol_ramas = self.vol_tronco = 0
            for intensidad in self.intensidades:
                lluvia_m = (intensidad * self.dt) / 1000
                self.vol_copa += lluvia_m * self.area_copa * self.coef_copa
                self.vol_ramas += lluvia_m * self.area_ramas * self.coef_ramas
                self.vol_tronco += lluvia_m * self.area_tronco * self.coef_tronco
                
            self.vol_interceptado = self.vol_copa + self.vol_ramas + self.vol_tronco
            self.vol_suelo_sin_cono = self.vol_total - self.vol_interceptado
            
            # Volume captured by cone
            self.area_cono = np.pi * radio_cono**2
            self.vol_captado_cono = sum((i * self.dt / 1000) * self.area_cono for i in self.intensidades)
            self.vol_suelo_con_cono = max(0, self.vol_total - self.vol_interceptado - self.vol_captado_cono)
            self.eficiencia_cono = (self.vol_captado_cono / self.vol_total) * 100
            
            # Optimization
            self.r_optimo, self.eficiencia_optima = self.calcular_radio_optimo()
            
            # Store calculation
            self.store_calculation()
            
            # Update displays
            self.update_results_display()
            self.create_efficiency_plot()
            self.create_2d_visualization()
            self.create_3d_visualization()
            self.create_hyetogram_plot()
            self.create_flow_diagram()
            self.create_advanced_analysis()
            
        except ValueError as e:
            messagebox.showerror("Input Error", 
                               "Please enter valid numerical values in all fields.\n\n"
                               "Error details: " + str(e))
        
    def calcular_radio_optimo(self):
        for r in np.arange(0.1, 5.0, 0.01):
            area = np.pi * r**2
            vol = sum((i * self.dt / 1000) * area for i in self.intensidades)
            ef = (vol / self.vol_total) * 100
            if ef >= 90:
                return r, ef
        return 4.5, 95.0
        
    def store_calculation(self):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        calculation = {
            'timestamp': timestamp,
            'volume_total': self.vol_total,
            'crown_radius': float(self.radio_copa_entry.get()),
            'cone_radius': float(self.radio_cono_entry.get()),
            'efficiency': self.eficiencia_cono,
            'optimal_radius': self.r_optimo
        }
        self.calculation_history.append(calculation)
        
    def update_results_display(self):
        crown_radius = float(self.radio_copa_entry.get())
        cone_radius = float(self.radio_cono_entry.get())
        
        # Calculate additional rain metrics
        total_rainfall_mm = sum(self.intensidades) * self.dt
        avg_intensity = np.mean(self.intensidades)
        max_intensity = np.max(self.intensidades)
        
        results = f"""
ANALYSIS RESULTS
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{'='*50}

RAIN CHARACTERISTICS:
• Hyetogram Type: {self.hyetogram_type.get().title()}
• Total Duration: {self.num_barras * self.dt} min
• Total Rainfall: {total_rainfall_mm:.1f} mm
• Average Intensity: {avg_intensity:.1f} mm/min
• Maximum Intensity: {max_intensity:.1f} mm/min
• Time Intervals: {self.num_barras}

INPUT PARAMETERS:
• Total Rain Volume: {self.vol_total:.4f} m³
• Crown Radius: {crown_radius:.2f} m
• Cone Radius: {cone_radius:.2f} m
• Interception Coefficients - Crown: {self.coef_copa*100:.0f}%, 
  Branches: {self.coef_ramas*100:.0f}%, Trunk: {self.coef_tronco*100:.0f}%

WATER DISTRIBUTION:
• Total Rain:           {self.vol_total:>10.4f} m³
• Intercepted by Tree: {self.vol_interceptado:>10.4f} m³
  - Crown:                 {self.vol_copa:>10.4f} m³
  - Branches:                {self.vol_ramas:>10.4f} m³
  - Trunk:               {self.vol_tronco:>10.4f} m³
• Captured by Cone:       {self.vol_captado_cono:>10.4f} m³
• Reaches Ground:
  - Without Cone:             {self.vol_suelo_sin_cono:>10.4f} m³
  - With Cone:             {self.vol_suelo_con_cono:>10.4f} m³

SYSTEM PERFORMANCE:
• Current Efficiency:      {self.eficiencia_cono:>10.2f}%
• Optimal Efficiency:      {self.eficiencia_optima:>10.2f}%
• Optimal Radius:           {self.r_optimo:>10.2f} m

EFFICIENCY IMPROVEMENT: {max(0, self.eficiencia_optima - self.eficiencia_cono):.2f}%
"""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, results)
        
    def create_efficiency_plot(self):
        for widget in self.efficiency_frame.winfo_children():
            widget.destroy()
            
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Calculate efficiency curve
        radios = np.arange(0.1, 4.5, 0.1)
        eficiencias = []
        for r in radios:
            area = np.pi * r**2
            vol = sum((i * self.dt / 1000) * area for i in self.intensidades)
            ef = (vol / self.vol_total) * 100
            eficiencias.append(ef)
            
        # Create plot
        ax.plot(radios, eficiencias, 'b-', linewidth=2.5, alpha=0.8, label='Efficiency Curve')
        ax.fill_between(radios, eficiencias, alpha=0.2, color='blue')
        
        # Add guides
        ax.axhline(y=90, color='r', linestyle='--', alpha=0.7, linewidth=2, label='90% Efficiency Target')
        ax.axvline(x=self.r_optimo, color='g', linestyle='--', alpha=0.7, linewidth=2, 
                  label=f'Optimal Radius: {self.r_optimo:.2f} m')
        
        # Mark current position
        radio_actual = float(self.radio_cono_entry.get())
        ax.plot(radio_actual, self.eficiencia_cono, 'ro', markersize=10, 
               label=f'Current: {self.eficiencia_cono:.1f}%')
        
        ax.set_xlabel('Cone Radius (m)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Collection Efficiency (%)', fontsize=10, fontweight='bold')
        ax.set_title('Cone Radius vs Collection Efficiency', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 105)
        
        canvas = FigureCanvasTkAgg(fig, self.efficiency_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_2d_visualization(self):
        for widget in self.visualization_frame.winfo_children():
            widget.destroy()
            
        fig = Figure(figsize=(10, 7), dpi=100)
        ax = fig.add_subplot(111)
        
        # Parameters
        r_cono = float(self.radio_cono_entry.get())
        r_tronco = 0.5
        r_copa = float(self.radio_copa_entry.get())
        altura_tronco = 2
        
        # Draw inverted collector cone (funnel)
        cone_height = 3
        # INVERTED: The cone now opens downward
        cone_x = [-r_cono, -0.3, 0.3, r_cono, -r_cono]
        cone_y = [5, 2, 2, 5, 5]  # Inverted: top at y=5, bottom at y=2
        ax.fill(cone_x, cone_y, '#4682B4', alpha=0.7, label='Collection Cone', edgecolor='navy', linewidth=2)
        
        # Draw trunk
        trunk = patches.Rectangle((-r_tronco, 0), 2*r_tronco, altura_tronco, 
                                 color='#8B4513', alpha=0.9, label='Trunk', linewidth=2)
        ax.add_patch(trunk)
        
        # Draw crown (ellipse for more natural appearance)
        crown = patches.Ellipse((0, 5 + r_copa/2), 2*r_copa, r_copa*1.5, 
                               color='#228B22', alpha=0.6, label='Crown')
        ax.add_patch(crown)
        
        # Add ground
        ax.axhline(y=0, color='#8B7355', linewidth=3, label='Ground')
        
        # Add water drops
        for i in range(20):
            x = np.random.uniform(-r_copa*1.5, r_copa*1.5)
            y = np.random.uniform(7, 9)
            ax.plot(x, y, 'b.', markersize=8, alpha=0.6)
        
        # Configuration
        ax.set_xlim(-7, 7)
        ax.set_ylim(-1, 10)
        ax.set_aspect('equal')
        ax.set_xlabel('Distance (m)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Height (m)', fontsize=11, fontweight='bold')
        ax.set_title('Rainwater Harvesting System - 2D Visualization', 
                    fontsize=13, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.2)
        
        # Move legend to upper left corner
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
        
        # Information box
        info_text = f"""System Configuration:
• Crown Radius: {r_copa:.1f} m
• Cone Radius: {r_cono:.1f} m
• Current Efficiency: {self.eficiencia_cono:.1f}%
• Optimal Radius: {self.r_optimo:.2f} m
• Optimal Efficiency: {self.eficiencia_optima:.1f}%"""
        
        ax.text(4, 1.5, info_text, fontsize=8, bbox=dict(boxstyle="round,pad=0.8", 
                facecolor="lightblue", alpha=0.8), verticalalignment='top')
        
        canvas = FigureCanvasTkAgg(fig, self.visualization_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_3d_visualization(self):
        """Create a complete 3D visualization of the rainwater harvesting system"""
        for widget in self.visualization_3d_frame.winfo_children():
            widget.destroy()
            
        fig = Figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        
        try:
            # Get parameters
            r_cono = float(self.radio_cono_entry.get())
            r_copa = float(self.radio_copa_entry.get())
            r_tronco = 0.5
            
            # Create trunk (cylinder)
            z_trunk = np.linspace(0, 2, 20)
            theta_trunk = np.linspace(0, 2*np.pi, 30)
            Z_trunk, Theta_trunk = np.meshgrid(z_trunk, theta_trunk)
            X_trunk = r_tronco * np.cos(Theta_trunk)
            Y_trunk = r_tronco * np.sin(Theta_trunk)
            
            # Create crown (sphere)
            u = np.linspace(0, 2 * np.pi, 25)
            v = np.linspace(0, np.pi, 25)
            U, V = np.meshgrid(u, v)
            X_crown = r_copa * np.cos(U) * np.sin(V)
            Y_crown = r_copa * np.sin(U) * np.sin(V)
            Z_crown = r_copa * np.cos(V) + 6  # Position above trunk
            
            # Create INVERTED collector cone (frustum of cone)
            z_cone = np.linspace(2, 5, 20)
            theta_cone = np.linspace(0, 2*np.pi, 30)
            Theta_cone, Z_cone = np.meshgrid(theta_cone, z_cone)
            
            # INVERTED: The cone radius increases from top to bottom
            # Radius varies linearly from 0.3 at top to r_cono at bottom
            r_cone_array = np.linspace(0.3, r_cono, len(z_cone))
            # Expand to match meshgrid shape
            R_cone = np.tile(r_cone_array.reshape(-1, 1), (1, len(theta_cone)))
            
            X_cone = R_cone * np.cos(Theta_cone)
            Y_cone = R_cone * np.sin(Theta_cone)
            
            # Create ground plane
            x_ground = np.linspace(-8, 8, 20)
            y_ground = np.linspace(-8, 8, 20)
            X_ground, Y_ground = np.meshgrid(x_ground, y_ground)
            Z_ground = np.zeros_like(X_ground)
            
            # Plot all components
            # Ground
            ax.plot_surface(X_ground, Y_ground, Z_ground, alpha=0.3, color='#8B7355')
            
            # Trunk
            ax.plot_surface(X_trunk, Y_trunk, Z_trunk, color='#8B4513', alpha=0.9)
            
            # Crown
            ax.plot_surface(X_crown, Y_crown, Z_crown, color='#228B22', alpha=0.6)
            
            # INVERTED Cone
            ax.plot_surface(X_cone, Y_cone, Z_cone, color='#4682B4', alpha=0.7)
            
            # Add rain drops (simplified as points)
            num_drops = 50
            x_rain = np.random.uniform(-r_copa*2, r_copa*2, num_drops)
            y_rain = np.random.uniform(-r_copa*2, r_copa*2, num_drops)
            z_rain = np.random.uniform(8, 10, num_drops)
            ax.scatter(x_rain, y_rain, z_rain, c='blue', alpha=0.6, s=20)
            
            # Configuration
            ax.set_xlabel('X (m)', fontsize=10, fontweight='bold', labelpad=10)
            ax.set_ylabel('Y (m)', fontsize=10, fontweight='bold', labelpad=10)
            ax.set_zlabel('Z (m)', fontsize=10, fontweight='bold', labelpad=10)
            ax.set_title('3D Visualization - Rainwater Harvesting System', 
                        fontsize=12, fontweight='bold', pad=20)
            
            # Set equal aspect ratio
            max_range = 8
            ax.set_xlim([-max_range, max_range])
            ax.set_ylim([-max_range, max_range])
            ax.set_zlim([0, 12])
            
            # Set viewing angle for better perspective
            ax.view_init(elev=25, azim=45)
            
            # Create custom legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#8B7355', alpha=0.3, label='Ground'),
                Patch(facecolor='#8B4513', alpha=0.9, label='Trunk'),
                Patch(facecolor='#228B22', alpha=0.6, label='Crown'),
                Patch(facecolor='#4682B4', alpha=0.7, label='Collection Cone')
            ]
            
            # Move legend to lower right corner
            ax.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.9, 0.1), 
                     fontsize=7, framealpha=0.9)
            
            # Add informational text
            info_text = f"""System Parameters:
• Crown Radius: {r_copa:.1f} m
• Cone Radius: {r_cono:.1f} m
• Efficiency: {self.eficiencia_cono:.1f}%
• Optimal: {self.eficiencia_optima:.1f}% 
  (r={self.r_optimo:.2f}m)

Use mouse to rotate"""
            
            # Move info box to upper left corner
            ax.text2D(0.02, 0.95, info_text, transform=ax.transAxes, fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
                     verticalalignment='top')
            
        except Exception as e:
            # Fallback in case of 3D rendering problems
            ax.text2D(0.5, 0.5, f"3D Visualization Error:\n{str(e)}", 
                     transform=ax.transAxes, ha='center', va='center', fontsize=11,
                     bbox=dict(boxstyle="round,pad=1", facecolor="red", alpha=0.7))
            print(f"3D Visualization Error: {e}")
        
        canvas = FigureCanvasTkAgg(fig, self.visualization_3d_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_hyetogram_plot(self):
        """Create a complete hyetogram (rainfall intensity over time)"""
        for widget in self.hyetogram_frame.winfo_children():
            widget.destroy()
            
        fig = Figure(figsize=(10, 8), dpi=100)
        
        # Create subplots for different hyetogram views
        ax1 = fig.add_subplot(211)  # Main hyetogram
        ax2 = fig.add_subplot(212)  # Cumulative rainfall
        
        # Time array
        time = np.arange(self.num_barras) * self.dt
        
        # Main hyetogram - bar chart
        bars = ax1.bar(time, self.intensidades, width=0.8, 
                      color=plt.cm.Blues(np.linspace(0.4, 1, len(self.intensidades))),
                      edgecolor='darkblue', alpha=0.7)
        
        # Add value labels on bars
        for bar, intensity in zip(bars, self.intensidades):
            height = bar.get_height()
            if height > 0:  # Only label non-zero bars
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{intensity:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        ax1.set_xlabel('Time (minutes)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Rainfall Intensity (mm/min)', fontsize=10, fontweight='bold')
        
        # Update title based on hyetogram type
        hyetogram_title = f'Rainfall Hyetogram - {self.hyetogram_type.get().title()} Pattern'
        ax1.set_title(hyetogram_title, fontsize=12, fontweight='bold', pad=20)
        
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_xticks(time)
        ax1.set_xlim(-0.5, self.num_barras - 0.5)
        
        # Add rainfall statistics to the plot
        total_rainfall = sum(self.intensidades) * self.dt
        avg_intensity = np.mean(self.intensidades)
        max_intensity = np.max(self.intensidades)
        
        stats_text = f"""Rainfall Statistics:
• Hyetogram Type: {self.hyetogram_type.get().title()}
• Total Duration: {self.num_barras * self.dt} min
• Total Rainfall: {total_rainfall:.1f} mm
• Average Intensity: {avg_intensity:.1f} mm/min
• Maximum Intensity: {max_intensity:.1f} mm/min
• Number of Intervals: {self.num_barras}"""
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
                verticalalignment='top')
        
        # Cumulative rainfall plot
        cumulative_rainfall = np.cumsum(self.intensidades) * self.dt
        ax2.plot(time, cumulative_rainfall, 'r-', linewidth=3, marker='o', markersize=6,
                label='Cumulative Rainfall')
        ax2.fill_between(time, cumulative_rainfall, alpha=0.3, color='red')
        
        # Add value labels for cumulative rainfall
        for i, (t, cumul) in enumerate(zip(time, cumulative_rainfall)):
            ax2.annotate(f'{cumul:.1f} mm', (t, cumul), textcoords="offset points",
                        xytext=(0,10), ha='center', fontsize=9, fontweight='bold')
        
        ax2.set_xlabel('Time (minutes)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Cumulative Rainfall (mm)', fontsize=11, fontweight='bold')
        ax2.set_title('Cumulative Rainfall Over Time', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        ax2.set_xticks(time)
        
        # Add design storm information
        design_storm_info = f"""Design Storm Characteristics:
• Pattern: {self.hyetogram_type.get().title()}
• Time Step: {self.dt} minute
• Duration: {self.num_barras * self.dt} minutes
• Peak Intensity: {max_intensity:.1f} mm/min at {np.argmax(self.intensidades) * self.dt} min"""
        
        ax2.text(0.02, 0.98, design_storm_info, transform=ax2.transAxes, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
                verticalalignment='top')
        
        fig.tight_layout(pad=3.0)
        
        canvas = FigureCanvasTkAgg(fig, self.hyetogram_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_flow_diagram(self):
        for widget in self.flow_frame.winfo_children():
            widget.destroy()
            
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Data for waterfall chart
        categories = ['Total Rain', 'Intercepted\nby Tree', 'Captured\nby Cone', 
                     'Reaches Ground\n(Without Cone)', 'Reaches Ground\n(With Cone)']
        valores = [self.vol_total, self.vol_interceptado, self.vol_captado_cono, 
                  self.vol_suelo_sin_cono, self.vol_suelo_con_cono]
        
        # Colors with better contrast
        colores = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3F7D20']
        
        # Create horizontal bar chart for better readability
        y_pos = np.arange(len(categories))
        
        bars = ax.barh(y_pos, valores, color=colores, alpha=0.8, height=0.6)
        
        # Add values on bars
        for bar, valor in zip(bars, valores):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{valor:.4f} m³', ha='left', va='center', fontweight='bold', fontsize=8)
        
        ax.set_xlabel('Volume (m³)', fontsize=11, fontweight='bold')
        ax.set_title('Rainwater Distribution Analysis', fontsize=12, fontweight='bold')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(categories, fontsize=8)
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim(0, max(valores) * 1.15)
        
        # Add efficiency annotation
        ax.text(0.05, 0.95, f'Overall System Efficiency: {self.eficiencia_cono:.1f}%', 
               transform=ax.transAxes, fontsize=11, bbox=dict(boxstyle="round,pad=0.5", 
               facecolor="yellow", alpha=0.7))
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, self.flow_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_advanced_analysis(self):
        for widget in self.analysis_frame.winfo_children():
            widget.destroy()
            
        fig = Figure(figsize=(12, 8), dpi=80)
        
        # Create subplots
        ax1 = fig.add_subplot(221)  # Area comparison
        ax2 = fig.add_subplot(222)  # Cost-benefit analysis
        ax3 = fig.add_subplot(223)  # Efficiency comparison
        ax4 = fig.add_subplot(224)  # Rainfall statistics
        
        # Plot 1: Area comparison
        areas = [self.area_copa, self.area_cono, self.area_ramas, self.area_tronco]
        labels = ['Crown', 'Cone', 'Branches', 'Trunk']
        colors = ['#228B22', '#4682B4', '#32CD32', '#8B4513']
        ax1.pie(areas, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Projected Area Distribution')
        
        # Plot 2: Cost-benefit analysis (simulated)
        cone_sizes = np.linspace(0.5, 4, 50)
        efficiencies = [(np.pi * r**2 * sum(self.intensidades) * self.dt / 1000 / self.vol_total) * 100 
                       for r in cone_sizes]
        # Simulate cost (linear with area)
        costs = [np.pi * r**2 * 100 for r in cone_sizes]  # $100 per m²
        
        ax2.plot(cone_sizes, efficiencies, 'g-', label='Efficiency')
        ax2_twin = ax2.twinx()
        ax2_twin.plot(cone_sizes, costs, 'r-', label='Cost')
        ax2.set_xlabel('Cone Radius (m)')
        ax2.set_ylabel('Efficiency (%)', color='g')
        ax2_twin.set_ylabel('Cost ($)', color='r')
        ax2.set_title('Cost-Benefit Analysis')
        ax2.legend(loc='upper left', fontsize=6)
        ax2_twin.legend(loc='upper right', fontsize=6)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Efficiency comparison
        current_eff = self.eficiencia_cono
        optimal_eff = self.eficiencia_optima
        interception_eff = (self.vol_interceptado / self.vol_total) * 100
        
        eff_data = [current_eff, optimal_eff, interception_eff]
        eff_labels = ['Current\nSystem', 'Optimal\nSystem', 'Tree\nInterception']
        
        bars = ax3.bar(eff_labels, eff_data, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax3.set_ylabel('Efficiency (%)')
        ax3.set_title('System Efficiency Comparison')
        
        # Add values on bars
        for bar, value in zip(bars, eff_data):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=6)
        
        ax3.set_ylim(0, 105)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Rainfall statistics
        rainfall_metrics = ['Total\nRainfall', 'Average\nIntensity', 'Maximum\nIntensity', 'Duration']
        rainfall_values = [sum(self.intensidades) * self.dt, np.mean(self.intensidades), 
                          np.max(self.intensidades), self.num_barras * self.dt]
        
        bars4 = ax4.bar(rainfall_metrics, rainfall_values, color=['#4A90E2', '#7ED321', '#D0021B', '#F5A623'])
        ax4.set_ylabel('Value')
        ax4.set_title('Rainfall Event Characteristics')
        
        # Add values on bars
        for bar, value in zip(bars4, rainfall_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        ax4.grid(True, alpha=0.3, axis='y')
        
        fig.tight_layout(pad=3.0)
        
        canvas = FigureCanvasTkAgg(fig, self.analysis_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def export_results(self):
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV Files", "*.csv"), ("Text Files", "*.txt"), ("All Files", "*.*")],
                title="Export Results"
            )
            
            if filename:
                with open(filename, 'w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    
                    # Write header
                    writer.writerow(["Rainwater Harvesting System Analysis"])
                    writer.writerow(["Export Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                    writer.writerow([])
                    
                    # Write rainfall characteristics
                    writer.writerow(["RAINFALL CHARACTERISTICS"])
                    writer.writerow(["Hyetogram Type:", self.hyetogram_type.get().title()])
                    writer.writerow(["Total Duration (min)", f"{self.num_barras * self.dt}"])
                    writer.writerow(["Total Rainfall (mm)", f"{sum(self.intensidades) * self.dt:.1f}"])
                    writer.writerow(["Average Intensity (mm/min)", f"{np.mean(self.intensidades):.1f}"])
                    writer.writerow(["Maximum Intensity (mm/min)", f"{np.max(self.intensidades):.1f}"])
                    writer.writerow(["Number of Intervals", f"{self.num_barras}"])
                    writer.writerow([])
                    
                    # Write hyetogram data
                    writer.writerow(["HYETOGRAM DATA"])
                    writer.writerow(["Time (min)", "Intensity (mm/min)"])
                    for i, intensity in enumerate(self.intensidades):
                        writer.writerow([f"{i * self.dt}", f"{intensity:.2f}"])
                    writer.writerow([])
                    
                    # Write parameters
                    writer.writerow(["INPUT PARAMETERS"])
                    writer.writerow(["Total Rain Volume (m³)", self.vol_total])
                    writer.writerow(["Crown Radius (m)", float(self.radio_copa_entry.get())])
                    writer.writerow(["Cone Radius (m)", float(self.radio_cono_entry.get())])
                    writer.writerow(["Crown Interception (%)", self.coef_copa * 100])
                    writer.writerow(["Branch Interception (%)", self.coef_ramas * 100])
                    writer.writerow(["Trunk Interception (%)", self.coef_tronco * 100])
                    writer.writerow([])
                    
                    # Write results
                    writer.writerow(["RESULTS"])
                    writer.writerow(["Total Rain (m³)", f"{self.vol_total:.4f}"])
                    writer.writerow(["Intercepted by Tree (m³)", f"{self.vol_interceptado:.4f}"])
                    writer.writerow(["  - Crown (m³)", f"{self.vol_copa:.4f}"])
                    writer.writerow(["  - Branches (m³)", f"{self.vol_ramas:.4f}"])
                    writer.writerow(["  - Trunk (m³)", f"{self.vol_tronco:.4f}"])
                    writer.writerow(["Captured by Cone (m³)", f"{self.vol_captado_cono:.4f}"])
                    writer.writerow(["Reaches Ground - Without Cone (m³)", f"{self.vol_suelo_sin_cono:.4f}"])
                    writer.writerow(["Reaches Ground - With Cone (m³)", f"{self.vol_suelo_con_cono:.4f}"])
                    writer.writerow([])
                    
                    writer.writerow(["PERFORMANCE METRICS"])
                    writer.writerow(["Current Efficiency (%)", f"{self.eficiencia_cono:.2f}"])
                    writer.writerow(["Optimal Efficiency (%)", f"{self.eficiencia_optima:.2f}"])
                    writer.writerow(["Optimal Radius (m)", f"{self.r_optimo:.2f}"])
                    writer.writerow(["Efficiency Improvement (%)", 
                                   f"{max(0, self.eficiencia_optima - self.eficiencia_cono):.2f}"])
                
                messagebox.showinfo("Export Successful", 
                                  f"Results successfully exported to:\n{filename}")
                
        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting results:\n{str(e)}")
            
    def reset_values(self):
        # Reset to default values
        self.vol_entry.delete(0, tk.END)
        self.vol_entry.insert(0, "1.7671")
        
        self.radio_copa_entry.delete(0, tk.END)
        self.radio_copa_entry.insert(0, "3")
        
        self.radio_cono_entry.delete(0, tk.END)
        self.radio_cono_entry.insert(0, "2.5")
        
        self.coef_copa_entry.delete(0, tk.END)
        self.coef_copa_entry.insert(0, "60")
        
        self.coef_ramas_entry.delete(0, tk.END)
        self.coef_ramas_entry.insert(0, "30")
        
        self.coef_tronco_entry.delete(0, tk.END)
        self.coef_tronco_entry.insert(0, "10")
        
        self.num_barras_entry.delete(0, tk.END)
        self.num_barras_entry.insert(0, "10")
        
        self.max_intensity_entry.delete(0, tk.END)
        self.max_intensity_entry.insert(0, "10")
        
        self.hyetogram_type.set("linear")
        
        self.calculate_results()
        
    def recalculate(self):
        self.calculate_results()

def main():
    root = tk.Tk()
    app = RainwaterHarvestingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()