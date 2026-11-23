import os
import joblib
import customtkinter as ctk
from tkinter import filedialog
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from prediccion.services.data_service import cargar_y_validar_csv
from prediccion.services.ml_service import entrenar_y_guardar_modelo, predecir_con_modelo
from prediccion.services.transform_service import aplicar_transformaciones

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BG_MAIN = "#090d16"
BG_CARD = "#111827"
BG_INPUT = "#0b0f19"
ACCENT_CYAN = "#22d3ee"
ACCENT_MAGENTA = "#e879f9"


class FinPredictDesktop(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("FinPredict ML — Predicción Financiera + Transformaciones 2D")
        self.geometry("1240x780")
        self.minsize(1060, 680)
        self.configure(fg_color=BG_MAIN)

        self.ruta_modelo_actual = None

        header = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=0, height=56)
        header.pack(fill="x")
        header.pack_propagate(False)

        ctk.CTkLabel(
            header, text="◈  FinPredict ML", text_color=ACCENT_CYAN,
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(side="left", padx=20)

        self.lbl_status_global = ctk.CTkLabel(
            header, text="Listo", text_color="#94a3b8", font=ctk.CTkFont(size=12)
        )
        self.lbl_status_global.pack(side="right", padx=20)

        self.tabs = ctk.CTkTabview(
            self, fg_color=BG_MAIN,
            segmented_button_fg_color=BG_CARD,
            segmented_button_selected_color="#0891b2",
            segmented_button_selected_hover_color="#06b6d4",
            segmented_button_unselected_color=BG_CARD,
            segmented_button_unselected_hover_color="#1e293b",
            corner_radius=12
        )
        self.tabs.pack(fill="both", expand=True, padx=16, pady=(8, 16))

        self.tab_ml = self.tabs.add("  1. Predicción Financiera (ML)  ")
        self.tab_2d = self.tabs.add("  2. Transformaciones Lineales 2D  ")

        self._construir_tab_ml()
        self._construir_tab_2d()

        self.bind("<Control-o>", lambda e: self._cargar_csv())

        csv_default = os.path.join(os.getcwd(), "datos_financieros.csv")
        if os.path.exists(csv_default):
            self.after(150, lambda: self._procesar_csv(csv_default))

    # ==================== PESTAÑA 1: PREDICCIÓN FINANCIERA ====================
    def _construir_tab_ml(self):
        self.tab_ml.grid_columnconfigure(0, weight=3)
        self.tab_ml.grid_columnconfigure(1, weight=2)
        self.tab_ml.grid_rowconfigure(1, weight=1)

        top_bar = ctk.CTkFrame(self.tab_ml, fg_color=BG_CARD, corner_radius=12)
        top_bar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=6, pady=(4, 10))

        self.lbl_archivo = ctk.CTkLabel(
            top_bar, text="Ningún dataset cargado — presiona Ctrl+O o haz clic en Cargar CSV",
            text_color="#94a3b8", font=ctk.CTkFont(size=13)
        )
        self.lbl_archivo.pack(side="left", padx=16, pady=10)

        ctk.CTkButton(
            top_bar, text="Cargar CSV Financiero (Ctrl+O)", command=self._cargar_csv,
            fg_color="#0891b2", hover_color="#06b6d4", corner_radius=8,
            font=ctk.CTkFont(size=13, weight="bold"), height=34
        ).pack(side="right", padx=16, pady=8)

        izq = ctk.CTkFrame(self.tab_ml, fg_color="transparent")
        izq.grid(row=1, column=0, sticky="nsew", padx=(6, 8), pady=4)
        izq.grid_rowconfigure(1, weight=3)
        izq.grid_rowconfigure(2, weight=2)
        izq.grid_columnconfigure(0, weight=1)

        kpi_row = ctk.CTkFrame(izq, fg_color="transparent")
        kpi_row.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        kpi_row.grid_columnconfigure((0, 1, 2), weight=1)

        self.kpi_mae = self._crear_kpi(kpi_row, 0, "MAE ($)", "--", ACCENT_CYAN)
        self.kpi_rmse = self._crear_kpi(kpi_row, 1, "RMSE ($)", "--", ACCENT_MAGENTA)
        self.kpi_r2 = self._crear_kpi(kpi_row, 2, "R² SCORE", "--", "#4ade80")

        card_graf = ctk.CTkFrame(izq, fg_color=BG_CARD, corner_radius=12)
        card_graf.grid(row=1, column=0, sticky="nsew", pady=(0, 10))

        self.fig_ml = Figure(figsize=(6.2, 2.7), dpi=95, facecolor=BG_CARD)
        self.ax_ml = self.fig_ml.add_subplot(111)
        self.canvas_ml = FigureCanvasTkAgg(self.fig_ml, master=card_graf)
        self.canvas_ml.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=8)
        self._dibujar_grafica_ml_vacia()

        card_math = ctk.CTkFrame(izq, fg_color=BG_CARD, corner_radius=12)
        card_math.grid(row=2, column=0, sticky="nsew")

        ctk.CTkLabel(
            card_math, text="Ecuación Normal:  β = (XᵀX)⁻¹XᵀY",
            text_color="#f8fafc", font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w", padx=14, pady=(10, 4))

        self.txt_metricas = ctk.CTkTextbox(
            card_math, fg_color=BG_INPUT, text_color="#e2e8f0",
            font=ctk.CTkFont(family="Monospace", size=11), corner_radius=8
        )
        self.txt_metricas.pack(fill="both", expand=True, padx=14, pady=(0, 12))
        self.txt_metricas.configure(state="disabled")

        der = ctk.CTkFrame(self.tab_ml, fg_color=BG_CARD, corner_radius=12)
        der.grid(row=1, column=1, sticky="nsew", padx=(8, 6), pady=4)

        head_pred = ctk.CTkFrame(der, fg_color="transparent")
        head_pred.pack(fill="x", padx=18, pady=(16, 8))

        ctk.CTkLabel(
            head_pred, text="Proyectar Nuevo Mes", text_color=ACCENT_CYAN,
            font=ctk.CTkFont(size=15, weight="bold")
        ).pack(side="left")

        ctk.CTkButton(
            head_pred, text="Ejemplo", width=74, height=28,
            fg_color="#1e293b", hover_color="#334155", text_color=ACCENT_CYAN,
            command=self._llenar_ejemplo
        ).pack(side="right")

        form = ctk.CTkFrame(der, fg_color="transparent")
        form.pack(fill="x", padx=18, pady=4)
        form.grid_columnconfigure((0, 1), weight=1)

        self.entries_ml = {}
        campos = [
            ("x1", "Marketing $ (X1)", "1200", 0, 0),
            ("x2", "Gastos Op. $ (X2)", "2700", 0, 1),
            ("x3", "Clientes Act. (X3)", "190", 1, 0),
            ("x4", "Ventas Prev. $ (X4)", "11500", 1, 1),
        ]
        for key, label, default, r, c in campos:
            box = ctk.CTkFrame(form, fg_color="transparent")
            box.grid(row=r, column=c, sticky="ew", padx=6, pady=6)
            ctk.CTkLabel(box, text=label, text_color="#94a3b8", font=ctk.CTkFont(size=12)).pack(anchor="w")
            e = ctk.CTkEntry(
                box, fg_color=BG_INPUT, border_color="#334155", corner_radius=8,
                height=38, font=ctk.CTkFont(family="Monospace", size=13)
            )
            e.insert(0, default)
            e.bind("<Return>", lambda ev: self._predecir_ml())
            e.pack(fill="x", pady=(4, 0))
            self.entries_ml[key] = e

        ctk.CTkButton(
            der, text="Calcular Ingreso (Enter)", command=self._predecir_ml,
            fg_color="#0891b2", hover_color="#06b6d4", corner_radius=10,
            height=42, font=ctk.CTkFont(size=14, weight="bold")
        ).pack(fill="x", padx=24, pady=(14, 14))

        res_card = ctk.CTkFrame(der, fg_color=BG_INPUT, corner_radius=12, border_width=1, border_color="#1e293b")
        res_card.pack(fill="both", expand=True, padx=24, pady=(0, 20))

        ctk.CTkLabel(
            res_card, text="INGRESO MENSUAL PROYECTADO", text_color="#94a3b8",
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(pady=(22, 2))

        self.lbl_resultado = ctk.CTkLabel(
            res_card, text="$--", text_color=ACCENT_CYAN,
            font=ctk.CTkFont(family="Monospace", size=38, weight="bold")
        )
        self.lbl_resultado.pack(pady=4)

        self.lbl_estado = ctk.CTkLabel(
            res_card, text="Presiona Enter para proyectar", text_color="#64748b",
            font=ctk.CTkFont(size=12, weight="bold"), wraplength=280
        )
        self.lbl_estado.pack(pady=(4, 18))

    def _crear_kpi(self, parent, col, titulo, valor, color):
        card = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=12)
        card.grid(row=0, column=col, sticky="ew", padx=4)
        ctk.CTkLabel(card, text=titulo, text_color="#94a3b8", font=ctk.CTkFont(size=10, weight="bold")).pack(pady=(10, 2))
        lbl = ctk.CTkLabel(card, text=valor, text_color=color, font=ctk.CTkFont(family="Monospace", size=19, weight="bold"))
        lbl.pack(pady=(0, 10))
        return lbl

    def _dibujar_grafica_ml_vacia(self):
        self.ax_ml.clear()
        self.ax_ml.set_facecolor(BG_INPUT)
        self.ax_ml.text(0.5, 0.5, "Carga un CSV financiero para visualizar Real vs. Predicho",
                        color="#64748b", ha="center", va="center", fontsize=10)
        self.ax_ml.set_xticks([])
        self.ax_ml.set_yticks([])
        self.canvas_ml.draw()

    def _llenar_ejemplo(self):
        for k, v in [("x1", "1200"), ("x2", "2700"), ("x3", "190"), ("x4", "11500")]:
            self.entries_ml[k].delete(0, "end")
            self.entries_ml[k].insert(0, v)
        self._predecir_ml()

    def _cargar_csv(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar archivo CSV financiero", initialdir=os.getcwd(),
            filetypes=[("Archivos CSV", "*.csv")]
        )
        if ruta:
            self._procesar_csv(ruta)

    def _procesar_csv(self, ruta):
        df, err = cargar_y_validar_csv(ruta)
        if err:
            self.lbl_archivo.configure(text=f"Error: {err}", text_color="#f43f5e")
            return

        res = entrenar_y_guardar_modelo(df, user_id="desktop")
        self.ruta_modelo_actual = res["ruta_modelo"]

        self.lbl_archivo.configure(
            text=f"Dataset activo: {os.path.basename(ruta)}  ({res['n_estudiantes']} periodos)",
            text_color="#4ade80"
        )
        self.lbl_status_global.configure(text=f"Modelo financiero listo (R² = {res['r2']})", text_color="#4ade80")
        self.kpi_mae.configure(text=f"${res['mae']}")
        self.kpi_rmse.configure(text=f"${res['rmse']}")
        self.kpi_r2.configure(text=f"{res['r2']} ({res['r2_pct']}%)")

        modelo = joblib.load(self.ruta_modelo_actual)
        Y_real = df["Y"].values
        Y_pred = modelo.predict(df[["X1", "X2", "X3", "X4"]])
        idx = np.arange(1, len(Y_real) + 1)

        self.ax_ml.clear()
        self.ax_ml.set_facecolor(BG_INPUT)
        self.ax_ml.plot(idx, Y_real, color=ACCENT_CYAN, marker="o", linewidth=2, markersize=5, label="Real (Y)")
        self.ax_ml.plot(idx, Y_pred, color=ACCENT_MAGENTA, marker="x", linestyle="--", linewidth=2, markersize=6, label="Predicho (Ŷ)")
        self.ax_ml.set_title("Ingresos Reales vs. Predichos ($ USD)", color="#f8fafc", fontsize=10, fontweight="bold")
        self.ax_ml.tick_params(colors="#94a3b8", labelsize=8)
        for spine in self.ax_ml.spines.values():
            spine.set_color("#1e293b")
        self.ax_ml.grid(color="#1e293b", linestyle=":", linewidth=1)
        self.ax_ml.legend(facecolor=BG_CARD, edgecolor="#334155", labelcolor="#f8fafc", fontsize=8)
        self.fig_ml.tight_layout()
        self.canvas_ml.draw()

        lineas = [
            f"{'Parámetro':<34} {'NumPy (β)':>12} {'Scikit-Learn':>14}",
            "-" * 62,
        ]
        for row in res["tabla_coef"]:
            lineas.append(f"{row['nombre']:<34} {row['numpy']:>12.4f} {row['sklearn']:>14.4f}")

        self.txt_metricas.configure(state="normal")
        self.txt_metricas.delete("1.0", "end")
        self.txt_metricas.insert("end", "\n".join(lineas))
        self.txt_metricas.configure(state="disabled")

        self._predecir_ml()

    def _predecir_ml(self):
        if not self.ruta_modelo_actual:
            self.lbl_estado.configure(text="Primero carga un archivo CSV.", text_color="#f59e0b")
            return
        try:
            vals = [float(self.entries_ml[f"x{i}"].get()) for i in range(1, 5)]
        except ValueError:
            self.lbl_estado.configure(text="Ingresa solo valores numéricos válidos.", text_color="#f43f5e")
            return

        if any(v < 0 for v in vals):
            self.lbl_estado.configure(text="Todos los montos y cantidades deben ser >= 0.", text_color="#f43f5e")
            return

        pred, aviso = predecir_con_modelo(self.ruta_modelo_actual, *vals)
        utilidad = pred - (vals[0] + vals[1])
        self.lbl_resultado.configure(text=f"${pred:,.2f}")

        if utilidad > 0:
            estado = f"✓ Rentable\nUtilidad neta est.: +${utilidad:,.2f}"
            color = "#4ade80"
        else:
            estado = f"⚠ Déficit estimado\nBalance: ${utilidad:,.2f}"
            color = "#f43f5e"

        if aviso:
            estado += "\n(Ajustado a piso $0)"
        self.lbl_estado.configure(text=estado, text_color=color)

    # ==================== PESTAÑA 2: TRANSFORMACIONES 2D ====================
    def _construir_tab_2d(self):
        self.tab_2d.grid_columnconfigure(0, weight=0)
        self.tab_2d.grid_columnconfigure(1, weight=1)
        self.tab_2d.grid_rowconfigure(0, weight=1)

        ctrl = ctk.CTkScrollableFrame(self.tab_2d, fg_color=BG_CARD, corner_radius=12, width=340)
        ctrl.grid(row=0, column=0, sticky="nsew", padx=(6, 10), pady=6)

        ctk.CTkLabel(
            ctrl, text="Presets Rápidos (Demo)", text_color="#94a3b8",
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(anchor="w", padx=10, pady=(6, 4))

        presets_frame = ctk.CTkFrame(ctrl, fg_color="transparent")
        presets_frame.pack(fill="x", padx=8, pady=(0, 10))
        presets_frame.grid_columnconfigure((0, 1), weight=1)

        presets = [
            ("Trasladar", {"tx": 3.0, "ty": 2.0, "sx": 1.0, "sy": 1.0, "ang": 0.0, "shx": 0.0, "shy": 0.0}),
            ("Escalar 1.5x", {"tx": 0.0, "ty": 0.0, "sx": 1.5, "sy": 1.5, "ang": 0.0, "shx": 0.0, "shy": 0.0}),
            ("Rotar 45°", {"tx": 0.0, "ty": 0.0, "sx": 1.0, "sy": 1.0, "ang": 45.0, "shx": 0.0, "shy": 0.0}),
            ("Sesgar X", {"tx": 0.0, "ty": 0.0, "sx": 1.0, "sy": 1.0, "ang": 0.0, "shx": 0.8, "shy": 0.0}),
            ("Combinada", {"tx": 2.0, "ty": -1.0, "sx": 1.2, "sy": 1.2, "ang": 30.0, "shx": 0.4, "shy": 0.0}),
            ("Resetear", {"tx": 0.0, "ty": 0.0, "sx": 1.0, "sy": 1.0, "ang": 0.0, "shx": 0.0, "shy": 0.0}),
        ]
        for i, (nombre, vals) in enumerate(presets):
            btn_color = "#334155" if nombre == "Resetear" else "#1e293b"
            ctk.CTkButton(
                presets_frame, text=nombre, height=28, fg_color=btn_color,
                hover_color="#0891b2", font=ctk.CTkFont(size=11, weight="bold"),
                command=lambda v=vals: self._aplicar_preset(v)
            ).grid(row=i // 2, column=i % 2, sticky="ew", padx=3, pady=3)

        self.vars_2d = {}
        self.val_labels = {}
        controles = [
            ("tx", "Traslación X", -5.0, 5.0, 0.0),
            ("ty", "Traslación Y", -5.0, 5.0, 0.0),
            ("sx", "Escalado X", 0.2, 3.0, 1.0),
            ("sy", "Escalado Y", 0.2, 3.0, 1.0),
            ("ang", "Rotación (°)", -180.0, 180.0, 0.0),
            ("shx", "Sesgado X", -2.0, 2.0, 0.0),
            ("shy", "Sesgado Y", -2.0, 2.0, 0.0),
        ]

        for key, label, vmin, vmax, vdef in controles:
            row_lbl = ctk.CTkFrame(ctrl, fg_color="transparent")
            row_lbl.pack(fill="x", padx=10, pady=(4, 0))
            ctk.CTkLabel(row_lbl, text=label, text_color="#cbd5e1", font=ctk.CTkFont(size=12)).pack(side="left")
            v_lbl = ctk.CTkLabel(row_lbl, text=f"{vdef:.1f}", text_color=ACCENT_CYAN, font=ctk.CTkFont(family="Monospace", size=12, weight="bold"))
            v_lbl.pack(side="right")
            self.val_labels[key] = v_lbl

            var = ctk.DoubleVar(value=vdef)
            self.vars_2d[key] = var
            ctk.CTkSlider(
                ctrl, from_=vmin, to=vmax, variable=var,
                command=lambda val, k=key: self._on_slider(k, val),
                progress_color=ACCENT_CYAN, button_color="#06b6d4", button_hover_color="#22d3ee"
            ).pack(fill="x", padx=10, pady=(2, 6))

        ctk.CTkLabel(
            ctrl, text="Matriz Compuesta M (3×3):", text_color="#94a3b8",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w", padx=10, pady=(8, 0))

        self.lbl_matriz = ctk.CTkLabel(
            ctrl, text="", fg_color=BG_INPUT, text_color=ACCENT_CYAN,
            font=ctk.CTkFont(family="Monospace", size=11), corner_radius=8,
            justify="left", padx=12, pady=10
        )
        self.lbl_matriz.pack(fill="x", padx=10, pady=(6, 10))

        graf_card = ctk.CTkFrame(self.tab_2d, fg_color=BG_CARD, corner_radius=12)
        graf_card.grid(row=0, column=1, sticky="nsew", padx=(6, 6), pady=6)

        self.fig_2d = Figure(figsize=(6.8, 5.6), dpi=105, facecolor=BG_CARD)
        self.ax_2d = self.fig_2d.add_subplot(111)
        self.canvas_2d = FigureCanvasTkAgg(self.fig_2d, master=graf_card)
        self.canvas_2d.get_tk_widget().pack(fill="both", expand=True, padx=12, pady=12)

        self._actualizar_2d()

    def _on_slider(self, key, val):
        self.val_labels[key].configure(text=f"{float(val):.1f}")
        self._actualizar_2d()

    def _aplicar_preset(self, vals):
        for k, v in vals.items():
            self.vars_2d[k].set(v)
            self.val_labels[k].configure(text=f"{v:.1f}")
        self._actualizar_2d()

    def _actualizar_2d(self):
        orig, trans, M = aplicar_transformaciones(
            tx=self.vars_2d["tx"].get(),
            ty=self.vars_2d["ty"].get(),
            sx=self.vars_2d["sx"].get(),
            sy=self.vars_2d["sy"].get(),
            angulo=self.vars_2d["ang"].get(),
            shx=self.vars_2d["shx"].get(),
            shy=self.vars_2d["shy"].get(),
        )

        self.ax_2d.clear()
        self.ax_2d.set_facecolor(BG_INPUT)
        self.ax_2d.axhline(0, color="#334155", linewidth=1)
        self.ax_2d.axvline(0, color="#334155", linewidth=1)
        self.ax_2d.grid(color="#1e293b", linestyle=":", linewidth=1)

        self.ax_2d.plot(orig[0], orig[1], color=ACCENT_CYAN, linewidth=2, linestyle="--", label="Original")
        self.ax_2d.fill(orig[0], orig[1], color=ACCENT_CYAN, alpha=0.12)

        self.ax_2d.plot(trans[0], trans[1], color=ACCENT_MAGENTA, linewidth=2.5, marker="o", markersize=4, label="Transformada")
        self.ax_2d.fill(trans[0], trans[1], color=ACCENT_MAGENTA, alpha=0.22)

        self.ax_2d.set_xlim(-8, 8)
        self.ax_2d.set_ylim(-8, 8)
        self.ax_2d.set_aspect("equal")
        self.ax_2d.set_title("Transformación Lineal 2D (M = T · R · Sh · S)", color="#f8fafc", fontsize=11, fontweight="bold")
        self.ax_2d.tick_params(colors="#94a3b8")
        for spine in self.ax_2d.spines.values():
            spine.set_color("#1e293b")
        self.ax_2d.legend(facecolor=BG_CARD, edgecolor="#334155", labelcolor="#f8fafc")

        self.canvas_2d.draw()

        np.set_printoptions(precision=2, suppress=True)
        self.lbl_matriz.configure(text=np.array2string(M))


if __name__ == "__main__":
    app = FinPredictDesktop()
    app.mainloop()
