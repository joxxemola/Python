"""
Sistema de Poleas — Simulador Interactivo
==========================================
  * Polea fija en el techo
  * De ella cuelgan: m1 (izquierda) y una polea móvil (derecha)
  * De la polea móvil cuelgan: m2 (izquierda) y m3 (derecha)

Controles:
  [ESPACIO]  Pausar / Reanudar
  [R]        Reiniciar
  [+/-]      Velocidad x2 / x0.5
  [Q/ESC]    Salir
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
import matplotlib.patheffects as pe

matplotlib.rcParams['toolbar'] = 'None'

# ─────────────────────────────────────────────
#  PALETA DE COLORES (tema oscuro estilo lab)
# ─────────────────────────────────────────────
BG       = '#0d0d14'
PANEL    = '#13131f'
GRID     = '#1e1e2e'
TEXT     = '#e8e8f8'
MUTED    = '#5a5a7a'
ROPE     = '#a0a0c0'
CEIL     = '#2a2a3a'
GOLD     = '#f0c040'
SILVER   = '#9090a8'
C_M1     = '#60a0f0'   # azul
C_M2     = '#f05060'   # rojo
C_M3     = '#4ff0b0'   # verde-cyan
C_TENS   = '#c080ff'   # violeta
C_VEL    = '#ffa040'   # naranja


# ─────────────────────────────────────────────
#  FÍSICA
# ─────────────────────────────────────────────
def calcular_fisica(m1, m2, m3, g=9.8):
    """Devuelve (a1, a2_rel, a3_rel, T1, T2, T3, caso)"""
    if m2 == 0 and m3 == 0:
        return g, 0, 0, 0, 0, 0, "CAÍDA LIBRE  (m₂ = m₃ = 0)"

    if m2 == 0:
        a1 = (m1 - 2*m3)*g / (m1 + 4*m3)
        a3 = -2*a1
        a3_abs = -a1 + a3
        T3 = abs(m3*(g - a3_abs))
        return a1, 0, a3, T3, 0, T3, "CASO m₂ = 0"

    if m3 == 0:
        a1 = (m1 - 2*m2)*g / (m1 + 4*m2)
        a2 = -2*a1
        a2_abs = -a1 + a2
        T2 = abs(m2*(g - a2_abs))
        return a1, a2, 0, T2, T2, 0, "CASO m₃ = 0"

    # Caso general
    A = np.array([
        [m1+m2+m3, m3-m2],
        [2*m2,    -(m2+m3)]
    ], dtype=float)
    B = np.array([(m2+m3-m1)*g, (m3-m2)*g])

    det = A[0,0]*A[1,1] - A[0,1]*A[1,0]
    if abs(det) < 1e-12:
        return 0, 0, 0, 0, 0, 0, "EQUILIBRIO ESTÁTICO"

    a1, a2 = np.linalg.solve(A, B)
    a3 = -a2
    a2_abs = -a1 + a2
    a3_abs = -a1 + a3
    T2 = m2*(g - a2_abs)
    T3 = m3*(g - a3_abs)
    T1 = T2 + T3
    return a1, a2, a3, abs(T1), abs(T2), abs(T3), "SISTEMA COMPLETO"


# ─────────────────────────────────────────────
#  HELPERS DE DIBUJO
# ─────────────────────────────────────────────
def pulley(ax, x, y, r, fijo=True):
    """Dibuja una polea con degradado concéntrico."""
    color_outer = GOLD if fijo else SILVER
    color_inner = '#a07800' if fijo else '#606070'

    for radio, color, lw, zo in [
        (r,      color_outer, 2.0, 3),
        (r*0.58, color_inner, 1.2, 4),
        (r*0.22, '#202030',   1.0, 5),
    ]:
        c = plt.Circle((x, y), radio, color=color,
                       linewidth=lw, edgecolor='#202030', zorder=zo)
        ax.add_patch(c)


def mass_box(ax, cx, cy, w, h, color, label, kg_label):
    """Caja con sombra y texto."""
    # Sombra
    s = FancyBboxPatch((cx - w/2 + 0.03, cy - h - 0.03), w, h,
                       boxstyle="round,pad=0.03",
                       facecolor='black', alpha=0.45, zorder=2)
    ax.add_patch(s)
    # Cuerpo
    box = FancyBboxPatch((cx - w/2, cy - h), w, h,
                         boxstyle="round,pad=0.03",
                         facecolor=color + '99', edgecolor=color,
                         linewidth=2, zorder=3)
    ax.add_patch(box)
    # Textos
    kw = dict(ha='center', va='center', zorder=6, fontfamily='monospace',
              path_effects=[pe.withStroke(linewidth=2, foreground='black')])
    ax.text(cx, cy - h/2 + 0.06, label, fontsize=12, fontweight='bold',
            color='white', **kw)
    ax.text(cx, cy - h/2 - 0.1, kg_label, fontsize=9, color=color + 'cc', **kw)


def tension_tag(ax, x, y, text, color):
    ax.text(x, y, text, ha='center', va='center', fontsize=8,
            color=color, fontfamily='monospace', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#0d0d14cc',
                      edgecolor=color, linewidth=1),
            zorder=10)


def velocity_arrow(ax, x, y_top, v, color):
    """Flecha de velocidad a la derecha de la masa."""
    if abs(v) < 0.05:
        return
    scale = min(abs(v) * 0.12, 0.55)
    dy = scale if v > 0 else -scale
    ax.annotate('', xy=(x, y_top - dy), xytext=(x, y_top),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=1.8, mutation_scale=10),
                zorder=9)
    ax.text(x + 0.08, y_top - dy/2,
            f'{abs(v):.1f} m/s', fontsize=7, color=color,
            fontfamily='monospace', va='center', zorder=9)


# ─────────────────────────────────────────────
#  SIMULADOR PRINCIPAL
# ─────────────────────────────────────────────
class Simulador:
    G = 9.8
    DT = 1/50

    # Geometría fija (coordenadas del mundo)
    PFX, PFY = 0.0, 3.6      # polea fija
    CEIL_Y   = 4.0
    Y0_M1    = 1.8            # posición inicial y de m1 (tope superior de la caja)
    Y0_PM    = 2.3            # posición inicial y de la polea móvil
    Y0_M2    = 1.4            # posición inicial y de m2 (relativa desde polea)
    Y0_M3    = 1.4
    XM1      = -1.5
    XPM      =  1.3
    XM2      =  0.6
    XM3      =  2.0
    BOX_W    =  0.55
    BOX_H    =  0.42
    PR_FIJA  =  0.28
    PR_MOV   =  0.22

    def __init__(self):
        self.m1 = 100.0
        self.m2 =  50.0
        self.m3 =  25.0
        self._reset_state()
        self._build_figure()
        self._recalculate()

    # ── Estado ──────────────────────────────
    def _reset_state(self):
        self.t         = 0.0
        self.paused    = False
        self.speed     = 1.0
        self.a1 = self.a2 = self.a3 = 0.0
        self.T1 = self.T2 = self.T3 = 0.0
        self.caso      = ''

    def _recalculate(self):
        self.a1, self.a2, self.a3, self.T1, self.T2, self.T3, self.caso = \
            calcular_fisica(self.m1, self.m2, self.m3, self.G)
        self.a2_abs = -self.a1 + self.a2
        self.a3_abs = -self.a1 + self.a3
        self._update_info_panel()

    # ── Posiciones en cada instante ─────────
    def _positions(self):
        t = self.t
        # Desplazamientos (positivo = hacia abajo)
        d1  =  0.5 * self.a1 * t**2
        dP  = -0.5 * self.a1 * t**2   # polea móvil opuesta a m1
        d2a =  0.5 * self.a2_abs * t**2
        d3a =  0.5 * self.a3_abs * t**2

        y1  = np.clip(self.Y0_M1 + d1,   0.45, self.CEIL_Y - 0.3)
        yP  = np.clip(self.Y0_PM + dP,   0.70, self.CEIL_Y - 0.3)
        y2  = np.clip(self.Y0_M2 + d2a,  0.45, yP - 0.25)
        y3  = np.clip(self.Y0_M3 + d3a,  0.45, yP - 0.25)

        v1  = self.a1 * t
        v2  = self.a2_abs * t
        v3  = self.a3_abs * t

        return y1, yP, y2, y3, v1, v2, v3

    # ── Figura ──────────────────────────────
    def _build_figure(self):
        plt.rcParams['axes.facecolor']   = BG
        plt.rcParams['figure.facecolor'] = BG

        self.fig = plt.figure(figsize=(14, 8), facecolor=BG)
        self.fig.canvas.manager.set_window_title('Sistema de Poleas — Simulador')

        # Layout: animación (izq) + panel info (der)
        gs = self.fig.add_gridspec(
            3, 2,
            left=0.03, right=0.97, top=0.93, bottom=0.14,
            width_ratios=[2.2, 1],
            height_ratios=[1, 1, 1],
            hspace=0.15, wspace=0.08
        )

        self.ax = self.fig.add_subplot(gs[:, 0])
        self._setup_ax()

        # Panel de resultados (3 filas dcha)
        self.ax_acc  = self.fig.add_subplot(gs[0, 1])
        self.ax_tens = self.fig.add_subplot(gs[1, 1])
        self.ax_info = self.fig.add_subplot(gs[2, 1])
        for a in [self.ax_acc, self.ax_tens, self.ax_info]:
            a.set_facecolor(PANEL)
            a.set_xticks([]); a.set_yticks([])
            for spine in a.spines.values():
                spine.set_edgecolor('#2a2a40')

        # Sliders (abajo)
        slider_kw = dict(color='#2a2a40', track_color='#1a1a2a')
        ax_s1 = self.fig.add_axes([0.07, 0.07, 0.22, 0.03])
        ax_s2 = self.fig.add_axes([0.07, 0.03, 0.22, 0.03])
        ax_s3 = self.fig.add_axes([0.38, 0.07, 0.22, 0.03])
        ax_sbm = self.fig.add_axes([0.38, 0.03, 0.22, 0.03])

        self.sl_m1 = Slider(ax_s1, 'm₁ (kg)', 1, 200, valinit=self.m1, **slider_kw)
        self.sl_m2 = Slider(ax_s2, 'm₂ (kg)', 0, 200, valinit=self.m2, **slider_kw)
        self.sl_m3 = Slider(ax_sbm,'m₃ (kg)', 0, 200, valinit=self.m3, **slider_kw)
        self.sl_sp = Slider(ax_s3, 'Vel ×',   0.1, 5.0, valinit=1.0,   **slider_kw)

        for sl in [self.sl_m1, self.sl_m2, self.sl_m3, self.sl_sp]:
            sl.label.set_color(TEXT)
            sl.valtext.set_color(GOLD)

        self.sl_m1.on_changed(self._on_slider)
        self.sl_m2.on_changed(self._on_slider)
        self.sl_m3.on_changed(self._on_slider)
        self.sl_sp.on_changed(lambda v: setattr(self, 'speed', v))

        # Botones
        btn_kw = dict(color='#1e1e2e', hovercolor='#2a2a40')
        ax_bp = self.fig.add_axes([0.67, 0.05, 0.10, 0.05])
        ax_br = self.fig.add_axes([0.79, 0.05, 0.10, 0.05])

        self.btn_pause  = Button(ax_bp, '⏸ Pausa',  **btn_kw)
        self.btn_reset  = Button(ax_br, '↺ Reiniciar', **btn_kw)

        for btn in [self.btn_pause, self.btn_reset]:
            btn.label.set_color(TEXT)
            btn.label.set_fontfamily('monospace')

        self.btn_pause.on_clicked(self._toggle_pause)
        self.btn_reset.on_clicked(self._reset)

        # Teclado
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        # Título
        self.fig.text(0.5, 0.97,
                      'SISTEMA DE POLEAS  —  Polea fija + Polea móvil',
                      ha='center', va='top', color=GOLD,
                      fontsize=14, fontfamily='monospace', fontweight='bold')

    def _setup_ax(self):
        ax = self.ax
        ax.set_xlim(-2.8, 3.2)
        ax.set_ylim(0.0, 4.4)
        ax.set_aspect('equal')
        ax.set_facecolor(BG)
        ax.tick_params(colors=MUTED, labelsize=7)
        ax.spines['bottom'].set_color(MUTED)
        ax.spines['left'].set_color(MUTED)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, color=GRID, linewidth=0.6, linestyle='--', alpha=0.6)

    # ── Callbacks ───────────────────────────
    def _on_slider(self, _):
        self.m1 = self.sl_m1.val
        self.m2 = self.sl_m2.val
        self.m3 = self.sl_m3.val
        self.t = 0.0
        self._recalculate()

    def _toggle_pause(self, _=None):
        self.paused = not self.paused
        self.btn_pause.label.set_text('▶ Reanudar' if self.paused else '⏸ Pausa')

    def _reset(self, _=None):
        self.t = 0.0
        self.paused = False
        self.btn_pause.label.set_text('⏸ Pausa')

    def _on_key(self, event):
        if event.key in (' ', 'p'):      self._toggle_pause()
        elif event.key == 'r':           self._reset()
        elif event.key == '+':           self.sl_sp.set_val(min(self.speed + 0.5, 5))
        elif event.key == '-':           self.sl_sp.set_val(max(self.speed - 0.5, 0.1))
        elif event.key in ('q', 'escape'): plt.close(self.fig)

    # ── Panel de resultados ─────────────────
    def _update_info_panel(self):
        for ax, title, rows in [
            (self.ax_acc,  'ACELERACIONES',
             [('a₁ (m1)', self.a1, C_M1),
              ('a₂ abs', self.a2_abs, C_M2),
              ('a₃ abs', self.a3_abs, C_M3),
              ('a_polea', -self.a1, GOLD)]),
            (self.ax_tens, 'TENSIONES (N)',
             [('T₁', self.T1, C_TENS),
              ('T₂', self.T2, C_M2),
              ('T₃', self.T3, C_M3),
              ('T₁−(T₂+T₃)', self.T1-(self.T2+self.T3), MUTED)]),
        ]:
            ax.clear()
            ax.set_facecolor(PANEL)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_edgecolor('#2a2a40')
            ax.text(0.5, 0.93, title, transform=ax.transAxes,
                    ha='center', va='top', color=MUTED,
                    fontsize=8, fontfamily='monospace')
            for i, (lbl, val, col) in enumerate(rows):
                y = 0.75 - i*0.20
                d = 'BAJA' if val > 0.001 else 'SUBE' if val < -0.001 else '●'
                ax.text(0.05, y, lbl + ':', transform=ax.transAxes,
                        ha='left', va='center', color=MUTED, fontsize=8,
                        fontfamily='monospace')
                ax.text(0.98, y, f'{val:+.3f}  {d}', transform=ax.transAxes,
                        ha='right', va='center', color=col, fontsize=8,
                        fontfamily='monospace', fontweight='bold')

        self.ax_info.clear()
        self.ax_info.set_facecolor(PANEL)
        self.ax_info.set_xticks([]); self.ax_info.set_yticks([])
        for sp in self.ax_info.spines.values(): sp.set_edgecolor('#2a2a40')

        lines = [
            ('CASO', self.caso, GOLD),
            (f'm₁={self.m1:.0f} kg', '', C_M1),
            (f'm₂={self.m2:.0f} kg', '', C_M2),
            (f'm₃={self.m3:.0f} kg', '', C_M3),
            ('g = 9.8 m/s²', '', MUTED),
            ('[ESPACIO] pausa', '', MUTED),
            ('[R] reiniciar', '', MUTED),
            ('[+/-] velocidad', '', MUTED),
        ]
        for i, (a, b, col) in enumerate(lines):
            self.ax_info.text(0.5, 0.95 - i*0.115, a + b,
                              transform=self.ax_info.transAxes,
                              ha='center', va='top', color=col,
                              fontsize=7.5, fontfamily='monospace')

    # ── Dibujo principal ────────────────────
    def _draw(self, frame):
        if not self.paused:
            self.t += self.DT * self.speed

        ax = self.ax
        ax.clear()
        self._setup_ax()

        y1, yP, y2, y3, v1, v2, v3 = self._positions()

        # ── Techo ──
        techo = mpatches.FancyBboxPatch((-2.5, self.CEIL_Y), 5.0, 0.28,
                                         boxstyle="square",
                                         facecolor=CEIL, edgecolor='#3a3a5a',
                                         linewidth=1.5, zorder=6)
        ax.add_patch(techo)
        for xh in np.arange(-2.5, 2.5, 0.28):
            ax.plot([xh, xh-0.15], [self.CEIL_Y, self.CEIL_Y+0.22],
                    color='#3a3a5a', lw=0.8, zorder=6)
        ax.text(0, self.CEIL_Y + 0.18, 'TECHO', ha='center', color=MUTED,
                fontsize=8, fontfamily='monospace', zorder=7)

        # ── Soporte polea fija ──
        ax.plot([self.PFX, self.PFX], [self.CEIL_Y, self.PFY + self.PR_FIJA],
                color=CEIL, lw=4, zorder=2)

        # ── Cuerdas ──
        rope_kw = dict(color=ROPE, lw=2.2, zorder=1, solid_capstyle='round')

        # cuerda m1
        ax.plot([self.PFX - self.PR_FIJA*0.7, self.XM1],
                [self.PFY - self.PR_FIJA*0.3,  y1],
                **rope_kw)
        # cuerda polea movil
        ax.plot([self.PFX + self.PR_FIJA*0.7, self.XPM],
                [self.PFY - self.PR_FIJA*0.3,  yP + self.PR_MOV],
                **rope_kw)
        # cuerdas m2 y m3
        if self.m2 > 0:
            ax.plot([self.XPM - self.PR_MOV*0.6, self.XM2],
                    [yP - self.PR_MOV*0.4,        y2],
                    **rope_kw)
        if self.m3 > 0:
            ax.plot([self.XPM + self.PR_MOV*0.6, self.XM3],
                    [yP - self.PR_MOV*0.4,        y3],
                    **rope_kw)

        # ── Poleas ──
        pulley(ax, self.PFX, self.PFY, self.PR_FIJA, fijo=True)
        pulley(ax, self.XPM, yP,        self.PR_MOV,  fijo=False)

        # ── Masas ──
        mass_box(ax, self.XM1, y1, self.BOX_W, self.BOX_H,
                 C_M1, 'm₁', f'{self.m1:.0f} kg')

        if self.m2 > 0:
            mass_box(ax, self.XM2, y2, self.BOX_W, self.BOX_H,
                     C_M2, 'm₂', f'{self.m2:.0f} kg')
        else:
            ax.text(self.XM2, 1.8, '(sin masa)', ha='center', color=MUTED,
                    fontsize=8, fontfamily='monospace')

        if self.m3 > 0:
            mass_box(ax, self.XM3, y3, self.BOX_W, self.BOX_H,
                     C_M3, 'm₃', f'{self.m3:.0f} kg')
        else:
            ax.text(self.XM3, 1.8, '(sin masa)', ha='center', color=MUTED,
                    fontsize=8, fontfamily='monospace')

        # ── Etiquetas de tensión ──
        if self.T1 > 0:
            xmid = (self.PFX + self.XM1) / 2 - 0.15
            ymid = (self.PFY + y1) / 2
            tension_tag(ax, xmid, ymid, f'T₁\n{self.T1:.0f} N', C_TENS)

        if self.T2 > 0 and self.m2 > 0:
            xmid = (self.XPM + self.XM2) / 2 - 0.15
            ymid = (yP + y2) / 2
            tension_tag(ax, xmid, ymid, f'T₂\n{self.T2:.0f} N', C_M2)

        if self.T3 > 0 and self.m3 > 0:
            xmid = (self.XPM + self.XM3) / 2 + 0.15
            ymid = (yP + y3) / 2
            tension_tag(ax, xmid, ymid, f'T₃\n{self.T3:.0f} N', C_M3)

        # ── Flechas de velocidad ──
        velocity_arrow(ax, self.XM1 + self.BOX_W/2 + 0.15, y1, v1, C_M1)
        if self.m2 > 0:
            velocity_arrow(ax, self.XM2 + self.BOX_W/2 + 0.15, y2, v2, C_M2)
        if self.m3 > 0:
            velocity_arrow(ax, self.XM3 + self.BOX_W/2 + 0.15, y3, v3, C_M3)

        # ── HUD: tiempo + estado ──
        estado_col = GOLD if not self.paused else C_VEL
        estado_txt = self.caso + ('  [PAUSA]' if self.paused else '')
        ax.text(0.02, 0.02, f't = {self.t:.2f} s',
                transform=ax.transAxes, color='#4ff0b0',
                fontsize=10, fontfamily='monospace', fontweight='bold')
        ax.text(0.5, 0.02, estado_txt,
                transform=ax.transAxes, ha='center', color=estado_col,
                fontsize=9, fontfamily='monospace')
        ax.text(0.98, 0.02, f'vel × {self.speed:.1f}',
                transform=ax.transAxes, ha='right', color=MUTED,
                fontsize=8, fontfamily='monospace')

        # ── Etiquetas de poleas ──
        ax.text(self.PFX, self.PFY + self.PR_FIJA + 0.12,
                'POLEA FIJA', ha='center', color=GOLD,
                fontsize=8, fontfamily='monospace', fontweight='bold', zorder=8)
        ax.text(self.XPM, yP + self.PR_MOV + 0.12,
                'POLEA MÓV', ha='center', color=SILVER,
                fontsize=8, fontfamily='monospace', fontweight='bold', zorder=8)

    # ── Animación ───────────────────────────
    def run(self):
        self._draw(0)
        self.anim = FuncAnimation(
            self.fig, self._draw,
            interval=20, cache_frame_data=False
        )
        plt.tight_layout(rect=[0, 0.12, 1, 0.95])
        plt.show()


# ─────────────────────────────────────────────
#  ENTRADA
# ─────────────────────────────────────────────
def intro():
    sep = '═' * 60
    print(f'\n{sep}')
    print('  SISTEMA DE POLEAS — Simulador interactivo')
    print(sep)
    print('  Esquema:')
    print('    TECHO')
    print('      │')
    print('    [POLEA FIJA]')
    print('    ╱          ╲')
    print('  [m₁]    [POLEA MÓVIL]')
    print('              ╱       ╲')
    print('           [m₂]      [m₃]')
    print(f'\n  Controles: ESPACIO=pausa  R=reset  +/-=velocidad  Q=salir')
    print(f'  Los sliders cambian masas en tiempo real.')
    print(f'{sep}\n')


if __name__ == '__main__':
    intro()

    try:
        m1 = float(input('  m₁ (kg) [100]: ') or '100')
        m2 = float(input('  m₂ (kg) [ 50]: ') or  '50')
        m3 = float(input('  m₃ (kg) [ 25]: ') or  '25')
        if any(v < 0 for v in [m1, m2, m3]):
            raise ValueError
    except ValueError:
        print('  → Usando valores por defecto: 100, 50, 25 kg')
        m1, m2, m3 = 100, 50, 25

    sim = Simulador()
    sim.m1, sim.m2, sim.m3 = m1, m2, m3
    sim.sl_m1.set_val(m1)
    sim.sl_m2.set_val(m2)
    sim.sl_m3.set_val(m3)
    sim._recalculate()
    sim.run()
