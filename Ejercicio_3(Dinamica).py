import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

class SistemaPoleaMovil:
    def __init__(self, m1=2, m2=3, m3=1, g=9.8):
        """
        Configuracion:
        - Polea fija en el techo (P1)
        - De P1 cuelgan: masa m1 (izquierda) y polea movil P2 (derecha)
        - De P2 cuelgan: masa m2 (izquierda) y masa m3 (derecha)
        """
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.g = g
        
        # Calcular aceleraciones y tensiones
        self.a1, self.a2, self.a3, self.T1, self.T2, self.T3 = self.calcular_aceleraciones_tensiones()
        
        # Verificar ligadura
        self.verificar_ligadura()
        
        # Posiciones iniciales
        self.y1 = 2.0      # altura inicial m1
        self.yP2 = 2.8     # altura inicial polea movil
        self.y2_rel = 0.8 if m2 > 0 else 0  # si m2=0, no hay masa
        self.y3_rel = 0.8 if m3 > 0 else 0
        
        # Velocidades iniciales
        self.v1 = 0
        self.v2 = 0
        self.v3 = 0
        self.vP2 = 0  # velocidad de la polea movil
        
        # Configuracion de la animacion
        self.fig, self.ax = plt.subplots(figsize=(14, 9))
        self.t = 0
        self.dt = 0.05
        
        # Posiciones fijas
        self.polea_fija_x = 0
        self.polea_fija_y = 4.0
        
        # Limites fisicos
        self.distancia_minima_polea = 0.6
        
        # Historial para graficas (opcional)
        self.tiempo_hist = []
        self.posiciones_hist = {'m1': [], 'm2': [], 'm3': [], 'polea': []}
        
        # Colores para cada masa
        self.colores = {'m1': 'blue', 'm2': 'red', 'm3': 'green'}
    
    def verificar_ligadura(self):
        """Verificar la ecuacion de ligadura del sistema"""
        if self.m2 > 0 or self.m3 > 0:
            # Aceleraciones absolutas
            a2_abs = -self.a1 + self.a2
            a3_abs = -self.a1 + self.a3
            ligadura = a2_abs + a3_abs + 2 * self.a1
            if abs(ligadura) > 1e-8:
                print(f"ADVERTENCIA: Violacion de ligadura: {ligadura:.2e}")
                if abs(ligadura) > 1e-5:
                    print("   Error significativo - revisar ecuaciones")
            else:
                print(f"Ligadura verificada: {ligadura:.2e}")
        else:
            print("Caso especial: sin ligadura (masas libres)")
    
    def calcular_aceleraciones_tensiones(self):
        """Calcula aceleraciones y tensiones para TODOS los casos posibles"""
        
        # CASO ESPECIAL: Sistema en equilibrio estatico
        if abs(self.m1 - 2*self.m2) < 1e-10 and abs(self.m2 - self.m3) < 1e-10 and self.m2 > 0:
            print("Sistema en equilibrio estatico")
            T2 = self.m2 * self.g
            T3 = self.m3 * self.g
            T1 = T2 + T3
            return 0, 0, 0, T1, T2, T3
        
        # CASO 1: m2 = 0 y m3 = 0
        if self.m2 == 0 and self.m3 == 0:
            print("Caso especial: m2 = m3 = 0 -> Caida libre")
            a1 = self.g
            a2 = 0
            a3 = 0
            T1 = 0
            T2 = 0
            T3 = 0
            return a1, a2, a3, T1, T2, T3
        
        # CASO 2: m2 = 0, m3 > 0
        elif self.m2 == 0 and self.m3 > 0:
            print("Caso especial: m2 = 0")
            a1 = (self.m1 - 2*self.m3) * self.g / (self.m1 + 4*self.m3)
            a2 = 0
            a3 = -2 * a1
            
            # Aceleraciones absolutas
            a3_abs = -a1 + a3
            T3 = self.m3 * (self.g - a3_abs)
            T1 = T3
            
            return a1, a2, a3, abs(T1), 0, abs(T3)
        
        # CASO 3: m3 = 0, m2 > 0
        elif self.m3 == 0 and self.m2 > 0:
            print("Caso especial: m3 = 0")
            a1 = (self.m1 - 2*self.m2) * self.g / (self.m1 + 4*self.m2)
            a2 = -2 * a1
            a3 = 0
            
            # Aceleraciones absolutas
            a2_abs = -a1 + a2
            T2 = self.m2 * (self.g - a2_abs)
            T1 = T2
            
            return a1, a2, a3, abs(T1), abs(T2), 0
        
        # CASO 4: Ambos m2 y m3 > 0 (caso general)
        else:
            print("Caso general: todas las masas > 0")
            
            # Sistema de ecuaciones para polea movil
            # Ecuación 1: (m1 + m2 + m3)a1 + (m3 - m2)a2 = (m2 + m3 - m1)g
            # Ecuación 2: 2m2 a1 - (m2 + m3)a2 = (m3 - m2)g
            
            A = np.array([
                [self.m1 + self.m2 + self.m3, self.m3 - self.m2],
                [2*self.m2, -(self.m2 + self.m3)]
            ])
            B = np.array([
                (self.m2 + self.m3 - self.m1) * self.g,
                (self.m3 - self.m2) * self.g
            ])
            
            try:
                a1, a2 = np.linalg.solve(A, B)
                print(f"   a1={a1:.3f}, a2={a2:.3f}")
            except np.linalg.LinAlgError:
                print("   Error en solucion numerica")
                a1, a2 = 0, 0
            
            a3 = -a2  # Por ligadura en la polea móvil
            
            # Aceleraciones absolutas
            a2_abs = -a1 + a2
            a3_abs = -a1 + a3
            
            print(f"   Absolutas: a2_abs={a2_abs:.3f}, a3_abs={a3_abs:.3f}")
            
            # Determinar quién baja realmente
            if a2_abs > 0:
                print(f"   m2 (masa={self.m2}kg) BAJA con a={a2_abs:.2f}")
            else:
                print(f"   m2 (masa={self.m2}kg) SUBE con a={a2_abs:.2f}")
                
            if a3_abs > 0:
                print(f"   m3 (masa={self.m3}kg) BAJA con a={a3_abs:.2f}")
            else:
                print(f"   m3 (masa={self.m3}kg) SUBE con a={a3_abs:.2f}")
            
            # Tensiones
            T2 = self.m2 * (self.g - a2_abs)
            T3 = self.m3 * (self.g - a3_abs)
            T1 = T2 + T3

            return a1, a2, a3, abs(T1), abs(T2), abs(T3)
    
    def actualizar_posiciones(self, frame):
        """Actualiza posiciones respetando limites fisicos"""
        self.t += self.dt
        
        # Movimiento de m1 (siempre presente)
        self.v1 = self.a1 * self.t
        dy1 = 0.5 * self.a1 * self.t**2
        nueva_y1 = 2.0 + dy1
        self.y1 = np.clip(nueva_y1, 0.5, 3.8)
        
        # ===== MOVIMIENTO DE LA POLEA MOVIL =====
        if self.m2 == 0 and self.m3 == 0:
            # CASO ESPECIAL: Polea movil cae libremente
            self.vP2 = self.g * self.t
            dyP2 = 0.5 * self.g * self.t**2
            nueva_yP2 = 2.8 + dyP2
        else:
            # CASO GENERAL: Polea movil se mueve segun ligadura
            self.vP2 = -self.a1 * self.t
            dyP2 = -0.5 * self.a1 * self.t**2
            nueva_yP2 = 2.8 + dyP2
        
        self.yP2 = np.clip(nueva_yP2, 1.0, 3.8)
        
        # Actualizar posiciones de m2 y m3
        if self.m2 > 0:
            self.v2 = self.a2 * self.t
            dy2_rel = 0.5 * self.a2 * self.t**2
            nueva_y2_rel = 0.8 + dy2_rel
            self.y2_rel = self.aplicar_limites_masa(nueva_y2_rel, self.yP2)
        
        if self.m3 > 0:
            self.v3 = self.a3 * self.t
            dy3_rel = 0.5 * self.a3 * self.t**2
            nueva_y3_rel = 0.8 + dy3_rel
            self.y3_rel = self.aplicar_limites_masa(nueva_y3_rel, self.yP2)
        
        # Verificar posiciones relativas
        if self.m2 > 0 and self.m3 > 0:
            a2_abs = -self.a1 + self.a2
            a3_abs = -self.a1 + self.a3
            
            # Debug opcional (comentar si no se quiere ver)
            # print(f"Debug - a2_abs={a2_abs:.2f}, a3_abs={a3_abs:.2f}")
            
            # Verificar que la masa más pesada tiende a bajar
            y2_abs = self.yP2 + self.y2_rel
            y3_abs = self.yP2 + self.y3_rel
            
            if self.m2 > self.m3 and a2_abs < 0 and a3_abs > 0:
                print("⚠ Atencion: m2 (pesada) sube y m3 (liviana) baja - verificar")
        
        # Guardar historial
        self.tiempo_hist.append(self.t)
        self.posiciones_hist['m1'].append(self.y1)
        self.posiciones_hist['polea'].append(self.yP2)
        if self.m2 > 0:
            self.posiciones_hist['m2'].append(self.y2)
        if self.m3 > 0:
            self.posiciones_hist['m3'].append(self.y3)
    
    def aplicar_limites_masa(self, y_rel, y_polea):
        """Aplica limites fisicos a la posicion relativa de una masa"""
        # Limite superior (no pasar polea)
        y_max = y_polea - self.distancia_minima_polea
        if y_polea + y_rel > y_max:
            y_rel = y_max - y_polea
        
        # Limite inferior
        if y_polea + y_rel < 0.5:
            y_rel = 0.5 - y_polea
        
        return y_rel
    
    def verificar_tensiones(self):
        """Verifica que las tensiones sean consistentes"""
        if self.m2 > 0 or self.m3 > 0:
            diferencia = abs(self.T1 - (self.T2 + self.T3))
            if diferencia > 1e-8 and self.T1 > 0:
                print(f"Verificacion tensiones: T1 - (T2+T3) = {diferencia:.2e}")
    
    @property
    def y2(self):
        """Posicion absoluta de m2 (solo si tiene masa)"""
        if self.m2 > 0:
            return self.yP2 + self.y2_rel
        return -10
    
    @property
    def y3(self):
        """Posicion absoluta de m3 (solo si tiene masa)"""
        if self.m3 > 0:
            return self.yP2 + self.y3_rel
        return -10
    
    def dibujar_polea(self, x, y, radio, color_face, titulo, es_fija=True):
        """Dibuja una polea con estilo mejorado"""
        polea = patches.Circle((x, y), radio, 
                              edgecolor='black', 
                              facecolor=color_face, 
                              linewidth=2,
                              zorder=3)
        self.ax.add_patch(polea)
        
        rueda = patches.Circle((x, y), radio*0.6, 
                              edgecolor='gray', 
                              facecolor='lightgray', 
                              linewidth=1,
                              zorder=4)
        self.ax.add_patch(rueda)
        
        eje = patches.Circle((x, y), radio*0.2, 
                            color='black', 
                            zorder=5)
        self.ax.add_patch(eje)
        
        if es_fija:
            self.ax.text(x, y + radio + 0.2, titulo, 
                        ha='center', fontsize=9, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        else:
            self.ax.text(x, y + radio + 0.2, titulo, 
                        ha='center', fontsize=9, color='darkred',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    def dibujar_masa(self, x, y, ancho, alto, masa, color, nombre, aceleracion, velocidad, tension):
        """Dibuja una masa con estilo mejorado y muestra la tension"""
        sombra = patches.Rectangle((x + 0.05, y - 0.4 - 0.05), ancho, alto, 
                                  facecolor='gray', alpha=0.3, zorder=1)
        self.ax.add_patch(sombra)
        
        masa_rect = FancyBboxPatch((x, y - 0.4), ancho, alto,
                                  boxstyle="round,pad=0.02,rounding_size=0.05",
                                  edgecolor='black', 
                                  facecolor=color, 
                                  linewidth=2,
                                  alpha=0.9,
                                  zorder=2)
        self.ax.add_patch(masa_rect)
        
        self.ax.text(x + ancho/2, y, f'm{nombre} = {masa}kg', 
                    ha='center', va='center', 
                    fontsize=9, weight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
        
        if tension > 0 and masa > 0:
            self.ax.text(x + ancho/2, y - 0.4, 
                        f'T{nombre}={tension:.1f} N', 
                        ha='center', fontsize=8, color='purple', weight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.9))
        
        if abs(velocidad) > 0.01 and masa > 0:
            self.ax.text(x + ancho/2, y - 0.8, 
                        f'v={velocidad:.2f} m/s', 
                        ha='center', fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.1", facecolor="lightcyan", alpha=0.7))
    
    def dibujar(self):
        """Dibuja el sistema completo"""
        self.ax.clear()
        
        # Limites y fondo
        self.ax.set_xlim(-4, 4)
        self.ax.set_ylim(0, 5)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#f0f0f0')
        self.ax.grid(True, alpha=0.3, linestyle='--')
        
        # Aceleraciones absolutas para mostrar
        if self.m2 > 0:
            a2_abs = -self.a1 + self.a2
            dir2_abs = "BAJA" if a2_abs > 0 else "SUBE" if a2_abs < 0 else "QUIETO"
        else:
            a2_abs = 0
            dir2_abs = "SIN MASA"
            
        if self.m3 > 0:
            a3_abs = -self.a1 + self.a3
            dir3_abs = "BAJA" if a3_abs > 0 else "SUBE" if a3_abs < 0 else "QUIETO"
        else:
            a3_abs = 0
            dir3_abs = "SIN MASA"
        
        dir1 = "BAJA" if self.a1 > 0 else "SUBE" if self.a1 < 0 else "QUIETO"
        
        # Determinar estado
        if self.m2 == 0 and self.m3 == 0:
            estado = "CASO ESPECIAL: m2 = m3 = 0 -> CAIDA LIBRE"
        elif self.m2 == 0:
            estado = "CASO ESPECIAL: m2 = 0"
        elif self.m3 == 0:
            estado = "CASO ESPECIAL: m3 = 0"
        else:
            estado = "SISTEMA COMPLETO"
            
            # Verificar quién es más pesado
            if self.m2 > self.m3:
                estado += f" - m2 ({self.m2}kg) es más pesada que m3 ({self.m3}kg)"
            elif self.m3 > self.m2:
                estado += f" - m3 ({self.m3}kg) es más pesada que m2 ({self.m2}kg)"
            else:
                estado += " - m2 = m3 (iguales)"
        
        titulo = f'{estado}\n'
        titulo += f'ACELERACIONES ABSOLUTAS: a1={self.a1:.2f} ({dir1}) | a2={a2_abs:.2f} ({dir2_abs}) | a3={a3_abs:.2f} ({dir3_abs})\n'
        titulo += f'MASAS: m1={self.m1}kg, m2={self.m2}kg, m3={self.m3}kg'
        
        self.ax.set_title(titulo, fontsize=11, pad=15)
        
        # Techo
        techo = patches.Rectangle((-2, 4.3), 4, 0.2, 
                                 facecolor='gray', edgecolor='darkgray', linewidth=2)
        self.ax.add_patch(techo)
        self.ax.text(0, 4.5, 'TECHO', ha='center', fontsize=12, weight='bold')
        
        # Soportes
        self.ax.plot([-1.5, -1.5], [4.3, 4.5], 'k-', linewidth=2)
        self.ax.plot([1.5, 1.5], [4.3, 4.5], 'k-', linewidth=2)
        
        # Polea fija
        self.dibujar_polea(self.polea_fija_x, self.polea_fija_y, 0.3, 
                          'gold', 'POLEA FIJA', es_fija=True)
        
        if self.T1 > 0:
            self.ax.text(0, self.polea_fija_y - 0.5, f'T1 = {self.T1:.1f} N', 
                        ha='center', fontsize=9, color='purple', weight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
        
        self.ax.plot([0, 0], [self.polea_fija_y, 4.3], 'k-', linewidth=2)
        
        # Polea movil
        self.dibujar_polea(2.0, self.yP2, 0.25, 'silver', 
                          'POLEA MOVIL', es_fija=False)
        
        if self.T2 > 0 or self.T3 > 0:
            self.ax.text(2.0, self.yP2 - 0.5, f'T2+T3 = {self.T2+self.T3:.1f} N', 
                        ha='center', fontsize=8, color='darkred',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9))
        
        # Limite fisico
        if self.m2 > 0 or self.m3 > 0:
            limite_y = self.yP2 - self.distancia_minima_polea
            self.ax.axhline(y=limite_y, xmin=0.4, xmax=0.7, 
                          color='red', linestyle='--', alpha=0.5)
            self.ax.text(2.8, limite_y, 'Limite fisico', 
                        color='red', fontsize=8, alpha=0.7,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
        
        # Cuerdas
        self.ax.plot([0, -1.5], [self.polea_fija_y - 0.1, self.y1 + 0.3], 
                    'k-', linewidth=2, alpha=0.8)
        self.ax.plot([0, 1.75], [self.polea_fija_y - 0.1, self.yP2 + 0.2], 
                    'k-', linewidth=2, alpha=0.8)
        
        if self.m2 > 0:
            self.ax.plot([1.75, 0.5], [self.yP2 - 0.1, self.y2 + 0.3], 
                        'k-', linewidth=2, alpha=0.8)
        if self.m3 > 0:
            self.ax.plot([2.25, 3.2], [self.yP2 - 0.1, self.y3 + 0.3], 
                        'k-', linewidth=2, alpha=0.8)
        
        # Masas
        self.dibujar_masa(-1.9, self.y1, 0.8, 0.8, self.m1, 
                         'lightblue', '1', self.a1, self.v1, self.T1)
        
        if self.m2 > 0:
            self.dibujar_masa(0.1, self.y2, 0.8, 0.8, self.m2, 
                            'lightcoral', '2', self.a2, self.v2, self.T2)
        else:
            self.ax.text(0.5, 2.5, '(sin masa)', 
                        color='gray', ha='center', alpha=0.7,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        
        if self.m3 > 0:
            self.dibujar_masa(2.7, self.y3, 0.8, 0.8, self.m3, 
                            'lightgreen', '3', self.a3, self.v3, self.T3)
        else:
            self.ax.text(3.1, 2.5, '(sin masa)', 
                        color='gray', ha='center', alpha=0.7,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        
        # Panel de tensiones
        tension_text = (f'TENSIONES:\n'
                       f'T1 = {self.T1:.1f} N\n'
                       f'T2 = {self.T2:.1f} N\n'
                       f'T3 = {self.T3:.1f} N')
        
        if self.m2 > 0 or self.m3 > 0:
            verificacion = self.T1 - (self.T2 + self.T3)
            tension_text += f'\nT1-(T2+T3) = {verificacion:.2e}'
        
        self.ax.text(-3.5, 4.0, tension_text, 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, 
                             edgecolor="purple"), 
                    fontsize=9)
        
        # Tiempo
        self.ax.text(-3.5, 0.2, f'Tiempo: {self.t:.1f} s',
                    bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.7),
                    fontsize=9)
        
        # Leyenda
        self.ax.text(2.5, 0.2, 'BAJA | SUBE | QUIETO | SIN MASA', 
                    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9,
                             edgecolor="orange"),
                    fontsize=8)
    
    def animar(self, frame):
        """Funcion de animacion"""
        self.actualizar_posiciones(frame)
        self.dibujar()
        return self.ax
    
    def ejecutar(self):
        """Ejecuta la animacion"""
        print("\n" + "=" * 70)
        print("INICIANDO ANIMACION")
        print("=" * 70)
        
        anim = FuncAnimation(self.fig, self.animar, interval=50, cache_frame_data=False)
        plt.tight_layout()
        plt.show()
        
        print("\n" + "=" * 70)
        print("ANIMACION FINALIZADA")
        print(f"Tiempo total: {self.t:.1f} s")
        print("=" * 70)


def mostrar_intro():
    print("=" * 70)
    print("SISTEMA DE POLEA FIJA + POLEA MOVIL")
    print("=" * 70)
    print("\nCONFIGURACION:")
    print("  * Polea fija en el techo")
    print("  * De ella cuelgan: m1 y una polea movil")
    print("  * De la polea movil cuelgan: m2 y m3")
    print("=" * 70)


if __name__ == "__main__":
    mostrar_intro()
    
    try:
        print("\nINGRESE LAS MASAS (kg):")
        m1 = float(input("m1 (kg) [ej: 100]: ") or "100")
        m2 = float(input("m2 (kg) [ej: 50]: ") or "50")
        m3 = float(input("m3 (kg) [ej: 25]: ") or "25")
        
        if m1 < 0 or m2 < 0 or m3 < 0:
            print("Error: Masas no pueden ser negativas")
            m1, m2, m3 = 100, 50, 25
    except ValueError:
        print("Error: Usando valores por defecto")
        m1, m2, m3 = 100, 50, 25
    
    print("\n" + "=" * 70)
    print("CONFIGURANDO SISTEMA...")
    print("=" * 70)
    
    sistema = SistemaPoleaMovil(m1, m2, m3)
    
    print("\n" + "=" * 70)
    print("RESULTADOS:")
    print(f"  a1 = {sistema.a1:>7.3f} m/s²")
    print(f"  a2 = {sistema.a2:>7.3f} m/s²")
    print(f"  a3 = {sistema.a3:>7.3f} m/s²")
    print("-" * 70)
    print(f"  T1 = {sistema.T1:>7.1f} N")
    print(f"  T2 = {sistema.T2:>7.1f} N")
    print(f"  T3 = {sistema.T3:>7.1f} N")
    print("=" * 70)
    
    # Mostrar aceleraciones absolutas
    if m2 > 0 and m3 > 0:
        a2_abs = -sistema.a1 + sistema.a2
        a3_abs = -sistema.a1 + sistema.a3
        print(f"\nACELERACIONES ABSOLUTAS:")
        print(f"  a2_abs = {a2_abs:.3f} m/s² ({'BAJA' if a2_abs>0 else 'SUBE'})")
        print(f"  a3_abs = {a3_abs:.3f} m/s² ({'BAJA' if a3_abs>0 else 'SUBE'})")
        
        if sistema.m2 > sistema.m3:
            if a2_abs < 0:
                print("\n⚠ ADVERTENCIA: m2 (pesada) está SUBIENDO - REVISAR ECUACIONES")
        elif sistema.m3 > sistema.m2:
            if a3_abs < 0:
                print("\n⚠ ADVERTENCIA: m3 (pesada) está SUBIENDO - REVISAR ECUACIONES")
    
    print("\n" + "=" * 70)
    sistema.ejecutar()