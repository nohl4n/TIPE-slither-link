import pygame
import sys
import random
import time

import SlitherLink as sl

class Slider:
    def __init__(self, x, y, w, min_val, max_val, initial, label, theme):
        self.rect = pygame.Rect(x, y, w, 16)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial
        self.label = label
        self.dragging = False
        self.theme = theme
        self.font = pygame.font.SysFont('Arial', 14)

    def draw(self, screen):
        pygame.draw.rect(screen, self.theme.slider_bg, self.rect)
        pygame.draw.rect(screen, self.theme.slider_border, self.rect, 1)
        knob_x = self.rect.x + (self.value - self.min_val) / (self.max_val - self.min_val) * self.rect.width
        knob_rect = pygame.Rect(knob_x - 6, self.rect.y, 12, self.rect.height)
        pygame.draw.rect(screen, self.theme.slider_knob, knob_rect)
        label_text = self.font.render(f"{self.label}: {self.value}", True, self.theme.text_color)
        screen.blit(label_text, (self.rect.x, self.rect.y - 18))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            x = max(self.rect.x, min(self.rect.x + self.rect.width, event.pos[0]))
            ratio = (x - self.rect.x) / self.rect.width
            self.value = int(self.min_val + ratio * (self.max_val - self.min_val))
            if self.value < self.min_val:
                self.value = self.min_val
            if self.value > self.max_val:
                self.value = self.max_val
            return True
        return False

class Theme:
    def __init__(self, dark=False):
        self.dark = dark
        self.update_colors()

    def update_colors(self):
        if self.dark:
            self.bg = (30, 30, 40)
            self.interface_bg = (45, 45, 55)
            self.text_color = (220, 220, 220)
            self.grid_color = (100, 100, 120)
            self.number_color = (255, 255, 255)
            self.line_color = (200, 200, 200)
            self.wrong_line_color = (255, 80, 80)
            self.hover_color = (80, 80, 100)
            self.edit_highlight = (200, 200, 100)
            self.button_color = (70, 70, 90)
            self.button_hover = (100, 100, 130)
            self.slider_bg = (60, 60, 80)
            self.slider_border = (120, 120, 140)
            self.slider_knob = (150, 150, 200)
        else:
            self.bg = (255, 255, 255)
            self.interface_bg = (220, 220, 235)
            self.text_color = (0, 0, 0)
            self.grid_color = (100, 100, 120)
            self.number_color = (0, 0, 0)
            self.line_color = (0, 0, 0)
            self.wrong_line_color = (200, 0, 0)
            self.hover_color = (180, 180, 200)
            self.edit_highlight = (255, 255, 150)
            self.button_color = (200, 200, 215)
            self.button_hover = (170, 170, 190)
            self.slider_bg = (200, 200, 215)
            self.slider_border = (100, 100, 120)
            self.slider_knob = (80, 80, 120)

    def toggle(self):
        self.dark = not self.dark
        self.update_colors()

class SlitherLinkGame:
    def __init__(self, rows=10, cols=10):
        pygame.init()
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.width, self.height = self.screen.get_size()

        self.interface_width = 220
        self.grid_left = 20
        self.grid_top = 20

        self.rows = rows
        self.cols = cols
        self.cell_size = self.compute_cell_size()

        # États
        self.numbers = [[-2 for _ in range(cols)] for _ in range(rows)]
        self.h_edges = [[0 for _ in range(cols + 1)] for _ in range(rows + 1)]
        self.v_edges = [[0 for _ in range(cols + 1)] for _ in range(rows + 1)]
        self.cell_colors = [[None for _ in range(cols)] for _ in range(rows)]

        self.edit_mode = True
        self.paint_mode = False
        self.selected_cell = (0, 0) if rows>0 and cols>0 else None

        self.theme = Theme(dark=False)

        # Curseurs
        slider_x = self.width - self.interface_width + 10
        slider_y = 200
        self.slider_rows = Slider(slider_x, slider_y, 180, 3, 50, rows, "Lignes", self.theme)
        self.slider_cols = Slider(slider_x, slider_y + 50, 180, 3, 50, cols, "Colonnes", self.theme)
        self.slider_density = Slider(slider_x, slider_y + 100, 180, 10, 100, 80, "Indices %", self.theme)

        # Boutons
        self.button_area_top = slider_y + 150
        self.button_scroll = 0
        self.max_button_scroll = 0
        self.init_buttons()

        self.message = "Bienvenue ! Appuyez sur E pour jouer."
        self.last_solve_time = None

        self.font = pygame.font.SysFont('Arial', 20, bold=True)
        self.small_font = pygame.font.SysFont('Arial', 14)

        self.interior_color = (240, 220, 120)
        self.exterior_color = (100, 150, 220)

        pygame.display.set_caption("Slither Link")

    def compute_cell_size(self):
        available_width = self.width - self.grid_left - self.interface_width - 20
        available_height = self.height - self.grid_top - 20
        if self.cols > 0 and self.rows > 0:
            return min(available_width // self.cols, available_height // self.rows, 55)
        return 30

    def update_cell_size(self):
        self.cell_size = self.compute_cell_size()

    def init_buttons(self):
        buttons_data = [
            ("Générer Carré", "square"),
            ("Générer Chemins", "simple"),
            ("Générer Complexe", "complex"),
            ("Générer Récursif", "recursive"),
            ("Générer Labyrinthe", "maze"),
            ("Générer Mutation", "mutation"),
            ("Générer Unique", "unique"),
            ("Résoudre Backtrack", "solve_back"),
            ("Résoudre Z3", "solve_z3"),
            ("Résoudre Heurist.", "solve_heur"),
            ("Vérifier Solution", "check_full"),
            ("Vérifier 2Couleurs", "check2color"),
            ("Mode Arêtes/Colo", "toggle_paint"),
            ("Mode Clair/Sombre", "toggle_theme"),
            ("Effacer Solution", "clear"),
            ("Effacer Tout", "clear_all"),
            ("Quitter", "quit")
        ]
        self.buttons = []
        x_base = self.width - self.interface_width + 10
        y_base = self.button_area_top
        btn_height = 26
        spacing = 2
        for i, (label, action) in enumerate(buttons_data):
            rect = pygame.Rect(x_base, y_base + i * (btn_height + spacing), 200, btn_height)
            self.buttons.append({"rect": rect, "label": label, "action": action})
        total_height = len(buttons_data) * (btn_height + spacing)
        visible_height = self.height - (self.button_area_top + 20)
        self.max_button_scroll = max(0, total_height - visible_height)

    def draw_buttons(self):
        mouse_pos = pygame.mouse.get_pos()
        for btn in self.buttons:
            rect = btn["rect"].move(0, -self.button_scroll)
            if rect.bottom < self.button_area_top or rect.top > self.height:
                continue
            color = self.theme.button_hover if rect.collidepoint(mouse_pos) else self.theme.button_color
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, self.theme.text_color, rect, 1)
            text = self.small_font.render(btn["label"], True, self.theme.text_color)
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)

    def handle_button_click(self, pos):
        for btn in self.buttons:
            rect = btn["rect"].move(0, -self.button_scroll)
            if rect.collidepoint(pos):
                act = btn["action"]
                if act == "square":
                    self.generate_square()
                elif act == "simple":
                    self.generate_simple_paths()
                elif act == "complex":
                    self.generate_complex_paths()
                elif act == "recursive":
                    self.generate_recursive()
                elif act == "maze":
                    self.generate_maze()
                elif act == "mutation":
                    self.generate_mutation()
                elif act == "unique":
                    self.generate_unique()
                elif act == "solve_back":
                    self.solve_puzzle("backtrack")
                elif act == "solve_z3":
                    self.solve_puzzle("z3")
                elif act == "solve_heur":
                    self.solve_puzzle("heuristique")
                elif act == "check_full":
                    self.check_full_solution()
                elif act == "check2color":
                    self.check_two_color()
                elif act == "toggle_paint":
                    if not self.edit_mode:
                        self.paint_mode = not self.paint_mode
                        mode = "Coloration" if self.paint_mode else "Arêtes"
                        self.message = f"Mode jeu : {mode}"
                        pygame.display.set_caption(f"Slither Link - Mode Jeu ({mode})")
                elif act == "toggle_theme":
                    self.theme.toggle()
                    for s in [self.slider_rows, self.slider_cols, self.slider_density]:
                        s.theme = self.theme
                elif act == "clear":
                    self.clear_solution()
                elif act == "clear_all":
                    self.clear_all()
                elif act == "quit":
                    pygame.quit()
                    sys.exit()
                return True
        return False

    # ---------- Génération ----------
    def generate_square(self):
        rows = self.slider_rows.value
        cols = self.slider_cols.value
        self.resize_grid(rows, cols)
        M = [[True]*cols for _ in range(rows)]
        sl.grignotage_carré(M, rows, cols)
        puzzle = sl.Nombre_Arrete(M)
        density = self.slider_density.value / 100.0
        puzzle = self.remove_random_indices(puzzle, density)
        self.load_puzzle(puzzle)
        self.clear_solution()
        self.edit_mode = False
        self.paint_mode = False
        self.message = "Puzzle généré (grignotage carré)"

    def generate_simple_paths(self):
        rows = self.slider_rows.value
        cols = self.slider_cols.value
        self.resize_grid(rows, cols)
        M = sl.generation_par_chemins(rows, cols)
        puzzle = sl.Nombre_Arrete(M)
        density = self.slider_density.value / 100.0
        puzzle = self.remove_random_indices(puzzle, density)
        self.load_puzzle(puzzle)
        self.clear_solution()
        self.edit_mode = False
        self.paint_mode = False
        self.message = "Généré par chemins simples"

    def generate_complex_paths(self):
        rows = self.slider_rows.value
        cols = self.slider_cols.value
        self.resize_grid(rows, cols)
        M = sl.generation_par_chemins_complexes(rows, cols)
        puzzle = sl.Nombre_Arrete(M)
        density = self.slider_density.value / 100.0
        puzzle = self.remove_random_indices(puzzle, density)
        self.load_puzzle(puzzle)
        self.clear_solution()
        self.edit_mode = False
        self.paint_mode = False
        self.message = "Généré par chemins complexes"

    def generate_recursive(self):
        rows = self.slider_rows.value
        cols = self.slider_cols.value
        self.resize_grid(rows, cols)
        M = sl.grignotage_rec(rows, cols)
        puzzle = sl.Nombre_Arrete(M)
        density = self.slider_density.value / 100.0
        puzzle = self.remove_random_indices(puzzle, density)
        self.load_puzzle(puzzle)
        self.clear_solution()
        self.edit_mode = False
        self.paint_mode = False
        self.message = "Généré par grignotage récursif"

    def generate_maze(self):
        rows = self.slider_rows.value
        cols = self.slider_cols.value
        if rows % 2 == 0: rows += 1
        if cols % 2 == 0: cols += 1
        self.resize_grid(rows, cols)
        M = sl.generation_par_labyrinthe(rows, cols)
        puzzle = sl.Nombre_Arrete(M)
        density = self.slider_density.value / 100.0
        puzzle = self.remove_random_indices(puzzle, density)
        self.load_puzzle(puzzle)
        self.clear_solution()
        self.edit_mode = False
        self.paint_mode = False
        self.message = "Généré par labyrinthe"

    def generate_mutation(self):
        rows = self.slider_rows.value
        cols = self.slider_cols.value
        self.resize_grid(rows, cols)
        M = sl.mutation(10, rows, cols)
        puzzle = sl.Nombre_Arrete(M)
        density = self.slider_density.value / 100.0
        puzzle = self.remove_random_indices(puzzle, density)
        self.load_puzzle(puzzle)
        self.clear_solution()
        self.edit_mode = False
        self.paint_mode = False
        self.message = "Généré par mutation"

    def generate_unique(self):
        rows = self.slider_rows.value
        cols = self.slider_cols.value
        self.resize_grid(rows, cols)
        # Appel à la fonction de génération unique du module SlitherLink
        # Elle retourne (puzzle, compteur) où puzzle est la matrice Na
        puzzle, compteur = sl.generer_unique_sol(rows, cols, tentative=100)
        # On ne retire pas d'indices supplémentaires, car la fonction le fait déjà
        self.load_puzzle(puzzle)
        self.clear_solution()
        self.edit_mode = False
        self.paint_mode = False
        self.message = f"Puzzle à solution unique généré ({compteur} indices retirés)"

    def remove_random_indices(self, puzzle, keep_ratio):
        n, m = len(puzzle), len(puzzle[0])
        total = n * m
        to_keep = int(total * keep_ratio)
        indices = [(i,j) for i in range(n) for j in range(m)]
        random.shuffle(indices)
        for (i,j) in indices[to_keep:]:
            puzzle[i][j] = -1
        return puzzle

    def load_puzzle(self, puzzle):
        for i in range(min(self.rows, len(puzzle))):
            for j in range(min(self.cols, len(puzzle[0]))):
                self.numbers[i][j] = puzzle[i][j]
        for i in range(self.rows):
            for j in range(self.cols):
                if i >= len(puzzle) or j >= len(puzzle[0]):
                    self.numbers[i][j] = -2

    # ---------- Solveurs ----------
    def solve_puzzle(self, method):
        Na = [[-1 if self.numbers[i][j] in (-1, -2) else self.numbers[i][j]
               for j in range(self.cols)] for i in range(self.rows)]
        start = time.time()
        solutions = []
        try:
            if method == "backtrack":
                solutions = sl.backtrack_solveur_all(Na)
            elif method == "z3":
                solutions = sl.z3_solveur(Na)
            elif method == "heuristique":
                res = sl.heuristique_solveur(Na, tentative=100)
                if res:
                    solutions = [res[0]]
        except Exception as e:
            self.message = f"Erreur: {e}"
            return
        elapsed = time.time() - start
        self.message = f"{method}: {elapsed:.3f}s"
        self.last_solve_time = elapsed
        if solutions:
            self.display_solution(solutions[0])
            self.edit_mode = False
            self.paint_mode = False
        else:
            self.message += " | Aucune solution"

    def display_solution(self, M):
        self.h_edges = [[0 for _ in range(self.cols + 1)] for _ in range(self.rows + 1)]
        self.v_edges = [[0 for _ in range(self.cols + 1)] for _ in range(self.rows + 1)]
        self.convert_M_to_edges(M)
        for i in range(self.rows):
            for j in range(self.cols):
                self.cell_colors[i][j] = self.interior_color if M[i][j] else self.exterior_color

    def convert_M_to_edges(self, M):
        n, m = self.rows, self.cols
        for i in range(n + 1):
            for j in range(m):
                top = M[i-1][j] if i > 0 else False
                bottom = M[i][j] if i < n else False
                self.h_edges[i][j] = 1 if top != bottom else 0
        for i in range(n):
            for j in range(m + 1):
                left = M[i][j-1] if j > 0 else False
                right = M[i][j] if j < m else False
                self.v_edges[i][j] = 1 if left != right else 0

    def clear_solution(self):
        self.h_edges = [[0 for _ in range(self.cols + 1)] for _ in range(self.rows + 1)]
        self.v_edges = [[0 for _ in range(self.cols + 1)] for _ in range(self.rows + 1)]
        self.cell_colors = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        self.message = "Solution effacée"

    def clear_all(self):
        """Efface tous les indices, les arêtes et les couleurs."""
        self.numbers = [[-2 for _ in range(self.cols)] for _ in range(self.rows)]
        self.h_edges = [[0 for _ in range(self.cols + 1)] for _ in range(self.rows + 1)]
        self.v_edges = [[0 for _ in range(self.cols + 1)] for _ in range(self.rows + 1)]
        self.cell_colors = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        self.message = "Grille entièrement effacée"

    # ---------- Vérifications (inchangées) ----------
    def check_full_solution(self):
        if self.paint_mode:
            M = [[False]*self.cols for _ in range(self.rows)]
            all_colored = True
            for i in range(self.rows):
                for j in range(self.cols):
                    if self.cell_colors[i][j] == self.interior_color:
                        M[i][j] = True
                    elif self.cell_colors[i][j] == self.exterior_color:
                        M[i][j] = False
                    else:
                        all_colored = False
            if not all_colored:
                self.message = "❌ Toutes les cases doivent être colorées"
                return
            ok = True
            for i in range(self.rows):
                for j in range(self.cols):
                    if self.numbers[i][j] in (-1, -2): continue
                    cnt = 0
                    for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ni, nj = i+di, j+dj
                        if 0 <= ni < self.rows and 0 <= nj < self.cols:
                            if M[ni][nj] != M[i][j]: cnt += 1
                        else:
                            if M[i][j] != False: cnt += 1
                    if cnt != self.numbers[i][j]:
                        ok = False
                        break
            if ok and sl.Verif(M):
                self.message = "✅ Solution parfaite !"
            else:
                self.message = "❌ Solution invalide"
        else:
            M = self.edges_to_M()
            if not sl.Verif(M):
                self.message = "❌ Boucle invalide"
                return
            ok = True
            for i in range(self.rows):
                for j in range(self.cols):
                    if self.numbers[i][j] in (-1, -2): continue
                    cnt = 0
                    for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ni, nj = i+di, j+dj
                        if 0 <= ni < self.rows and 0 <= nj < self.cols:
                            if M[ni][nj] != M[i][j]: cnt += 1
                        else:
                            if M[i][j] != False: cnt += 1
                    if cnt != self.numbers[i][j]:
                        ok = False
                        break
            if ok:
                self.message = "✅ Solution valide"
            else:
                self.message = "❌ Indices non respectés"

    def edges_to_M(self):
        n, m = self.rows, self.cols
        M = [[False]*m for _ in range(n)]
        visited = [[False]*m for _ in range(n)]
        queue = [(0,0)]
        visited[0][0] = True
        while queue:
            i, j = queue.pop(0)
            if i > 0 and not visited[i-1][j]:
                has_edge = (self.h_edges[i][j] == 1)
                M[i-1][j] = not M[i][j] if has_edge else M[i][j]
                visited[i-1][j] = True
                queue.append((i-1, j))
            if i < n-1 and not visited[i+1][j]:
                has_edge = (self.h_edges[i+1][j] == 1)
                M[i+1][j] = not M[i][j] if has_edge else M[i][j]
                visited[i+1][j] = True
                queue.append((i+1, j))
            if j > 0 and not visited[i][j-1]:
                has_edge = (self.v_edges[i][j] == 1)
                M[i][j-1] = not M[i][j] if has_edge else M[i][j]
                visited[i][j-1] = True
                queue.append((i, j-1))
            if j < m-1 and not visited[i][j+1]:
                has_edge = (self.v_edges[i][j+1] == 1)
                M[i][j+1] = not M[i][j] if has_edge else M[i][j]
                visited[i][j+1] = True
                queue.append((i, j+1))
        return M

    def check_two_color(self):
        M = self.edges_to_M()
        if sl.Verif(M):
            self.message = "✅ 2-coloration valide (boucle unique)"
            for i in range(self.rows):
                for j in range(self.cols):
                    self.cell_colors[i][j] = self.interior_color if M[i][j] else self.exterior_color
        else:
            self.message = "❌ 2-coloration invalide"

    def toggle_cell_color(self, i, j):
        if self.cell_colors[i][j] is None:
            self.cell_colors[i][j] = self.interior_color
        elif self.cell_colors[i][j] == self.interior_color:
            self.cell_colors[i][j] = self.exterior_color
        else:
            self.cell_colors[i][j] = None
        all_colored = all(self.cell_colors[i][j] is not None for i in range(self.rows) for j in range(self.cols))
        if all_colored:
            M = [[self.cell_colors[i][j] == self.interior_color for j in range(self.cols)] for i in range(self.rows)]
            self.convert_M_to_edges(M)

    # ---------- Dessin (inchangé) ----------
    def draw(self):
        self.screen.fill(self.theme.bg)
        interface_rect = pygame.Rect(self.width - self.interface_width, 0, self.interface_width, self.height)
        pygame.draw.rect(self.screen, self.theme.interface_bg, interface_rect)

        total_grid_width = self.cols * self.cell_size
        total_grid_height = self.rows * self.cell_size
        self.grid_left = (self.width - self.interface_width - total_grid_width) // 2
        self.grid_top = (self.height - total_grid_height) // 2
        if self.grid_left < 10: self.grid_left = 10
        if self.grid_top < 10: self.grid_top = 10

        # Cases colorées
        for i in range(self.rows):
            for j in range(self.cols):
                if self.cell_colors[i][j]:
                    x = self.grid_left + j * self.cell_size
                    y = self.grid_top + i * self.cell_size
                    pygame.draw.rect(self.screen, self.cell_colors[i][j], (x, y, self.cell_size, self.cell_size))

        # Nombres
        for i in range(self.rows):
            for j in range(self.cols):
                val = self.numbers[i][j]
                if val not in (-1, -2):
                    x = self.grid_left + j * self.cell_size + self.cell_size // 2
                    y = self.grid_top + i * self.cell_size + self.cell_size // 2
                    num_text = self.font.render(str(val), True, self.theme.number_color)
                    text_rect = num_text.get_rect(center=(x, y))
                    self.screen.blit(num_text, text_rect)

        # Grille
        for i in range(self.rows + 1):
            y = self.grid_top + i * self.cell_size
            pygame.draw.line(self.screen, self.theme.grid_color,
                             (self.grid_left, y), (self.grid_left + total_grid_width, y), 1)
        for j in range(self.cols + 1):
            x = self.grid_left + j * self.cell_size
            pygame.draw.line(self.screen, self.theme.grid_color,
                             (x, self.grid_top), (x, self.grid_top + total_grid_height), 1)
        for i in range(self.rows + 1):
            for j in range(self.cols + 1):
                x = self.grid_left + j * self.cell_size
                y = self.grid_top + i * self.cell_size
                pygame.draw.circle(self.screen, self.theme.grid_color, (x, y), 2)

        # Arêtes
        for i in range(self.rows + 1):
            for j in range(self.cols):
                if self.h_edges[i][j] == 1:
                    x1 = self.grid_left + j * self.cell_size
                    y = self.grid_top + i * self.cell_size
                    x2 = x1 + self.cell_size
                    pygame.draw.line(self.screen, self.theme.line_color, (x1, y), (x2, y), 3)
                elif self.h_edges[i][j] == 2:
                    x1 = self.grid_left + j * self.cell_size
                    y = self.grid_top + i * self.cell_size
                    x2 = x1 + self.cell_size
                    cx = (x1 + x2)//2
                    cs = self.cell_size//8
                    pygame.draw.line(self.screen, self.theme.wrong_line_color,
                                     (cx-cs, y-cs), (cx+cs, y+cs), 3)
                    pygame.draw.line(self.screen, self.theme.wrong_line_color,
                                     (cx-cs, y+cs), (cx+cs, y-cs), 3)
        for i in range(self.rows):
            for j in range(self.cols + 1):
                if self.v_edges[i][j] == 1:
                    x = self.grid_left + j * self.cell_size
                    y1 = self.grid_top + i * self.cell_size
                    y2 = y1 + self.cell_size
                    pygame.draw.line(self.screen, self.theme.line_color, (x, y1), (x, y2), 3)
                elif self.v_edges[i][j] == 2:
                    x = self.grid_left + j * self.cell_size
                    y1 = self.grid_top + i * self.cell_size
                    y2 = y1 + self.cell_size
                    cy = (y1 + y2)//2
                    cs = self.cell_size//8
                    pygame.draw.line(self.screen, self.theme.wrong_line_color,
                                     (x-cs, cy-cs), (x+cs, cy+cs), 3)
                    pygame.draw.line(self.screen, self.theme.wrong_line_color,
                                     (x-cs, cy+cs), (x+cs, cy-cs), 3)

        # Surbrillances
        if self.selected_cell and (self.edit_mode or (not self.edit_mode and self.paint_mode)):
            i, j = self.selected_cell
            x = self.grid_left + j * self.cell_size
            y = self.grid_top + i * self.cell_size
            pygame.draw.rect(self.screen, self.theme.edit_highlight, (x, y, self.cell_size, self.cell_size), 2)

        if not self.edit_mode and not self.paint_mode and hasattr(self, 'hover_edge') and self.hover_edge:
            etype, i, j = self.hover_edge
            if etype == 'h':
                x1 = self.grid_left + j * self.cell_size
                y = self.grid_top + i * self.cell_size
                x2 = x1 + self.cell_size
                pygame.draw.line(self.screen, self.theme.hover_color, (x1, y), (x2, y), 6)
            else:
                x = self.grid_left + j * self.cell_size
                y1 = self.grid_top + i * self.cell_size
                y2 = y1 + self.cell_size
                pygame.draw.line(self.screen, self.theme.hover_color, (x, y1), (x, y2), 6)

        self.draw_interface()
        self.slider_rows.draw(self.screen)
        self.slider_cols.draw(self.screen)
        self.slider_density.draw(self.screen)
        self.draw_buttons()

    def draw_interface(self):
        x_interface = self.width - self.interface_width + 10
        mode = "ÉDITION" if self.edit_mode else ("JEU - COLORATION" if self.paint_mode else "JEU - ARÊTES")
        mode_surf = self.font.render(f"MODE {mode}", True, self.theme.text_color)
        self.screen.blit(mode_surf, (x_interface, 20))

        instructions = []
        if self.edit_mode:
            instructions = [
                "Flèches: Déplacer",
                "0-9: Chiffre",
                "6: 0, 7: Effacer",
                "C: Couleur case",
                "E: Mode jeu",
                "D: Thème"
            ]
        elif self.paint_mode:
            instructions = [
                "Clic case: Cycle",
                "Intérieur/Extérieur",
                "E: Mode édition",
                "P: Mode arêtes"
            ]
        else:
            instructions = [
                "Clic gauche: Noir",
                "Clic droit: Rouge",
                "Clic milieu: Effacer",
                "E: Mode édition",
                "P: Mode coloration"
            ]
        y_inst = 60
        for line in instructions:
            txt = self.small_font.render(line, True, self.theme.text_color)
            self.screen.blit(txt, (x_interface, y_inst))
            y_inst += 20

        msg_txt = self.small_font.render(self.message, True, self.theme.text_color)
        self.screen.blit(msg_txt, (x_interface, self.height - 60))
        if self.last_solve_time is not None:
            time_txt = self.small_font.render(f"Temps: {self.last_solve_time:.3f}s", True, self.theme.text_color)
            self.screen.blit(time_txt, (x_interface, self.height - 35))

    # ---------- Gestion événements (inchangée, avec curseurs corrigés) ----------
    def handle_sliders(self, event):
        if self.slider_rows.handle_event(event):
            self.resize_grid(self.slider_rows.value, self.slider_cols.value)
            return True
        if self.slider_cols.handle_event(event):
            self.resize_grid(self.slider_rows.value, self.slider_cols.value)
            return True
        if self.slider_density.handle_event(event):
            return True
        return False

    def resize_grid(self, new_rows, new_cols):
        if new_rows == self.rows and new_cols == self.cols:
            return
        old_numbers = [row[:] for row in self.numbers]
        old_colors = [row[:] for row in self.cell_colors]
        self.rows, self.cols = new_rows, new_cols
        self.numbers = [[-2 for _ in range(new_cols)] for _ in range(new_rows)]
        self.cell_colors = [[None for _ in range(new_cols)] for _ in range(new_rows)]
        self.h_edges = [[0 for _ in range(new_cols+1)] for _ in range(new_rows+1)]
        self.v_edges = [[0 for _ in range(new_cols+1)] for _ in range(new_rows+1)]
        for i in range(min(len(old_numbers), new_rows)):
            for j in range(min(len(old_numbers[0]), new_cols)):
                self.numbers[i][j] = old_numbers[i][j]
                self.cell_colors[i][j] = old_colors[i][j]
        if self.selected_cell and (self.selected_cell[0] >= new_rows or self.selected_cell[1] >= new_cols):
            self.selected_cell = (min(self.selected_cell[0], new_rows-1), min(self.selected_cell[1], new_cols-1))
        self.update_cell_size()
        # Recalculer le défilement des boutons après redimensionnement
        self.max_button_scroll = max(0, len(self.buttons)*28 - (self.height - self.button_area_top - 20))

    def run(self):
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_e:
                        self.edit_mode = not self.edit_mode
                        if not self.edit_mode:
                            self.paint_mode = False
                            self.message = "Mode jeu (arêtes)"
                        else:
                            self.message = "Mode édition"
                    elif event.key == pygame.K_p and not self.edit_mode:
                        self.paint_mode = not self.paint_mode
                        mode = "Coloration" if self.paint_mode else "Arêtes"
                        self.message = f"Mode jeu : {mode}"
                    elif event.key == pygame.K_d:
                        self.theme.toggle()
                        for s in [self.slider_rows, self.slider_cols, self.slider_density]:
                            s.theme = self.theme
                    elif self.edit_mode and self.selected_cell:
                        i, j = self.selected_cell
                        if event.key == pygame.K_UP and i > 0:
                            self.selected_cell = (i-1, j)
                        elif event.key == pygame.K_DOWN and i < self.rows-1:
                            self.selected_cell = (i+1, j)
                        elif event.key == pygame.K_LEFT and j > 0:
                            self.selected_cell = (i, j-1)
                        elif event.key == pygame.K_RIGHT and j < self.cols-1:
                            self.selected_cell = (i, j+1)
                        elif event.key == pygame.K_c:
                            colors = [None,
                                    (255, 200, 200),   # rouge clair
                                    (200, 255, 200),   # vert clair
                                    (200, 200, 255),   # bleu clair
                                    (255, 255, 200),   # jaune clair
                                    (255, 200, 255),   # magenta clair
                                    (200, 255, 255),   # cyan clair
                                    (255, 220, 180),   # orange clair
                                    (220, 255, 220),   # vert pâle
                                    (220, 220, 255)]   # lavande
                            cur = self.cell_colors[i][j]
                            if cur is None:
                                self.cell_colors[i][j] = colors[1]
                            else:
                                try:
                                    idx = colors.index(cur)
                                    nxt = (idx + 1) % len(colors)
                                    self.cell_colors[i][j] = colors[nxt] if nxt != 0 else None
                                except ValueError:
                                    self.cell_colors[i][j] = colors[1]
                        elif event.key in range(pygame.K_0, pygame.K_9+1):
                            if event.key == pygame.K_6:
                                self.numbers[i][j] = 0
                            elif event.key == pygame.K_7:
                                self.numbers[i][j] = -2
                            else:
                                self.numbers[i][j] = event.key - pygame.K_0
                elif event.type == pygame.MOUSEWHEEL:
                    self.button_scroll -= event.y * 20
                    self.button_scroll = max(0, min(self.button_scroll, self.max_button_scroll))
                elif event.type == pygame.MOUSEMOTION:
                    if self.handle_sliders(event):
                        continue
                    if not self.edit_mode and not self.paint_mode:
                        mx, my = event.pos
                        if (self.grid_left <= mx <= self.grid_left + self.cols * self.cell_size and
                            self.grid_top <= my <= self.grid_top + self.rows * self.cell_size):
                            gx = (mx - self.grid_left) / self.cell_size
                            gy = (my - self.grid_top) / self.cell_size
                            self.hover_edge = None
                            for i in range(self.rows + 1):
                                for j in range(self.cols):
                                    if abs(gy - i) < 0.2 and j <= gx <= j+1:
                                        self.hover_edge = ('h', i, j)
                                        break
                                if self.hover_edge: break
                            if not self.hover_edge:
                                for i in range(self.rows):
                                    for j in range(self.cols + 1):
                                        if abs(gx - j) < 0.2 and i <= gy <= i+1:
                                            self.hover_edge = ('v', i, j)
                                            break
                                    if self.hover_edge: break
                        else:
                            self.hover_edge = None
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.handle_sliders(event):
                        continue
                    if self.handle_button_click(event.pos):
                        continue
                    mx, my = event.pos
                    if (self.grid_left <= mx <= self.grid_left + self.cols * self.cell_size and
                        self.grid_top <= my <= self.grid_top + self.rows * self.cell_size):
                        gx = (mx - self.grid_left) // self.cell_size
                        gy = (my - self.grid_top) // self.cell_size
                        if 0 <= gx < self.cols and 0 <= gy < self.rows:
                            if self.edit_mode:
                                self.selected_cell = (gy, gx)
                            else:
                                if self.paint_mode:
                                    self.toggle_cell_color(gy, gx)
                                    self.selected_cell = (gy, gx)
                                else:
                                    edge = self.get_edge_at_pixel(mx, my)
                                    if edge:
                                        if event.button == 1:
                                            self.set_edge(edge, 1)
                                        elif event.button == 3:
                                            self.set_edge(edge, 2)
                                        elif event.button == 2:
                                            self.set_edge(edge, 0)
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.handle_sliders(event)

            self.update_cell_size()
            self.draw()
            pygame.display.flip()
            clock.tick(60)
        pygame.quit()
        sys.exit()

    def get_edge_at_pixel(self, mx, my):
        if not (self.grid_left <= mx <= self.grid_left + self.cols * self.cell_size and
                self.grid_top <= my <= self.grid_top + self.rows * self.cell_size):
            return None
        gx = (mx - self.grid_left) / self.cell_size
        gy = (my - self.grid_top) / self.cell_size
        for i in range(self.rows + 1):
            for j in range(self.cols):
                if abs(gy - i) < 0.2 and j <= gx <= j+1:
                    return ('h', i, j)
        for i in range(self.rows):
            for j in range(self.cols + 1):
                if abs(gx - j) < 0.2 and i <= gy <= i+1:
                    return ('v', i, j)
        return None

    def set_edge(self, edge_info, edge_type):
        if not edge_info:
            return
        kind, i, j = edge_info
        if kind == 'h':
            self.h_edges[i][j] = edge_type
        else:
            self.v_edges[i][j] = edge_type

if __name__ == "__main__":
    game = SlitherLinkGame(rows=10, cols=10)
    game.run()
