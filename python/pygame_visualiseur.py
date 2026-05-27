import pygame
import sys
import math

class SlitherLink:
    def __init__(self, rows=5, cols=5):
        # Option cadrillage
        self.show_grid = False  # False = points, True = cadrillage
        
        self.rows = rows
        self.cols = cols
        self.cell_size = 45
        self.margin = 50
        self.width = 2 * self.margin + self.cols * self.cell_size + 250
        self.height = 2 * self.margin + self.rows * self.cell_size
        
        # État du jeu
        self.numbers = [[-2 for _ in range(cols)] for _ in range(rows)]
        self.h_edges = [[0 for _ in range(cols + 1)] for _ in range(rows + 1)]
        self.v_edges = [[0 for _ in range(cols + 1)] for _ in range(rows + 1)]
        
        # Couleurs des cases
        self.cell_colors = [[None for _ in range(cols)] for _ in range(rows)]
        
        # Mode édition
        self.edit_mode = True
        self.selected_cell = (0, 0)  # Commence à la première case
        
        # Champs de saisie pour la taille
        self.input_rows = str(rows)
        self.input_cols = str(cols)
        self.active_input = None  # 'rows' ou 'cols'
        
        # État de la souris - INITIALISATION AJOUTÉE
        self.hover_edge = None  # Cette ligne manquait!
        
        # Couleurs
        self.bg_color = (255, 255, 255)
        self.grid_color = (100, 100, 100)
        self.number_color = (0, 0, 0)
        self.line_color = (0, 0, 0)
        self.wrong_line_color = (255, 0, 0)
        self.hover_color = (150, 150, 150)
        self.dot_color = (0, 0, 0)
        self.edit_highlight = (255, 200, 200)
        self.interface_bg = (240, 240, 240)
        self.input_bg = (255, 255, 255)
        self.input_active_bg = (220, 240, 255)
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Slither Link - Mode Édition")
        self.font = pygame.font.SysFont('Arial', 24, bold=True)
        self.small_font = pygame.font.SysFont('Arial', 16)
        self.large_font = pygame.font.SysFont('Arial', 32, bold=True)
        
    def draw_grid(self):
        self.screen.fill(self.bg_color)
        
        # Dessiner le fond de l'interface
        interface_rect = pygame.Rect(self.width - 250, 0, 250, self.height)
        pygame.draw.rect(self.screen, self.interface_bg, interface_rect)
        
        # Dessiner les couleurs de fond des cases
        for i in range(self.rows):
            for j in range(self.cols):
                if self.cell_colors[i][j]:
                    x = self.margin + j * self.cell_size
                    y = self.margin + i * self.cell_size
                    pygame.draw.rect(self.screen, self.cell_colors[i][j], 
                                   (x, y, self.cell_size, self.cell_size))
        
        # Dessiner les nombres
        for i in range(self.rows):
            for j in range(self.cols):
                if self.numbers[i][j] != -2:  # Affiche tout sauf la valeur "effacée" (-2)
                    x = self.margin + j * self.cell_size + self.cell_size // 2
                    y = self.margin + i * self.cell_size + self.cell_size // 2
                    
                    # Fond avec la couleur de la case
                    if self.cell_colors[i][j]:
                        # Utiliser la couleur de la case comme fond
                        text_bg = pygame.Surface((30, 30), pygame.SRCALPHA)
                        text_bg.fill((*self.cell_colors[i][j], 200))  # Même couleur avec transparence
                    else:
                        # Fond blanc si pas de couleur
                        text_bg = pygame.Surface((30, 30), pygame.SRCALPHA)
                        text_bg.fill((255, 255, 255, 200))
                    
                    self.screen.blit(text_bg, (x-15, y-15))
                    
                    number_str = str(self.numbers[i][j]) if self.numbers[i][j] != -1 else "?"
                    number_text = self.font.render(number_str, True, self.number_color)
                    text_rect = number_text.get_rect(center=(x, y))
                    self.screen.blit(number_text, text_rect)
                
        # Dessiner les points ronds (plus gros)
        # Dessiner les points ou le cadrillage
        if not self.show_grid:
            # Mode points (actuel)
            for i in range(self.rows + 1):
                for j in range(self.cols + 1):
                    x = self.margin + j * self.cell_size
                    y = self.margin + i * self.cell_size
                    pygame.draw.circle(self.screen, self.dot_color, (x, y), 7)
        else:
            # Mode cadrillage
            for i in range(self.rows + 1):
                for j in range(self.cols + 1):
                    x = self.margin + j * self.cell_size
                    y = self.margin + i * self.cell_size
                    # Points plus petits et discrets
                    pygame.draw.circle(self.screen, self.grid_color, (x, y), 3)
            
            # Lignes de grille
            for i in range(self.rows + 1):
                # Lignes horizontales
                y = self.margin + i * self.cell_size
                pygame.draw.line(self.screen, self.grid_color, 
                                (self.margin, y), 
                                (self.margin + self.cols * self.cell_size, y), 1)
            
            for j in range(self.cols + 1):
                # Lignes verticales
                x = self.margin + j * self.cell_size
                pygame.draw.line(self.screen, self.grid_color, 
                                (x, self.margin), 
                                (x, self.margin + self.rows * self.cell_size), 1)
        
        # Dessiner les arêtes (noires ou rouges si incorrectes)
# Dessiner les arêtes (noires ou rouges si incorrectes)
# Dessiner les arêtes (noires ou rouges si incorrectes)
        for i in range(self.rows + 1):
            for j in range(self.cols):
                if self.h_edges[i][j] == 1:
                    x1 = self.margin + j * self.cell_size
                    y = self.margin + i * self.cell_size
                    x2 = x1 + self.cell_size
                    pygame.draw.line(self.screen, self.line_color, (x1, y), (x2, y), 4)
                elif self.h_edges[i][j] == 2:  # Mauvaise liaison
                    x1 = self.margin + j * self.cell_size
                    y = self.margin + i * self.cell_size
                    x2 = x1 + self.cell_size
                    # Dessiner une croix en "X" avec extrémités perpendiculaires
                    cross_size = self.cell_size // 9
                    cross_thickness = 5
                    center_x = (x1 + x2) // 2
                    
                    # Première diagonale (\) avec extrémités carrées
                    start_x1 = center_x - cross_size
                    start_y1 = y - cross_size
                    end_x1 = center_x + cross_size
                    end_y1 = y + cross_size
                    pygame.draw.line(self.screen, self.wrong_line_color,
                                   (start_x1, start_y1), (end_x1, end_y1),
                                   cross_thickness)
                    
                    # Deuxième diagonale (/) avec extrémités carrées
                    start_x2 = center_x - cross_size
                    start_y2 = y + cross_size
                    end_x2 = center_x + cross_size
                    end_y2 = y - cross_size
                    pygame.draw.line(self.screen, self.wrong_line_color,
                                   (start_x2, start_y2), (end_x2, end_y2),
                                   cross_thickness)
                    
                    # Dessiner des petits carrés aux extrémités pour les rendre perpendiculaires
                    square_size = cross_thickness - 2
                    # Extrémités de la première diagonale
                    pygame.draw.rect(self.screen, self.wrong_line_color,
                                   (start_x1 - square_size//2, start_y1 - square_size//2,       square_size, square_size))
                    pygame.draw.rect(self.screen, self.wrong_line_color,
                                   (end_x1 - square_size//2, end_y1 - square_size//2,       square_size, square_size))
                    # Extrémités de la deuxième diagonale
                    pygame.draw.rect(self.screen, self.wrong_line_color,
                                   (start_x2 - square_size//2, start_y2 - square_size//2,       square_size, square_size))
                    pygame.draw.rect(self.screen, self.wrong_line_color,
                                   (end_x2 - square_size//2, end_y2 - square_size//2,       square_size, square_size))
        
        for i in range(self.rows):
            for j in range(self.cols + 1):
                if self.v_edges[i][j] == 1:
                    x = self.margin + j * self.cell_size
                    y1 = self.margin + i * self.cell_size
                    y2 = y1 + self.cell_size
                    pygame.draw.line(self.screen, self.line_color, (x, y1), (x, y2), 4)
                elif self.v_edges[i][j] == 2:  # Mauvaise liaison
                    x = self.margin + j * self.cell_size
                    y1 = self.margin + i * self.cell_size
                    y2 = y1 + self.cell_size
                    # Dessiner une croix en "X" avec extrémités perpendiculaires
                    cross_size = self.cell_size // 9
                    cross_thickness = 5
                    center_y = (y1 + y2) // 2
                    
                    # Première diagonale (\) avec extrémités carrées
                    start_x1 = x - cross_size
                    start_y1 = center_y - cross_size
                    end_x1 = x + cross_size
                    end_y1 = center_y + cross_size
                    pygame.draw.line(self.screen, self.wrong_line_color,
                                   (start_x1, start_y1), (end_x1, end_y1),
                                   cross_thickness)
                    
                    # Deuxième diagonale (/) avec extrémités carrées
                    start_x2 = x - cross_size
                    start_y2 = center_y + cross_size
                    end_x2 = x + cross_size
                    end_y2 = center_y - cross_size
                    pygame.draw.line(self.screen, self.wrong_line_color,
                                   (start_x2, start_y2), (end_x2, end_y2),
                                   cross_thickness)
                    
                    # Dessiner des petits carrés aux extrémités pour les rendre perpendiculaires
                    square_size = cross_thickness - 2
                    # Extrémités de la première diagonale
                    pygame.draw.rect(self.screen, self.wrong_line_color,
                                   (start_x1 - square_size//2, start_y1 - square_size//2,       square_size, square_size))
                    pygame.draw.rect(self.screen, self.wrong_line_color,
                                   (end_x1 - square_size//2, end_y1 - square_size//2,       square_size, square_size))
                        # Extrémités de la deuxième diagonale
                    pygame.draw.rect(self.screen, self.wrong_line_color,
                                   (start_x2 - square_size//2, start_y2 - square_size//2,       square_size, square_size))
                    pygame.draw.rect(self.screen, self.wrong_line_color,
                                   (end_x2 - square_size//2, end_y2 - square_size//2,       square_size, square_size))
                                        
    # Dessiner l'arête survolée (seulement en mode jeu) - CORRECTION: vérifier si hover_edge existe
        if hasattr(self, 'hover_edge') and self.hover_edge and not self.edit_mode:
            edge_type, i, j = self.hover_edge
            if edge_type == 'h':
                x1 = self.margin + j * self.cell_size
                y = self.margin + i * self.cell_size
                x2 = x1 + self.cell_size
                pygame.draw.line(self.screen, self.hover_color, (x1, y), (x2, y), 6)
            else:  # 'v'
                x = self.margin + j * self.cell_size
                y1 = self.margin + i * self.cell_size
                y2 = y1 + self.cell_size
                pygame.draw.line(self.screen, self.hover_color, (x, y1), (x, y2), 6)
        
        # Mode édition : surbrillance de la case sélectionnée
        if self.edit_mode and self.selected_cell:
            i, j = self.selected_cell
            x = self.margin + j * self.cell_size
            y = self.margin + i * self.cell_size
            pygame.draw.rect(self.screen, self.edit_highlight, 
                           (x, y, self.cell_size, self.cell_size), 3)
        
        # Dessiner l'interface
        self.draw_interface()
    
    def draw_interface(self):
        x_interface = self.width - 240
        
        # Titre du mode
        mode_text = "MODE ÉDITION" if self.edit_mode else "MODE JEU"
        mode_surface = self.font.render(mode_text, True, (0, 100, 0))
        self.screen.blit(mode_surface, (x_interface, 20))
        
        # Instructions mode édition
        if self.edit_mode:
            instructions = [
                "CONTROLES ÉDITION: 📝",
                "Flèches: Déplacer",
                "0-9: Chiffre 0-9",
                "5 ou ?: Point d'interrogation",
                "C: Changer couleur",
                "E: Mode jeu",
                "",
                "TAILLE GRILLE:"
            ]
        else:
            instructions = [
                "CONTROLES JEU: ✅ ",
                "Clic arête: Ligne noire",
                "Clic droit: Croix rouge",
                "Clic milieu: Effacer",
                "E: Mode édition"
            ]
        
        for idx, instruction in enumerate(instructions):
            text = self.small_font.render(instruction, True, (0, 0, 0))
            self.screen.blit(text, (x_interface, 60 + idx * 25))
        
        # Champs de saisie pour la taille de la grille
        if self.edit_mode:
            # Lignes
            rows_label = self.small_font.render("Lignes:", True, (0, 0, 0))
            self.screen.blit(rows_label, (x_interface, 220))
            
            rows_bg_color = self.input_active_bg if self.active_input == 'rows' else self.input_bg
            pygame.draw.rect(self.screen, rows_bg_color, (x_interface + 70, 215, 50, 30))
            pygame.draw.rect(self.screen, (0, 0, 0), (x_interface + 70, 215, 50, 30), 1)
            
            rows_text = self.small_font.render(self.input_rows, True, (0, 0, 0))
            self.screen.blit(rows_text, (x_interface + 75, 220))
            
            # Colonnes
            cols_label = self.small_font.render("Colonnes:", True, (0, 0, 0))
            self.screen.blit(cols_label, (x_interface, 260))
            
            cols_bg_color = self.input_active_bg if self.active_input == 'cols' else self.input_bg
            pygame.draw.rect(self.screen, cols_bg_color, (x_interface + 70, 255, 50, 30))
            pygame.draw.rect(self.screen, (0, 0, 0), (x_interface + 70, 255, 50, 30), 1)
            
            cols_text = self.small_font.render(self.input_cols, True, (0, 0, 0))
            self.screen.blit(cols_text, (x_interface + 75, 260))
            
            # Bouton Appliquer
            apply_bg = (180, 220, 180)
            pygame.draw.rect(self.screen, apply_bg, (x_interface + 130, 235, 80, 30))
            pygame.draw.rect(self.screen, (0, 0, 0), (x_interface + 130, 235, 80, 30), 1)
            apply_text = self.small_font.render("Appliquer", True, (0, 0, 0))
            self.screen.blit(apply_text, (x_interface + 140, 240))
        
        # Case sélectionnée
        if self.edit_mode and self.selected_cell:
            i, j = self.selected_cell
            selected_text = f"Case: ({i+1},{j+1})"
            value_text = f"Valeur: {self.numbers[i][j] if self.numbers[i][j] != -1 else '?'}"
            
            sel_surface = self.small_font.render(selected_text, True, (0, 0, 0))
            val_surface = self.small_font.render(value_text, True, (0, 0, 0))
            
            self.screen.blit(sel_surface, (x_interface, self.height - 80))
            self.screen.blit(val_surface, (x_interface, self.height - 55))
    
    def get_edge_at_pos(self, pos):
        x, y = pos
        if (x < self.margin or x > self.width - 250 - self.margin or 
            y < self.margin or y > self.height - self.margin):
            return None
        
        grid_x = (x - self.margin) / self.cell_size
        grid_y = (y - self.margin) / self.cell_size
        
        # Vérifier les arêtes horizontales
        for i in range(self.rows + 1):
            for j in range(self.cols):
                edge_y = i
                edge_x1 = j
                edge_x2 = j + 1
                
                if (abs(grid_y - edge_y) < 0.2 and 
                    edge_x1 <= grid_x <= edge_x2):
                    return ('h', i, j)
        
        # Vérifier les arêtes verticales
        for i in range(self.rows):
            for j in range(self.cols + 1):
                edge_x = j
                edge_y1 = i
                edge_y2 = i + 1
                
                if (abs(grid_x - edge_x) < 0.2 and 
                    edge_y1 <= grid_y <= edge_y2):
                    return ('v', i, j)
        
        return None
    
    def get_cell_at_pos(self, pos):
        x, y = pos
        if (x < self.margin or x > self.width - 250 - self.margin or 
            y < self.margin or y > self.height - self.margin):
            return None
        
        grid_x = int((x - self.margin) / self.cell_size)
        grid_y = int((y - self.margin) / self.cell_size)
        
        if 0 <= grid_x < self.cols and 0 <= grid_y < self.rows:
            return (grid_y, grid_x)
        return None
    
    def set_edge(self, edge_info, edge_type):
        if not edge_info:
            return
        
        edge_kind, i, j = edge_info
        
        if edge_kind == 'h':
            self.h_edges[i][j] = edge_type
        else:  # 'v'
            self.v_edges[i][j] = edge_type
    
    def cycle_cell_color(self, cell):
        if not cell:
            return
        
        i, j = cell
        colors = [None, (255, 200, 200), (200, 255, 200), (200, 200, 255), 
                 (255, 255, 200), (255, 200, 255), (200, 255, 255),
                 (255, 220, 180), (220, 255, 220), (220, 220, 255)]
        
        current_color = self.cell_colors[i][j]
        if current_color is None:
            self.cell_colors[i][j] = colors[1]
        else:
            try:
                current_index = colors.index(current_color)
                next_index = (current_index + 1) % len(colors)
                self.cell_colors[i][j] = colors[next_index] if next_index != 0 else None
            except ValueError:
                self.cell_colors[i][j] = colors[1]
    
    def resize_grid(self, new_rows, new_cols):
        if new_rows < 3: new_rows = 3
        if new_cols < 3: new_cols = 3
        if new_rows > 20: new_rows = 20
        if new_cols > 20: new_cols = 20
        
        # Sauvegarder l'état actuel
        old_rows, old_cols = self.rows, self.cols
        old_numbers = [row[:] for row in self.numbers]
        old_colors = [row[:] for row in self.cell_colors]
        
        # Redimensionner
        self.rows, self.cols = new_rows, new_cols
        self.numbers = [[0 for _ in range(new_cols)] for _ in range(new_rows)]
        self.cell_colors = [[None for _ in range(new_cols)] for _ in range(new_rows)]
        self.h_edges = [[0 for _ in range(new_cols + 1)] for _ in range(new_rows + 1)]
        self.v_edges = [[0 for _ in range(new_cols + 1)] for _ in range(new_rows + 1)]
        
        # Copier les anciennes valeurs
        for i in range(min(old_rows, new_rows)):
            for j in range(min(old_cols, new_cols)):
                self.numbers[i][j] = old_numbers[i][j]
                self.cell_colors[i][j] = old_colors[i][j]
        
        # Ajuster la sélection si nécessaire
        if self.selected_cell:
            i, j = self.selected_cell
            if i >= new_rows or j >= new_cols:
                self.selected_cell = (min(i, new_rows-1), min(j, new_cols-1))
        
        # Mettre à jour les champs de saisie
        self.input_rows = str(new_rows)
        self.input_cols = str(new_cols)
        
        # Recalculer la taille de la fenêtre
        self.width = 2 * self.margin + self.cols * self.cell_size + 250
        self.height = 2 * self.margin + self.rows * self.cell_size
        self.screen = pygame.display.set_mode((self.width, self.height))
    
    def move_selection(self, direction):
        if not self.selected_cell:
            return
        
        i, j = self.selected_cell
        
        if direction == 'up' and i > 0:
            self.selected_cell = (i-1, j)
        elif direction == 'down' and i < self.rows-1:
            self.selected_cell = (i+1, j)
        elif direction == 'left' and j > 0:
            self.selected_cell = (i, j-1)
        elif direction == 'right' and j < self.cols-1:
            self.selected_cell = (i, j+1)
    
    def check_input_fields_click(self, pos):
        x, y = pos
        x_interface = self.width - 240
        
        # Vérifier le champ lignes
        if x_interface + 70 <= x <= x_interface + 120 and 215 <= y <= 245:
            self.active_input = 'rows'
            return True
        
        # Vérifier le champ colonnes
        if x_interface + 70 <= x <= x_interface + 120 and 255 <= y <= 285:
            self.active_input = 'cols'
            return True
        
        # Vérifier le bouton Appliquer
        if x_interface + 130 <= x <= x_interface + 210 and 235 <= y <= 265:
            try:
                new_rows = int(self.input_rows)
                new_cols = int(self.input_cols)
                self.resize_grid(new_rows, new_cols)
            except ValueError:
                pass  # Ignorer les valeurs non numériques
            return True
        
        self.active_input = None
        return False
    
    def handle_text_input(self, event):
        if self.active_input == 'rows':
            if event.key == pygame.K_BACKSPACE:
                self.input_rows = self.input_rows[:-1]
            elif event.key == pygame.K_RETURN:
                try:
                    new_rows = int(self.input_rows)
                    self.resize_grid(new_rows, self.cols)
                except ValueError:
                    pass
            elif event.unicode.isdigit():
                if len(self.input_rows) < 2:  # Limiter à 2 chiffres
                    self.input_rows += event.unicode
        
        elif self.active_input == 'cols':
            if event.key == pygame.K_BACKSPACE:
                self.input_cols = self.input_cols[:-1]
            elif event.key == pygame.K_RETURN:
                try:
                    new_cols = int(self.input_cols)
                    self.resize_grid(self.rows, new_cols)
                except ValueError:
                    pass
            elif event.unicode.isdigit():
                if len(self.input_cols) < 2:  # Limiter à 2 chiffres
                    self.input_cols += event.unicode
    
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.MOUSEMOTION:
                    if self.edit_mode:
                        self.hover_edge = None
                    else:
                        self.hover_edge = self.get_edge_at_pos(event.pos)
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.edit_mode:
                        # Vérifier d'abord les champs de saisie
                        if self.check_input_fields_click(event.pos):
                            continue
                        
                        # Sinon, sélectionner une case
                        cell = self.get_cell_at_pos(event.pos)
                        if cell:
                            self.selected_cell = cell
                    else:
                        # Mode jeu : interagir avec les arêtes
                        if event.button == 1:  # Clic gauche - ligne noire
                            self.set_edge(self.hover_edge, 1)
                        elif event.button == 3:  # Clic droit - croix rouge
                            self.set_edge(self.hover_edge, 2)
                        elif event.button == 2:  # Clic milieu - effacer
                            self.set_edge(self.hover_edge, 0)
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_e:
                        # Basculer entre mode édition et mode jeu
                        self.edit_mode = not self.edit_mode
                        caption = "Slither Link - Mode Édition" if self.edit_mode else "Slither Link - Mode Jeu"
                        pygame.display.set_caption(caption)
    
                    elif event.key == pygame.K_g:  # Touche G pour basculer le cadrillage
                        self.show_grid = not self.show_grid

                    
                    elif self.edit_mode:
                        # Gestion des champs de saisie
                        if self.active_input:
                            self.handle_text_input(event)
                            continue
                        
                        # Déplacement avec les flèches
                        if event.key == pygame.K_UP:
                            self.move_selection('up')
                        elif event.key == pygame.K_DOWN:
                            self.move_selection('down')
                        elif event.key == pygame.K_LEFT:
                            self.move_selection('left')
                        elif event.key == pygame.K_RIGHT:
                            self.move_selection('right')
                        
                        # Modification des valeurs
                        elif event.key == pygame.K_c and self.selected_cell:
                            self.cycle_cell_color(self.selected_cell)
                        
                        elif (event.key == pygame.K_5 or event.key == pygame.K_SLASH or 
                              (event.key == pygame.K_QUESTION and event.mod & pygame.KMOD_SHIFT)):
                            if self.selected_cell:
                                i, j = self.selected_cell
                                self.numbers[i][j] = -1  # Point d'interrogation
                        
                        elif event.key in [pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3, 
                                            pygame.K_4, pygame.K_5, pygame.K_6, pygame.K_7, 
                                            pygame.K_8, pygame.K_9] and self.selected_cell:
                            i, j = self.selected_cell
                            if event.key == pygame.K_6:
                                self.numbers[i][j] = 0  # 6 met un 0
                            elif event.key == pygame.K_7:
                                self.numbers[i][j] = -2 # efface
                            else:
                                self.numbers[i][j] = event.key - pygame.K_0
            
            self.draw_grid()
            pygame.display.flip()
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = SlitherLink(rows=5, cols=5)
    game.run()
