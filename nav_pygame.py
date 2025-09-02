"""
MEDBO AI — Navigation Simulation with LIDAR-like Scan + A* replanning
Save as: nav_pygame_lidar.py
Dependencies: pygame, pyttsx3 (optional for voice)
Place robot.png and pharmacy.png in same folder (or replace icons)
Controls:
    SPACE - start navigation
    P     - pause/resume
    R     - reset
    O     - toggle dynamic obstacles
    ESC   - quit
"""

import pygame, sys, random, math, heapq, threading, time, queue
from dataclasses import dataclass

# ---------- Config ----------
CELL = 36
ROWS = 14
COLS = 20
WIDTH = COLS * CELL
HEIGHT = ROWS * CELL + 120
FPS = 60
ROBOT_SPEED = 140.0   # pixels / second
OBSTACLE_COUNT = 10
LIDAR_RANGE_CELLS = 5    # how many grid cells LIDAR can see (radius)
LIDAR_ANGLE_STEP = 6     # degrees per LIDAR beam

# Colors
BG = (12, 12, 18)
GRID = (28, 30, 46)
CELL_FREE = (18, 18, 24)
WALL = (60, 60, 64)
DYN_OBS = (220, 80, 80)
ROBOT_CLR = (60, 200, 230)
PATH_CLR = (0, 190, 140)
GOAL_CLR = (255, 200, 70)
START_CLR = (70, 200, 100)
HUD_BG = (16, 18, 28)
LIDAR_COLOR = (120, 255, 180)
DETECTED_COLOR = (255, 160, 60)
TEXT = (230, 230, 230)

# ---------- A* (grid) ----------
def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    openq = []
    heapq.heappush(openq, (heuristic(start,goal), 0, start))
    came = {}
    g = {start: 0}
    visited = set()
    while openq:
        f, cost, current = heapq.heappop(openq)
        if current == goal:
            path = [current]
            while current in came:
                current = came[current]
                path.append(current)
            return path[::-1]
        if current in visited:
            continue
        visited.add(current)
        r,c = current
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr, nc = r+dr, c+dc
            if 0<=nr<rows and 0<=nc<cols and grid[nr][nc]==0:
                tentative = g[current] + 1
                if tentative < g.get((nr,nc), 1e9):
                    came[(nr,nc)] = current
                    g[(nr,nc)] = tentative
                    heapq.heappush(openq, (tentative + heuristic((nr,nc), goal), tentative, (nr,nc)))
    return None

# ---------- Speech queue (safe pyttsx3) ----------
speech_q = queue.Queue()
try:
    import pyttsx3
    def tts_worker():
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        try:
            voices = engine.getProperty('voices')
            if len(voices) > 1:
                engine.setProperty('voice', voices[1].id)
        except Exception:
            pass
        while True:
            text = speech_q.get()
            if text is None:
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception:
                pass
            speech_q.task_done()
    threading.Thread(target=tts_worker, daemon=True).start()
    def speak(text):
        speech_q.put(text)
except Exception:
    def speak(text):
        print("[SAY]", text)

# ---------- Simulation class ----------
@dataclass
class Config:
    rows: int = ROWS
    cols: int = COLS
    cell: int = CELL

class NavSim:
    def __init__(self, cfg: Config):
        pygame.init()
        pygame.display.set_caption("MEDBO AI — LIDAR Navigation")
        self.cfg = cfg
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 16)
        self.big = pygame.font.SysFont("Consolas", 20, bold=True)

        # occupancy grid: 0 free, 1 static wall
        self.grid = [[0]*self.cfg.cols for _ in range(self.cfg.rows)]
        # add demo static walls
        for r in range(3, 11):
            self.grid[r][7] = 1
        for c in range(6, 16):
            self.grid[9][c] = 1
        self.grid[9][10] = 0

        # start / goal
        self.start = (self.cfg.rows - 2, 2)
        self.goal = (2, self.cfg.cols - 3)

        # dynamic obstacles
        self.dynamic = set()
        self.spawn_dynamic(OBSTACLE_COUNT)

        # robot state
        self.robot_cell = self.start
        self.robot_pos = self.cell_center(self.robot_cell)
        self.path = []
        self.path_index = 0
        self.moving = False
        self.paused = False
        self.returning = False

        self.last_obs_move = time.time()
        self.last_lidar_angle = 0.0
        self.detected_cells = set()   # cells LIDAR currently sees as obstacles

        self.log_lines = []
        self.add_log("Ready. Press SPACE to start navigation.")

    # helpers
    def add_log(self, txt):
        self.log_lines.append(f"{time.strftime('%H:%M:%S')} | {txt}")
        if len(self.log_lines) > 6:
            self.log_lines.pop(0)
        print(txt)

    def spawn_dynamic(self, n):
        free = [(r,c) for r in range(self.cfg.rows) for c in range(self.cfg.cols)
                if self.grid[r][c]==0 and (r,c)!=self.start and (r,c)!=self.goal]
        random.shuffle(free)
        for i in range(min(n, len(free))):
            self.dynamic.add(free[i])

    def cell_center(self, cell):
        r,c = cell
        x = c*self.cfg.cell + self.cfg.cell//2
        y = r*self.cfg.cell + self.cfg.cell//2
        return [float(x), float(y)]

    def current_grid(self):
        g = [row[:] for row in self.grid]
        for (r,c) in self.dynamic:
            g[r][c] = 1
        # incorporate detected (LIDAR) obstacles too (simulated sensor)
        for (r,c) in self.detected_cells:
            if 0 <= r < self.cfg.rows and 0 <= c < self.cfg.cols:
                g[r][c] = 1
        return g

    def toggle_dynamic(self):
        if self.dynamic:
            self.dynamic.clear(); self.add_log("Dynamic obstacles OFF")
        else:
            self.spawn_dynamic(OBSTACLE_COUNT); self.add_log("Dynamic obstacles ON")

    def compute_path(self, start_cell, goal_cell):
        g = self.current_grid()
        return astar(g, start_cell, goal_cell)

    # dynamic obstacles movement
    def move_dynamic(self):
        newset = set()
        for (r,c) in list(self.dynamic):
            if random.random() < 0.6:
                dirs = [(0,0),(1,0),(-1,0),(0,1),(0,-1)]
                random.shuffle(dirs)
                moved = False
                for dr,dc in dirs:
                    nr, nc = r+dr, c+dc
                    if 0<=nr<self.cfg.rows and 0<=nc<self.cfg.cols and self.grid[nr][nc]==0 and (nr,nc) not in self.dynamic and (nr,nc)!=self.start and (nr,nc)!=self.goal:
                        newset.add((nr,nc)); moved=True; break
                if not moved:
                    newset.add((r,c))
            else:
                newset.add((r,c))
        self.dynamic = newset

    # LIDAR-like scan: rotate a beam sweep and mark detected obstacle cells
    def lidar_scan(self):
        # compute robot cell center and angle sweep
        cx, cy = self.robot_pos
        sensors = []
        detected = set()
        for angle_deg in range(0, 360, LIDAR_ANGLE_STEP):
            angle = math.radians(angle_deg + int(self.last_lidar_angle))
            # cast ray cell-by-cell up to range
            for d in range(1, LIDAR_RANGE_CELLS+1):
                rr = int((cy + math.sin(angle)*(d*self.cfg.cell)) // self.cfg.cell)
                cc = int((cx + math.cos(angle)*(d*self.cfg.cell)) // self.cfg.cell)
                if 0 <= rr < self.cfg.rows and 0 <= cc < self.cfg.cols:
                    sensors.append(((cx, cy), (cc*self.cfg.cell + self.cfg.cell//2, rr*self.cfg.cell + self.cfg.cell//2)))
                    # if obstacle present at (rr,cc) either static or dynamic -> detection
                    if self.grid[rr][cc] == 1 or (rr,cc) in self.dynamic:
                        detected.add((rr,cc))
                        break
                else:
                    break
        # rotate sweep slowly
        self.last_lidar_angle = (self.last_lidar_angle + 6) % 360
        self.detected_cells = detected
        return sensors, detected

    # robot animation along path
    def animate_robot(self, path, speed):
        if not path:
            self.add_log("No path found.")
            speak("No path found.")
            self.moving = False
            return
        self.add_log(f"Path found ({len(path)} steps). Starting navigation.")
        speak("Starting navigation to pharmacy.")
        self.moving = True
        self.returning = False
        self.path = path
        self.path_index = 0
        # iterate steps
        for _ in range(1000000):
            if not self.moving:
                self.add_log("Navigation aborted.")
                return
            if self.paused:
                time.sleep(0.05); continue
            if self.path_index >= len(self.path):
                # reached end
                if not self.returning:
                    self.add_log("Reached pharmacy.")
                    speak("Reached pharmacy.")
                    time.sleep(0.6)
                    self.returning = True
                    # compute return path
                    newpath = self.compute_path(self.goal, self.start)
                    if not newpath:
                        self.add_log("No return path found.")
                        speak("No return path found.")
                        self.moving = False
                        return
                    self.path = newpath
                    self.path_index = 0
                    self.add_log("Return path computed. Heading back.")
                    speak("Returning to patient room.")
                    continue
                else:
                    self.add_log("Returned to start. Navigation complete.")
                    speak("Navigation complete.")
                    self.moving = False
                    return
            # move towards next cell
            target = self.path[self.path_index]
            tx, ty = self.cell_center(target)
            rx, ry = self.robot_pos
            dx = tx - rx; dy = ty - ry
            dist = math.hypot(dx, dy)
            step = ROBOT_SPEED * speed
            if dist < 4:
                # arrive at cell
                self.robot_cell = target
                self.robot_pos = [tx, ty]
                self.path_index += 1
                # If next cell blocked by dynamic or detected obstacle -> replan
                if self.path_index < len(self.path):
                    nxt = self.path[self.path_index]
                    if nxt in self.dynamic or self.grid[nxt[0]][nxt[1]]==1 or nxt in self.detected_cells:
                        self.add_log("Obstacle detected on path. Replanning...")
                        speak("Obstacle detected on path. Recalculating route.")
                        goalcell = self.start if self.returning else self.goal
                        newp = self.compute_path(self.robot_cell, goalcell)
                        if newp:
                            self.path = newp
                            self.path_index = 0
                            self.add_log("New path found, resuming.")
                            continue
                        else:
                            self.add_log("No alternative path found.")
                            speak("No alternative path found. Aborting navigation.")
                            self.moving = False
                            return
                continue
            # interpolation move
            self.robot_pos[0] += dx / dist * step
            self.robot_pos[1] += dy / dist * step
            time.sleep(speed)

    def start_navigation(self, speed=0.02):
        if self.moving:
            self.add_log("Already navigating")
            return
        # compute path considering detected obstacles
        grid = self.current_grid()
        path = astar(grid, self.robot_cell, self.goal)
        if not path:
            self.add_log("No initial path found.")
            speak("No initial path found.")
            return
        # run animate in thread to not block UI
        threading.Thread(target=self.animate_robot, args=(path, speed), daemon=True).start()

    def stop_navigation(self):
        self.moving = False

    def reset(self):
        self.dynamic.clear()
        self.spawn_dynamic(OBSTACLE_COUNT)
        self.robot_cell = self.start
        self.robot_pos = self.cell_center(self.robot_cell)
        self.path = []
        self.path_index = 0
        self.moving = False
        self.returning = False
        self.detected_cells.clear()
        self.add_log("Simulation reset.")

    def draw(self):
        self.screen.fill(BG)
        # draw grid cells
        for r in range(self.cfg.rows):
            for c in range(self.cfg.cols):
                x = c*self.cfg.cell; y = r*self.cfg.cell
                rect = pygame.Rect(x,y,self.cfg.cell,self.cfg.cell)
                color = WALL if self.grid[r][c]==1 else CELL_FREE
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, GRID, rect, 1)
        # draw dynamic obstacles
        for (r,c) in self.dynamic:
            pygame.draw.rect(self.screen, DYN_OBS, (c*self.cfg.cell+6, r*self.cfg.cell+6, self.cfg.cell-12, self.cfg.cell-12), border_radius=6)
        # draw goal/start
        sx = self.start[1]*self.cfg.cell + self.cfg.cell//2; sy = self.start[0]*self.cfg.cell + self.cfg.cell//2
        gx = self.goal[1]*self.cfg.cell + self.cfg.cell//2; gy = self.goal[0]*self.cfg.cell + self.cfg.cell//2
        pygame.draw.circle(self.screen, START_CLR, (sx, sy), self.cfg.cell//3)
        pygame.draw.circle(self.screen, GOAL_CLR, (gx, gy), self.cfg.cell//3)
        # draw current planned path
        if self.path:
            pts = []
            for (r,c) in self.path:
                pts.append((c*self.cfg.cell + self.cfg.cell//2, r*self.cfg.cell + self.cfg.cell//2))
            if len(pts) > 1:
                pygame.draw.lines(self.screen, PATH_CLR, False, pts, 4)
        # draw robot
        rx, ry = int(self.robot_pos[0]), int(self.robot_pos[1])
        pygame.draw.circle(self.screen, ROBOT_CLR, (rx, ry), self.cfg.cell//3)
        # draw lidar rays and mark detected cells
        sensors, detected = self.lidar_scan()
        for ray in sensors:
            pygame.draw.line(self.screen, LIDAR_COLOR, ray[0], ray[1], 1)
        for (r,c) in detected:
            cx = c*self.cfg.cell + self.cfg.cell//2; cy = r*self.cfg.cell + self.cfg.cell//2
            pygame.draw.circle(self.screen, DETECTED_COLOR, (cx, cy), 6)
        # hud
        hud_y = self.cfg.rows*self.cfg.cell
        pygame.draw.rect(self.screen, HUD_BG, (0, hud_y, WIDTH, HEIGHT - hud_y))
        info = f"SPACE:Start  P:Pause  R:Reset  O:Toggle Obstacles  ESC:Quit"
        self.screen.blit(self.big.render(info, True, TEXT), (8, hud_y + 6))
        # logs
        for i, line in enumerate(self.log_lines):
            self.screen.blit(self.font.render(line, True, TEXT), (8, hud_y + 36 + i*18))
        # sensor status
        sens = f"Detected Cells: {len(self.detected_cells)}"
        self.screen.blit(self.font.render(sens, True, TEXT), (WIDTH - 260, hud_y + 36))
        # robot cell & path length
        cur = f"Robot cell: {self.robot_cell}   Path steps: {len(self.path)}   Returning: {self.returning}"
        self.screen.blit(self.font.render(cur, True, TEXT), (8, hud_y + 36 + 7*18))
        pygame.display.flip()

    def run(self):
        self.last_obs_move = time.time()
        while True:
            dt = self.clock.tick(FPS) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if 'speech_q' in globals():
                        speech_q.put(None)
                    pygame.quit(); sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.start_navigation(speed=0.02)
                    elif event.key == pygame.K_p:
                        self.paused = not self.paused
                        self.add_log(f"Paused={self.paused}")
                    elif event.key == pygame.K_r:
                        self.reset()
                    elif event.key == pygame.K_o:
                        self.toggle_dynamic()
                    elif event.key == pygame.K_ESCAPE:
                        if 'speech_q' in globals():
                            speech_q.put(None)
                        pygame.quit(); sys.exit()
            # move dynamic obstacles periodically
            if time.time() - self.last_obs_move > 0.8:
                if self.dynamic:
                    self.move_dynamic()
                self.last_obs_move = time.time()
            # draw
            self.draw()

# ---------- run ----------
if __name__ == "__main__":
    sim = NavSim(Config())
    sim.run()
