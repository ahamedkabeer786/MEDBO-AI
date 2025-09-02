import math
import time
import random
from collections import deque
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ---------------------------- PAGE CONFIG & THEME ----------------------------
st.set_page_config(
    page_title="MEDBO AI — Monitoring & Navigation",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Gradient / UI polish via CSS
st.markdown("""
<style>
/* App background gradient */
.main, .block-container {
  background: linear-gradient(135deg, #0b0f1a 0%, #0f1f3a 50%, #0b0f1a 100%) !important;
  color: #e8f0ff !important;
}
/* Headings */
h1, h2, h3, h4 {
  color: #d4e6ff !important;
}
/* Metric cards tweak */
[data-testid="stMetric"] div {
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 8px 12px;
}
/* Buttons */
.stButton>button {
  background: linear-gradient(90deg, #1dd1a1, #48dbfb);
  color: #06233b;
  border: 0;
  border-radius: 12px;
  padding: 0.6rem 1.1rem;
  font-weight: 700;
}
.stButton>button:hover {
  filter: brightness(0.95);
}
/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  gap: 4px;
}
.stTabs [data-baseweb="tab"] {
  background: rgba(255,255,255,0.06);
  padding: 10px 16px;
  border-radius: 12px;
  color: #d4e6ff;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(90deg, #0fb9b1 0%, #2e86de 100%);
  color: #06233b !important;
}
hr {
  border: none;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------- SESSION STATE SETUP ----------------------------
def init_state():
    ss = st.session_state
    # Monitoring flags & data
    ss.setdefault("monitoring_on", False)
    ss.setdefault("hb_data", deque(maxlen=300))   # heart-beat signal
    ss.setdefault("spo2_data", deque(maxlen=120))
    ss.setdefault("bp_sys_data", deque(maxlen=120))
    ss.setdefault("bp_dia_data", deque(maxlen=120))
    ss.setdefault("last_update_ts", 0.0)

    # Navigation flags & data
    ss.setdefault("nav_on", False)
    ss.setdefault("grid_w", 30)
    ss.setdefault("grid_h", 18)
    ss.setdefault("cell_px", 20)
    ss.setdefault("obstacles", set())
    ss.setdefault("start", (2, 9))       # patient room
    ss.setdefault("goal", (27, 9))       # pharmacy
    ss.setdefault("robot_pos", (2, 9))
    ss.setdefault("path", [])
    ss.setdefault("lidar_range", 10)     # cells
    ss.setdefault("lidar_rays", 72)      # 5 degrees
    ss.setdefault("last_nav_update", 0.0)

init_state()

# ---------------------------- HELPERS: MONITORING ----------------------------
def generate_ecg_point(t, bpm=78):
    """Simple synthetic ECG-like waveform (not medical-accurate)."""
    # Period in seconds
    period = 60.0 / bpm
    phase = (t % period) / period  # 0..1
    # Create a stylized P-QRS-T shape using piecewise bumps
    # baseline
    y = 0.02 * np.sin(2 * np.pi * 5 * phase)

    # P wave
    if 0.08 < phase < 0.16:
        y += 0.1 * math.exp(-((phase - 0.12) ** 2) / 0.0008)

    # QRS complex
    if 0.18 < phase < 0.24:
        y -= 0.6 * math.exp(-((phase - 0.20) ** 2) / 0.00009)  # Q small dip
    if 0.20 < phase < 0.22:
        y += 1.4 * math.exp(-((phase - 0.21) ** 2) / 0.000015) # R spike
    if 0.22 < phase < 0.26:
        y -= 0.3 * math.exp(-((phase - 0.24) ** 2) / 0.00009)  # S dip

    # T wave
    if 0.34 < phase < 0.50:
        y += 0.25 * math.exp(-((phase - 0.42) ** 2) / 0.0025)

    # Add small noise
    y += np.random.normal(0, 0.01)
    # Scale to typical mV-ish and offset
    return y

def simulate_spo2_value():
    base = 97 + np.random.normal(0, 0.4)
    return max(92, min(100, base))

def simulate_bp_values():
    # Systolic ~ 110-125, Diastolic ~ 70-85
    sys = 118 + np.random.normal(0, 3.5)
    dia = 78 + np.random.normal(0, 2.5)
    return int(sys), int(dia)

# ---------------------------- HELPERS: NAVIGATION (A*) ----------------------------
def neighbors(pt, w, h):
    (x, y) = pt
    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
        nx, ny = x+dx, y+dy
        if 0 <= nx < w and 0 <= ny < h:
            yield (nx, ny)

def heuristic(a, b):
    # Manhattan distance
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def a_star(start, goal, w, h, obstacles):
    open_set = {start}
    came_from = {}
    g = {start: 0}
    f = {start: heuristic(start, goal)}
    while open_set:
        current = min(open_set, key=lambda n: f.get(n, float('inf')))
        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        open_set.remove(current)
        for nb in neighbors(current, w, h):
            if nb in obstacles:
                continue
            tentative_g = g[current] + 1
            if tentative_g < g.get(nb, float('inf')):
                came_from[nb] = current
                g[nb] = tentative_g
                f[nb] = tentative_g + heuristic(nb, goal)
                open_set.add(nb)
    return []

def cast_lidar_rays(pos, obstacles, w, h, rays=72, max_range=10):
    """Return list of (end_x, end_y) cells where the ray stopped or max range reached."""
    ox, oy = pos
    hits = []
    for i in range(rays):
        angle = (2 * math.pi) * (i / rays)
        dx = math.cos(angle)
        dy = math.sin(angle)
        dist = 0.0
        cx, cy = ox, oy
        hit = (ox, oy)
        while dist < max_range:
            # step in small increments and map to nearest cell
            cx_f = ox + dx * dist
            cy_f = oy + dy * dist
            cell = (min(max(int(round(cx_f)), 0), w-1),
                    min(max(int(round(cy_f)), 0), h-1))
            if cell in obstacles:
                hit = cell
                break
            hit = cell
            dist += 0.25
        hits.append(hit)
    return hits

def randomize_obstacles(w, h, start, goal, density=0.14, margin=2):
    obs = set()
    for x in range(w):
        for y in range(h):
            if (x, y) in [start, goal]:
                continue
            # Keep clear corridor around start & goal
            if abs(x-start[0])<=margin and abs(y-start[1])<=margin:
                continue
            if abs(x-goal[0])<=margin and abs(y-goal[1])<=margin:
                continue
            if random.random() < density:
                obs.add((x, y))
    return obs

# ---------------------------- HEADER / NAV ----------------------------
st.title("🤖 MEDBO AI — Pro Dashboard")
st.caption("Autonomous Medical Delivery Robot • Real-Time Patient Monitoring & A* Navigation with LIDAR")

tabs = st.tabs(["🏠 Overview", "🫀 Patient Monitoring", "🧭 Robot Navigation"])

# ---------------------------- TAB: OVERVIEW ----------------------------
with tabs[0]:
    st.subheader("Project Overview")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.markdown("""
**MEDBO AI** combines **real-time patient monitoring** with **autonomous navigation**:
- Live ECG-like heart signal, SpO₂, and BP
- Threshold-based alerts
- A* pathfinding on hospital grid
- LIDAR-like scan for nearby obstacles
- Start/Stop control for both modules
        """)
        st.markdown("---")
        st.markdown("**Quick Controls**")
        c1, c2, c3 = st.columns(3)
        with c1:
            if not st.session_state.monitoring_on:
                if st.button("▶ Start Monitoring"):
                    st.session_state.monitoring_on = True
            else:
                if st.button("⏸ Stop Monitoring"):
                    st.session_state.monitoring_on = False
        with c2:
            if not st.session_state.nav_on:
                if st.button("🚀 Start Navigation"):
                    st.session_state.nav_on = True
            else:
                if st.button("🛑 Stop Navigation"):
                    st.session_state.nav_on = False
        with c3:
            if st.button("🔄 Reset All"):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                init_state()
                st.success("State reset.")
                st.rerun()

    with col2:
        st.markdown("**Live Status**")
        m1, m2, m3 = st.columns(3)
        hb = st.session_state.hb_data[-1] if st.session_state.hb_data else 0.0
        spo2 = int(st.session_state.spo2_data[-1]) if st.session_state.spo2_data else 98
        bps = st.session_state.bp_sys_data[-1] if st.session_state.bp_sys_data else 118
        bpd = st.session_state.bp_dia_data[-1] if st.session_state.bp_dia_data else 78
        m1.metric("ECG Signal (mV)", f"{hb:+.2f}")
        m2.metric("SpO₂ (%)", f"{spo2}")
        m3.metric("BP (mmHg)", f"{int(bps)}/{int(bpd)}")

# ---------------------------- TAB: PATIENT MONITORING ----------------------------
with tabs[1]:
    st.subheader("Real-Time Patient Monitoring")

    # Controls
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if not st.session_state.monitoring_on:
            if st.button("▶ Start Monitoring", key="mon_start"):
                st.session_state.monitoring_on = True
        else:
            if st.button("⏸ Stop Monitoring", key="mon_stop"):
                st.session_state.monitoring_on = False
    with c2:
        if st.button("🧹 Clear Graphs"):
            st.session_state.hb_data.clear()
            st.session_state.spo2_data.clear()
            st.session_state.bp_sys_data.clear()
            st.session_state.bp_dia_data.clear()
    with c3:
        bpm = st.slider("Target Heart Rate (BPM)", 55, 110, 78)
    with c4:
        speed = st.selectbox("Update Speed", ["Slow", "Normal", "Fast"], index=1)

    # Live Metrics
    m1, m2, m3 = st.columns(3)
    hb_val = st.session_state.hb_data[-1] if st.session_state.hb_data else 0.0
    m1.metric("ECG (mV)", f"{hb_val:+.2f}")
    last_spo2 = int(st.session_state.spo2_data[-1]) if st.session_state.spo2_data else 98
    m2.metric("SpO₂ (%)", f"{last_spo2}")
    bps = st.session_state.bp_sys_data[-1] if st.session_state.bp_sys_data else 118
    bpd = st.session_state.bp_dia_data[-1] if st.session_state.bp_dia_data else 78
    m3.metric("BP (mmHg)", f"{int(bps)}/{int(bpd)}")

    # Placeholders for charts
    ecg_ph = st.empty()
    trio_ph = st.empty()
    alert_ph = st.empty()

    # Update loop (short burst, relies on reruns from Streamlit)
    burst_iterations = {"Slow": 10, "Normal": 20, "Fast": 35}[speed]
    if st.session_state.monitoring_on:
        for i in range(burst_iterations):
            t = time.time()
            st.session_state.hb_data.append(generate_ecg_point(t, bpm=bpm))
            if i % 5 == 0:
                sp = simulate_spo2_value()
                st.session_state.spo2_data.append(sp)
                sy, di = simulate_bp_values()
                st.session_state.bp_sys_data.append(sy)
                st.session_state.bp_dia_data.append(di)

            # ECG-like line
            fig, ax = plt.subplots(figsize=(10, 2.6))
            ax.plot(list(st.session_state.hb_data), linewidth=1.6)
            ax.set_ylim(-0.8, 1.8)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title("ECG-like Heart Signal", color="#d4e6ff")
            ax.set_facecolor("#0b1326")
            for spine in ax.spines.values():
                spine.set_color("#2b3b5a")
            ecg_ph.pyplot(fig, clear_figure=True)

            # Trio chart: SpO2 + BP
            fig2, ax2 = plt.subplots(figsize=(10, 2.4))
            ax2.plot(list(st.session_state.spo2_data), label="SpO₂", linewidth=1.6)
            ax2.plot(list(st.session_state.bp_sys_data), label="BP Systolic", linewidth=1.2)
            ax2.plot(list(st.session_state.bp_dia_data), label="BP Diastolic", linewidth=1.2)
            ax2.set_ylim(60, 140)
            ax2.set_title("SpO₂ and Blood Pressure", color="#d4e6ff")
            ax2.set_facecolor("#0b1326")
            ax2.legend()
            for spine in ax2.spines.values():
                spine.set_color("#2b3b5a")
            trio_ph.pyplot(fig2, clear_figure=True)

            # Alerts
            alerts = []
            if last_spo2 < 94:
                alerts.append(f"⚠ Low SpO₂ detected: {last_spo2}%")
            if bps > 135 or bpd > 90:
                alerts.append(f"⚠ High BP detected: {int(bps)}/{int(bpd)}")

            if alerts:
                for a in alerts:
                    alert_ph.warning(a)
            else:
                alert_ph.info("All vitals in safe range.")

            time.sleep(0.08)

# ---------------------------- TAB: ROBOT NAVIGATION ----------------------------
with tabs[2]:
    st.subheader("Autonomous Navigation — A* Pathfinding + LIDAR")

    w = st.session_state.grid_w
    h = st.session_state.grid_h
    start = st.session_state.start
    goal = st.session_state.goal

    c1, c2, c3, c4 = st.columns([1.1,1.1,1,1.2])
    with c1:
        if not st.session_state.nav_on:
            if st.button("🚀 Start Navigation", key="nav_start"):
                st.session_state.nav_on = True
                # Compute path if needed
                if not st.session_state.path:
                    st.session_state.path = a_star(start, goal, w, h, st.session_state.obstacles)
        else:
            if st.button("🛑 Stop Navigation", key="nav_stop"):
                st.session_state.nav_on = False
    with c2:
        if st.button("🎲 Randomize Obstacles"):
            st.session_state.obstacles = randomize_obstacles(w, h, start, goal, density=0.15)
            st.session_state.path = a_star(start, goal, w, h, st.session_state.obstacles)
            st.session_state.robot_pos = start
    with c3:
        if st.button("🔄 Reset Map"):
            st.session_state.obstacles = set()
            st.session_state.path = a_star(start, goal, w, h, st.session_state.obstacles)
            st.session_state.robot_pos = start
    with c4:
        st.session_state.lidar_range = st.slider("LIDAR Range (cells)", 5, 18, st.session_state.lidar_range)

    # Ensure path is available
    if not st.session_state.path:
        st.session_state.path = a_star(start, goal, w, h, st.session_state.obstacles)

    # Plot grid, obstacles, path, robot, goal + lidar
    nav_placeholder = st.empty()
    status_placeholder = st.empty()

    def draw_nav():
        cell_px = st.session_state.cell_px
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.set_facecolor("#0b1326")
        # Grid
        for x in range(w+1):
            ax.plot([x, x], [0, h], color="#1f2a44", linewidth=0.5)
        for y in range(h+1):
            ax.plot([0, w], [y, y], color="#1f2a44", linewidth=0.5)

        # Obstacles
        for (ox, oy) in st.session_state.obstacles:
            ax.add_patch(plt.Rectangle((ox, oy), 1, 1, color="#2e3b5a"))

        # Path
        if st.session_state.path:
            px = [p[0]+0.5 for p in st.session_state.path]
            py = [p[1]+0.5 for p in st.session_state.path]
            ax.plot(px, py, linewidth=2.2, color="#1dd1a1", alpha=0.9, label="A* Path")

        # LIDAR rays
        rp = st.session_state.robot_pos
        hits = cast_lidar_rays(rp, st.session_state.obstacles, w, h, rays=st.session_state.lidar_rays, max_range=st.session_state.lidar_range)
        for hx, hy in hits:
            ax.plot([rp[0]+0.5, hx+0.5], [rp[1]+0.5, hy+0.5], color="#10ac84", alpha=0.35, linewidth=1)

        # Start / Goal / Robot
        ax.add_patch(plt.Circle((start[0]+0.5, start[1]+0.5), 0.38, color="#54a0ff"))
        ax.text(start[0]+0.5, start[1]+0.5, "Room", color="white", ha="center", va="center", fontsize=9)

        ax.add_patch(plt.Circle((goal[0]+0.5, goal[1]+0.5), 0.38, color="#ff9f43"))
        ax.text(goal[0]+0.5, goal[1]+0.5, "Pharmacy", color="black", ha="center", va="center", fontsize=9)

        ax.add_patch(plt.Circle((rp[0]+0.5, rp[1]+0.5), 0.35, color="#5f27cd"))
        ax.text(rp[0]+0.5, rp[1]+0.5, "🤖", ha="center", va="center", fontsize=10)

        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Hospital Grid • A* Pathfinding with LIDAR Scan", color="#d4e6ff")
        for spine in ax.spines.values():
            spine.set_color("#2b3b5a")
        nav_placeholder.pyplot(fig, clear_figure=True)

    # Animate robot along path (short burst for each run)
    if st.session_state.nav_on and st.session_state.path:
        # If robot at goal, reverse path to return
        if st.session_state.robot_pos == st.session_state.goal:
            st.session_state.path = list(reversed(st.session_state.path))
            st.session_state.goal, st.session_state.start = st.session_state.start, st.session_state.goal
            status_placeholder.success("Reached Pharmacy ✅ — Returning to Patient Room...")

        steps_per_burst = 8
        for _ in range(steps_per_burst):
            # If path empty, recompute
            if not st.session_state.path:
                st.session_state.path = a_star(st.session_state.robot_pos, st.session_state.goal, w, h, st.session_state.obstacles)
                if not st.session_state.path:
                    status_placeholder.error("No path available! Adjust obstacles or reset map.")
                    break

            # Move one step along path
            if len(st.session_state.path) > 1:
                st.session_state.robot_pos = st.session_state.path[1]
                st.session_state.path = st.session_state.path[1:]
            draw_nav()
            time.sleep(0.08)

            # Status line
            if st.session_state.robot_pos == st.session_state.goal:
                if st.session_state.goal == (27, 9):
                    status_placeholder.success("Reached Pharmacy ✅")
                else:
                    status_placeholder.success("Returned to Patient Room ✅")

    else:
        draw_nav()
        if not st.session_state.nav_on:
            status_placeholder.info("Click **Start Navigation** to begin route.")
