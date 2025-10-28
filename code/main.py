"""
===============================================================================
Autonomous Mobile Robot Navigation App/Pipeline
Author: Jonathan Loo
Version: 1.0
Date: October 2025
===============================================================================
Purpose
--------
Implements a synchronous Sense→Think→Act control loop for autonomous maze navigation.
Each loop iteration reads the robot pose and LiDAR scan, refines pose via
ICP scan matching, updates the occupancy grid map (OGM), computes or updates a path
(A* or frontier-based), generates a lookahead setpoint, applies it to the simulated
robot, and visualises/logs the result.

Core Concept
-------------
Demonstrates a compact “SLAM” pipeline:
    ICP-aided localisation + OGM mapping + goal/frontier navigation
executed in real time within a single blocking loop.

Simulation vs Real Operation
----------------------------
- **SIMULATION (default):** 
  `apply_setpoint()` advances robot pose internally via unicycle kinematics.
- **REAL MODE:** 
  `apply_setpoint()` transmits setpoints to hardware; display updates only from
  robot-reported pose/scan data. Loop remains synchronous and blocking.

Main Loop Sequence
------------------
SENSE → (ICP) → FUSE → MAP → PLAN → ACT → LOG/VIZ

1) Pose & LiDAR acquisition  
2) ICP alignment and gated fusion  
3) Occupancy grid update  
4) Path planning (`determine_navigation_path()`)  
5) Setpoint computation (`compute_setpoint()`)  
6) Motion update (`apply_setpoint()`)  
7) Visualisation and CSV logging  

Modes
-----
- **KNOWN:** Preplanned A* path to fixed goal.  
- **UNKNOWN:** Frontier-based exploration until goal discovered.  
- **GOALSEEKING:** Path-following using lookahead setpoints.  

Termination
------------
Loop ends when the robot reaches the goal (`arrival_tol_m`) or user quits ('q').

Notes
-----
- All localisation, mapping, and control logic run in one synchronous loop.
- For real-robot use, implement:
      get_pose(), get_scan(), apply_setpoint()
- Candidates only modify `determine_frontier_path()` for the unknown-world task.
"""

from util import *

# -----------------------------------------------------------------------------
# This is the main simulation configuration
# -----------------------------------------------------------------------------
DEFAULTS: Dict[str, Dict] = {
    "world": {
        "wall_half_thickness_m": 0.005,
        "border_thickness_m": 0.01,
    },
    "snake_maze": {
        "size_m": 1.80,
        "cell_size_m": 0.45,
        "num_walls": 4,
        "gap_cells": 1,
    },
    "random_maze": {
        "size_m": 1.80,
        "cell_size_m": 0.45,
        "random_wall_count": 5,
        "random_seed": None,
        "candidates_to_list": 3,
        "seed_scan_start": 0,
        "seed_scan_stride": 1,
        "max_attempts_per_page": 10000,
        "segment_len_cells_min": 1,
        "segment_len_cells_max": 2,
        "orientation_bias": 0.5,
    },
    "planning": {
        "sample_step_m": 0.03,
        "resample_ds_m": 0.05,
        "equal_eps": 1e-6,
        "seg_eps": 1e-9,
    },
    "lidar": {
        "num_rays": 360,
        "max_range_m": 3.0,
        "raycast_eps": 1e-6,
    },
    "ogm": {
        "xyreso_m": 0.03,
        "l_free": -0.4,
        "l_occ": 0.85,
        "l_min": -4.0,
        "l_max": 4.0,
        "hit_margin_m": 1e-3,
        "prob_free_max": 0.35,
        "prob_occ_min": 0.65,
        "size_eps": 1e-9,
        "gray_free": 0.9,
        "gray_occ": 0.0,
        "gray_unk": 1.0,
    },

    "icp_fusion": {
        "enabled": True,
        "alpha": 0.1,
        "max_trans_m": 0.20,
        "max_rot_deg": 20.0,
        "min_points": 50,
        "max_rmse_m": 0.05,
        "snap_trans_m": 0.02,
        "snap_rot_deg": 2.0,
    },
    "viz": {
        "main_figsize_in": (14, 10),
        "robot_arrow_len_m": 0.05,
        "robot_arrow_head_m": (0.03, 0.03),
        "ogm_arrow_len_m": 0.05,
        "ogm_arrow_head_m": (0.03, 0.03),
        "lidar_alpha": 0.2,
        "lidar_lw": 0.5,
        "thumb_size_in": (3, 3),
        "pause_s": 0.01,
    },
    "logging": {
        "level": logging.INFO,
        "format": "[%(levelname)s] %(message)s",
        "pose_csv": "pose.csv",
        "lidar_csv": "lidar.csv",
    },
    "app": {
        "arrival_tolerance_m": 0.1,
        "mode": "GOALSEEKING",  # fixed mode
        "map_type": "RANDOM",  # RANDOM | SNAKE
        "entrance_cell": (0, 0),
        "snake_goal_cell": (3, 3),
        "random_goal_cell": (3, 3),
    },
    "robot": {
        "robot_radius_m": 0.15,
        "turn_angle_rad": math.radians(36),
        "k_ang": 10,
        "v_max_mps": 1.0,  # may be 0.35 for real robot
        "dt_s": 0.1,
        "dt_guard_s": 1e-3,
    },
    "setpoing_cfg": {
        "lookahead_m": 0.3,
    },    
}

def install_key_to_viz(viz: Dict) -> None:
    """Attach keyboard listeners for the live plot window."""
    def _on_key(event):
        globals()["_LAST_KEY"] = event.key
    viz["fig"].canvas.mpl_connect("key_press_event", _on_key)

logging.basicConfig(level=DEFAULTS["logging"]["level"], format=DEFAULTS["logging"]["format"])
log = logging.getLogger("maze_app")

# -----------------------------------------------------------------------------
# This is the main application loop
# -----------------------------------------------------------------------------

def main() -> None:

# -----------------------------------------------------------------------------
# The following is the initial setup including user input, maze world generation, entrance and goal "cell" coordinates,
# initial path planning (mainly for the known maze), lidar, occupancy grid map (OGM), visualisation and logging setup. 
# -----------------------------------------------------------------------------
    settings = copy.deepcopy(DEFAULTS)
    app = ask_options(settings)
    nav_mode = choose_navigation_mode(settings)

    world, entrance, goal_cell = build_world(settings, app)
    planner = create_planner(world, settings["planning"]["sample_step_m"], settings["robot"]["robot_radius_m"])
    path = initialise_navigation_path(planner, entrance, goal_cell, settings, nav_mode)
    sensor = create_lidar(settings["lidar"])
    ogm = create_ogm(settings["ogm"], 0.0, 0.0, world["size_m"], world["size_m"])
    viz = create_viz(world["size_m"], world["cell_size_m"], settings["viz"], settings["robot"]["robot_radius_m"])
    logger_dict = create_logger(settings["lidar"]["num_rays"], settings["logging"])
    start_x, start_y = cell_center(entrance, world["cell_size_m"])
    start_heading = math.atan2(path[1][1] - start_y, path[1][0] - start_x) if len(path) >= 2 else 0.0
    astar_pts = planner["cspace"] if planner["cspace"] else planner["obstacles"]

    state = SimulationState(
        world=world,
        entrance=entrance,
        goal=make_goal(goal_cell),
        path=path,
        sensor=sensor,
        ogm=ogm,
        viz=viz,
        logger=logger_dict,
        pose=make_pose(start_x, start_y, start_heading),
        settings=settings,
        icp_prev_pts=None,
        icp_prev_pose=None,
        step=0,
        astar_pts=astar_pts,
        ctrl=settings["setpoing_cfg"].copy(),
        planner=planner,
    )
    state.robot_iface = load_robot_interface(state.settings)

    install_key_to_viz(state.viz)

    while True:
        key = globals().get("_LAST_KEY", None)
        globals()["_LAST_KEY"] = None
        if key == "q":
            print("Quit requested.")
            break
# -----------------------------------------------------------------------------
# Interface to simulated robot data or real robot data
# For real robot data, simply load the real robot data via the load_robot_interface()
# -----------------------------------------------------------------------------
        robot = state.robot_iface
        if robot is None:
            robot = state.robot_iface = load_robot_interface(state.settings)

# -----------------------------------------------------------------------------
# Main navigation pipeline
# read robot (pose, lidar) --> ICP matching (pose estimation) --> pose fusion --> update OGM --> path planning --> setpoint control --> apply to robot --> map visualisation
# -----------------------------------------------------------------------------
        pose = robot.get_pose(state)
        state.pose = pose
        scan_data = robot.get_scan(state, pose)
        curr_pts = icp_points(pose, scan_data, state.settings["lidar"])
        state.icp_prev_pts, state.icp_prev_pose = curr_pts, pose
        icp_pose, rmse, n_pts, tf_pts = icp_match_step(state.icp_prev_pts, curr_pts, state.icp_prev_pose)
        pose = fuse_icp_pose(state.settings, pose, icp_pose, rmse, n_pts)
        state.pose = pose
        update_ogm(state.ogm, scan_data, pose)
        determine_navigation_path(state)
        setpoint = compute_setpoint(state.ctrl, state.path, pose)

        new_pose = robot.apply_setpoint(state, pose, setpoint)
        state.pose = new_pose
        state.step += 1
# -----------------------------------------------------------------------------
# Visualisation and Logging
# -----------------------------------------------------------------------------
        render(state.viz, state.world, state.ogm, pose, scan_data, state.goal, state.step, state.path, state.entrance, state.icp_prev_pts, curr_pts, tf_pts, state.astar_pts, state.frontier_goal, state.frontier_candidates)

        with state.logger["pose"].open("a", newline="") as handle:
            csv.writer(handle).writerow([state.step, new_pose["x"], new_pose["y"], math.degrees(new_pose["theta"]), state.settings["app"]["mode"]])

        nav_mode = state.settings.get("navigation", {}).get("mode", "KNOWN")
        if state.frontier_goal:
            fgx, fgy = cell_center(state.frontier_goal, state.world["cell_size_m"])
            fg_dist = math.hypot(fgx - new_pose["x"], fgy - new_pose["y"])
        else:
            fgx = fgy = fg_dist = float("nan")
        frontier_cells = ";".join(f"{cell[0]}:{cell[1]}" for cell in state.frontier_candidates) if state.frontier_candidates else ""
        path_length = len(state.path)
        if state.path:
            path_first_x, path_first_y = state.path[0]
        else:
            path_first_x = path_first_y = float("nan")

        diag_icp_x = diag_icp_y = diag_icp_theta = float("nan")
        diag_rmse = float("nan")
        diag_pts = 0
        diag_icp_x = icp_pose["x"]
        diag_icp_y = icp_pose["y"]
        diag_icp_theta = math.degrees(icp_pose["theta"])
        diag_rmse = rmse if rmse is not None else float("nan")
        diag_pts = n_pts

        with state.logger["diag"].open("a", newline="") as handle:
            csv.writer(handle).writerow(
                [
                    state.step, nav_mode, new_pose["x"], new_pose["y"], math.degrees(new_pose["theta"]), fgx, fgy, fg_dist,
                    f"{state.frontier_goal[0]}:{state.frontier_goal[1]}" if state.frontier_goal else "", len(state.frontier_candidates),
                    frontier_cells, path_length, path_first_x, path_first_y, diag_icp_x, diag_icp_y, diag_icp_theta, diag_rmse, diag_pts,
                ]
            )
        row = [state.step]
        for angle, distance in zip(scan_data["angles"], scan_data["ranges"]):
            row.extend([math.degrees(angle), distance])

        with state.logger["lidar"].open("a", newline="") as handle:
            csv.writer(handle).writerow(row)

        icp_info = f" | icp_pose=({icp_pose['x']:.3f},{icp_pose['y']:.3f},{math.degrees(icp_pose['theta']):.1f}°)"
        
        log.info("Step %05d | Maze World = %s | pose=(%.2f,%.2f,%.1f°)%s | setpoint=(%.2f,%.2f,%.1f°)", state.step, state.settings.get("navigation", {}).get("mode", "KNOWN").upper(), new_pose["x"], new_pose["y"], math.degrees(new_pose["theta"]), icp_info, setpoint["x"], setpoint["y"], math.degrees(setpoint["theta"])) 

# -----------------------------------------------------------------------------
# Stopping condition
# -----------------------------------------------------------------------------
        goal_x, goal_y = cell_center(state.goal["cell"], state.world["cell_size_m"])
        if math.hypot(goal_x - pose["x"], goal_y - pose["y"]) <= state.settings["app"]["arrival_tolerance_m"]:
            print("Simulation complete: Robot reached the goal.")
            log.info("Reached goal; stopping.")
            plt.show(block=True)
            break

    print("Done.")
    plt.close("all")

# -----------------------------------------------------------------------------
# WinterHack 2025: Candidate Selection Challenge
# The following function is to be completed by candidates as part of the challenge.
# Candidates only modify the code within the designated section. Candidates should not
# change the function signature, docstring, or any code outside the designated section.
# -----------------------------------------------------------------------------

def determine_frontier_path(state: SimulationState) -> None:
    """
    Determines and sets the frontier path for robot navigation in an unknown environment.
    This function identifies the next frontier cell to explore and plans a path to it. If the current
    frontier goal matches the ultimate goal cell, it returns that cell. Otherwise, it detects new
    frontiers and selects the most promising one based on various criteria including heading alignment,
    forward progress towards the goal, and distances.
    
    Args:
        state (SimulationState): The current simulation state containing robot pose, world information,
                                goals, and other navigation parameters.
    Returns:
         None. Modifies the state in place by setting the `frontier_goal` and `path` attributes.
         `frontier_goal` is a cell representing the chosen frontier to explore.
         `path` is a list of cells representing the plan to reach the `frontier_goal`.

    The function is expected to perform the following key steps:
    1. Checks if current frontier matches the overall goal.
    2. If not, detects new frontiers and their distances 
    3. Select a frontier based on:
       - Heading alignment with robot's current orientation
       - Forward progress towards goal
       - Distance from robot
       - Proximity to ultimate goal
    3. Plans a path to the selected frontier. 
    4. Update:
        - state.frontier_goal with selected frontier
        - state.path with planned path to the selected frontier
    """

    if state.frontier_goal == state.goal["cell"]:
        print("Found matched frontier to goal")
        return state.frontier_goal
    
    else:
        frontiers, distances = detect_frontiers(state)
        goal_cell = state.goal["cell"]
    
        state.frontier_candidates = frontiers
        state.frontier_distances = distances
          #-----START: To be completed by candidate-----
        import math

        def _wrap(a: float) -> float:
            return (a + math.pi) % (2 * math.pi) - math.pi

        def _los_clear_to_goal_ogm(state: SimulationState) -> bool:

            ogm = state.ogm
            grid = ogm["grid"]
            res  = ogm["res"]
            minx = ogm["minx"]; miny = ogm["miny"]
            occ_th = ogm["cfg"].get("prob_occ_min", 0.65)
            h, w = grid.shape

            px, py = state.pose["x"], state.pose["y"]
            gx, gy = cell_center(state.goal["cell"], state.world["cell_size_m"])
            dx, dy = gx - px, gy - py
            L = math.hypot(dx, dy)
            if L <= 1e-9:
                return True

            steps = max(2, int(L / max(1e-6, 0.5 * res)))
            for i in range(1, steps + 1):
                t = i / steps
                x = px + dx * t
                y = py + dy * t
                ix = int((x - minx) / res)
                iy = int((y - miny) / res)
                if not (0 <= ix < w and 0 <= iy < h):
                    return False
                p = 1.0 / (1.0 + math.exp(-float(grid[iy, ix])))
                if p >= occ_th:
                    return False
            return True

        start_cell = pose_to_cell(state.world, state.pose)
        cell_size  = state.world["cell_size_m"]
        step_scale = max(cell_size, 1e-6)
        robot_x, robot_y = state.pose["x"], state.pose["y"]
        robot_heading    = state.pose["theta"]
        goal_x, goal_y   = cell_center(goal_cell, cell_size)

        best_frontier_cell = None


        prev_target = state.frontier_goal
        if prev_target in frontiers:
            px_f, py_f = cell_center(prev_target, cell_size)
            if math.hypot(px_f - robot_x, py_f - robot_y) > 0.35 * cell_size:
                best_frontier_cell = prev_target


        if best_frontier_cell is None and _los_clear_to_goal_ogm(state) and goal_cell != start_cell:
            best_frontier_cell = goal_cell

        if best_frontier_cell is None:

            if not hasattr(state, "_frontier_visits"):
                state._frontier_visits = {}
            visits = state._frontier_visits
            step_now = getattr(state, "step", 0)
            cooldown = 8  # N 步内重复访问 → 惩罚

            # 前向锥参数
            cone_primary_deg = 60.0
            cone_relax_deg   = 90.0
            cone_primary_rad = math.radians(cone_primary_deg)
            cone_relax_rad   = math.radians(cone_relax_deg)

            forward_primary = []  # 60°
            forward_relax   = []  # 90°
            any_dir         = []  # 全部兜底

            for cell in frontiers:
                if cell == start_cell:
                    continue
                bfs_steps = distances.get(cell)
                if bfs_steps is None:
                    continue

                fx, fy = cell_center(cell, cell_size)
                to_goal      = math.hypot(goal_x - fx,  goal_y - fy) / step_scale
                to_frontier  = math.hypot(fx - robot_x, fy - robot_y) / step_scale


                desired_head   = math.atan2(fy - robot_y, fx - robot_x)
                delta_heading  = abs(_wrap(desired_head - robot_heading))  # 弧度
                heading_delta  = delta_heading / math.pi                   # 归一化 0..1


                goal_heading    = math.atan2(goal_y - robot_y, goal_x - robot_x)
                goal_alignment  = abs(_wrap(desired_head - goal_heading)) / math.pi


                revisit_penalty = 0.0
                last_visit = visits.get(cell)
                if last_visit is not None:
                    age = step_now - last_visit
                    if age <= cooldown:
                        revisit_penalty = 5.0 * (1.0 - age / max(1, cooldown))
                score = (
                    0.70 * to_goal +
                    0.30 * float(bfs_steps) +
                    0.20 * goal_alignment +
                    0.15 * heading_delta +
                    0.10 * to_frontier +
                    revisit_penalty
                )

                item = (score, cell)
            
                if delta_heading <= cone_primary_rad:
                    forward_primary.append(item)
                elif delta_heading <= cone_relax_rad:
                    forward_relax.append(item)
                else:
                    any_dir.append(item)

            if forward_primary:
                forward_primary.sort(key=lambda it: (it[0], it[1]))
                best_frontier_cell = forward_primary[0][1]
            elif forward_relax:
                forward_relax.sort(key=lambda it: (it[0], it[1]))
                best_frontier_cell = forward_relax[0][1]
            elif any_dir:
                any_dir.sort(key=lambda it: (it[0], it[1]))
                best_frontier_cell = any_dir[0][1]
            else:
                best_frontier_cell = goal_cell

            visits[best_frontier_cell] = step_now
            state._frontier_visits = visits
        #-----END: To be completed by candidate-----

    state.frontier_goal = best_frontier_cell
    start_cell = pose_to_cell(state.world, state.pose)
    state.path = plan_unknown_world(state, start_cell, state.frontier_goal)
    return


def detect_frontiers(state: SimulationState) -> Tuple[List[Cell], Dict[Cell, int]]:
    """
    Detect frontier cells in an occupancy grid map using a breadth-first search from the robot pose.
    Parameters
    ----------
    state : SimulationState
        The simulation state object providing the world and map information required for frontier
        detection. Expected fields and structure:
          - state.settings: a dict; navigation mode is read from
            state.settings.get("navigation", {}).get("mode", "KNOWN"). Mode must be the string
            "UNKNOWN" (case-insensitive) for frontier detection to run; otherwise the function
            returns ([], {}).
          - state.ogm: a dict describing the occupancy grid map with keys:
              - "grid": 2D numpy array (float) of log-odds or similar values. The code converts this to
                probabilities using the logistic/sigmoid function: p = 1 / (1 + exp(-grid)).
              - "cfg": a dict of optional configuration thresholds:
                  - "prob_free_max" (float, default 0.35) — cells with p <= prob_free_max are treated as free.
                  - "prob_occ_min"  (float, default 0.65) — cells with p >= prob_occ_min are treated as occupied.
              - "minx", "miny" (float) — origin of the occupancy grid in world coordinates.
              - "res" (float) — grid resolution (meters per grid cell).
          - state.world: a dict with world/grid parameters:
              - "cell_size_m" (float) — cell size used by pose_to_cell / cell_center.
              - either "grid_size" (int) or "size_m" (float). If "grid_size" not present, an integer grid
                size is computed as round(size_m / cell_size_m). grid_size must be > 0.
          - state.pose: robot pose used as the BFS start, converted to a starting grid cell using
            pose_to_cell(state.world, state.pose).
    Returns
    -------
    Tuple[List[Cell], Dict[Cell, int]]
        - frontier_cells: list of Cell (tuples of ints, e.g. (cx, cy)) that are reachable free cells
          adjacent (4-connected) to at least one "unknown" cell. The list is sorted by descending
          distance (farthest reachable first) and then by the cell coordinates as a tie-breaker.
        - frontier_distances: dict mapping each returned frontier cell to its integer Manhattan-style
          distance (number of 4-connected steps) from the start cell discovered by the BFS.
    Behavior and details
    --------------------
    - Early exits:
        - If navigation mode is not "UNKNOWN" (after uppercasing), returns ([], {}).
        - If state.ogm is missing or ogm["grid"] is empty, returns ([], {}).
        - If grid_size <= 0, returns ([], {}).
        - If the start cell (pose_to_cell(state.world, state.pose)) classifies as "occupied",
          returns ([], {}).
    - Cell classification:
        - The inner classification converts a cell index to world coordinates using cell_center(cell, cell_size),
          converts to occupancy grid indices (ix, iy) with (wx - minx)/res and (wy - miny)/res,
          and returns:
            - "occupied" if (ix, iy) is out of the occupancy-grid bounds or p >= prob_occ_min
            - "free"     if p <= prob_free_max
            - "unknown"  otherwise (probability between the free and occupied thresholds)
        - Occupancy probabilities are obtained with a sigmoid applied to the raw grid values.
    - Search and frontier definition:
        - Performs a BFS (4-connected neighbors) starting from the robot cell, exploring only cells
          classified as "free" and bounded by the provided grid_size.
        - A frontier cell is any reachable free cell that has at least one 4-connected neighbor
          classified as "unknown".
        - Only reachable free cells are considered when forming frontiers; occupied or out-of-bounds
          neighbors block traversal.
    - Output ordering and contents:
        - frontier_cells is sorted by (-distance, cell) so that cells farther from the start appear first.
        - frontier_distances contains distances only for those cells present in frontier_cells.
    Complexity
    ----------
    - Time: O(V) where V is the number of free cells visited by the BFS (bounded by grid_size^2 in worst-case).
      Each visited cell checks up to four neighbors and classification uses constant-time operations (array access).
    - Space: O(V) for the BFS queue and the distances mapping.
    Notes
    -----
    - This function relies on helper functions/constructs not defined here: pose_to_cell(world, pose)
      and cell_center(cell, cell_size). The type alias Cell is assumed to be a 2-tuple of ints.
    - The exact numeric behavior depends on how the occupancy grid (ogm["grid"]) stores values
      (log-odds or other); this function treats those values as inputs to a sigmoid to obtain a
      probability in [0, 1].
    - The thresholds prob_free_max and prob_occ_min are inclusive as implemented (<= free and >= occ).
    """

    from collections import deque

    def classify(cell: Cell) -> str:
        cx, cy = cell
        wx, wy = cell_center(cell, cell_size)
        ix = int((wx - minx) / res)
        iy = int((wy - miny) / res)
        if not (0 <= ix < width and 0 <= iy < height):
            return "occupied"
        p = prob[iy, ix]
        if p >= occ_thresh:
            return "occupied"
        if p <= free_thresh:
            return "free"
        return "unknown"
    
    mode = state.settings.get("navigation", {}).get("mode", "KNOWN").upper()
    if mode != "UNKNOWN":
        return [], {}

    ogm = state.ogm
    if not ogm or ogm["grid"].size == 0:
        return [], {}

    grid = ogm["grid"]
    cfg = ogm["cfg"]
    prob = 1 / (1 + np.exp(-grid))
    free_thresh = cfg.get("prob_free_max", 0.35)
    occ_thresh = cfg.get("prob_occ_min", 0.65)

    cell_size = state.world["cell_size_m"]
    grid_size = state.world.get("grid_size", int(round(state.world["size_m"] / cell_size)))
    if grid_size <= 0:
        return [], {}

    width = grid.shape[1]
    height = grid.shape[0]
    minx = ogm["minx"]
    miny = ogm["miny"]
    res = ogm["res"]

    start_cell = pose_to_cell(state.world, state.pose)
    if classify(start_cell) == "occupied":
        return [], {}

    queue: "deque[Cell]" = deque([start_cell])
    distances: Dict[Cell, int] = {start_cell: 0}

    while queue:
        cx, cy = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < grid_size and 0 <= ny < grid_size):
                continue
            cell = (nx, ny)
            if cell in distances:
                continue
            if classify(cell) != "free":
                continue
            distances[cell] = distances[(cx, cy)] + 1
            queue.append(cell)

    frontier_cells: List[Cell] = []
    for cell, dist in distances.items():
        cx, cy = cell
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (cx + dx, cy + dy)
            if not (0 <= nb[0] < grid_size and 0 <= nb[1] < grid_size):
                continue
            if classify(nb) == "unknown":
                frontier_cells.append(cell)
                break

    if not frontier_cells:
        return [], {}

    frontier_cells.sort(key=lambda cell: (-distances[cell], cell))
    frontier_distances = {cell: distances[cell] for cell in frontier_cells}
    return frontier_cells, frontier_distances


def determine_navigation_path(state: SimulationState) -> None:
    """
    Determines the navigation path to the goal cell based on the current simulation state.
    If the navigation mode is set to "UNKNOWN", computes a path to the frontier using
    `determine_frontier_path`. Otherwise, assumes the world is known and the path to the
    goal cell has already been determined during initialization.
    Args:
        state (SimulationState): The current simulation state containing settings and goal information.
    Returns:
        None
    """

    mode = state.settings.get("navigation", {}).get("mode", "KNOWN").upper()

    if mode == "UNKNOWN":
        determine_frontier_path(state)
        return
    else:
        #------------------------------
        # Known world: path to the goal cell already determined at initialisation
        #------------------------------
        if not state.path:
            determine_goal_path(state)
        return

if __name__ == "__main__":
    main()
