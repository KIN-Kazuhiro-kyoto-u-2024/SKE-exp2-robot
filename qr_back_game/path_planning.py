import heapq
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
from .geometry import cell_of_xy, xy_of_cell

Cell = Tuple[int, int]

# Grid-aligned A*: only horizontal/vertical moves are allowed.
# The returned trajectory is therefore a sequence of adjacent cell centers.
NEIGHBORS_4 = [
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
]


def in_bounds(c: Cell, grid_w: int, grid_h: int) -> bool:
    return 0 <= c[0] < grid_w and 0 <= c[1] < grid_h


def heuristic(a: Cell, b: Cell) -> float:
    # Manhattan distance is admissible for 4-connected grid motion.
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


def blocked_cells_around(center_xy: np.ndarray, radius: float, cfg) -> set:
    cx, cy = cell_of_xy(float(center_xy[0]), float(center_xy[1]), cfg.cell_size, cfg.grid_w, cfg.grid_h)
    r_cells = int(math.ceil(radius / cfg.cell_size))
    blocked = set()
    for ix in range(cx - r_cells, cx + r_cells + 1):
        for iy in range(cy - r_cells, cy + r_cells + 1):
            c = (ix, iy)
            if not in_bounds(c, cfg.grid_w, cfg.grid_h):
                continue
            xy = xy_of_cell(c, cfg.cell_size)
            if float(np.linalg.norm(xy - center_xy)) <= radius + cfg.cell_size * 0.5:
                blocked.add(c)
    return blocked


def unblock_escape_neighbors(start: Cell, avoid_xy: np.ndarray, blocked: set, cfg) -> None:
    """Allow one-step escape moves when the start is inside a blocked region.

    The opponent neighborhood is treated as an A* obstacle, but if the attacker
    is already too close, blocking every neighbor makes A* return only [start].
    This opens neighbors that increase distance from the opponent.
    """
    start_xy = xy_of_cell(start, cfg.cell_size)
    start_dist = float(np.linalg.norm(start_xy - avoid_xy))
    for dx, dy, _ in NEIGHBORS_4:
        nxt = (start[0] + dx, start[1] + dy)
        if not in_bounds(nxt, cfg.grid_w, cfg.grid_h):
            continue
        nxt_xy = xy_of_cell(nxt, cfg.cell_size)
        nxt_dist = float(np.linalg.norm(nxt_xy - avoid_xy))
        if nxt_dist > start_dist:
            blocked.discard(nxt)


def rear_goal_cells(target_pose: np.ndarray, cfg, n_rings: int = 7) -> List[Cell]:
    """Candidate goal cells inside the target rear fan, near cells first."""
    goals: List[Cell] = []
    rear = float(target_pose[2]) + math.pi
    for r in range(1, n_rings + 1):
        dist = r * cfg.cell_size
        # Sampling is only for choosing grid cells; final path stays grid-aligned.
        for da in np.linspace(-cfg.rear_sector_half_angle * 0.85, cfg.rear_sector_half_angle * 0.85, 9):
            x = float(target_pose[0]) + dist * math.cos(rear + float(da))
            y = float(target_pose[1]) + dist * math.sin(rear + float(da))
            c = cell_of_xy(x, y, cfg.cell_size, cfg.grid_w, cfg.grid_h)
            if c not in goals:
                goals.append(c)
    return goals


def _reconstruct(came_from: Dict[Cell, Optional[Cell]], current: Cell) -> List[Cell]:
    path = []
    c: Optional[Cell] = current
    while c is not None:
        path.append(c)
        c = came_from[c]
    return list(reversed(path))


def astar(
    start: Cell,
    goals: List[Cell],
    cfg,
    blocked: Optional[set] = None,
    preferred_y: Optional[float] = None,
) -> List[Cell]:
    
    """4-connected A* on the 10 cm grid.

    preferred_y is a soft tie-breaker to break symmetric head-on behavior. It does
    not introduce continuous waypoints; all returned cells remain grid-adjacent.
    """

    if blocked is None:
        blocked = set()
    valid_goals = [g for g in goals if in_bounds(g, cfg.grid_w, cfg.grid_h) and g not in blocked]
    if not valid_goals:
        valid_goals = [g for g in goals if in_bounds(g, cfg.grid_w, cfg.grid_h)]
    if not valid_goals:
        return [start]

    goal_set = set(valid_goals)
    frontier = [(0.0, start)]
    came_from: Dict[Cell, Optional[Cell]] = {start: None}
    cost_so_far: Dict[Cell, float] = {start: 0.0}
    best_cell = start
    best_h = min(heuristic(start, g) for g in valid_goals)

    while frontier:
        _, current = heapq.heappop(frontier)
        current_h = min(heuristic(current, g) for g in valid_goals)
        if current_h < best_h:
            best_h = current_h
            best_cell = current

        if current in goal_set:
            return _reconstruct(came_from, current)

        for dx, dy, step_cost in NEIGHBORS_4:
            nxt = (current[0] + dx, current[1] + dy)
            if not in_bounds(nxt, cfg.grid_w, cfg.grid_h) or nxt in blocked:
                continue

            # A tiny lane preference is used only as a cost tie-breaker.
            lane_cost = 0.0
            if preferred_y is not None:
                nxt_y = xy_of_cell(nxt, cfg.cell_size)[1]
                lane_cost = 0.015 * abs(float(nxt_y) - float(preferred_y)) / max(cfg.field_h, 1e-6)

            new_cost = cost_so_far[current] + step_cost + lane_cost
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                h = min(heuristic(nxt, g) for g in valid_goals)
                heapq.heappush(frontier, (new_cost + h, nxt))
                came_from[nxt] = current
    return _reconstruct(came_from, best_cell)


def simplify_grid_path(cells: List[Cell]) -> List[Cell]:
    """Keep all turns and segment ends, while preserving grid-cell centers.

    For pure cell-by-cell visualization/control, use the unsimplified cells. This
    helper is currently not used by the controller, but is useful for experiments.
    """
    if len(cells) <= 2:
        return cells[:]
    out = [cells[0]]
    prev_dir = (cells[1][0] - cells[0][0], cells[1][1] - cells[0][1])
    for i in range(1, len(cells) - 1):
        d = (cells[i + 1][0] - cells[i][0], cells[i + 1][1] - cells[i][1])
        if d != prev_dir:
            out.append(cells[i])
            prev_dir = d
    out.append(cells[-1])
    return out

# A* アルゴリズムを用いて，グリッドベースで自分（attacker）から相手（target）への最短経路を探索
def plan_cells_to_rear(
    attacker_pose: np.ndarray,
    target_pose: np.ndarray,
    cfg,
    avoid_xy: Optional[np.ndarray] = None,
    preferred_y: Optional[float] = None,
) -> List[Cell]:
    
    # 経路の始点（start）と終点（goals）を設定
    # 始点は自分の位置，終点は相手の後方の位置
    start = cell_of_xy(float(attacker_pose[0]), float(attacker_pose[1]), cfg.cell_size, cfg.grid_w, cfg.grid_h)
    goals = rear_goal_cells(target_pose, cfg)

    # 障害物の位置を設定
    blocked = set()
    if avoid_xy is not None:
        blocked |= blocked_cells_around(avoid_xy, 1.5 * cfg.collision_radius, cfg)
        blocked.discard(start)
        unblock_escape_neighbors(start, avoid_xy, blocked, cfg)

    # A* アルゴリズムの実行
    return astar(start, goals, cfg, blocked, preferred_y=preferred_y)

# plan_cells_to_rear で得られたグリッドベースの経路を xy 座標における表現に変換
def plan_path_to_rear(
    attacker_pose: np.ndarray,
    target_pose: np.ndarray,
    cfg,
    avoid_xy: Optional[np.ndarray] = None,
    preferred_y: Optional[float] = None,
) -> List[np.ndarray]:
    cells = plan_cells_to_rear(attacker_pose, target_pose, cfg, avoid_xy=avoid_xy, preferred_y=preferred_y)
    return [xy_of_cell(c, cfg.cell_size) for c in cells]


def path_is_grid_aligned(path: List[np.ndarray], cfg, tol: float = 1e-8) -> bool:
    """Validation helper: every waypoint is a cell center and moves are 4-neighbor."""
    if not path:
        return True
    cells = [cell_of_xy(float(p[0]), float(p[1]), cfg.cell_size, cfg.grid_w, cfg.grid_h) for p in path]
    centers = [xy_of_cell(c, cfg.cell_size) for c in cells]
    if any(float(np.linalg.norm(p - q)) > tol for p, q in zip(path, centers)):
        return False
    for a, b in zip(cells[:-1], cells[1:]):
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            return False
    return True
