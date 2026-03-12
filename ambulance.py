import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from queue import PriorityQueue, Queue
import random
import time

GRID_SIZE = 20
TRAFFIC_PROB = 0.15
BLOCK_PROB = 0.15

# symbols / cell types
ROAD = 0
TRAFFIC = 1
BLOCK = 2

# overlay values for visualization
BFS_PATH = 3
GREEDY_PATH = 4
ASTAR_PATH = 5
START = 6
GOAL = 7
HOSPITAL = 8

start = (0, 0)
goal = (GRID_SIZE - 1, GRID_SIZE - 1)
hospitals = []

# ------------------------------------
# GRID GENERATION
# ------------------------------------

def generate_grid():
    """Create a random city grid with roads, traffic and blocked cells."""
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):

            r = random.random()

            if r < BLOCK_PROB:
                grid[i][j] = BLOCK

            elif r < BLOCK_PROB + TRAFFIC_PROB:
                grid[i][j] = TRAFFIC

            else:
                grid[i][j] = ROAD

    # ensure start and goal are always roads
    grid[start] = ROAD
    grid[goal] = ROAD

    return grid


def generate_hospitals(grid, num_hospitals: int = 3):
    """Randomly choose a few non-blocked cells to act as hospitals."""
    positions = []
    attempts = 0
    max_attempts = GRID_SIZE * GRID_SIZE * 2

    while len(positions) < num_hospitals and attempts < max_attempts:
        attempts += 1
        r = random.randrange(GRID_SIZE)
        c = random.randrange(GRID_SIZE)

        if grid[r, c] == BLOCK:
            continue

        pos = (r, c)
        if pos in positions or pos == start:
            continue

        positions.append(pos)

    return positions


# ------------------------------------
# HELPER FUNCTIONS
# ------------------------------------

def configure_simulation():
    """Optionally let the user override basic simulation parameters."""
    global GRID_SIZE, TRAFFIC_PROB, BLOCK_PROB, start, goal

    print("Ambulance Route Optimization")
    print("----------------------------")
    print("Press Enter to keep defaults shown in brackets.\n")

    try:
        size_input = input(f"Grid size (default {GRID_SIZE}): ").strip()
        if size_input:
            GRID_SIZE = max(5, int(size_input))

        traffic_input = input(f"Traffic probability 0-1 (default {TRAFFIC_PROB}): ").strip()
        if traffic_input:
            TRAFFIC_PROB = float(traffic_input)

        block_input = input(f"Block probability 0-1 (default {BLOCK_PROB}): ").strip()
        if block_input:
            BLOCK_PROB = float(block_input)

        seed_input = input("Random seed (empty for random): ").strip()
        if seed_input:
            seed = int(seed_input)
            random.seed(seed)
            np.random.seed(seed)
            print(f"Using seed = {seed}")
    except ValueError:
        print("Invalid input detected. Falling back to default configuration.")

    # update start and goal after potential GRID_SIZE change
    start = (0, 0)
    goal = (GRID_SIZE - 1, GRID_SIZE - 1)


def neighbors(node, grid):
    """4-connected neighbors that are not blocked."""
    dirs = [(1,0),(-1,0),(0,1),(0,-1)]

    result = []

    for d in dirs:

        r = node[0] + d[0]
        c = node[1] + d[1]

        if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:

            if grid[r][c] != BLOCK:
                result.append((r,c))

    return result


def heuristic(a,b):
    """Manhattan distance."""
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


def reconstruct(came_from,current):

    path = [current]

    while current in came_from:
        current = came_from[current]
        path.append(current)

    path.reverse()

    return path


# ------------------------------------
# BFS
# ------------------------------------

def bfs(grid):

    q = Queue()
    q.put(start)

    visited = set()
    visited.add(start)

    came = {}

    nodes = 0

    while not q.empty():

        current = q.get()
        nodes += 1

        if current == goal:
            return reconstruct(came,current),nodes

        for n in neighbors(current,grid):

            if n not in visited:
                visited.add(n)
                came[n] = current
                q.put(n)

    return None,nodes


def compute_travel_time(path, grid):
    """Aggregate travel time along a path, considering traffic delays."""
    if not path:
        return None

    total = 0
    # cost to move into each next cell
    for node in path[1:]:
        total += 3 if grid[node] == TRAFFIC else 1
    return total


# ------------------------------------
# GREEDY SEARCH
# ------------------------------------

def greedy(grid):

    pq = PriorityQueue()
    pq.put((0,start))

    visited = set()

    came = {}

    nodes = 0

    while not pq.empty():

        current = pq.get()[1]

        nodes += 1

        if current == goal:
            return reconstruct(came,current),nodes

        visited.add(current)

        for n in neighbors(current,grid):

            if n not in visited:

                h = heuristic(n,goal)

                pq.put((h,n))

                if n not in came:
                    came[n] = current

    return None,nodes


# ------------------------------------
# A STAR
# ------------------------------------

def astar(grid):

    pq = PriorityQueue()
    pq.put((0,start))

    came = {}

    g = {start:0}

    nodes = 0

    while not pq.empty():

        current = pq.get()[1]
        nodes += 1

        if current == goal:
            return reconstruct(came,current),nodes

        for n in neighbors(current,grid):

            cost = 3 if grid[n]==TRAFFIC else 1

            new_g = g[current] + cost

            if n not in g or new_g < g[n]:

                g[n] = new_g

                f = new_g + heuristic(n,goal)

                pq.put((f,n))

                came[n] = current

    return None,nodes


# ------------------------------------
# VISUALIZATION
# ------------------------------------

def draw(grid,results):

    img = np.copy(grid)

    for name, path, *_ in results:
        if not path:
            continue
        if name == "BFS":
            value = BFS_PATH
        elif name == "Greedy":
            value = GREEDY_PATH
        elif name == "A*":
            value = ASTAR_PATH
        else:
            value = BFS_PATH

        for node in path:
            img[node] = value

    # mark hospitals (except the chosen goal, which gets its own color)
    for hr, hc in hospitals:
        if (hr, hc) != goal:
            img[hr, hc] = HOSPITAL

    img[start] = START
    img[goal] = GOAL

    # discrete color map for better interpretation
    cmap = colors.ListedColormap(
        [
            "#ffffff",  # 0 road
            "#ffcc66",  # 1 traffic
            "#333333",  # 2 block
            "#1f77b4",  # 3 BFS path
            "#2ca02c",  # 4 Greedy path
            "#d62728",  # 5 A* path
            "#ffff00",  # 6 start
            "#800080",  # 7 goal (chosen hospital)
            "#00ffff",  # 8 other hospitals
        ]
    )
    bounds = [i - 0.5 for i in range(0, 10)]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(6,6))
    plt.imshow(img,cmap=cmap,norm=norm)
    plt.title("Ambulance Route Optimization")
    plt.xticks([])
    plt.yticks([])

    proxy_patches = [
        plt.Line2D([0], [0], color="#1f77b4", lw=4),
        plt.Line2D([0], [0], color="#2ca02c", lw=4),
        plt.Line2D([0], [0], color="#d62728", lw=4),
        plt.Line2D([0], [0], color="#ffcc66", lw=4),
        plt.Line2D([0], [0], color="#333333", lw=4),
        plt.Line2D([0], [0], color="#00ffff", lw=4),
    ]
    plt.legend(
        proxy_patches,
        ["BFS path", "Greedy path", "A* path", "Traffic", "Blocked", "Hospital"],
        loc="upper right",
        bbox_to_anchor=(1.4, 1.0),
    )

    plt.tight_layout()
    plt.show()


# ------------------------------------
# MAIN
# ------------------------------------

def run_algorithms(grid):
    """Run all algorithms and collect results."""
    algorithms = [
        ("BFS", bfs),
        ("Greedy", greedy),
        ("A*", astar),
    ]

    results = []

    for name, func in algorithms:
        print(f"Running {name}...")
        t0 = time.time()
        path, nodes = func(grid)
        elapsed = time.time() - t0
        travel_time = compute_travel_time(path, grid)
        results.append((name, path, nodes, elapsed, travel_time))

    return results


def print_results(results):
    print("\nRESULTS\n")
    best_algo = None
    best_travel = None

    for name, path, nodes, elapsed, travel_time in results:
        print(name)
        if path:
            print("  Path length:", len(path))
            print("  Nodes explored:", nodes)
            print("  Runtime (s):", round(elapsed, 4))
            if travel_time is not None:
                print("  Travel time (cost):", travel_time)
                if best_travel is None or travel_time < best_travel:
                    best_travel = travel_time
                    best_algo = name
        else:
            print("  No path found.")
        print()

    if best_algo is not None:
        print(f"Most efficient route (by travel time): {best_algo} (cost = {best_travel})")
    else:
        print("No algorithm found a route from start to goal.")


def print_dataset_sample(grid, goal, sample_size: int = 10):
    """Print a small sample of the underlying dataset (cost + heuristic)."""
    rows = []
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            cell_type = grid[r, c]
            if cell_type == BLOCK:
                continue  # skip completely blocked cells
            cost = 3 if cell_type == TRAFFIC else 1
            h = heuristic((r, c), goal)
            rows.append(((r, c), cell_type, cost, h))

    if not rows:
        print("Dataset is empty (all cells blocked).")
        return

    random.shuffle(rows)
    print("\nSample of dataset (row, col, type, cost, heuristic to goal):")
    print("  (r, c)   TYPE     cost  h")
    for (r, c), t, cost, h in rows[:sample_size]:
        cell_name = "ROAD" if t == ROAD else "TRAFFIC"
        print(f"  ({r:2d}, {c:2d})  {cell_name:7s}  {cost:4d}  {h}")


def find_best_hospital_for_start(grid, start_pos, hospital_list):
    """Pick the hospital with the lowest A* travel time from a given start."""
    global start, goal

    best_h = None
    best_cost = None

    original_start = start
    original_goal = goal

    start = start_pos

    for h in hospital_list:
        if grid[h] == BLOCK:
            continue

        goal = h
        grid[start] = ROAD
        grid[goal] = ROAD

        path, _ = astar(grid)
        travel_time = compute_travel_time(path, grid)

        if path and travel_time is not None:
            if best_cost is None or travel_time < best_cost:
                best_cost = travel_time
                best_h = h

    # restore (not strictly necessary here, but safer)
    start = original_start
    goal = original_goal

    return best_h, best_cost


def main():
    global start, goal, hospitals

    configure_simulation()
    grid = generate_grid()
    hospitals = generate_hospitals(grid)

    print(f"\nGenerated grid of size {GRID_SIZE} x {GRID_SIZE}.")
    print(f"Hospitals (row, col): {hospitals}")
    print("You can run in two modes:")
    print("  1) Manual destination (you choose exact goal).")
    print("  2) Nearest hospital (system picks best hospital by A* travel time).")
    mode_input = input("Choose mode [1/2, default 1]: ").strip()
    use_hospital_mode = mode_input == "2"

    print("\nEnter coordinates as: row col  (0-based indices).")
    print("Type 'n' to create a new random grid (with new hospitals), or 'q' to quit.\n")

    while True:
        user_input = input("Start (row col) or 'n' new grid / 'q' quit: ").strip().lower()

        if user_input == "q":
            break
        if user_input == "n":
            grid = generate_grid()
            hospitals = generate_hospitals(grid)
            print(f"\nNew grid generated. Hospitals: {hospitals}")
            continue

        try:
            sr, sc = map(int, user_input.split())
        except ValueError:
            print("Please enter two integers like '0 0', or 'n' / 'q'.")
            continue

        # bounds check for start
        if not (0 <= sr < GRID_SIZE and 0 <= sc < GRID_SIZE):
            print(f"Coordinates must be between 0 and {GRID_SIZE - 1}.")
            continue

        if grid[sr, sc] == BLOCK:
            print("Start is on a blocked cell. Choose different coordinates.")
            continue

        if use_hospital_mode:
            # automatically pick the best hospital as goal
            best_h, best_cost = find_best_hospital_for_start(grid, (sr, sc), hospitals)
            if best_h is None:
                print("No reachable hospital from this start. Try another start or new grid.")
                continue

            print(f"Chosen hospital {best_h} with estimated travel cost {best_cost}.")
            gr, gc = best_h
        else:
            goal_input = input("Goal (row col): ").strip().lower()
            if goal_input == "q":
                break
            if goal_input == "n":
                grid = generate_grid()
                hospitals = generate_hospitals(grid)
                print(f"\nNew grid generated. Hospitals: {hospitals}")
                continue

            try:
                gr, gc = map(int, goal_input.split())
            except ValueError:
                print("Please enter two integers like '10 10'.")
                continue

            # bounds check for goal
            if not (0 <= gr < GRID_SIZE and 0 <= gc < GRID_SIZE):
                print(f"Coordinates must be between 0 and {GRID_SIZE - 1}.")
                continue

            if grid[gr, gc] == BLOCK:
                print("Goal is on a blocked cell. Choose different coordinates.")
                continue

        # update global start/goal used by the algorithms and drawing
        start = (sr, sc)
        goal = (gr, gc)

        # ensure walkable
        grid[start] = ROAD
        grid[goal] = ROAD

        # show a small dataset sample for this goal
        print_dataset_sample(grid, goal)

        # run and show algorithms
        results = run_algorithms(grid)
        print_results(results)
        draw(grid, results)


if __name__ == "__main__":
    main()
