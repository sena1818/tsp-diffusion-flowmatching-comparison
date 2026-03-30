"""Generate DIFUSCO-style TSP instances in text format."""

import argparse
import numpy as np
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm


def compute_distance_matrix(coords):
    """Compute pairwise Euclidean distance matrix, shape (N, N)."""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def solve_tsp_elkai(coords):
    """
    Solve TSP using elkai (LKH-3 bindings).
    coords: (N, 2) numpy array in [0, 1]
    Returns: 0-indexed tour list of length N (no return node)
    """
    import elkai
    SCALE = 1_000_000
    dist_matrix = compute_distance_matrix(coords)
    int_dist = (dist_matrix * SCALE).astype(int).tolist()
    cities = elkai.DistanceMatrix(int_dist)
    tour = cities.solve_tsp()  # returns [0, ..., 0]
    return tour[:-1]  # drop the trailing return node


def solve_tsp_python_tsp(coords, method='lk'):
    """
    Solve TSP using python-tsp (fallback option).
    method: 'dp' (exact, N<=20), 'lk' (Lin-Kernighan), 'sa' (simulated annealing)
    """
    dist_matrix = compute_distance_matrix(coords)
    if method == 'dp':
        from python_tsp.exact import solve_tsp_dynamic_programming
        permutation, _ = solve_tsp_dynamic_programming(dist_matrix)
    elif method == 'lk':
        from python_tsp.heuristics import solve_tsp_lin_kernighan
        permutation, _ = solve_tsp_lin_kernighan(dist_matrix)
    elif method == 'sa':
        from python_tsp.heuristics import solve_tsp_simulated_annealing
        permutation, _ = solve_tsp_simulated_annealing(dist_matrix)
    else:
        raise ValueError(f"Unknown method: {method}")
    return list(permutation)


def format_instance(coords, tour):
    """
    Format one instance as a DIFUSCO-compatible single-line string.
    coords: (N, 2), tour: 0-indexed list of length N
    Output: "x1 y1 x2 y2 ... xN yN output t1 t2 ... tN t1"
    """
    coord_strs = []
    for x, y in coords:
        coord_strs.append(f"{x:.8f}")
        coord_strs.append(f"{y:.8f}")

    # Convert to 1-indexed and append the return edge.
    tour_1indexed = [str(t + 1) for t in tour]
    tour_1indexed.append(str(tour[0] + 1))

    return " ".join(coord_strs) + " output " + " ".join(tour_1indexed)


def generate_single_instance(idx, num_nodes, solver, seed_base):
    """Generate one TSP instance and return the formatted string."""
    rng = np.random.RandomState(seed_base + idx)
    coords = rng.uniform(0, 1, size=(num_nodes, 2))

    if solver == 'elkai':
        tour = solve_tsp_elkai(coords)
    elif solver == 'python-tsp-lk':
        tour = solve_tsp_python_tsp(coords, method='lk')
    elif solver == 'python-tsp-dp':
        tour = solve_tsp_python_tsp(coords, method='dp')
    elif solver == 'python-tsp-sa':
        tour = solve_tsp_python_tsp(coords, method='sa')
    else:
        raise ValueError(f"Unknown solver: {solver}")

    return format_instance(coords, tour)


def validate_line(line):
    """Validate one data line. Returns (is_valid, error_message)."""
    parts = line.strip().split(" output ")
    if len(parts) != 2:
        return False, "missing 'output' separator"

    coords_vals = list(map(float, parts[0].split()))
    tour_vals = list(map(int, parts[1].split()))
    N = len(coords_vals) // 2

    if len(coords_vals) % 2 != 0:
        return False, "odd number of coordinate values"
    if len(tour_vals) != N + 1:
        return False, f"tour length {len(tour_vals)}, expected {N + 1}"
    if tour_vals[0] != tour_vals[-1]:
        return False, "tour doesn't return to start"
    if set(tour_vals[:-1]) != set(range(1, N + 1)):
        return False, "tour doesn't visit all cities exactly once"
    return True, ""


def main():
    parser = argparse.ArgumentParser(description='Generate TSP dataset (DIFUSCO format)')
    parser.add_argument('--num_nodes', type=int, default=20)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--solver', type=str, default='elkai',
                        choices=['elkai', 'python-tsp-lk', 'python-tsp-dp', 'python-tsp-sa'])
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    print(f"Generating {args.num_samples} TSP-{args.num_nodes} instances "
          f"using {args.solver} solver...")

    worker_fn = partial(
        generate_single_instance,
        num_nodes=args.num_nodes,
        solver=args.solver,
        seed_base=args.seed,
    )

    results = []
    with Pool(processes=args.num_workers) as pool:
        for line in tqdm(pool.imap(worker_fn, range(args.num_samples)),
                         total=args.num_samples, desc=f"TSP-{args.num_nodes}"):
            results.append(line)

    with open(args.output_file, 'w') as f:
        for line in results:
            f.write(line + '\n')

    print(f"Saved {len(results)} instances to {args.output_file}")

    # Validate a small prefix before finishing.
    errors = 0
    for i, line in enumerate(results[:5]):
        ok, msg = validate_line(line)
        if not ok:
            print(f"  [FAIL] Line {i}: {msg}")
            errors += 1
    if errors == 0:
        print("Validation: first 5 lines OK")


if __name__ == '__main__':
    main()
