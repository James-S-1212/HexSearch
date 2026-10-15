import math 
import time
from scipy.stats import multinomial
from scipy import integrate
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np


# x is top, y is right...
class invalid_hex_error(Exception):
    pass


def r_uniform(x, N):
    """
    Uniform marginal distribution function r, resulting from uniform prior.
    """

    num_points = (N+1)*(N+2)/2
    return np.full(x.shape[0], 1/num_points)


def get_cell_params(cell, N, r=r_uniform):
    """
    Compute the theta estimate for a given trinomial cell which minimises the SMML codelength. 
    Formula is as follows:
    theta_j_i = (sum_{x in cell}(x_i*r(x)))/N*(sum_{x in cell}r(x))
    
    Input: 
         cell: Array of datapoints. 
        N: Number of trials. 
        r_x: Marginal probability of a datapoint x (same for all x here since uniform assumed). 
    Output: 
        theta_j: Theta estimate for the cell \theta^*_j
        r_cell: Marginal probability of the cell, equivalent to q(\theta^*_j)
    """

    r_vals = r(cell, N)
    assert r_vals.shape == (cell.shape[0],) # Ensure r is vectorized

    numerator = np.sum(r_vals[:, None]*cell, axis=0)
    r_cell = np.sum(r_vals)
    theta_j = numerator/(N*r_cell)
    theta_j = theta_j / theta_j.sum() # Normalise arithmetic errors from computational anomalies
    return theta_j, r_cell, r_vals


def SMML(partition, N, r=r_uniform, convert_to_bits=False):
    """
    Measured in nats. Can convert to bits with log2. This computes the expected SMML codelength of a given partition (expectation is with respect to r). 
    Input: 
        partition: Array of arrays representing the dataspace partition for a model m. 
        N: Number of trials, where n=(N+1)(N+2)/2
    Output: 
        Expected SMML codelength given by assertion + detail lengths. 
    """
    
    assertion = 0
    detail = 0
    for cell in partition:
        np_cell = np.array(cell)
        theta_j, q_j, r_vals = get_cell_params(np_cell, N, r)
        if q_j == 0:
            continue

        assertion += -q_j*math.log(q_j)
        detail += -np.sum(r_vals*multinomial.logpmf(np_cell, n=N, p=theta_j))
        
    if convert_to_bits:
        return (assertion+detail)/math.log(2)
    else:
        return assertion + detail


def initialise_hex(dx, dy, r1, r2, r3, N):
    """
    Create first hexagon given specifications. This determines the entire lattice. To avoid hex edges cutting through points, the centre coordinates are offset by 
    small irrational values in all directions. Radii must be integers. This way, every edge has a coordinate which is irrational plus some multiple
    of an integer, and thus is always irrational. Since data points have all integer coordinates, no data points will have hex edges cutting through them. 

    Input: 
        x, y: Translation offset for the lattice. Cartesian coordinate pair, representing x and y translation. 
        N: Simplex bound. 
    Output: 
        centre: Simplex coordinates of hexagon centre which generates the lattice. 
    """

    offset_r1 = r1 + math.sqrt(2)/100
    offset_r2 = r2 + math.sqrt(3)/100
    offset_r3 = r3 + math.sqrt(5)/100
    centre = dx, dy, N-dx-dy
    return centre, offset_r1, offset_r2, offset_r3


def valid_hex(r1,r2,r3):
    """
    Input: 
        r1, r2, r3: Hexagon radii in the x, y, and z simplex directions respectively. 
    Output: 
        True if radii represent a 6-sided polygon and False otherwise. 
    """

    return r1<r2+r3 and r2<r1+r3 and r3<r1+r2


def hex_jump(centre, radii, direction):
    """
    Find the centre of a hexagon after a series of jumps, starting from a hexagon centred at "centre". 
    
    Input:
        centre: Centre coordinate of start hexagon. 
        radii: 3-tuple of positive integer radii for the hex lattice. 
        direction: 3-tuple representing vector of hex jumps to make (i.e. (1,-2,4) would be jumping one hexagon in the x direction, two away from the y direction,
            and 4 toward the z direction)
    Output:
        new_centre: Centre coordinate of destination hexagon. 
    
    """

    r1, r2, r3 = radii
    x_old, y_old, z_old = centre
    coeff_x, coeff_y, coeff_z = direction
    
    x_new = x_old + coeff_x*(2*r1)          + coeff_y*(-r1 - r2 + r3)   +  coeff_z*(-r1 + r2 - r3)
    y_new = y_old + coeff_x*(-r1 - r2 + r3) + coeff_y*(2*r2)            + coeff_z*(r1 - r2 - r3)
    z_new = z_old + coeff_x*(-r1 + r2 - r3) + coeff_y*(r1 - r2 - r3)    + coeff_z*(2*r3)
    new_centre = np.array([x_new, y_new, z_new])
    return new_centre


def hex_on_simplex(centre, r1,r2,r3, N):
    """
    Determine whether a given hexagon overlaps with the simplex. 

    Input: 
        centre: Hexagon centre coordinate triple. 
        r1, r2, r3: Hexagon radii. 
        N: Simplex bound. 
    Output: 
        True if the hexagon overlaps with the simplex and False if it is entirely disjoint. 
    """

    x_mid, y_mid, z_mid = centre
    if (x_mid-r1 > N or x_mid+r1 < 0 or # Checking if bottom of hexagon is above simplex, if top is below, and so on. 
        y_mid-r2 > N or y_mid+r2 < 0 or
        z_mid-r3 > N or z_mid+r3 < 0):
        return False
    else:
        return True


def get_hex_interior(centre, x_rad, y_rad, z_rad, N):
    """
    Input: 
        centre: 3-tuple of (x,y,z) coordinates in the simplex. Hexagon centre.
        x_rad: int+offset representing radius of hexagon in x direction. The offset is so no point falls on the boundary. 
        y_rad: int+offset representing radius of hexagon in y direction. The offset is so no point falls on the boundary. 
        z_rad: int+offset representing radius of hexagon in z direction. The offset is so no point falls on the boundary. 
        N: int representing number of points in the simplex.
    Output:
        points: array of all (x,y,z) simplex points inside the hexagon. 
    """

    points=[]
    x_mid, y_mid, z_mid = centre
    # Check if hexagon is entirely off the edge of the simplex (empty) to save time
    if not hex_on_simplex(centre, x_rad, y_rad, z_rad, N): 
        return points 

    x_low = max(math.floor(x_mid-x_rad)+1, 0) # Don't go out of bounds
    x_high = min(math.floor(x_mid+x_rad), N)
    for x in range(x_low, x_high+1):
        y_low = max(math.floor(y_mid-y_rad)+1, math.floor(N-z_mid-z_rad-x)+1, 0) # Since y_mid-y_rad <= y <= y_mid+y_rad, and z_mid-z_rad <= N-x-y <= z_mid+z_rad (since z=N-x-y)
                                   # Which means N-z_mid+z_rad-x >= y >= N-z_mid-z_rad-x 
        y_high = min(math.floor(y_mid+y_rad), math.floor(N-z_mid+z_rad-x), N-x) # Also don't go out of bounds (the last arg checks this)
        for y in range(y_low, y_high+1):
            points.append(np.array([x, y, N-x-y]))
    return points

# r1=n+sqrt(2)/10, r2=n+sqrt(3)/10, r3=sqrt(5)/10. Linearly independent so that no edge falls on a simplex integer lattice point. 
def find_start(centre, radii, N): 
    """
    Navigate to bottom-left hex on simplex.

    Inputs:
        centre: Centre of generating hexagon for lattice. (x,y,z) simplex coordinates. 
        radii: (r1,r2,r3) hexagon radii for lattice. 
        N: Simplex bound. 
    Output: 
        centre: Centre is updated to the (x,y,z) simplex coordinate of the bottom-left-most hexagon which still has some area on the simplex.  
    """
    r1, r2, r3 = radii
    if not valid_hex(r1,r2,r3):
        raise invalid_hex_error("Not a valid hexagon. Radii give shape with fewer than six sides.")
    
    # Need to go down, check hex bottom
    while centre[0]-r1 > 0:
        centre = hex_jump(centre, radii, (-1,0,0)) # Jump to the hexagon 1 step down in x direction
    # Need to go up, check hex top
    while centre[0]+r1 < 0:
        centre = hex_jump(centre, radii, (1,0,0)) # Jump to hexagon 1 up in x direction
    
    # Need to go left 
    while centre[2]+r3 < N:
        centre = hex_jump(centre, radii, (0,0,1)) # Jump to down-left hexagon neighbour
        if centre[0]+r1 < 0: # Need to go up again, hex top drifted below simplex
            centre = hex_jump(centre, radii, (1,0,0))
    # Need to go right
    while centre[2]-r3 > N:
        centre = hex_jump(centre, radii, (0,0,-1)) # Jump to up-right hexagon neighbour
        if centre[0]-r1 > 0: # Need to go down again, hex bottom drifted above simplex bottom
            centre = hex_jump(centre, radii, (-1,0,0))

    # Correct y as well in case of strange edge cases
    while centre[1]-r2 > 0:
        centre = hex_jump(centre, radii, (0,-1,0))
        if centre[0]-r1 > 0:
            centre = hex_jump(centre, radii, (-1,0,0))
        elif centre[0]+r1 < 0:
            centre = hex_jump(centre, radii, (1,0,0))
    while centre[1]+r2 < 0:
        centre = hex_jump(centre, radii, (0,1,0))
        if centre[0]+r1 < 0:
            centre = hex_jump(centre, radii, (1,0,0))
        elif centre[0]-r1 > 0:
            centre = hex_jump(centre, radii, (-1,0,0))
    return centre


def iterate_hex_cover(dx, dy, r1, r2, r3, N):
    """
    Given a generating hexagon, generate the lattice and collect the partition cells. 
    
    Input: 
        dx,dy: Coordinates of generating hexagon. 
        r1,r2,r3: The radii of the generating hexagon.
        N: Number of trials. 
    Output: 
        partition: List of lists representing the cells of the partition induced by the lattice induced by the generating hexagon.  
    """

    centre, r1, r2, r3 = initialise_hex(dx, dy, r1, r2, r3, N)
    if not valid_hex(r1,r2,r3):
            raise invalid_hex_error("Not a valid hex. Radii give shape with fewer than six sides.")
    partition = []

    bottom = find_start(centre, (r1,r2,r3), N)

    while hex_on_simplex(bottom, r1, r2, r3, N): # Slide bottom along the bottom of the simplex so we can go up the x columns
        curr = bottom
        while curr[0]-r1<=N: # Slide all the way up the x column collecting cells
            cell = get_hex_interior(curr, r1, r2, r3, N)
            if len(cell) > 0:
                partition.append(cell) # Get the hex cell and add to partition if nonempty
            curr = hex_jump(curr, (r1,r2,r3), (1,0,0))
        bottom = hex_jump(bottom, (r1,r2,r3), (0,1,0))
        if not hex_on_simplex(bottom, r1, r2, r3, N):
            bottom = hex_jump(bottom, (r1,r2,r3), (1,0,0))
    return partition


##########################################################################################
# Non-parallel. Need to call externally with "from trinomial_hex1 import lattice_search"
# Call with "best_partition, best_score, best_params = lattice_search(N)"


def lattice_search(N, r=r_uniform, timing=False):
    if timing:
        start = time.time()
    best_score = math.inf 
    best_partition = None
    best_params = None
    r1_bound = math.floor(N/2)
    for r1 in range(0, r1_bound+1):
        if timing:
            print("r1=", r1, " at time ", time.time()-start, best_score)
        r2_bound = math.floor(N/2)
        for r2 in range(0, r2_bound+1):
            r3_upper_bound = min(r1+r2-1, math.floor(N/2)) # Max value allowed for r3 based on valid_hex
            r3_lower_bound = abs(r2-r1) # Min value allowed for r3 based on valid_hex
            for r3 in range(r3_lower_bound+1, r3_upper_bound+1): 
                if not valid_hex(r1, r2, r3):
                    continue
            
                dx_bound = N
                for dx in range(0, (2*dx_bound)+1):
                    dx = dx/2
                    dy_bound = N
                    for dy in range(0, (2*dy_bound)+1):
                        dy = dy/2
                        partition = iterate_hex_cover(dx, dy, r1, r2, r3, N)
                        partition_score = SMML(partition, N, r)
                        if partition_score < best_score:
                            best_partition = partition 
                            best_score = partition_score
                            best_params = dx, dy, r1, r2, r3
    if timing: 
        return best_partition, best_score, (time.time()-start), best_params
    return best_partition, best_score, best_params



##########################################################################################
# Parallelised version. Need to call externally with "from trinomial_hex1 import lattice_search_parallel"
# "best_partition, best_score, best_params = lattice_search_parallel(N)"
 
 
def build_tasks(N, r=r_uniform):
    """
    Enumerate every valid (r1, r2, r3, dx) combo up front.
    """

    tasks = []
    r_bound = math.floor(N / 2)
    for r1 in range(0, r_bound + 1):
        for r2 in range(0, r_bound + 1):
            r3_upper = min(r1 + r2 - 1, r_bound)
            r3_lower = abs(r2 - r1)
            for r3 in range(r3_lower + 1, r3_upper + 1):
                if not valid_hex(r1, r2, r3):
                    continue
                for dx2 in range(0, (2 * N) + 1):
                    dx = dx2 / 2
                    tasks.append((r1, r2, r3, dx, N, r))
    return tasks


def _search_one_dx(args):
    """
    Given a dx value, search over all possible dy values for the generating hexagon. Used in parallel search. 
    """

    r1, r2, r3, dx, N, r = args
    best_score = math.inf
    best_params = None

    for dy2 in range(0, (2*N)+1):
        dy = dy2/2
        partition = iterate_hex_cover(dx, dy, r1, r2, r3, N)
        score = SMML(partition, N, r)
        if score < best_score:
            best_score = score
            best_params = (dx, dy, r1, r2, r3)

    return best_score, best_params


def lattice_search_parallel(N, r=r_uniform, n_workers=None,
                                  checkpoint_path="search_checkpoint.pkl",
                                  checkpoint_every=200):
    """
    Parallel version of lattice search.
    """

    import pickle

    if n_workers is None:
        n_workers = os.cpu_count()-2

    tasks = build_tasks(N, r)
    total = len(tasks)
    print(f"{total} total tasks across {n_workers} workers")

    best_score = math.inf
    best_params = None
    completed = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_search_one_dx, t) for t in tasks]
        for future in as_completed(futures):
            score, params = future.result()
            completed += 1
            if params is not None and score < best_score:
                best_score = score
                best_params = params

            if completed % checkpoint_every == 0 or completed == total:
                elapsed = time.time() - t0
                print(f"{completed}/{total} ({100*completed/total:.1f}%), "
                      f"best so far: {best_score:.4f}, elapsed: {elapsed:.0f}s")
                with open(checkpoint_path, "wb") as f:
                    pickle.dump({
                        "best_score": best_score,
                        "best_params": best_params,
                        "completed": completed,
                        "total": total,
                    }, f)

    # Recompute the actual winning partition just once, at the end
    dx, dy, r1, r2, r3 = best_params
    best_partition = iterate_hex_cover(dx, dy, r1, r2, r3, N)

    with open(checkpoint_path, "wb") as f:
        pickle.dump({
            "best_score": best_score,
            "best_params": best_params,
            "best_partition": best_partition,
            "completed": total,
            "total": total,
            "finished": True,
        }, f)

    return best_partition, best_score, best_params
