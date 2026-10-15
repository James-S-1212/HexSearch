from trinomial_hex import lattice_search_parallel, lattice_search
from trinomial_hex_regular import lattice_search_parallel_regular2_early_stop
import math
N=100
# best_partition, best_score, best_params = lattice_search(N)
# best_partition, best_score, best_params = lattice_search_parallel(N)
best_partition, best_score, best_params = lattice_search_parallel_regular2_early_stop(N, patience=2, lower=1, upper=None)
