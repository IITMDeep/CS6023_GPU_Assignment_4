%%writefile example.cu
// optimized_evacuation.cu
#include <bits/stdc++.h>
#include <curand_kernel.h>
using namespace std;

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

// Bit operations for compact visited set and cycle detection
#define SET_BIT(array, bit) array[bit/32] |= (1u << (bit%32))
#define GET_BIT(array, bit) ((array[bit/32] >> (bit%32)) & 1u)
#define MAX_VISITS_PER_CITY 3  // Allow revisiting cities this many times
#define CYCLE_DETECTION_WINDOW 10  // Look back this many steps to detect cycles

// Forward declarations
__global__ void setup_kernel(curandState *state, unsigned int seed, int n);
__global__ void random_walk_batch_kernel(
    int batch_offset, int batch_size, int numCities,
    int *rowPtr, int *colIdx, int *edgeLen, int *edgeCap,
    int *shelters, int *shelter_cap, int *shelter_fill,
    int max_dist_elderly,
    int *populated_city_ids, int *prime_age, int *elderly,
    int *path_counts, int *paths_flat, int *paths_offset,
    int *drop_counts, int *drops_flat, int *drops_offset,
    unsigned char *visit_counts,  // Using unsigned char to count visits per city
    curandState *rand_state
);

// Configuration parameters
struct SimConfig {
    int batch_size;
    int max_path_len;
    int max_drop_len;
    bool use_edge_len;
    bool use_edge_cap;
};

// Kernel constants
__constant__ SimConfig d_config;

// Device functions
__device__ int get_random_neighbor(int city, const int* rowPtr, const int* colIdx, curandState *state) {
    int start = rowPtr[city];
    int end = rowPtr[city + 1];
    if (start == end) return -1;
    int idx = start + (curand(state) % (end - start));
    return colIdx[idx];
}

__global__ void setup_kernel(curandState *state, unsigned int seed, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // Use different seed per thread
        curand_init(seed, tid, 0, &state[tid]);
    }
}

__global__ void random_walk_batch_kernel(
    int batch_offset, int batch_size, int numCities,
    int *rowPtr, int *colIdx, int *edgeLen, int *edgeCap,
    int *shelters, int *shelter_cap, int *shelter_fill,
    int max_dist_elderly,
    int *populated_city_ids, int *prime_age, int *elderly,
    int *path_counts, int *paths_flat, int *paths_offset,
    int *drop_counts, int *drops_flat, int *drops_offset,
    unsigned char *visit_counts,  // Using unsigned char to count visits per city
    curandState *rand_state
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;
    
    int global_tid = batch_offset + tid;
    int city = populated_city_ids[global_tid];
    int pa = prime_age[global_tid];
    int el = elderly[global_tid];
    
    // Calculate the base offsets for this thread
    int path_base = paths_offset[tid];
    int drop_base = drops_offset[tid];
    
    // Initialize visit counts to 0
    unsigned char *my_visit_counts = &visit_counts[tid * numCities];
    for (int i = 0; i < numCities; i++) {
        my_visit_counts[i] = 0;
    }
    
    // Small circular buffer to detect cycles
    int recent_cities[CYCLE_DETECTION_WINDOW];
    for (int i = 0; i < CYCLE_DETECTION_WINDOW; i++) {
        recent_cities[i] = -1;
    }
    
    int path_len = 0, drop_len = 0, total_dist = 0;
    curandState local_state = rand_state[global_tid];
    int curr = city;
    
    // Store the starting city immediately
    paths_flat[path_base + path_len++] = curr;
    my_visit_counts[curr]++;
    
    // Initialize cycle detection buffer with starting city
    recent_cities[0] = curr;
    
    int cycle_pos = 1; // Position in cycle detection buffer
    bool should_continue = true;
    bool all_dropped = false;
    
    while (should_continue && path_len < d_config.max_path_len) {
        // Check if current location is a shelter
        if (shelters[curr]) {
            // Try to accommodate as many people as possible
            int old_fill = atomicAdd(&shelter_fill[curr], pa + el);
            int remaining = shelter_cap[curr] - old_fill;
            
            // Calculate how many people we can save
            int can_save = min(pa + el, max(0, remaining));
            int save_pa = min(can_save, pa);
            int save_el = can_save - save_pa;
            
            // Record the drop
            if (drop_base + drop_len * 3 + 2 < drops_offset[tid+1]) {
                drops_flat[drop_base + drop_len * 3] = curr;
                drops_flat[drop_base + drop_len * 3 + 1] = save_pa;
                drops_flat[drop_base + drop_len * 3 + 2] = save_el;
                drop_len++;
            }
            
            // Update remaining population
            pa -= save_pa;
            el -= save_el;
            
            // Only end if everyone is saved
            if (pa == 0 && el == 0) {
                all_dropped = true;
                should_continue = false;
                break;
            }
            
            // Otherwise continue to find shelter for remaining people
        }
        
        // Find a neighbor with minimal visits
        int next = -1;
        int min_visits = MAX_VISITS_PER_CITY + 1;
        int edge_dist = 1;
        int attempts = min(10, rowPtr[curr+1] - rowPtr[curr]); // Try more neighbors
        
        // First pass: try to find neighbors with fewer visits
        for (int retry = 0; retry < attempts; retry++) {
            int n = get_random_neighbor(curr, rowPtr, colIdx, &local_state);
            if (n != -1) {
                // Check if adding this city would create a short cycle
                bool creates_cycle = false;
                for (int i = 0; i < CYCLE_DETECTION_WINDOW; i++) {
                    if (recent_cities[i] == n) {
                        // Found the same city in recent history - potential cycle
                        // Only consider it a cycle if it's not just the immediate previous city
                        // (which would be a simple back-and-forth)
                        if (i < CYCLE_DETECTION_WINDOW - 1 && recent_cities[(i+1) % CYCLE_DETECTION_WINDOW] != curr) {
                            creates_cycle = true;
                            break;
                        }
                    }
                }
                
                // If this city has been visited fewer times than our current best
                // and doesn't create a cycle, consider it
                if (!creates_cycle && my_visit_counts[n] < min_visits) {
                    // Get edge distance to check if it would exceed elderly distance limit
                    int candidate_edge_dist = 1;
                    for (int e = rowPtr[curr]; e < rowPtr[curr+1]; e++) {
                        if (colIdx[e] == n) {
                            candidate_edge_dist = d_config.use_edge_len ? edgeLen[e] : 1;
                            break;
                        }
                    }
                    
                    // Check if taking this edge would exceed the elderly distance limit
                    // Only consider this edge if:
                    // 1. We have no elderly OR
                    // 2. Taking this edge won't exceed the max elderly distance
                    if (el == 0 || total_dist + candidate_edge_dist <= max_dist_elderly) {
                        next = n;
                        min_visits = my_visit_counts[n];
                        edge_dist = candidate_edge_dist;
                        
                        // If we found an unvisited city, we can stop looking
                        if (min_visits == 0) break;
                    }
                }
            }
        }
        
        // If we have elderly and couldn't find a suitable path within distance limit,
        // drop the elderly here before continuing
        if (el > 0 && (next == -1 || total_dist + edge_dist > max_dist_elderly)) {
            // Drop elderly here due to distance constraint
            if (drop_base + drop_len * 3 + 2 < drops_offset[tid+1]) {
                drops_flat[drop_base + drop_len * 3] = curr;
                drops_flat[drop_base + drop_len * 3 + 1] = 0;
                drops_flat[drop_base + drop_len * 3 + 2] = el;
                drop_len++;
            }
            el = 0; // All elderly dropped
            
            // Now try again to find a path without elderly constraints
            // Reset these variables to try again
            next = -1;
            min_visits = MAX_VISITS_PER_CITY + 1;
            
            // Try to find neighbors without elderly constraint
            for (int retry = 0; retry < attempts; retry++) {
                int n = get_random_neighbor(curr, rowPtr, colIdx, &local_state);
                if (n != -1) {
                    // Check for cycles
                    bool creates_cycle = false;
                    for (int i = 0; i < CYCLE_DETECTION_WINDOW; i++) {
                        if (recent_cities[i] == n && 
                            i < CYCLE_DETECTION_WINDOW - 1 && 
                            recent_cities[(i+1) % CYCLE_DETECTION_WINDOW] != curr) {
                            creates_cycle = true;
                            break;
                        }
                    }
                    
                    // Consider this neighbor if it doesn't create a cycle
                    if (!creates_cycle && my_visit_counts[n] < min_visits) {
                        next = n;
                        min_visits = my_visit_counts[n];
                        
                        // Get edge distance
                        for (int e = rowPtr[curr]; e < rowPtr[curr+1]; e++) {
                            if (colIdx[e] == n) {
                                edge_dist = d_config.use_edge_len ? edgeLen[e] : 1;
                                break;
                            }
                        }
                        
                        // If we found an unvisited city, we can stop looking
                        if (min_visits == 0) break;
                    }
                }
            }
        }
        
        // If we still couldn't find any suitable neighbor or all have been visited MAX_VISITS_PER_CITY times
        if (next == -1 || min_visits >= MAX_VISITS_PER_CITY) {
            // Drop remaining population at current city before terminating
            should_continue = false;
        } else {
            // Update total distance with the selected edge distance
            total_dist += edge_dist;
            
            // Update the current city and add to path
            curr = next;
            if (path_base + path_len < paths_offset[tid+1]) {
                paths_flat[path_base + path_len++] = curr;
            }
            
            // Increment visit count for this city
            my_visit_counts[curr]++;
            
            // Update cycle detection buffer (circular buffer)
            recent_cities[cycle_pos % CYCLE_DETECTION_WINDOW] = curr;
            cycle_pos++;
        }
    }
    
    // Drop any remaining population at the last visited city
    if ((pa > 0 || el > 0) && !all_dropped && drop_base + drop_len * 3 + 2 < drops_offset[tid+1]) {
        drops_flat[drop_base + drop_len * 3] = curr;
        drops_flat[drop_base + drop_len * 3 + 1] = pa;
        drops_flat[drop_base + drop_len * 3 + 2] = el;
        drop_len++;
    }
    
    // Save the final results
    path_counts[tid] = path_len;
    drop_counts[tid] = drop_len;
    rand_state[global_tid] = local_state;
}

// Host utility functions
template<typename T>
void allocDevice(T **ptr, size_t size) {
    cudaError_t err = cudaMalloc(ptr, size);
    if (err != cudaSuccess) {
        cerr << "Failed to allocate device memory: " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

template<typename T>
void copyToDevice(T *dst, const T *src, size_t size) {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cerr << "Failed to copy to device: " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

template<typename T>
void copyFromDevice(T *dst, const T *src, size_t size) {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cerr << "Failed to copy from device: " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

// Calculate prefixsum for compact storage
vector<int> calculatePrefixSum(const vector<int>& counts, int max_items) {
    vector<int> offsets(counts.size() + 1, 0);
    for (size_t i = 0; i < counts.size(); i++) {
        offsets[i+1] = offsets[i] + min(counts[i], max_items);
    }
    return offsets;
}

// Main function
int main() {
    // Read input parameters
    int numCities, numRoads, numShelters, numPopCities;
    int maxDistElderly;
    cin >> numCities >> numRoads;
    
    ofstream outFile("output.txt");
    // Determine device properties for optimization
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    
    // Create graph in CSR format
    vector<int> rowPtr(numCities + 1), colIdx, edgeLen, edgeCap;
    vector<vector<pair<int, pair<int, int>>>> adj(numCities);
    
    for (int i = 0; i < numRoads; ++i) {
        int u, v, l, c;
        cin >> u >> v >> l >> c;
        adj[u].emplace_back(v, make_pair(l, c));
        adj[v].emplace_back(u, make_pair(l, c));
    }
    
    int edgeCount = 0;
    for (int i = 0; i < numCities; ++i) {
        rowPtr[i] = edgeCount;
        for (auto& edge : adj[i]) {
            colIdx.push_back(edge.first);
            edgeLen.push_back(edge.second.first);
            edgeCap.push_back(edge.second.second);
            ++edgeCount;
        }
    }
    rowPtr[numCities] = edgeCount;
    
    // Read shelter information
    cin >> numShelters;
    vector<int> shelter_city(numCities, 0), shelter_cap(numCities, 0);
    for (int i = 0; i < numShelters; ++i) {
        int city, cap;
        cin >> city >> cap;
        shelter_city[city] = 1;
        shelter_cap[city] = cap;
    }
    
    // Read population information
    cin >> numPopCities;
    vector<int> pop_city_ids(numPopCities), prime(numPopCities), old(numPopCities);
    for (int i = 0; i < numPopCities; ++i) {
        int city, p, o;
        cin >> city >> p >> o;
        pop_city_ids[i] = city;
        prime[i] = p;
        old[i] = o;
    }
    
    cin >> maxDistElderly;
    
    // Configure simulation parameters based on available memory
    SimConfig config;
    config.max_path_len = 512;
    config.max_drop_len = 128;
    config.use_edge_len = true;
    config.use_edge_cap = true;
    
    // Calculate batch size based on available memory
    size_t mem_per_city = sizeof(int) * config.max_path_len * 2 + 
                         sizeof(int) * config.max_drop_len * 3 * 2 +
                         sizeof(unsigned int) * CEIL_DIV(numCities, 32);
    
    // Reserve 20% of free memory for other allocations
    size_t available_mem = free_mem * 0.8;
    int max_batch_size = available_mem / mem_per_city;
    
    // Limit batch size to a reasonable number
    config.batch_size = min(max_batch_size, numPopCities);
    config.batch_size = max(1, min(config.batch_size, 10000));
    
    // Copy configuration to constant memory
    cudaMemcpyToSymbol(d_config, &config, sizeof(SimConfig));
    
    // Allocate GPU memory for graph data
    int *d_rowPtr, *d_colIdx, *d_edgeLen, *d_edgeCap;
    allocDevice(&d_rowPtr, sizeof(int) * rowPtr.size());
    allocDevice(&d_colIdx, sizeof(int) * colIdx.size());
    allocDevice(&d_edgeLen, sizeof(int) * edgeLen.size());
    allocDevice(&d_edgeCap, sizeof(int) * edgeCap.size());
    
    copyToDevice(d_rowPtr, rowPtr.data(), sizeof(int) * rowPtr.size());
    copyToDevice(d_colIdx, colIdx.data(), sizeof(int) * colIdx.size());
    copyToDevice(d_edgeLen, edgeLen.data(), sizeof(int) * edgeLen.size());
    copyToDevice(d_edgeCap, edgeCap.data(), sizeof(int) * edgeCap.size());
    
    // Allocate GPU memory for shelter data
    int *d_shelters, *d_shelter_cap, *d_shelter_fill;
    allocDevice(&d_shelters, sizeof(int) * numCities);
    allocDevice(&d_shelter_cap, sizeof(int) * numCities);
    allocDevice(&d_shelter_fill, sizeof(int) * numCities);
    
    copyToDevice(d_shelters, shelter_city.data(), sizeof(int) * numCities);
    copyToDevice(d_shelter_cap, shelter_cap.data(), sizeof(int) * numCities);
    cudaMemset(d_shelter_fill, 0, sizeof(int) * numCities);
    
    // Allocate GPU memory for population data
    int *d_populated_city_ids, *d_prime_age, *d_elderly;
    allocDevice(&d_populated_city_ids, sizeof(int) * numPopCities);
    allocDevice(&d_prime_age, sizeof(int) * numPopCities);
    allocDevice(&d_elderly, sizeof(int) * numPopCities);
    
    copyToDevice(d_populated_city_ids, pop_city_ids.data(), sizeof(int) * numPopCities);
    copyToDevice(d_prime_age, prime.data(), sizeof(int) * numPopCities);
    copyToDevice(d_elderly, old.data(), sizeof(int) * numPopCities);
    
    // CURAND setup for all populated cities
    curandState *d_states;
    allocDevice(&d_states, sizeof(curandState) * numPopCities);
    
    // Use time as random seed
    unsigned int seed = time(NULL);
    int numBlocks = CEIL_DIV(numPopCities, THREADS_PER_BLOCK);
    setup_kernel<<<numBlocks, THREADS_PER_BLOCK>>>(d_states, seed, numPopCities);
    
    // Output vectors for paths and drops
    vector<vector<int>> all_paths(numPopCities);
    vector<vector<vector<int>>> all_drops(numPopCities);
    
    // Process in batches
    for (int batch_start = 0; batch_start < numPopCities; batch_start += config.batch_size) {
        int current_batch_size = min(config.batch_size, numPopCities - batch_start);
        
        // Initial allocation with max sizes
        vector<int> path_count_estimate(current_batch_size, config.max_path_len / 2);
        vector<int> drop_count_estimate(current_batch_size, config.max_drop_len / 2);
        
        // Calculate offsets for compact storage
        vector<int> path_offsets = calculatePrefixSum(path_count_estimate, config.max_path_len);
        vector<int> drop_offsets = calculatePrefixSum(drop_count_estimate, config.max_drop_len);
        
        // Allocate memory for results
        int *d_path_counts, *d_paths_flat, *d_paths_offset;
        int *d_drop_counts, *d_drops_flat, *d_drops_offset;
        unsigned char *d_visit_counts;
        
        allocDevice(&d_path_counts, sizeof(int) * current_batch_size);
        allocDevice(&d_drop_counts, sizeof(int) * current_batch_size);
        allocDevice(&d_paths_flat, sizeof(int) * path_offsets.back());
        allocDevice(&d_drops_flat, sizeof(int) * drop_offsets.back() * 3);
        allocDevice(&d_paths_offset, sizeof(int) * (current_batch_size + 1));
        allocDevice(&d_drops_offset, sizeof(int) * (current_batch_size + 1));
        
        // Allocate visit count array (one byte per city per batch item)
        allocDevice(&d_visit_counts, sizeof(unsigned char) * current_batch_size * numCities);
        cudaMemset(d_visit_counts, 0, sizeof(unsigned char) * current_batch_size * numCities);
        
        // Copy offsets to device
        copyToDevice(d_paths_offset, path_offsets.data(), sizeof(int) * (current_batch_size + 1));
        copyToDevice(d_drops_offset, drop_offsets.data(), sizeof(int) * (current_batch_size + 1));
        
        // Launch kernel for current batch
        int batch_blocks = CEIL_DIV(current_batch_size, THREADS_PER_BLOCK);
        random_walk_batch_kernel<<<batch_blocks, THREADS_PER_BLOCK>>>(
            batch_start, current_batch_size, numCities,
            d_rowPtr, d_colIdx, d_edgeLen, d_edgeCap,
            d_shelters, d_shelter_cap, d_shelter_fill,
            maxDistElderly,
            d_populated_city_ids, d_prime_age, d_elderly,
            d_path_counts, d_paths_flat, d_paths_offset,
            d_drop_counts, d_drops_flat, d_drops_offset,
            d_visit_counts, d_states
        );
        
        // Check for kernel errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cerr << "Kernel launch error: " << cudaGetErrorString(err) << endl;
            exit(EXIT_FAILURE);
        }
        
        cudaDeviceSynchronize();
        
        // Copy results back
        vector<int> path_counts(current_batch_size), drop_counts(current_batch_size);
        copyFromDevice(path_counts.data(), d_path_counts, sizeof(int) * current_batch_size);
        copyFromDevice(drop_counts.data(), d_drop_counts, sizeof(int) * current_batch_size);
        
        // Allocate host memory for paths and drops based on actual sizes
        vector<int> paths_flat(path_offsets.back());
        vector<int> drops_flat(drop_offsets.back() * 3);
        
        // Copy paths and drops
        copyFromDevice(paths_flat.data(), d_paths_flat, sizeof(int) * path_offsets.back());
        copyFromDevice(drops_flat.data(), d_drops_flat, sizeof(int) * drop_offsets.back() * 3);
        
        // Extract individual paths and drops
        for (int i = 0; i < current_batch_size; i++) {
            int path_start = path_offsets[i];
            int path_end = path_start + min(path_counts[i], config.max_path_len);
            
            // Store this path
            all_paths[batch_start + i].assign(
                paths_flat.begin() + path_start,
                paths_flat.begin() + path_end
            );
            
            // Process drops
            all_drops[batch_start + i].clear();
            int drop_start = drop_offsets[i];
            for (int j = 0; j < min(drop_counts[i], config.max_drop_len); j++) {
                int idx = drop_start + j * 3;
                all_drops[batch_start + i].push_back({
                    drops_flat[idx],
                    drops_flat[idx + 1],
                    drops_flat[idx + 2]
                });
            }
        }
        
        // Free batch resources
        cudaFree(d_path_counts);
        cudaFree(d_paths_flat);
        cudaFree(d_paths_offset);
        cudaFree(d_drop_counts);
        cudaFree(d_drops_flat);
        cudaFree(d_drops_offset);
        cudaFree(d_visit_counts);
    }
    
    // Output results to file
    // outFile << "paths" << endl;
    for (int i = 0; i < numPopCities; ++i) {
        for (size_t j = 0; j < all_paths[i].size(); ++j) {
            outFile << all_paths[i][j];
            if (j < all_paths[i].size() - 1) outFile <<" ";
        }
        outFile << endl;
    }
    
    // outFile << "drops" << endl;
    for (int i = 0; i < numPopCities; ++i) {
        for (size_t j = 0; j < all_drops[i].size(); ++j) {
            int cityId = all_drops[i][j][0];
            int primeCount = all_drops[i][j][1];
            int elderCount = all_drops[i][j][2];
            outFile << cityId <<" "<< primeCount <<" "<< elderCount;
            if (j < all_drops[i].size() - 1) outFile <<" ";
        }
        outFile << endl;
    }
    
    // Close the output file
    outFile.close();
    
    // Free device memory
    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_edgeLen);
    cudaFree(d_edgeCap);
    cudaFree(d_shelters);
    cudaFree(d_shelter_cap);
    cudaFree(d_shelter_fill);
    cudaFree(d_populated_city_ids);
    cudaFree(d_prime_age);
    cudaFree(d_elderly);
    cudaFree(d_states);
    
    return 0;
}