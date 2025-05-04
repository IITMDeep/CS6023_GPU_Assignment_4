%%writefile example.cu
// Parallel Evacuation Simulation - CUDA (Optimized with City Batching)
#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>  // Added for file output
using namespace std;

#define ll long long
#define INF 1e18
#define MAX_CITIES 5000  // Reduced for batch processing
#define MAX_PATH 15000
#define MAX_DROPS 15000
#define BATCH_SIZE 5000  // Process cities in batches of this size

__device__ ll atomicSubLL(ll* address, ll val) {
    return (ll)atomicAdd((unsigned long long*)address, (unsigned long long)(-val));
}

// Modified kernel to work with city batches
__global__ void evacuationKernel(
    int batchOffset,
    int numCitiesTotal,
    int numCitiesInBatch,
    const int* rowPtr,
    const int* colIdx,
    const ll* weights,
    int numShel,
    const int* shelCity,
    unsigned long long* shelCap,
    int numPop,
    const int* popCity,
    const ll* primeInit,
    const ll* elderInit,
    ll maxDistElder,
    ll* pathLens,
    ll* paths,
    int* numDrops,
    ll* drops,
    bool* processedPop
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numPop) return;
    
    // Skip populations that have already been processed or don't belong to this batch
    int start = popCity[tid];
    if (processedPop[tid] || start < batchOffset || start >= batchOffset + numCitiesInBatch) return;
    
    ll prime = primeInit[tid];
    ll elder = elderInit[tid];
    ll remainingPrime = prime, remainingElder = elder;

    // Dijkstra setup - only for cities in this batch and global shelters
    ll dist[MAX_CITIES];
    bool vis[MAX_CITIES];
    int par[MAX_CITIES];
    for (int i = 0; i < numCitiesInBatch; ++i) {
        // Map global city ID to batch-local ID
        int localCityId = i;
        dist[localCityId] = INF;
        vis[localCityId] = false;
        par[localCityId] = -1;
    }
    
    // Start city is now relative to the batch
    int startLocal = start - batchOffset;
    dist[startLocal] = 0;

    // Modified Dijkstra to work within the batch
    for (int i = 0; i < numCitiesInBatch; ++i) {
        int u = -1;
        ll best = INF;
        for (int j = 0; j < numCitiesInBatch; ++j) {
            if (!vis[j] && dist[j] < best) {
                best = dist[j];
                u = j;
            }
        }
        if (u == -1) break;
        vis[u] = true;
        
        // Global city ID for adjacency list access
        int globalU = u + batchOffset;
        
        for (int e = rowPtr[globalU]; e < rowPtr[globalU+1]; ++e) {
            int v = colIdx[e];
            
            // Only process edges to cities in this batch
            if (v >= batchOffset && v < batchOffset + numCitiesInBatch) {
                int localV = v - batchOffset;
                ll w = weights[e];
                if (dist[localV] > dist[u] + w) {
                    dist[localV] = dist[u] + w;
                    par[localV] = u;
                }
            }
        }
    }

    // Find shelter distances within this batch
    bool visitedShelter[MAX_CITIES] = {false};
    int shelOrder[MAX_CITIES];
    int numShelFound = 0;
    
    for (int i = 0; i < numShel; ++i) {
        int shelGlobalCity = shelCity[i];
        // Only include shelters in this batch
        if (shelGlobalCity >= batchOffset && shelGlobalCity < batchOffset + numCitiesInBatch) {
            shelOrder[numShelFound++] = i;
        }
    }
    
    // Sort shelters by distance
    for (int i = 0; i < numShelFound; ++i) {
        for (int j = i + 1; j < numShelFound; ++j) {
            int s1Local = shelCity[shelOrder[i]] - batchOffset;
            int s2Local = shelCity[shelOrder[j]] - batchOffset;
            if (dist[s1Local] > dist[s2Local]) {
                int temp = shelOrder[i];
                shelOrder[i] = shelOrder[j];
                shelOrder[j] = temp;
            }
        }
    }

    int fullPath[MAX_PATH];
    int fullPathLen = 0;
    int dropIdx = 0;
    ll travelDist = 0;
    bool elderlyDropped = false;
    
    // Initialize path with starting city (using global city ID)
    fullPath[fullPathLen++] = start;
    
    // Try each shelter in order of proximity
    for (int idx = 0; idx < numShelFound && (remainingPrime > 0 || remainingElder > 0); ++idx) {
        int shelIdx = shelOrder[idx];
        int targetGlobalCity = shelCity[shelIdx];
        int targetLocalCity = targetGlobalCity - batchOffset;
        
        // Skip if we've already processed this shelter
        if (visitedShelter[targetLocalCity]) continue;
        
        // Check if we're already at this shelter
        int lastPathCity = fullPath[fullPathLen - 1];
        if (lastPathCity == targetGlobalCity) {
            visitedShelter[targetLocalCity] = true;
            
            // Try to drop evacuees at this shelter
            unsigned long long cap = atomicAdd(&shelCap[shelIdx], 0ULL); // read capacity
            if (cap > 0) {
                ll elderToDrop = 0;
                ll primeToDrop = 0;
                
                // First prioritize elderly
                if (remainingElder > 0) {
                    elderToDrop = min(remainingElder, (ll)cap);
                    atomicSubLL((ll*)&shelCap[shelIdx], elderToDrop);
                    cap -= elderToDrop;
                    remainingElder -= elderToDrop;
                }
                
                // Then handle prime age if there's still capacity
                if (remainingPrime > 0 && cap > 0) {
                    primeToDrop = min(remainingPrime, (ll)cap);
                    atomicSubLL((ll*)&shelCap[shelIdx], primeToDrop);
                    remainingPrime -= primeToDrop;
                }
                
                // Only record the drop if people were actually dropped
                if (elderToDrop > 0 || primeToDrop > 0) {
                    drops[tid*MAX_DROPS*3 + dropIdx*3 + 0] = targetGlobalCity;
                    drops[tid*MAX_DROPS*3 + dropIdx*3 + 1] = primeToDrop;
                    drops[tid*MAX_DROPS*3 + dropIdx*3 + 2] = elderToDrop;
                    dropIdx++;
                }
            }
            
            continue; // Move to next shelter
        }
        
        // We need to travel to this shelter
        int pathToShelter[MAX_PATH];
        int pathToShelterLen = 0;
        
        // Trace path from target to current position (using local city IDs)
        int currCitylocal = targetLocalCity;
        int lastPathCityLocal = -1;
        
        // Convert last path city to local if it's in this batch
        if (lastPathCity >= batchOffset && lastPathCity < batchOffset + numCitiesInBatch) {
            lastPathCityLocal = lastPathCity - batchOffset;
        }
        
        while (currCitylocal != -1 && currCitylocal != lastPathCityLocal && pathToShelterLen < MAX_PATH) {
            pathToShelter[pathToShelterLen++] = currCitylocal + batchOffset; // Convert back to global ID
            currCitylocal = par[currCitylocal];
        }
        
        // If we couldn't trace back to our current position, skip this shelter
        if (currCitylocal != lastPathCityLocal) continue;
        
        // Now add each city in the path in reverse order (from current to target)
        for (int i = pathToShelterLen - 1; i >= 0; --i) {
            int nextCity = pathToShelter[i];

            // Calculate distance to the next city
            for (int e = rowPtr[lastPathCity]; e < rowPtr[lastPathCity+1]; ++e) {
                if (colIdx[e] == nextCity) {
                    travelDist += weights[e];
                    break;
                }
            }
            
            // Add to our full path
            fullPath[fullPathLen++] = nextCity;
            
            // Check if elderly need to be dropped due to max travel distance
            if (!elderlyDropped && remainingElder > 0 && travelDist > maxDistElder) {
                // Drop elderly at the previous city
                int dropCity = fullPath[fullPathLen - 2]; // Previous city
                drops[tid*MAX_DROPS*3 + dropIdx*3 + 0] = dropCity;
                drops[tid*MAX_DROPS*3 + dropIdx*3 + 1] = 0;
                drops[tid*MAX_DROPS*3 + dropIdx*3 + 2] = remainingElder;
                elderlyDropped = true;
                remainingElder = 0;
                dropIdx++;
            }
            
            // Check if this intermediate city is a shelter we haven't visited
            for (int j = 0; j < numShel; ++j) {
                if (shelCity[j] == nextCity) {
                    int nextCityLocal = nextCity - batchOffset;
                    if (nextCityLocal >= 0 && nextCityLocal < numCitiesInBatch && !visitedShelter[nextCityLocal]) {
                        visitedShelter[nextCityLocal] = true;
                        
                        unsigned long long cap = atomicAdd(&shelCap[j], 0ULL); // read capacity
                        if (cap > 0) {
                            ll elderToDrop = 0;
                            ll primeToDrop = 0;
                            
                            // First prioritize elderly
                            if (remainingElder > 0) {
                                elderToDrop = min(remainingElder, (ll)cap);
                                atomicSubLL((ll*)&shelCap[j], elderToDrop);
                                cap -= elderToDrop;
                                remainingElder -= elderToDrop;
                            }
                            
                            // Then handle prime age if there's still capacity
                            if (remainingPrime > 0 && cap > 0) {
                                primeToDrop = min(remainingPrime, (ll)cap);
                                atomicSubLL((ll*)&shelCap[j], primeToDrop);
                                remainingPrime -= primeToDrop;
                            }
                            
                            // Only record the drop if people were actually dropped
                            if (elderToDrop > 0 || primeToDrop > 0) {
                                drops[tid*MAX_DROPS*3 + dropIdx*3 + 0] = nextCity;
                                drops[tid*MAX_DROPS*3 + dropIdx*3 + 1] = primeToDrop;
                                drops[tid*MAX_DROPS*3 + dropIdx*3 + 2] = elderToDrop;
                                dropIdx++;
                            }
                        }
                    }
                    break;
                }
            }
            
            lastPathCity = nextCity;
            
            // If everyone has been evacuated, we can stop
            if (remainingPrime == 0 && (remainingElder == 0 || elderlyDropped)) {
                break;
            }
        }
        
        // Mark the target shelter as visited
        visitedShelter[targetLocalCity] = true;
        
        // Try to drop evacuees at this shelter if we haven't placed everyone yet
        if ((remainingPrime > 0 || remainingElder > 0) && lastPathCity == targetGlobalCity) {
            unsigned long long cap = atomicAdd(&shelCap[shelIdx], 0ULL); // read capacity
            if (cap > 0) {
                ll elderToDrop = 0;
                ll primeToDrop = 0;
                
                // First prioritize elderly
                if (remainingElder > 0) {
                    elderToDrop = min(remainingElder, (ll)cap);
                    atomicSubLL((ll*)&shelCap[shelIdx], elderToDrop);
                    cap -= elderToDrop;
                    remainingElder -= elderToDrop;
                }
                
                // Then handle prime age if there's still capacity
                if (remainingPrime > 0 && cap > 0) {
                    primeToDrop = min(remainingPrime, (ll)cap);
                    atomicSubLL((ll*)&shelCap[shelIdx], primeToDrop);
                    remainingPrime -= primeToDrop;
                }
                
                // Only record the drop if people were actually dropped
                if (elderToDrop > 0 || primeToDrop > 0) {
                    drops[tid*MAX_DROPS*3 + dropIdx*3 + 0] = targetGlobalCity;
                    drops[tid*MAX_DROPS*3 + dropIdx*3 + 1] = primeToDrop;
                    drops[tid*MAX_DROPS*3 + dropIdx*3 + 2] = elderToDrop;
                    dropIdx++;
                }
            }
        }
        
        // If everyone has been evacuated, we can stop
        if (remainingPrime == 0 && (remainingElder == 0 || elderlyDropped)) {
            break;
        }
    }

    // Final fallback for elderly if they haven't been dropped yet due to max distance
    if (!elderlyDropped && remainingElder > 0) {
        // Find the last legal city within maxDistElder
        ll distAcc = 0;
        int lastLegalCity = start;
        
        for (int i = 1; i < fullPathLen; ++i) {
            int from = fullPath[i - 1];
            int to = fullPath[i];
            
            ll segmentDist = 0;
            for (int e = rowPtr[from]; e < rowPtr[from + 1]; ++e) {
                if (colIdx[e] == to) {
                    segmentDist = weights[e];
                    break;
                }
            }
            
            if (distAcc + segmentDist <= maxDistElder) {
                distAcc += segmentDist;
                lastLegalCity = to;
            } else {
                break;
            }
        }
        
        drops[tid * MAX_DROPS * 3 + dropIdx * 3 + 0] = lastLegalCity;
        drops[tid * MAX_DROPS * 3 + dropIdx * 3 + 1] = 0;
        drops[tid * MAX_DROPS * 3 + dropIdx * 3 + 2] = remainingElder;
        dropIdx++;
        elderlyDropped = true;
    }

    // IMPROVED CODE: Handle remaining evacuees - ensure all evacuees are dropped somewhere
    if (remainingPrime > 0 || remainingElder > 0) {
        int currentCity = fullPathLen > 0 ? fullPath[fullPathLen - 1] : start;
        
        // Check if current city is a shelter
        bool isShelter = false;
        for (int i = 0; i < numShel; ++i) {
            if (shelCity[i] == currentCity) {
                isShelter = true;
                break;
            }
        }
        
        // If current city is a shelter, try to find a neighboring non-shelter city
        if (isShelter) {
            // First attempt: Try to find the closest non-shelter neighboring city
            bool foundNeighbor = false;
            int neighborCity = -1;
            ll shortestDistance = INF;
            
            for (int e = rowPtr[currentCity]; e < rowPtr[currentCity + 1]; ++e) {
                int candidate = colIdx[e];
                ll distance = weights[e];
                
                // Check if candidate is not a shelter
                bool candidateIsShelter = false;
                for (int i = 0; i < numShel; ++i) {
                    if (shelCity[i] == candidate) {
                        candidateIsShelter = true;
                        break;
                    }
                }
                
                if (!candidateIsShelter && distance < shortestDistance) {
                    neighborCity = candidate;
                    shortestDistance = distance;
                    foundNeighbor = true;
                }
            }
            
            if (foundNeighbor) {
                // Add neighbor to path and drop evacuees there
                travelDist += shortestDistance; // Update travel distance
                fullPath[fullPathLen++] = neighborCity;
                
                // Check for elderly max distance again
                if (!elderlyDropped && remainingElder > 0 && travelDist > maxDistElder) {
                    // Drop elderly at current city before moving to neighbor
                    drops[tid * MAX_DROPS * 3 + dropIdx * 3 + 0] = currentCity;
                    drops[tid * MAX_DROPS * 3 + dropIdx * 3 + 1] = 0;
                    drops[tid * MAX_DROPS * 3 + dropIdx * 3 + 2] = remainingElder;
                    dropIdx++;
                    elderlyDropped = true;
                    
                    // Only prime age people continue to neighbor
                    drops[tid * MAX_DROPS * 3 + dropIdx * 3 + 0] = neighborCity;
                    drops[tid * MAX_DROPS * 3 + dropIdx * 3 + 1] = remainingPrime;
                    drops[tid * MAX_DROPS * 3 + dropIdx * 3 + 2] = 0;
                    dropIdx++;
                    remainingPrime = 0;
                    remainingElder = 0;
                } else {
                    // Drop both groups at neighbor city
                    drops[tid * MAX_DROPS * 3 + dropIdx * 3 + 0] = neighborCity;
                    drops[tid * MAX_DROPS * 3 + dropIdx * 3 + 1] = remainingPrime;
                    drops[tid * MAX_DROPS * 3 + dropIdx * 3 + 2] = remainingElder;
                    dropIdx++;
                    remainingPrime = 0;
                    remainingElder = 0;
                }
            } else {
                // Second attempt: Try neighbors of neighbors (2-hop search)
                bool foundTwoHopNeighbor = false;
                int twoHopCity = -1;
                int intermediateCity = -1;
                ll totalDistance = INF;
                
                // For each direct neighbor (even shelters)
                for (int e1 = rowPtr[currentCity]; e1 < rowPtr[currentCity + 1]; ++e1) {
                    int firstHop = colIdx[e1];
                    ll firstDist = weights[e1];
                    
                    // For each neighbor of this neighbor
                    for (int e2 = rowPtr[firstHop]; e2 < rowPtr[firstHop + 1]; ++e2) {
                        int secondHop = colIdx[e2];
                        ll secondDist = weights[e2];
                        
                        // Skip if it's the current city or another shelter
                        if (secondHop == currentCity) continue;
                        
                        bool secondHopIsShelter = false;
                        for (int i = 0; i < numShel; ++i) {
                            if (shelCity[i] == secondHop) {
                                secondHopIsShelter = true;
                                break;
                            }
                        }
                        
                        if (!secondHopIsShelter && firstDist + secondDist < totalDistance) {
                            twoHopCity = secondHop;
                            intermediateCity = firstHop;
                            totalDistance = firstDist + secondDist;
                            foundTwoHopNeighbor = true;
                        }
                    }
                }
                
                if (foundTwoHopNeighbor) {
                    // Add both hops to path
                    ll firstHopDist = 0;
                    for (int e = rowPtr[currentCity]; e < rowPtr[currentCity + 1]; ++e) {
                        if (colIdx[e] == intermediateCity) {
                            firstHopDist = weights[e];
                            break;
                        }
                    }
                    
                    travelDist += firstHopDist;
                    fullPath[fullPathLen++] = intermediateCity;
                    
                    // Check max distance for elderly after first hop
                    bool elderlyStoppedAtIntermediate = false;
                    if (!elderlyDropped && remainingElder > 0 && travelDist > maxDistElder) {
                        // Drop elderly at intermediate city
                        drops[tid * MAX_DROPS * 3 + dropIdx * 3 + 0] = intermediateCity;
                        drops[tid * MAX_DROPS * 3 + dropIdx * 3 + 1] = 0;
                        drops[tid * MAX_DROPS * 3 + dropIdx * 3 + 2] = remainingElder;
                        dropIdx++;
                        elderlyDropped = true;
                        elderlyStoppedAtIntermediate = true;
                        remainingElder = 0;
                    }
                    
                    // Add second hop
                    ll secondHopDist = 0;
                    for (int e = rowPtr[intermediateCity]; e < rowPtr[intermediateCity + 1]; ++e) {
                        if (colIdx[e] == twoHopCity) {
                            secondHopDist = weights[e];
                            break;
                        }
                    }
                    
                    travelDist += secondHopDist;
                    fullPath[fullPathLen++] = twoHopCity;
                    
                    // Check max distance again if elderly still with the group
                    if (!elderlyDropped && remainingElder > 0 && travelDist > maxDistElder) {
                        // Drop elderly at intermediate city (backtrack)
                        drops[tid * MAX_DROPS * 3 + dropIdx * 3 + 0] = intermediateCity;
                        drops[tid * MAX_DROPS * 3 + dropIdx * 3 + 1] = 0;
                        drops[tid * MAX_DROPS * 3 + dropIdx * 3 + 2] = remainingElder;
                        dropIdx++;
                        elderlyDropped = true;
                        
                        // Only prime age people continue to final destination
                        if (remainingPrime > 0) {
                            drops[tid * MAX_DROPS * 3 + dropIdx * 3 + 0] = twoHopCity;
                            drops[tid * MAX_DROPS * 3 + dropIdx * 3 + 1] = remainingPrime;
                            drops[tid * MAX_DROPS * 3 + dropIdx * 3 + 2] = 0;
                            dropIdx++;
                            remainingPrime = 0;
                        }
                    } else {
                        // Drop everyone (or just prime if elderly already dropped) at final destination
                        if (!elderlyStoppedAtIntermediate && remainingElder > 0) {
                            drops[tid * MAX_DROPS * 3 + dropIdx * 3 + 0] = twoHopCity;
                            drops[tid * MAX_DROPS * 3 + dropIdx * 3 + 1] = remainingPrime;
                            drops[tid * MAX_DROPS * 3 + dropIdx * 3 + 2] = remainingElder;
                            dropIdx++;
                            remainingPrime = 0;
                            remainingElder = 0;
                        } else if (remainingPrime > 0) {
                            drops[tid * MAX_DROPS * 3 + dropIdx * 3 + 0] = twoHopCity;
                            drops[tid * MAX_DROPS * 3 + dropIdx * 3 + 1] = remainingPrime;
                            drops[tid * MAX_DROPS * 3 + dropIdx * 3 + 2] = 0;
                            dropIdx++;
                            remainingPrime = 0;
                        }
                    }
                } else {
                    // Last resort: Use the current city even if it's a shelter
                    drops[tid * MAX_DROPS * 3 + dropIdx * 3 + 0] = currentCity;
                    drops[tid * MAX_DROPS * 3 + dropIdx * 3 + 1] = remainingPrime;
                    drops[tid * MAX_DROPS * 3 + dropIdx * 3 + 2] = remainingElder;
                    dropIdx++;
                    remainingPrime = 0;
                    remainingElder = 0;
                }
            }
        } else {
            // If current city is not a shelter, just drop everyone there
            drops[tid * MAX_DROPS * 3 + dropIdx * 3 + 0] = currentCity;
            drops[tid * MAX_DROPS * 3 + dropIdx * 3 + 1] = remainingPrime;
            drops[tid * MAX_DROPS * 3 + dropIdx * 3 + 2] = remainingElder;
            dropIdx++;
            remainingPrime = 0;
            remainingElder = 0;
        }
    }

    // Debug check - this should never happen now
    if (remainingPrime > 0 || remainingElder > 0) {
        // Emergency drop at start city as absolute last resort
        drops[tid * MAX_DROPS * 3 + dropIdx * 3 + 0] = start;
        drops[tid * MAX_DROPS * 3 + dropIdx * 3 + 1] = remainingPrime;
        drops[tid * MAX_DROPS * 3 + dropIdx * 3 + 2] = remainingElder;
        dropIdx++;
    }

    // Write final path length and actual path taken
    pathLens[tid] = fullPathLen;
    for (int i = 0; i < fullPathLen; ++i) {
        paths[tid * MAX_PATH + i] = fullPath[i];
    }
    
    numDrops[tid] = dropIdx;
    
    // Mark this population as processed
    processedPop[tid] = true;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    // Open output file
    ofstream outFile("output.txt");
    if (!outFile.is_open()) {
        cerr << "Error: Could not open output.txt for writing!" << endl;
        return 1;
    }
    
    int numCities, numEdges;
    cin >> numCities >> numEdges;

    vector<vector<pair<int, ll>>> adj(numCities);
    for (int i = 0; i < numEdges; ++i) {
        int u, v; ll l, c;
        cin >> u >> v >> l >> c;
        adj[u].push_back({v, l});
        adj[v].push_back({u, l});
    }

    vector<int> rowPtr(numCities + 1), colIdx;
    vector<ll> weights;
    int ptr = 0;
    for (int u = 0; u < numCities; ++u) {
        rowPtr[u] = ptr;
        for (auto& [v, w] : adj[u]) {
            colIdx.push_back(v);
            weights.push_back(w);
            ++ptr;
        }
    }
    rowPtr[numCities] = ptr;

    int numShel;
    cin >> numShel;
    vector<int> shelCity(numShel);
    vector<unsigned long long> shelCap(numShel);
    for (int i = 0; i < numShel; ++i)
        cin >> shelCity[i] >> shelCap[i];

    int numPop;
    cin >> numPop;
    vector<int> popCity(numPop);
    vector<ll> primeInit(numPop), elderInit(numPop);
    for (int i = 0; i < numPop; ++i)
        cin >> popCity[i] >> primeInit[i] >> elderInit[i];

    ll maxDistElder;
    cin >> maxDistElder;

    // Output vectors for results
    vector<ll> pathLens(numPop, 0);
    vector<ll> paths(numPop * MAX_PATH, 0);
    vector<int> numDrops(numPop, 0);
    vector<ll> drops(numPop * MAX_DROPS * 3, 0);
    vector<bool> processedPop(numPop, false);

    // Device allocations for static data (shared across all batches)
    int *d_rowPtr, *d_colIdx, *d_popCity, *d_shelCity;
    ll *d_weights, *d_primeInit, *d_elderInit;
    unsigned long long *d_shelCap;
    
    // Device allocations for output data
    ll *d_pathLens, *d_paths;
    int *d_numDrops;
    ll *d_drops;
    bool *d_processedPop;
    
    auto start = std::chrono::high_resolution_clock::now();

    cudaMalloc(&d_rowPtr, sizeof(int) * (numCities + 1));
    cudaMalloc(&d_colIdx, sizeof(int) * colIdx.size());
    cudaMalloc(&d_weights, sizeof(ll) * weights.size());
    cudaMemcpy(d_rowPtr, rowPtr.data(), sizeof(int) * (numCities + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, colIdx.data(), sizeof(int) * colIdx.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights.data(), sizeof(ll) * weights.size(), cudaMemcpyHostToDevice);

    cudaMalloc(&d_popCity, sizeof(int) * numPop);
    cudaMalloc(&d_primeInit, sizeof(ll) * numPop);
    cudaMalloc(&d_elderInit, sizeof(ll) * numPop);
    cudaMemcpy(d_popCity, popCity.data(), sizeof(int) * numPop, cudaMemcpyHostToDevice);
    cudaMemcpy(d_primeInit, primeInit.data(), sizeof(ll) * numPop, cudaMemcpyHostToDevice);
    cudaMemcpy(d_elderInit, elderInit.data(), sizeof(ll) * numPop, cudaMemcpyHostToDevice);

    cudaMalloc(&d_shelCity, sizeof(int) * numShel);
    cudaMalloc(&d_shelCap, sizeof(unsigned long long) * numShel);
    cudaMemcpy(d_shelCity, shelCity.data(), sizeof(int) * numShel, cudaMemcpyHostToDevice);
    cudaMemcpy(d_shelCap, shelCap.data(), sizeof(unsigned long long) * numShel, cudaMemcpyHostToDevice);

    cudaMalloc(&d_pathLens, sizeof(ll) * numPop);
    cudaMalloc(&d_paths, sizeof(ll) * numPop * MAX_PATH);
    cudaMalloc(&d_numDrops, sizeof(int) * numPop);
    cudaMalloc(&d_drops, sizeof(ll) * numPop * MAX_DROPS * 3);
    cudaMalloc(&d_processedPop, sizeof(bool) * numPop);
    
    // Initialize output arrays with zeros
    cudaMemset(d_pathLens, 0, sizeof(ll) * numPop);
    cudaMemset(d_paths, 0, sizeof(ll) * numPop * MAX_PATH);
    cudaMemset(d_numDrops, 0, sizeof(int) * numPop);
    cudaMemset(d_drops, 0, sizeof(ll) * numPop * MAX_DROPS * 3);
    cudaMemset(d_processedPop, 0, sizeof(bool) * numPop);

    // Process cities in batches
    for (int batchOffset = 0; batchOffset < numCities; batchOffset += BATCH_SIZE) {
        int numCitiesInBatch = min(BATCH_SIZE, numCities - batchOffset);
        
        // Also print to console for progress monitoring
        cout << "Processing batch: Cities " << batchOffset << " to " << (batchOffset + numCitiesInBatch - 1) << endl;
        
        // Only launch kernel if there are cities to process in this batch
        if (numCitiesInBatch > 0) {
            evacuationKernel<<<(numPop + 255) / 256, 256>>>(
                batchOffset,
                numCities,
                numCitiesInBatch,
                d_rowPtr,
                d_colIdx,
                d_weights,
                numShel,
                d_shelCity,
                d_shelCap,
                numPop,
                d_popCity,
                d_primeInit,
                d_elderInit,
                maxDistElder,
                d_pathLens,
                d_paths,
                d_numDrops,
                d_drops,
                d_processedPop
            );
            cudaDeviceSynchronize();
        }
    }

    // Copy results back to host
    cudaMemcpy(pathLens.data(), d_pathLens, sizeof(ll) * numPop, cudaMemcpyDeviceToHost);
    cudaMemcpy(paths.data(), d_paths, sizeof(ll) * numPop * MAX_PATH, cudaMemcpyDeviceToHost);
    cudaMemcpy(numDrops.data(), d_numDrops, sizeof(int) * numPop, cudaMemcpyDeviceToHost);
    cudaMemcpy(drops.data(), d_drops, sizeof(ll) * numPop * MAX_DROPS * 3, cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end - start;

    // Write execution time to both console and file
    std::cout << "Total execution time: " << duration_ms.count() << " ms\n";

    // Output results to file
    vector<string> pathStrings;
vector<string> dropStrings;

// Format paths
outFile << "path_sizes = [";
for (int i = 0; i < numPop; ++i) {
    outFile << pathLens[i];
    if (i < numPop - 1) outFile << ", ";
}
outFile << "]\n";

outFile << "paths = [";
for (int i = 0; i < numPop; ++i) {
    outFile << "[";
    for (int j = 0; j < pathLens[i]; ++j) {
        outFile << paths[i * MAX_PATH + j];
        if (j < pathLens[i] - 1) outFile << ", ";
    }
    outFile << "]";
    if (i < numPop - 1) outFile << ", ";
}
outFile << "]\n";

// Format drop counts
outFile << "num_drops = [";
for (int i = 0; i < numPop; ++i) {
    outFile << numDrops[i];
    if (i < numPop - 1) outFile << ", ";
}
outFile << "]\n";

// Format drops
outFile << "drops = [";
for (int i = 0; i < numPop; ++i) {
    outFile << "[";
    for (int j = 0; j < numDrops[i]; ++j) {
        ll cityId = drops[i * MAX_DROPS * 3 + j * 3 + 0];
        ll primeCount = drops[i * MAX_DROPS * 3 + j * 3 + 1];
        ll elderCount = drops[i * MAX_DROPS * 3 + j * 3 + 2];
        outFile << "(" << cityId << ", " << primeCount << ", " << elderCount << ")";
        if (j < numDrops[i] - 1) outFile << ", ";
    }
    outFile << "]";
    if (i < numPop - 1) outFile << ", ";
}
outFile << "]\n";

// Close the output file
outFile.close();
    
    // Free device memory
    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_weights);
    cudaFree(d_popCity);
    cudaFree(d_primeInit);
    cudaFree(d_elderInit);
    cudaFree(d_shelCity);
    cudaFree(d_shelCap);
    cudaFree(d_pathLens);
    cudaFree(d_paths);
    cudaFree(d_numDrops);
    cudaFree(d_drops);
    cudaFree(d_processedPop);
    
    return 0;
}
