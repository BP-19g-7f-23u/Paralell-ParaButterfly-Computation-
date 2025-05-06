#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <tuple>
#include <numeric>
#include <omp.h>

using namespace std;

// Graph Representation
class BipartiteGraph {
public:
    int num_vertices;
    vector<vector<int>> adj_list; // Adjacency list for each vertex
    
    BipartiteGraph(int vertices) : num_vertices(vertices) {
        adj_list.resize(vertices);
    }
    
    // Add an edge to the graph
    void addEdge(int u, int v) {
        adj_list[u].push_back(v);
        adj_list[v].push_back(u);
    }
    
    const vector<int>& neighbors(int u) const {
        return adj_list[u];
    }
    
    int degree(int u) const {
        return adj_list[u].size();
    }
};

// Structure to represent a wedge
struct Wedge {
    int u1;      // First endpoint
    int u2;      // Second endpoint
    int center;  // Center vertex (v)
    int weight;  // Weight (1 in the algorithm)
    
    Wedge(int _u1, int _u2, int _center, int _weight = 1) 
        : u1(_u1), u2(_u2), center(_center), weight(_weight) {}
    
    Wedge() : u1(-1), u2(-1), center(-1), weight(0) {} // Default constructor
};

// Function to compute prefix sum for wedge indexing
vector<int> computePrefixSum(const vector<int>& values) {
    vector<int> prefixSum(values.size() + 1, 0);
    for (int i = 0; i < values.size(); i++) {
        prefixSum[i + 1] = prefixSum[i] + values[i];
    }
    return prefixSum;
}

// Calculate total wedges per vertex
vector<int> calculateWedgesPerVertex(const BipartiteGraph& graph) {
    int n = graph.num_vertices;
    vector<int> wedgesPerVertex(n, 0);
    
    for (int u1 = 0; u1 < n; u1++) {
        int totalWedges = 0;
        for (int v : graph.neighbors(u1)) {
            // For each neighbor v, count u2 where u1 < u2
            for (int u2 : graph.neighbors(v)) {
                if (u1 < u2) {
                    totalWedges++;
                }
            }
        }
        wedgesPerVertex[u1] = totalWedges;
    }
    
    return wedgesPerVertex;
}

// Parallel wedge retrieval using OpenMP tasks
vector<Wedge> getWedgesUsingTasks(const BipartiteGraph& graph) {
    int n = graph.num_vertices;
    
    // Step 1: Calculate wedges per vertex
    vector<int> wedgesPerVertex = calculateWedgesPerVertex(graph);
    
    // Step 2: Compute prefix sum for wedge indexing
    vector<int> wedgeIndexPrefix = computePrefixSum(wedgesPerVertex);
    int totalWedges = wedgeIndexPrefix.back();
    
    // Step 3: Initialize wedge array
    vector<Wedge> wedges(totalWedges);
    
    // Step 4-10: Use OpenMP tasks for parallelism
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int u1 = 0; u1 < n; u1++) {
                #pragma omp task
                {
                    // Process this vertex
                    int baseIndex = wedgeIndexPrefix[u1];
                    int currentIndex = baseIndex;
                    
                    for (int v : graph.neighbors(u1)) {
                        for (int u2 : graph.neighbors(v)) {
                            // Ensure u1 < u2 to avoid counting the same wedge twice
                            if (u1 < u2) {
                                // Place wedge at the calculated index
                                wedges[currentIndex++] = Wedge(u1, u2, v);
                            }
                        }
                    }
                }
            }
        }
    }
    
    return wedges;
}

// Count butterflies from wedges
int countButterflies(const vector<Wedge>& wedges) {
    // Map to count wedges between endpoint pairs
    unordered_map<pair<int, int>, int, pair_hash> endpointCount;
    
    // Count wedges with same endpoints
    for (const auto& wedge : wedges) {
        pair<int, int> endpoints = make_pair(wedge.u1, wedge.u2);
        endpointCount[endpoints]++;
    }
    
    // Count butterflies
    int butterflyCount = 0;
    for (const auto& pair : endpointCount) {
        int count = pair.second;
        // Each pair of wedges with the same endpoints forms a butterfly
        butterflyCount += (count * (count - 1)) / 2;
    }
    
    return butterflyCount;
}

// Custom hash function for pair
struct pair_hash {
    template <class T1, class T2>
    size_t operator()(const pair<T1, T2>& p) const {
        auto h1 = hash<T1>{}(p.first);
        auto h2 = hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

int main() {
    // Example graph initialization
    BipartiteGraph graph(6);  // 6 vertices in the graph
    
    // Add edges to the graph (creating butterflies)
    graph.addEdge(0, 2);
    graph.addEdge(0, 3);
    graph.addEdge(1, 2);
    graph.addEdge(1, 3);
    graph.addEdge(2, 4);
    graph.addEdge(3, 4);
    graph.addEdge(3, 5);
    graph.addEdge(4, 5);
    
    // Retrieve wedges using OpenMP tasks
    vector<Wedge> wedges = getWedgesUsingTasks(graph);
    
    // Count butterflies from wedges
    int butterflyCount = countButterflies(wedges);
    
    // Output the results
    cout << "Total wedges: " << wedges.size() << endl;
    cout << "Total butterflies: " << butterflyCount << endl;
    
    // Print some wedges for verification
    cout << "\nSample wedges (u1, u2, center):" << endl;
    for (int i = 0; i < min(10, (int)wedges.size()); i++) {
        cout << "(" << wedges[i].u1 << ", " << wedges[i].u2 << ", " << wedges[i].center << ")" << endl;
    }
    
    return 0;
}
