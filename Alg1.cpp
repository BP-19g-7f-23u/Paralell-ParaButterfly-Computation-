#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <functional>

using namespace std;

// Structure to represent a processed graph
struct ProcessedGraph {
    vector<int> vertices;                  // Sorted vertices
    vector<vector<int>> adjacencyList;     // Sorted adjacency lists
    vector<int> degreeU;                   // Degree in U for each vertex
    vector<int> degreeV;                   // Degree in V for each vertex
};

// Preprocessing algorithm for butterfly counting
ProcessedGraph preprocess(const vector<pair<int, int>>& edges, function<int(int)> rankFunction) {
    // Step 1: Extract all unique vertices from edges
    unordered_map<int, bool> vertexMap;
    for (const auto& edge : edges) {
        vertexMap[edge.first] = true;
        vertexMap[edge.second] = true;
    }
    
    // Convert to vector for sorting
    vector<int> allVertices;
    for (const auto& entry : vertexMap) {
        allVertices.push_back(entry.first);
    }
    
    // Step 2: Sort vertices based on the ranking function (Line 2)
    sort(allVertices.begin(), allVertices.end(), 
        [&rankFunction](int a, int b) { return rankFunction(a) < rankFunction(b); });
    
    // Step 3: Create mapping from original vertex IDs to ranks (Line 3)
    unordered_map<int, int> vertexToRank;
    for (int i = 0; i < allVertices.size(); i++) {
        vertexToRank[allVertices[i]] = i;
    }
    
    // Step 4: Rename edges using vertex ranks (Line 4)
    vector<pair<int, int>> rankedEdges;
    for (const auto& edge : edges) {
        rankedEdges.push_back({vertexToRank[edge.first], vertexToRank[edge.second]});
    }
    
    // Step 5: Create the adjacency list for the processed graph
    int n = allVertices.size();
    vector<vector<int>> adjacencyList(n);
    vector<int> degreeU(n, 0);
    vector<int> degreeV(n, 0);
    
    // Add edges to adjacency lists
    for (const auto& edge : rankedEdges) {
        int u = edge.first;
        int v = edge.second;
        adjacencyList[u].push_back(v);
        adjacencyList[v].push_back(u);
        
        // Update degrees (line 8)
        degreeU[u]++;
        degreeV[v]++;
    }
    
    // Step 6: Sort neighbors in each adjacency list (parallel for loop in line 6-7)
    // In C++, we'll use a sequential for loop, but in a real implementation
    // this could be parallelized using OpenMP or similar
    #pragma omp parallel for
    for (int u = 0; u < n; u++) {
        // Sort neighbors by their rank (decreasing order as requested in line 7)
        sort(adjacencyList[u].begin(), adjacencyList[u].end(), std::greater<int>());
    }
    
    // Step 7: Return the processed graph
    ProcessedGraph result;
    result.vertices = allVertices;
    result.adjacencyList = adjacencyList;
    result.degreeU = degreeU;
    result.degreeV = degreeV;
    
    return result;
}

// Example ranking function (can be customized)
int degreeRank(int vertex, const unordered_map<int, vector<int>>& graph) {
    if (graph.find(vertex) != graph.end()) {
        return graph.at(vertex).size();  // Rank by degree
    }
    return 0;
}

int main() {
    // Example graph
    vector<pair<int, int>> edges = {
        {1, 6}, {1, 7}, {1, 8},
        {2, 6}, {2, 7}, {2, 8},
        {3, 7}, {3, 8}, {3, 9},
        {4, 8}, {4, 9}, {4, 10},
        {5, 9}, {5, 10}
    };
    
    // Build original graph for degree calculation
    unordered_map<int, vector<int>> originalGraph;
    for (const auto& edge : edges) {
        originalGraph[edge.first].push_back(edge.second);
        originalGraph[edge.second].push_back(edge.first);
    }
    
    // Create ranking function using the degree information
    auto rankFunc = [&originalGraph](int v) { return degreeRank(v, originalGraph); };
    
    // Perform preprocessing
    ProcessedGraph processedGraph = preprocess(edges, rankFunc);
    
    // Print results
    cout << "Sorted Vertices (by rank):" << endl;
    for (int i = 0; i < processedGraph.vertices.size(); i++) {
        cout << "Rank " << i << ": Original Vertex " << processedGraph.vertices[i] 
             << " (Degree: " << originalGraph[processedGraph.vertices[i]].size() << ")" << endl;
    }
    
    cout << "\nSorted Adjacency Lists:" << endl;
    for (int i = 0; i < processedGraph.adjacencyList.size(); i++) {
        cout << "Vertex with rank " << i << " (Original: " << processedGraph.vertices[i] << "): ";
        for (int neighbor : processedGraph.adjacencyList[i]) {
            cout << neighbor << " ";
        }
        cout << endl;
    }
    
    cout << "\nDegree Information:" << endl;
    for (int i = 0; i < processedGraph.vertices.size(); i++) {
        cout << "Vertex with rank " << i << " (Original: " << processedGraph.vertices[i] 
             << "): U-degree = " << processedGraph.degreeU[i] 
             << ", V-degree = " << processedGraph.degreeV[i] << endl;
    }
    
    return 0;
}
