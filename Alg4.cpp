#include <iostream>
#include <vector>
#include <unordered_map>
#include <tuple>
#include <omp.h>

using namespace std;

// Function to aggregate wedge frequencies (GET-FREQ)
pair<vector<pair<pair<int, int>, int>>, vector<int>> getFreq(const vector<tuple<int, int, int>>& wedges) {
    unordered_map<pair<int, int>, int> freqMap;
    for (const auto& wedge : wedges) {
        int u1 = get<0>(wedge);
        int u2 = get<1>(wedge);
        freqMap[{u1, u2}]++;
    }

    // Convert the map to vectors
    vector<pair<pair<int, int>, int>> R;
    vector<int> F;
    int index = 0;
    for (const auto& entry : freqMap) {
        R.push_back(entry);
        F.push_back(index);
        index++;
    }
    F.push_back(R.size()); // The final value for F to indicate the end of the list.

    return {R, F};
}

// Function to retrieve wedges (GET-WEDGES)
vector<tuple<int, int, int>> getWedges(const vector<vector<int>>& adjacencyList) {
    vector<tuple<int, int, int>> wedges;

    #pragma omp parallel for
    for (int u1 = 0; u1 < adjacencyList.size(); u1++) {
        for (int i = 0; i < adjacencyList[u1].size(); i++) {
            int v = adjacencyList[u1][i];
            for (int j = 0; j < adjacencyList[v].size(); j++) {
                int u2 = adjacencyList[v][j];
                if (u1 != u2) {
                    // Store the wedge (u1, u2, v)
                    wedges.push_back(make_tuple(u1, u2, v));
                }
            }
        }
    }

    return wedges;
}

// Count butterflies per edge using the wedges
vector<int> countE_Wedges(const vector<tuple<int, int, int>>& wedges) {
    auto [R, F] = getFreq(wedges); // Retrieve frequencies of wedges
    vector<int> B(R.size(), 0);    // Butterfly counts per edge

    // Step 4: Parallel for each frequency in R
    #pragma omp parallel for
    for (int i = 0; i < R.size(); i++) {
        auto [u1_u2, d] = R[i]; // u1, u2, and d (wedge frequency)
        int u1 = u1_u2.first;
        int u2 = u1_u2.second;

        // Step 6: Parallel for each wedge center
        for (int j = F[i]; j < F[i + 1]; j++) {
            int v = get<2>(wedges[j]); // v is the wedge center

            #pragma omp critical
            {
                B[u1]--;
                B[u2]--;
            }
        }
    }

    return B;
}

int main() {
    // Example graph with vertices and edges
    int n = 6;
    vector<vector<int>> adjacencyList(n);

    // Add edges (bipartite graph: vertices in U and V)
    adjacencyList[0] = {1, 2}; // Vertex 0 is connected to 1, 2
    adjacencyList[1] = {0, 3}; // Vertex 1 is connected to 0, 3
    adjacencyList[2] = {0, 3}; // Vertex 2 is connected to 0, 3
    adjacencyList[3] = {1, 2, 4}; // Vertex 3 is connected to 1, 2, 4
    adjacencyList[4] = {3, 5}; // Vertex 4 is connected to 3, 5
    adjacencyList[5] = {4}; // Vertex 5 is connected to 4

    // Retrieve wedges from the graph
    vector<tuple<int, int, int>> wedges = getWedges(adjacencyList);

    // Count butterflies per edge
    vector<int> edgeButterflies = countE_Wedges(wedges);
    cout << "Butterflies per edge:" << endl;
    for (int i = 0; i < edgeButterflies.size(); i++) {
        cout << "Edge " << i << ": " << edgeButterflies[i] << " butterflies" << endl;
    }

    return 0;
}

