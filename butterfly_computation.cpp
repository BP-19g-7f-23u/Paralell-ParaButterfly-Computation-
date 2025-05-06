#include<iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <map>
#include <set>
#include <algorithm>
#include <utility>
#include <tuple>
#include <string>
#include <sstream>
#include <chrono>
#include <omp.h>
#include <cmath>
#include <queue>
#include <memory>
#include <functional>
#include <random>
#include <atomic>
#include <iomanip>
using namespace std ;
// Custom hash function for pairs
struct pair_hash {
    template <class T1, class T2>
    size_t operator()(const pair<T1, T2>& p) const {
        auto h1 = hash<T1>{}(p.first);
        auto h2 = hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

// Bipartite graph class to represent author-paper relationships
class BipartiteGraph {
private:
    int numAuthors;
    int numPapers;
    int numEdges;
    vector<vector<int>> authorAdj; // Authors to papers
    vector<vector<int>> paperAdj;  // Papers to authors

public:
    BipartiteGraph(int authors, int papers) : numAuthors(authors), numPapers(papers), numEdges(0) {
        authorAdj.resize(authors);
        paperAdj.resize(papers);
    }
    void addEdge(int author, int paper) {
        if (author >= 0 && author < numAuthors && paper >= 0 && paper < numPapers) {
            authorAdj[author].push_back(paper);
            paperAdj[paper].push_back(author);
            numEdges++;
        }
    }

    int getNumAuthors() const { return numAuthors; }
    int getNumPapers() const { return numPapers; }
    int getNumEdges() const { return numEdges; }

    const vector<int>& getAuthorAdjList(int author) const {
        return authorAdj[author];
    }

    const vector<int>& getPaperAdjList(int paper) const {
        return paperAdj[paper];
    }

    // Extract subgraph based on author partition
    BipartiteGraph getSubgraph(const vector<int>& authorPartition) {
        BipartiteGraph subgraph(numAuthors, numPapers);
        for (int author : authorPartition) {
            for (int paper : authorAdj[author]) {
                subgraph.addEdge(author, paper);
            }
        }
        return subgraph;
    }

    // Reorder vertices for better cache locality
    void reorderVertices(const vector<int>& authorOrder, const vector<int>& paperOrder) {
        vector<vector<int>> newAuthorAdj(numAuthors);
        vector<vector<int>> newPaperAdj(numPapers);
        vector<int> authorMap(numAuthors);
        vector<int> paperMap(numPapers);

        for (int i = 0; i < numAuthors; ++i) authorMap[authorOrder[i]] = i;
        for (int i = 0; i < numPapers; ++i) paperMap[paperOrder[i]] = i;

        for (int oldAuthor = 0; oldAuthor < numAuthors; ++oldAuthor) {
            int newAuthor = authorMap[oldAuthor];
            for (int oldPaper : authorAdj[oldAuthor]) {
                int newPaper = paperMap[oldPaper];
                newAuthorAdj[newAuthor].push_back(newPaper);
                newPaperAdj[newPaper].push_back(newAuthor);
            }
        }

        authorAdj = move(newAuthorAdj);
        paperAdj = move(newPaperAdj);
    }
};

// Utility function to load bipartite graph from edge list file
BipartiteGraph loadGraph(const string& filename, int numAuthors, int numPapers) {
    BipartiteGraph graph(numAuthors, numPapers);
    ifstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }
    
    int author, paper;
    while (file >> author >> paper) {
        graph.addEdge(author, paper);
    }
    
    file.close();
    return graph;
}

// Load partitioning information from METIS output
vector<vector<int>> loadPartitioning(const string& filename, int numAuthors, int numParts) {
    vector<vector<int>> partitions(numParts);
    ifstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Error opening partitioning file: " << filename << endl;
        exit(1);
    }
    
    int part;
    for (int i = 0; i < numAuthors; i++) {
        file >> part;
        partitions[part].push_back(i);
    }
    
    file.close();
    return partitions;
}

// Available ranking methods for vertex ordering
enum class RankingMethod {
    SIDE_ORDER,
    DEGREE_ORDER,
    APPROXIMATE_DEGREE_ORDER,
    COMPLEMENT_DEGENERACY_ORDER,
    APPROXIMATE_COMPLEMENT_DEGENERACY_ORDER
};

// Available methods for wedge aggregation
enum class AggregationMethod {
    SORTING,
    HASHING,
    HISTOGRAMMING,
    SIMPLE_BATCHING,
    WEDGE_AWARE_BATCHING
};

// Wedge structure for representing 2-paths
struct Wedge {
    int endpoint1;
    int endpoint2;
    int center;
    
    Wedge(int e1, int e2, int c) : endpoint1(e1), endpoint2(e2), center(c) {}
};

// ----- Algorithm 3: Parallel Butterfly Counting Per Vertex -----

// Compute vertex ranking based on selected method
vector<int> computeVertexRanking(
    const BipartiteGraph& graph, 
    RankingMethod rankMethod) {
    
    int numAuthors = graph.getNumAuthors();
    vector<int> ranking(numAuthors);
    
    // Initialize ranking with vertex IDs
    for (int i = 0; i < numAuthors; i++) {
        ranking[i] = i;
    }

    switch (rankMethod) {
        case RankingMethod::SIDE_ORDER:
            // No sorting needed, keep vertices in natural order
            break;
            
        case RankingMethod::DEGREE_ORDER:
            // Sort by decreasing order of degree
            sort(ranking.begin(), ranking.end(), 
                [&](int a, int b) {
                    return graph.getAuthorAdjList(a).size() > 
                           graph.getAuthorAdjList(b).size();
                });
            break;
            
        case RankingMethod::APPROXIMATE_DEGREE_ORDER:
            // Sort by log-degree
            sort(ranking.begin(), ranking.end(), 
                [&](int a, int b) {
                    int degA = graph.getAuthorAdjList(a).size();
                    int degB = graph.getAuthorAdjList(b).size();
                    int logA = degA > 0 ? (int)log2(degA) : 0;
                    int logB = degB > 0 ? (int)log2(degB) : 0;
                    return logA > logB || (logA == logB && a < b);
                });
            break;
            
        case RankingMethod::COMPLEMENT_DEGENERACY_ORDER: {
            // Implement complement degeneracy ordering
            vector<int> degrees(numAuthors);
            vector<bool> removed(numAuthors, false);
            
            // Calculate initial degrees
            for (int i = 0; i < numAuthors; i++) {
                degrees[i] = graph.getAuthorAdjList(i).size();
            }
            
            // Compute ordering by repeatedly removing highest degree vertex
            for (int i = 0; i < numAuthors; i++) {
                // Find vertex with maximum degree
                int maxVertex = -1;
                int maxDegree = -1;
                
                #pragma omp parallel
                {
                    int localMaxVertex = -1;
                    int localMaxDegree = -1;
                    
                    #pragma omp for schedule(dynamic, 1000)
                    for (int v = 0; v < numAuthors; v++) {
                        if (!removed[v] && degrees[v] > localMaxDegree) {
                            localMaxDegree = degrees[v];
                            localMaxVertex = v;
                        }
                    }
                    
                    #pragma omp critical
                    {
                        if (localMaxDegree > maxDegree) {
                            maxDegree = localMaxDegree;
                            maxVertex = localMaxVertex;
                        }
                    }
                }
                
                if (maxVertex == -1) break;
                
                // Add to ordering and mark as removed
                ranking[i] = maxVertex;
                removed[maxVertex] = true;
                
                // Update degrees of neighbors
                for (int paper : graph.getAuthorAdjList(maxVertex)) {
                    for (int neighbor : graph.getPaperAdjList(paper)) {
                        if (!removed[neighbor]) {
                            #pragma omp atomic update
                            degrees[neighbor]--;
                        }
                    }
                }
            }
            break;
        }
            
        case RankingMethod::APPROXIMATE_COMPLEMENT_DEGENERACY_ORDER: {
            // Implement approximate complement degeneracy ordering
            vector<int> logDegrees(numAuthors);
            vector<bool> removed(numAuthors, false);
            
            // Calculate initial log-degrees
            for (int i = 0; i < numAuthors; i++) {
                int deg = graph.getAuthorAdjList(i).size();
                logDegrees[i] = deg > 0 ? (int)log2(deg) : 0;
            }
            
            // Compute ordering by repeatedly removing highest log-degree vertex
            for (int i = 0; i < numAuthors; i++) {
                // Find vertex with maximum log-degree
                int maxVertex = -1;
                int maxLogDegree = -1;
                
                #pragma omp parallel
                {
                    int localMaxVertex = -1;
                    int localMaxLogDegree = -1;
                    
                    #pragma omp for schedule(dynamic, 1000)
                    for (int v = 0; v < numAuthors; v++) {
                        if (!removed[v] && logDegrees[v] > localMaxLogDegree) {
                            localMaxLogDegree = logDegrees[v];
                            localMaxVertex = v;
                        }
                    }
                    
                    #pragma omp critical
                    {
                        if (localMaxLogDegree > maxLogDegree) {
                            maxLogDegree = localMaxLogDegree;
                            maxVertex = localMaxVertex;
                        }
                    }
                }
                
                if (maxVertex == -1) break;
                
                // Add to ordering and mark as removed
                ranking[i] = maxVertex;
                removed[maxVertex] = true;
            }
            break;
        }
    }
    
    return ranking;
}

// Retrieve wedges based on vertex ranking
vector<Wedge> retrieveWedges(
    const BipartiteGraph& graph, 
    const vector<int>& ranking) {
    
    int numAuthors = graph.getNumAuthors();
    vector<int> rankMap(numAuthors);
    
    for (int i = 0; i < numAuthors; i++) {
        rankMap[ranking[i]] = i;
    }

    vector<Wedge> wedges;
    
    #pragma omp parallel
    {
        vector<Wedge> localWedges;
        
        #pragma omp for schedule(dynamic, 100)
        for (int i = 0; i < numAuthors; i++) {
            int u = ranking[i];
            
            // Retrieve all wedges where the center and second endpoint have higher rank
            for (int paper : graph.getAuthorAdjList(u)) {
                for (int author2 : graph.getPaperAdjList(paper)) {
                    if (rankMap[author2] > i) { // author2 has higher rank than u
                        localWedges.emplace_back(u, author2, paper);
                    }
                }
            }
        }
        
        #pragma omp critical
        {
            wedges.insert(wedges.end(), localWedges.begin(), localWedges.end());
        }
    }
    
    return wedges;
}

// Aggregate wedges by endpoints using specified method
unordered_map<pair<int, int>, int, pair_hash> aggregateWedges(
    const vector<Wedge>& wedges,
    AggregationMethod method) {
    
    unordered_map<pair<int, int>, int, pair_hash> counts;
    
    switch (method) {
        case AggregationMethod::SORTING: {
            // Sort wedges by endpoints and count
            vector<pair<pair<int, int>, int>> sortedWedges;
            
            #pragma omp parallel
            {
                vector<pair<pair<int, int>, int>> localSorted;
                
                #pragma omp for schedule(static)
                for (size_t i = 0; i < wedges.size(); i++) {
                    const Wedge& w = wedges[i];
                    localSorted.push_back({{w.endpoint1, w.endpoint2}, w.center});
                }
                
                sort(localSorted.begin(), localSorted.end());
                
                #pragma omp critical
                {
                    sortedWedges.insert(sortedWedges.end(), 
                                       localSorted.begin(), 
                                       localSorted.end());
                }
            }
            
            // Now count consecutive identical pairs
            sort(sortedWedges.begin(), sortedWedges.end());
            
            if (!sortedWedges.empty()) {
                auto currentPair = sortedWedges[0].first;
                int count = 1;
                
                for (size_t i = 1; i < sortedWedges.size(); i++) {
                    if (sortedWedges[i].first == currentPair) {
                        count++;
                    } else {
                        counts[currentPair] = count;
                        currentPair = sortedWedges[i].first;
                        count = 1;
                    }
                }
                
                counts[currentPair] = count;
            }
            break;
        }
        
        case AggregationMethod::HASHING: {
            // Use parallel hashing
            #pragma omp parallel
            {
                unordered_map<pair<int, int>, int, pair_hash> localCounts;
                
                #pragma omp for schedule(dynamic, 1000)
                for (size_t i = 0; i < wedges.size(); i++) {
                    const Wedge& w = wedges[i];
                    localCounts[{w.endpoint1, w.endpoint2}]++;
                }
                
                #pragma omp critical
                {
                    for (const auto& pair : localCounts) {
                        counts[pair.first] += pair.second;
                    }
                }
            }
            break;
        }
        
        case AggregationMethod::HISTOGRAMMING: {
            // Similar to hashing but uses histogramming
            const int NUM_BUCKETS = 1024;
            vector<unordered_map<pair<int, int>, int, pair_hash>> 
                localCounts(NUM_BUCKETS);
            
            #pragma omp parallel for schedule(dynamic, 1000)
            for (size_t i = 0; i < wedges.size(); i++) {
                const Wedge& w = wedges[i];
                auto pair = make_pair(w.endpoint1, w.endpoint2);
                size_t hash = pair_hash()(pair) % NUM_BUCKETS;
                
                #pragma omp atomic update
                localCounts[hash][pair]++;
            }
            
            // Merge all histograms
            for (const auto& bucketMap : localCounts) {
                for (const auto& pair : bucketMap) {
                    counts[pair.first] += pair.second;
                }
            }
            break;
        }
        
        case AggregationMethod::SIMPLE_BATCHING: {
            // Process in fixed-size batches
            const int BATCH_SIZE = 128;
            int numVertices = 0;
            
            // Find max vertex ID for proper sizing
            for (const auto& wedge : wedges) {
                numVertices = max(numVertices, max(wedge.endpoint1, wedge.endpoint2) + 1);
            }
            
            vector<int> vertices(numVertices);
            for (int i = 0; i < numVertices; i++) {
                vertices[i] = i;
            }
            
            for (int start = 0; start < numVertices; start += BATCH_SIZE) {
                int end = min(start + BATCH_SIZE, numVertices);
                vector<int> batch(vertices.begin() + start, vertices.begin() + end);
                
                #pragma omp parallel
                {
                    unordered_map<pair<int, int>, int, pair_hash> localCounts;
                    
                    #pragma omp for schedule(dynamic)
                    for (int i = 0; i < batch.size(); i++) {
                        int u = batch[i];
                        
                        // Count wedges with u as an endpoint
                        unordered_map<int, int> neighbors;
                        for (const auto& wedge : wedges) {
                            if (wedge.endpoint1 == u) {
                                neighbors[wedge.endpoint2]++;
                            } else if (wedge.endpoint2 == u) {
                                neighbors[wedge.endpoint1]++;
                            }
                        }
                        
                        // Add wedge counts
                        for (const auto& pair : neighbors) {
                            if (pair.first > u) { // Avoid double counting
                                localCounts[{u, pair.first}] += pair.second;
                            }
                        }
                    }
                    
                    #pragma omp critical
                    {
                        for (const auto& pair : localCounts) {
                            counts[pair.first] += pair.second;
                        }
                    }
                }
            }
            break;
        }
        
        case AggregationMethod::WEDGE_AWARE_BATCHING: {
            // Dynamically adjust batch size based on wedge density
            const int MAX_WEDGES_PER_BATCH = 10000000;
            int numVertices = 0;
            
            // Find max vertex ID and count wedges per vertex
            for (const auto& wedge : wedges) {
                numVertices = max(numVertices, max(wedge.endpoint1, wedge.endpoint2) + 1);
            }
            
            vector<int> vertices(numVertices);
            vector<int> wedgeEstimates(numVertices, 0);
            
            for (int i = 0; i < numVertices; i++) {
                vertices[i] = i;
            }
            
            // Count wedges per vertex
            for (const auto& wedge : wedges) {
                wedgeEstimates[wedge.endpoint1]++;
                wedgeEstimates[wedge.endpoint2]++;
            }
            
            // Create batches based on wedge estimates
            vector<vector<int>> batches;
            vector<int> currentBatch;
            int currentBatchWedges = 0;
            
            for (int i = 0; i < numVertices; i++) {
                if (currentBatchWedges + wedgeEstimates[i] > MAX_WEDGES_PER_BATCH && 
                    !currentBatch.empty()) {
                    batches.push_back(currentBatch);
                    currentBatch.clear();
                    currentBatchWedges = 0;
                }
                
                currentBatch.push_back(i);
                currentBatchWedges += wedgeEstimates[i];
            }
            
            if (!currentBatch.empty()) {
                batches.push_back(currentBatch);
            }
            
            // Process each batch
            for (const auto& batch : batches) {
                #pragma omp parallel
                {
                    unordered_map<pair<int, int>, int, pair_hash> localCounts;
                    
                    #pragma omp for schedule(dynamic)
                    for (size_t i = 0; i < batch.size(); i++) {
                        int u = batch[i];
                        
                        // Count wedges with u as an endpoint
                        unordered_map<int, int> neighbors;
                        for (const auto& wedge : wedges) {
                            if (wedge.endpoint1 == u) {
                                neighbors[wedge.endpoint2]++;
                            } else if (wedge.endpoint2 == u) {
                                neighbors[wedge.endpoint1]++;
                            }
                        }
                        
                        // Add wedge counts
                        for (const auto& pair : neighbors) {
                            if (pair.first > u) { // Avoid double counting
                                localCounts[{u, pair.first}] += pair.second;
                            }
                        }
                    }
                    
                    #pragma omp critical
                    {
                        for (const auto& pair : localCounts) {
                            counts[pair.first] += pair.second;
                        }
                    }
                }
            }
            break;
        }
    }
    
    return counts;
}

// Count butterflies per vertex from wedge counts
vector<long long> countVertexButterflies(
    const unordered_map<pair<int, int>, int, pair_hash>& wedgeCounts,
    const vector<Wedge>& wedges,
    int numVertices) {
    
    vector<long long> butterflyCounts(numVertices, 0);
    
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < wedgeCounts.bucket_count(); i++) {
        for (auto it = wedgeCounts.begin(i); it != wedgeCounts.end(i); ++it) {
            int u = it->first.first;
            int v = it->first.second;
            int count = it->second;
            
            if (count > 1) {
                // Calculate butterfly count from wedge count
                long long butterflies = (count * (count - 1)) / 2;
                
                #pragma omp atomic
                butterflyCounts[u] += butterflies;
                
                #pragma omp atomic
                butterflyCounts[v] += butterflies;
            }
        }
    }
    
    // Count butterflies for center vertices
    unordered_map<int, vector<pair<int, int>>> centerToEndpoints;
    
    for (const auto& wedge : wedges) {
        centerToEndpoints[wedge.center].push_back({wedge.endpoint1, wedge.endpoint2});
    }
    
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < centerToEndpoints.bucket_count(); i++) {
        for (auto it = centerToEndpoints.begin(i); it != centerToEndpoints.end(i); ++it) {
            int center = it->first;
            const auto& endpointPairs = it->second;
            
            unordered_map<pair<int, int>, int, pair_hash> localWedgeCounts;
            for (const auto& pair : endpointPairs) {
                localWedgeCounts[pair]++;
            }
            
            for (const auto& pair : localWedgeCounts) {
                int count = pair.second;
                if (count > 1) {
                    #pragma omp atomic
                    butterflyCounts[center] += (count - 1);
                }
            }
        }
    }
    
    return butterflyCounts;
}

// Algorithm 3: Main function for butterfly counting per vertex
vector<long long> countButterflyPerVertex(
    const BipartiteGraph& graph,
    RankingMethod rankMethod = RankingMethod::DEGREE_ORDER,
    AggregationMethod aggMethod = AggregationMethod::HASHING) {
    
    // Step 1: Compute vertex ranking
    vector<int> ranking = computeVertexRanking(graph, rankMethod);
    
    // Step 2: Retrieve wedges based on ranking
    vector<Wedge> wedges = retrieveWedges(graph, ranking);
    
    // Step 3: Aggregate wedges by endpoints
    auto wedgeCounts = aggregateWedges(wedges, aggMethod);
    
    // Step 4: Count butterflies per vertex
    int maxVertex = max(graph.getNumAuthors(), graph.getNumPapers());
    return countVertexButterflies(wedgeCounts, wedges, maxVertex);
}

// ----- Algorithm 4: Parallel Butterfly Counting Per Edge -----

// Count butterflies per edge from wedge counts
unordered_map<pair<int, int>, int, pair_hash> countEdgeButterflies(
    const unordered_map<pair<int, int>, int, pair_hash>& wedgeCounts,
    const vector<Wedge>& wedges,
    const BipartiteGraph& graph) {
    
    unordered_map<pair<int, int>, int, pair_hash> edgeCounts;
    
    // Create a map from endpoints to wedge centers
    unordered_map<pair<int, int>, vector<int>, pair_hash> endpointsToCenters;
    
    for (const auto& wedge : wedges) {
        endpointsToCenters[{wedge.endpoint1, wedge.endpoint2}].push_back(wedge.center);
    }
    
    #pragma omp parallel
    {
        unordered_map<pair<int, int>, int, pair_hash> localCounts;
        
        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < wedgeCounts.bucket_count(); i++) {
            for (auto it = wedgeCounts.begin(i); it != wedgeCounts.end(i); ++it) {
                auto endpoints = it->first;
                int count = it->second;
                
                if (count <= 1) continue;
                
                // Find all centers (papers) that connect these endpoints
                const auto& centers = endpointsToCenters[endpoints];
                
                for (int center : centers) {
                    // Add butterfly count to the edges that form the wedges
                    localCounts[{min(endpoints.first, center), 
                                max(endpoints.first, center)}] += (count - 1);
                    
                    localCounts[{min(endpoints.second, center), 
                                max(endpoints.second, center)}] += (count - 1);
                }
            }
        }
        
        #pragma omp critical
        {
            for (const auto& pair : localCounts) {
                edgeCounts[pair.first] += pair.second;
            }
        }
    }
    
    return edgeCounts;
}

// Algorithm 4: Main function for butterfly counting per edge
unordered_map<pair<int, int>, int, pair_hash> countButterflyPerEdge(
    const BipartiteGraph& graph,
    RankingMethod rankMethod = RankingMethod::DEGREE_ORDER,
    AggregationMethod aggMethod = AggregationMethod::HASHING) {
    
    // Step 1: Compute vertex ranking
    vector<int> ranking = computeVertexRanking(graph, rankMethod);
    
    // Step 2: Retrieve wedges based on ranking
    vector<Wedge> wedges = retrieveWedges(graph, ranking);
    
    // Step 3: Aggregate wedges by endpoints
    auto wedgeCounts = aggregateWedges(wedges, aggMethod);
    
    // Step 4: Count butterflies per edge
    return countEdgeButterflies(wedgeCounts, wedges, graph);
}

// ----- Algorithm 5: Parallel Vertex Peeling (Tip Decomposition) -----

class BucketingStructure {
private:
    vector<vector<int>> buckets;
    vector<int> position;
    int maxValue;

public:
    // n = number of vertices, maxVal = maximum bucket index
    BucketingStructure(int n, int maxVal)
      : buckets(maxVal + 1), position(n, -1), maxValue(maxVal) {}

    // Place vertex into bucket[level]
    void insert(int vertex, int level) {
        if (level < 0 || level > maxValue) return;
        position[vertex] = level;
        buckets[level].push_back(vertex);
    }

    // Move vertex from its old bucket to new level
    void update(int vertex, int newLevel) {
        int oldLevel = position[vertex];
        if (oldLevel == newLevel ||
            newLevel < 0 || newLevel > maxValue) return;

        auto &oldB = buckets[oldLevel];
        oldB.erase(remove(oldB.begin(), oldB.end(), vertex), oldB.end());

        position[vertex] = newLevel;
        buckets[newLevel].push_back(vertex);
    }

    // Return & remove all vertices in the *first* non-empty bucket
    vector<int> getNextBucket() {
        cout << "Getting next bucket of vertices..." << endl;
        for (int lvl = 0; lvl <= maxValue; ++lvl) {
            if (!buckets[lvl].empty()) {
                auto result = move(buckets[lvl]);
                buckets[lvl].clear();
                return result;
            }
        }
        cout << "No more vertices in the buckets!" << endl;
        return {};
    }

    // Check if all buckets are empty
    bool isEmpty() const {
        for (auto &b : buckets)
            if (!b.empty()) return false;
        return true;
    }
};

// Algorithm 5: Update butterfly counts after peeling vertices
vector<long long> updateV(
    const BipartiteGraph& graph, 
    const vector<long long>& butterflyCounts,
    const vector<int>& peeledVertices,
    AggregationMethod aggMethod = AggregationMethod::HASHING) {
    
    // Initialize wedges array
    vector<Wedge> wedges;
    cout << "Updating butterfly counts after peeling..." << endl;
    
    // Collect wedges from peeled vertices in parallel
    #pragma omp parallel
    {
        vector<Wedge> localWedges;
        
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < peeledVertices.size(); i++) {
            int u1 = peeledVertices[i];
            for (int p : graph.getAuthorAdjList(u1)) {
                for (int u2 : graph.getPaperAdjList(p)) {
                    if (u2 != u1) {
                        localWedges.emplace_back(u1, u2, p);
                    }
                }
            }
        }
        
        // Add local wedges to the global vector in a thread-safe way
        #pragma omp critical
        {
            wedges.insert(wedges.end(), localWedges.begin(), localWedges.end());
        }
    }

    // Debugging: Check the number of wedges collected
    cout << "Number of wedges collected: " << wedges.size() << endl;

    // Aggregate wedges and count butterflies
    auto wedgeCounts = aggregateWedges(wedges, aggMethod);
    
    // Count butterflies to subtract
    int maxVertex = max(graph.getNumAuthors(), graph.getNumPapers());
    vector<long long> deltaButterflyCounts = countVertexButterflies(wedgeCounts, wedges, maxVertex);
    
    // Subtract from original counts to get updated counts
    vector<long long> updatedCounts = butterflyCounts;
    for (size_t i = 0; i < updatedCounts.size(); i++) {
        if (i < deltaButterflyCounts.size()) {
            updatedCounts[i] -= deltaButterflyCounts[i];
            if (updatedCounts[i] < 0) updatedCounts[i] = 0; // Ensure no negative counts
        }
    }
    
    return updatedCounts;
}

// Algorithm 5: Vertex peeling (tip decomposition)
vector<int> peelV(
    const BipartiteGraph& graph, 
    const vector<long long>& initialButterflyCounts,
    AggregationMethod aggMethod = AggregationMethod::HASHING) {
    
    int numAuthors = graph.getNumAuthors();
    
    // Find maximum butterfly count
    long long maxCount = 0;
    for (int i = 0; i < numAuthors; i++) {
        maxCount = max(maxCount, initialButterflyCounts[i]);
    }
    
    // Initialize bucketing structure
    BucketingStructure buckets(numAuthors, maxCount);
    for (int u = 0; u < numAuthors; u++) {
        buckets.insert(u, initialButterflyCounts[u]);
    }
    
    // Initialize result array (tip numbers)
    vector<int> tipNumbers(numAuthors, 0);
    
    // Variables for peeling process
    int peelingRound = 0;
    int processedCount = 0;
    vector<long long> currentCounts = initialButterflyCounts;
    cout << "Starting peeling process..." << endl;

    // Peeling process
    while (processedCount < numAuthors) {
        // Get vertices with minimum butterfly count
        vector<int> toRemove = buckets.getNextBucket();
        if (toRemove.empty()) break;
        
        peelingRound++;
        
        // Set tip numbers for peeled vertices
        for (int v : toRemove) {
            tipNumbers[v] = peelingRound;
        }
        
        processedCount += toRemove.size();
        
        // Update butterfly counts
        currentCounts = updateV(graph, currentCounts, toRemove, aggMethod);
        
        // Update buckets with new counts
        for (int u = 0; u < numAuthors; u++) {
            if (tipNumbers[u] == 0) { // Not yet processed
                buckets.update(u, currentCounts[u]);
            }
        }
    }
    
    return tipNumbers;
    
}

// Bucketing structure for managing vertex levels
class BucketingStructure {
private:
    vector<vector<int>> buckets;
    vector<int> position;
    int maxValue;

public:
    // Constructor: n = number of vertices, maxVal = maximum bucket index
    BucketingStructure(int n, int maxVal)
        : buckets(maxVal + 1), position(n, -1), maxValue(maxVal) {}

    // Place vertex into bucket[level]
    void insert(int vertex, int level) {
        if (level < 0 || level > maxValue) return;
        position[vertex] = level;
        buckets[level].push_back(vertex);
    }

    // Move vertex from its old bucket to new level
    void update(int vertex, int newLevel) {
        int oldLevel = position[vertex];
        if (oldLevel == newLevel || newLevel < 0 || newLevel > maxValue) return;

        // Remove from old bucket
        auto& oldB = buckets[oldLevel];
        oldB.erase(remove(oldB.begin(), oldB.end(), vertex), oldB.end());

        // Add to new bucket
        position[vertex] = newLevel;
        buckets[newLevel].push_back(vertex);
    }

    // Return & remove all vertices in the *first* non-empty bucket
    vector<int> getNextBucket() {
        for (int lvl = 0; lvl <= maxValue; ++lvl) {
            if (!buckets[lvl].empty()) {
                vector<int> result = move(buckets[lvl]);
                buckets[lvl].clear();
                return result;
            }
        }
        return {};  // Return empty vector if no bucket found
    }

    // Check if all buckets are empty
    bool isEmpty() const {
        for (const auto& b : buckets)
            if (!b.empty()) return false;
        return true;
    }
    };


// Compute intersection of two sorted lists
vector<int> intersect(const vector<int>& a, const vector<int>& b) {
    vector<int> result;
    size_t i = 0, j = 0;
    
    while (i < a.size() && j < b.size()) {
        if (a[i] < b[j]) {
            i++;
        } else if (a[i] > b[j]) {
            j++;
        } else {
            result.push_back(a[i]);
            i++;
            j++;
        }
    }
    
    return result;
}

// Algorithm 6: Update butterfly counts after peeling edges
unordered_map<pair<int, int>, int, pair_hash> updateE(
    const BipartiteGraph& graph, 
    const unordered_map<pair<int, int>, int, pair_hash>& butterflyCounts,
    const vector<pair<int, int>>& peeledEdges) {
    
    // Initialize storage for butterfly count updates
    vector<tuple<pair<int, int>, int, int>> updates;
    
    // Process each peeled edge
    #pragma omp parallel
    {
        vector<tuple<pair<int, int>, int, int>> localUpdates;
        
        #pragma omp for schedule(dynamic)
        for (const auto& edge : peeledEdges) {
            int u1 = edge.first;
            int v1 = edge.second;
            
            // Process all neighbors of v1
            for (int u2 : graph.getPaperAdjList(v1)) {
                if (u2 != u1) {
                    // Find intersection of neighbors
                    vector<int> intersection = intersect(
                        graph.getAuthorAdjList(u1),
                        graph.getAuthorAdjList(u2)
                    );
                    
                    // Add update for edge (u2, v1)
                    if (intersection.size() > 1) {
                        localUpdates.emplace_back(
                            make_pair(u2, v1), 
                            intersection.size() - 1, 
                            0
                        );
                    }
                    
                    // Process each vertex in the intersection
                    for (int v2 : intersection) {
                        if (v2 != v1) {
                            // Add updates for edges (u1, v2) and (u2, v2)
                            localUpdates.emplace_back(
                                make_pair(u1, v2), 
                                1, 
                                0
                            );
                            
                            localUpdates.emplace_back(
                                make_pair(u2, v2), 
                                1, 
                                0
                            );
                        }
                    }
                }
            }
        }
        
        #pragma omp critical
        {
            updates.insert(updates.end(), localUpdates.begin(), localUpdates.end());
        }
    }
    
    // Aggregate updates
    unordered_map<pair<int, int>, int, pair_hash> deltaCounts;
    
    for (const auto& update : updates) {
        auto edge = get<0>(update);
        int count = get<1>(update);
        deltaCounts[edge] += count;
    }
    
    // Apply updates to butterfly counts
    unordered_map<pair<int, int>, int, pair_hash> updatedCounts = butterflyCounts;
    
    for (const auto& pair : deltaCounts) {
        auto edge = pair.first;
        int delta = pair.second;
        
        // Apply the update
        auto it = updatedCounts.find(edge);
        if (it != updatedCounts.end()) {
            it->second -= delta;
            if (it->second <= 0) {
                updatedCounts.erase(it);
            }
        }
    }
    
    return updatedCounts;
}

// Algorithm 6: Edge peeling (wing decomposition)
unordered_map<pair<int, int>, int, pair_hash> peelE(
    const BipartiteGraph& graph, 
    const unordered_map<pair<int, int>, int, pair_hash>& initialButterflyCounts) {
    
    // Find maximum butterfly count
    int maxCount = 0;
    for (const auto& pair : initialButterflyCounts) {
        maxCount = max(maxCount, pair.second);
    }
    
    // Initialize bucketing structure
    BucketingStructure buckets(numAuthors, maxCount);   // If you want to use the same class
    for (const auto& pair : initialButterflyCounts) {
        buckets.insert(pair.first, someLevel); // Where `someLevel` is an int (use appropriate logic to get the level)

    }
    
    // Initialize result (wing numbers)
    unordered_map<pair<int, int>, int, pair_hash> wingNumbers;
    
    // Variables for peeling process
    int peelingRound = 0;
    int totalEdges = initialButterflyCounts.size();
    int processedCount = 0;
    unordered_map<pair<int, int>, int, pair_hash> currentCounts = initialButterflyCounts;
    
    // Peeling process
    while (processedCount < totalEdges) {
        // Get edges with minimum butterfly count
        vector<pair<int, int>> toRemove = buckets.getNextBucket();
        if (toRemove.empty()) break;
        
        peelingRound++;
        
        // Set wing numbers for peeled edges
        for (const auto& edge : toRemove) {
            wingNumbers[edge] = peelingRound;
        }
        
        processedCount += toRemove.size();
        
        // Update butterfly counts
        currentCounts = updateE(graph, currentCounts, toRemove);
        
        // Update buckets with new counts
        for (const auto& pair : currentCounts) {
            if (wingNumbers.find(pair.first) == wingNumbers.end()) { // Not yet processed
                buckets.update(pair.first, pair.second);
            }
        }
    }
    
    return wingNumbers;
}

// ----- Utility functions for counting total butterflies -----

// Count total butterflies in the graph
long long countTotalButterflies(const BipartiteGraph& graph, 
                              RankingMethod rankMethod = RankingMethod::DEGREE_ORDER,
                              AggregationMethod aggMethod = AggregationMethod::HASHING) {
    
    // Step 1: Compute vertex ranking
    vector<int> ranking = computeVertexRanking(graph, rankMethod);
    
    // Step 2: Retrieve wedges based on ranking
    vector<Wedge> wedges = retrieveWedges(graph, ranking);
    
    // Step 3: Aggregate wedges by endpoints
    auto wedgeCounts = aggregateWedges(wedges, aggMethod);
    
    // Step 4: Count total butterflies
    long long total = 0;
    
    for (const auto& pair : wedgeCounts) {
        int count = pair.second;
        if (count > 1) {
            total += (count * (count - 1)) / 2;
        }
    }
    
    return total;
}

// Graph sparsification for approximate counting
BipartiteGraph sparsifyGraph(const BipartiteGraph& graph, double probability) {
    BipartiteGraph sparsified(graph.getNumAuthors(), graph.getNumPapers());
    
    // Edge sparsification
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);
    
    for (int u = 0; u < graph.getNumAuthors(); u++) {
        for (int v : graph.getAuthorAdjList(u)) {
            if (dis(gen) < probability) {
                sparsified.addEdge(u, v);
            }
        }
    }
    
    return sparsified;
}

// Colorful sparsification for approximate butterfly counting
long long approximateCountColorful(
    const BipartiteGraph& graph, 
    int numColors,
    RankingMethod rankMethod = RankingMethod::DEGREE_ORDER,
    AggregationMethod aggMethod = AggregationMethod::HASHING) {
    
    // Assign random colors to vertices
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, numColors - 1);
    
    int maxVertex = max(graph.getNumAuthors(), graph.getNumPapers());
    vector<int> colors(maxVertex);
    for (int i = 0; i < maxVertex; i++) {
        colors[i] = dis(gen);
    }
    
    // Count monochromatic butterflies
    
    // Step 1: Compute vertex ranking
    vector<int> ranking = computeVertexRanking(graph, rankMethod);
    
    // Step 2: Retrieve colored wedges
    vector<Wedge> wedges;
    
    #pragma omp parallel
    {
        vector<Wedge> localWedges;
        
        #pragma omp for schedule(dynamic, 100)
        for (int i = 0; i < graph.getNumAuthors(); i++) {
            int u = ranking[i];
            
            // Only include wedges with matching colors
            for (int paper : graph.getAuthorAdjList(u)) {
                if (colors[u] != colors[paper]) continue;
                
                for (int v : graph.getPaperAdjList(paper)) {
                    if (colors[v] != colors[u]) continue;
                    
                    if (find(ranking.begin(), ranking.begin() + i, v) == ranking.begin() + i) {
                        localWedges.emplace_back(u, v, paper);
                    }
                }
            }
        }
        
        #pragma omp critical
        {
            wedges.insert(wedges.end(), localWedges.begin(), localWedges.end());
        }
    }
    
    // Step 3: Aggregate wedges
    auto wedgeCounts = aggregateWedges(wedges, aggMethod);
    
    // Step 4: Count monochromatic butterflies
    long long monochromaticCount = 0;
    
    for (const auto& pair : wedgeCounts) {
        int count = pair.second;
        if (count > 1) {
            monochromaticCount += (count * (count - 1)) / 2;
        }
    }
    
    // Scale up by color factor
    double scaleFactor = pow(numColors, 4);
    return static_cast<long long>(monochromaticCount * scaleFactor);
}

// Cache-aware optimization
void optimizeForCache(BipartiteGraph& graph) {
    int numAuthors = graph.getNumAuthors();
    int numPapers = graph.getNumPapers();

    vector<int> authorOrder(numAuthors);
    vector<int> paperOrder(numPapers);

    for (int i = 0; i < numAuthors; i++) authorOrder[i] = i;
    for (int i = 0; i < numPapers; i++) paperOrder[i] = i;

    // Sort authors based on first paper to improve locality
    sort(authorOrder.begin(), authorOrder.end(), [&](int a, int b) {
        const auto& la = graph.getAuthorAdjList(a);
        const auto& lb = graph.getAuthorAdjList(b);
        if (la.empty()) return false;
        if (lb.empty()) return true;
        return la[0] < lb[0];
    });

    // Sort papers by number of authors (descending)
    sort(paperOrder.begin(), paperOrder.end(), [&](int a, int b) {
        return graph.getPaperAdjList(a).size() > graph.getPaperAdjList(b).size();
    });

    // Reorder vertices
    graph.reorderVertices(authorOrder, paperOrder);
}

// Utility function to save butterfly counts to a file
void saveVertexCounts(const vector<long long>& counts, const string& filename) {
    ofstream outFile(filename);
    if (!outFile.is_open()) {
        cerr << "Failed to open file for writing: " << filename << endl;
        return;
    }
    
    for (size_t i = 0; i < counts.size(); i++) {
        outFile << i << " " << counts[i] << "\n";
    }
    
    outFile.close();
}

// Utility function to save edge butterfly counts to a file
void saveEdgeCounts(const unordered_map<pair<int, int>, int, pair_hash>& counts, 
                   const string& filename) {
    ofstream outFile(filename);
    if (!outFile.is_open()) {
        cerr << "Failed to open file for writing: " << filename << endl;
        return;
    }
    
    for (const auto& pair : counts) {
        outFile << pair.first.first << " " << pair.first.second << " " << pair.second << "\n";
    }
    
    outFile.close();
}

// Utility function to save tip numbers to a file
void saveTipNumbers(const vector<int>& tipNumbers, const string& filename) {
    ofstream outFile(filename);
    if (!outFile.is_open()) {
        cerr << "Failed to open file for writing: " << filename << endl;
        return;
    }
    
    for (size_t i = 0; i < tipNumbers.size(); i++) {
        outFile << i << " " << tipNumbers[i] << "\n";
    }
    
    outFile.close();
}

// Utility function to save wing numbers to a file
void saveWingNumbers(const unordered_map<pair<int, int>, int, pair_hash>& wingNumbers, 
                    const string& filename) {
    ofstream outFile(filename);
    if (!outFile.is_open()) {
        cerr << "Failed to open file for writing: " << filename << endl;
        return;
    }
    
    for (const auto& pair : wingNumbers) {
        outFile << pair.first.first << " " << pair.first.second << " " << pair.second << "\n";
    }
    
    outFile.close();
}

// ----- Main function -----
int main(int argc, char* argv[]) {
    // Parse command line arguments
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <edge_file> <num_authors> <num_papers> [operation] [ranking_method] [agg_method]" << endl;
        cerr << "Operations: total, vertex, edge, tip, wing, sparsify, colorful" << endl;
        cerr << "Ranking methods: side, degree, approx_degree, comp_deg, approx_comp_deg" << endl;
        cerr << "Aggregation methods: sort, hash, hist, batch_simple, batch_wedge" << endl;
        return 1;
    }
    
    string edgeFile = argv[1];
    int numAuthors = stoi(argv[2]);
    int numPapers = stoi(argv[3]);
    
    string operation = (argc > 4) ? argv[4] : "total";
    
    // Parse ranking method
    RankingMethod rankMethod = RankingMethod::DEGREE_ORDER;
    if (argc > 5) {
        string rankStr = argv[5];
        if (rankStr == "side") {
            rankMethod = RankingMethod::SIDE_ORDER;
        } else if (rankStr == "degree") {
            rankMethod = RankingMethod::DEGREE_ORDER;
        } else if (rankStr == "approx_degree") {
            rankMethod = RankingMethod::APPROXIMATE_DEGREE_ORDER;
        } else if (rankStr == "comp_deg") {
            rankMethod = RankingMethod::COMPLEMENT_DEGENERACY_ORDER;
        } else if (rankStr == "approx_comp_deg") {
            rankMethod = RankingMethod::APPROXIMATE_COMPLEMENT_DEGENERACY_ORDER;
        }
    }
    
    // Parse aggregation method
    AggregationMethod aggMethod = AggregationMethod::HASHING;
    if (argc > 6) {
        string aggStr = argv[6];
        if (aggStr == "sort") {
            aggMethod = AggregationMethod::SORTING;
        } else if (aggStr == "hash") {
            aggMethod = AggregationMethod::HASHING;
        } else if (aggStr == "hist") {
            aggMethod = AggregationMethod::HISTOGRAMMING;
        } else if (aggStr == "batch_simple") {
            aggMethod = AggregationMethod::SIMPLE_BATCHING;
        } else if (aggStr == "batch_wedge") {
            aggMethod = AggregationMethod::WEDGE_AWARE_BATCHING;
        }
    }
    
    // Set number of threads
    int numThreads = omp_get_max_threads();
    cout << "Using " << numThreads << " threads" << endl;
    
    // Load graph
    auto startLoad = chrono::high_resolution_clock::now();
    BipartiteGraph graph = loadGraph(edgeFile, numAuthors, numPapers);
    auto endLoad = chrono::high_resolution_clock::now();
    
    cout << "Loaded graph with " << graph.getNumAuthors() << " authors, "
              << graph.getNumPapers() << " papers, and "
              << graph.getNumEdges() << " edges in "
              << chrono::duration_cast<chrono::milliseconds>(endLoad - startLoad).count()
              << " ms" << endl;
    
    // Apply cache optimization
    auto startCache = chrono::high_resolution_clock::now();
    optimizeForCache(graph);
    auto endCache = chrono::high_resolution_clock::now();
    
    cout << "Applied cache optimization in "
              << chrono::duration_cast<chrono::milliseconds>(endCache - startCache).count()
              << " ms" << endl;
    
    // Perform operation
    if (operation == "total") {
        auto startCount = chrono::high_resolution_clock::now();
        long long totalButterflies = countTotalButterflies(graph, rankMethod, aggMethod);
        auto endCount = chrono::high_resolution_clock::now();
        
        cout << "Total butterfly count: " << totalButterflies << endl;
        cout << "Counting time: "
                  << chrono::duration_cast<chrono::milliseconds>(endCount - startCount).count()
                  << " ms" << endl;
    }
    else if (operation == "vertex") {
        auto startCount = chrono::high_resolution_clock::now();
        vector<long long> vertexCounts = countButterflyPerVertex(graph, rankMethod, aggMethod);
        auto endCount = chrono::high_resolution_clock::now();
        
        long long totalCount = 0;
        for (auto count : vertexCounts) totalCount += count;
        totalCount /= 4; // Each butterfly is counted 4 times (once per vertex)
        
        cout << "Total butterfly count (derived from vertex counts): " << totalCount << endl;
        cout << "Counting time: "
                  << chrono::duration_cast<chrono::milliseconds>(endCount - startCount).count()
                  << " ms" << endl;
        
        // Save results
        saveVertexCounts(vertexCounts, "butterfly_counts.txt");
        cout << "Vertex butterfly counts saved to butterfly_counts.txt" << endl;
    }
    else if (operation == "edge") {
        auto startCount = chrono::high_resolution_clock::now();
        auto edgeCounts = countButterflyPerEdge(graph, rankMethod, aggMethod);
        auto endCount = chrono::high_resolution_clock::now();
        
        long long totalCount = 0;
        for (const auto& pair : edgeCounts) totalCount += pair.second;
        totalCount /= 4; // Each butterfly is counted 4 times (once per edge)
        
        cout << "Total butterfly count (derived from edge counts): " << totalCount << endl;
        cout << "Counting time: "
                  << chrono::duration_cast<chrono::milliseconds>(endCount - startCount).count()
                  << " ms" << endl;
        
        // Save results
        saveEdgeCounts(edgeCounts, "edge_butterfly_counts.txt");
        cout << "Edge butterfly counts saved to edge_butterfly_counts.txt" << endl;
    }
    else if (operation == "tip") {
        // First count butterflies per vertex
        auto startCount = chrono::high_resolution_clock::now();
        vector<long long> vertexCounts = countButterflyPerVertex(graph, rankMethod, aggMethod);
        auto endCount = chrono::high_resolution_clock::now();
        
        cout << "Counted butterflies per vertex in "
                  << chrono::duration_cast<chrono::milliseconds>(endCount - startCount).count()
                  << " ms" << endl;
        
        // Then perform tip decomposition
        auto startPeel = chrono::high_resolution_clock::now();
        vector<int> tipNumbers = peelV(graph, vertexCounts, aggMethod);
        auto endPeel = chrono::high_resolution_clock::now();
        
        cout << "Performed tip decomposition in "
                  << chrono::duration_cast<chrono::milliseconds>(endPeel - startPeel).count()
                  << " ms" << endl;
        
        // Save results
        saveTipNumbers(tipNumbers, "tip_numbers.txt");
        cout << "Tip numbers saved to tip_numbers.txt" << endl;
    }
    else if (operation == "wing") {
        // First count butterflies per edge
        auto startCount = chrono::high_resolution_clock::now();
        auto edgeCounts = countButterflyPerEdge(graph, rankMethod, aggMethod);
        auto endCount = chrono::high_resolution_clock::now();
        
        cout << "Counted butterflies per edge in "
                  << chrono::duration_cast<chrono::milliseconds>(endCount - startCount).count()
                  << " ms" << endl;
        
        // Then perform wing decomposition
        auto startPeel = chrono::high_resolution_clock::now();
        auto wingNumbers = peelE(graph, edgeCounts);
        auto endPeel = chrono::high_resolution_clock::now();
        
        cout << "Performed wing decomposition in "
                  << chrono::duration_cast<chrono::milliseconds>(endPeel - startPeel).count()
                  << " ms" << endl;
        
        // Save results
        saveWingNumbers(wingNumbers, "wing_numbers.txt");
        cout << "Wing numbers saved to wing_numbers.txt" << endl;
    }
    else if (operation == "sparsify") {
        if (argc < 8) {
            cerr << "Usage for sparsify: " << argv[0] << " <edge_file> <num_authors> <num_papers> sparsify <ranking_method> <agg_method> <probability>" << endl;
            return 1;
        }
        
        double probability = stod(argv[7]);
        
        // Count exact total
        auto startExact = chrono::high_resolution_clock::now();
        long long exactCount = countTotalButterflies(graph, rankMethod, aggMethod);
        auto endExact = chrono::high_resolution_clock::now();
        
        cout << "Exact butterfly count: " << exactCount << " (in "
                  << chrono::duration_cast<chrono::milliseconds>(endExact - startExact).count()
                  << " ms)" << endl;
        
        // Sparsify and count approximate
        auto startSparsify = chrono::high_resolution_clock::now();
        BipartiteGraph sparsified = sparsifyGraph(graph, probability);
        auto endSparsify = chrono::high_resolution_clock::now();
        
        cout << "Sparsified graph with probability " << probability << " in "<< chrono::duration_cast<chrono::milliseconds>(endSparsify - startSparsify).count()
                 << " ms" << endl;
        
        auto startApprox = chrono::high_resolution_clock::now();
        long long approxCount = countTotalButterflies(sparsified, rankMethod, aggMethod) / (probability * probability * probability * probability);
        auto endApprox = chrono::high_resolution_clock::now();
        
        cout << "Approximate butterfly count: " << approxCount << " (in "
                  << chrono::duration_cast<chrono::milliseconds>(endApprox - startApprox).count()
                  << " ms)" << endl;
        
        double error = abs(approxCount - exactCount) / static_cast<double>(exactCount) * 100.0;
        cout << "Relative error: " << error << "%" << endl;
    }
    else if (operation == "colorful") {
        if (argc < 8) {
            cerr << "Usage for colorful: " << argv[0] << " <edge_file> <num_authors> <num_papers> colorful <ranking_method> <agg_method> <num_colors>" << endl;
            return 1;
        }
        
        int numColors = stoi(argv[7]);
        
        // Count exact total
        auto startExact = chrono::high_resolution_clock::now();
        long long exactCount = countTotalButterflies(graph, rankMethod, aggMethod);
        auto endExact = chrono::high_resolution_clock::now();
        
        cout << "Exact butterfly count: " << exactCount << " (in "
                  << chrono::duration_cast<chrono::milliseconds>(endExact - startExact).count()
                  << " ms)" << endl;
        
        // Count using colorful sparsification
        auto startApprox = chrono::high_resolution_clock::now();
        long long approxCount = approximateCountColorful(graph, numColors, rankMethod, aggMethod);
        auto endApprox = chrono::high_resolution_clock::now();
        
        cout << "Approximate butterfly count (colorful): " << approxCount << " (in "
                  << chrono::duration_cast<chrono::milliseconds>(endApprox - startApprox).count()
                  << " ms)" << endl;
        
        double error = abs(approxCount - exactCount) / static_cast<double>(exactCount) * 100.0;
        cout << "Relative error: " << error << "%" << endl;
    }
    else {
        cerr << "Unknown operation: " << operation << endl;
        return 1;
    }
    
    return 0;
}

