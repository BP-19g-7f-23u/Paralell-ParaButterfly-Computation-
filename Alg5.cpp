// Algorithm 5: Parallel vertex peeling (tip decomposition)
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <omp.h>
#include <queue>
#include <utility>
#include <tuple>

struct pair_hash {
    template <class T1, class T2>
    size_t operator()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

class BipartiteGraph {
private:
    int numLeft, numRight;
    std::vector<std::vector<int>> leftAdj;
    std::vector<std::vector<int>> rightAdj;

public:
    BipartiteGraph(int left, int right) : numLeft(left), numRight(right) {
        leftAdj.resize(left);
        rightAdj.resize(right);
    }

    void addEdge(int left, int right) {
        leftAdj[left].push_back(right);
        rightAdj[right].push_back(left);
    }

    const std::vector<int>& getLeftNeighbors(int u) const {
        return leftAdj[u];
    }

    const std::vector<int>& getRightNeighbors(int v) const {
        return rightAdj[v];
    }

    int getNumLeft() const { return numLeft; }
    int getNumRight() const { return numRight; }
};

// Bucketing structure for peeling
class BucketingStructure {
private:
    std::vector<std::vector<int>> buckets;
    std::vector<int> position;
    int maxValue;

public:
    BucketingStructure(int n, int maxVal) : maxValue(maxVal + 1) {
        buckets.resize(maxVal + 1);
        position.resize(n, -1);
    }

    void insert(int vertex, int value) {
        if (value >= 0 && value <= maxValue) {
            position[vertex] = value;
            buckets[value].push_back(vertex);
        }
    }

    void update(int vertex, int newValue) {
        int oldValue = position[vertex];
        if (oldValue == newValue || newValue < 0 || newValue > maxValue) return;

        // Remove vertex from old bucket
        if (oldValue >= 0 && oldValue <= maxValue) {
            auto& oldBucket = buckets[oldValue];
            oldBucket.erase(std::remove(oldBucket.begin(), oldBucket.end(), vertex), oldBucket.end());
        }

        // Add to new bucket
        position[vertex] = newValue;
        buckets[newValue].push_back(vertex);
    }

    std::vector<int> getNextBucket() {
        for (int i = 0; i <= maxValue; i++) {
            if (!buckets[i].empty()) {
                std::vector<int> result = std::move(buckets[i]);
                buckets[i].clear();
                return result;
            }
        }
        return {};
    }

    bool isEmpty() const {
        for (const auto& bucket : buckets) {
            if (!bucket.empty()) return false;
        }
        return true;
    }
};

// Get frequency of wedges
std::pair<std::vector<std::pair<int, int>>, std::vector<int>> 
getFreq(const std::vector<std::tuple<int, int, int>>& wedges) {
    // Sort wedges by endpoints
    std::vector<std::pair<std::pair<int, int>, int>> sortedWedges;
    
    #pragma omp parallel
    {
        std::vector<std::pair<std::pair<int, int>, int>> localSorted;
        
        #pragma omp for schedule(static)
        for (size_t i = 0; i < wedges.size(); i++) {
            int u1, u2, v;
            std::tie(u1, u2, v) = wedges[i];
            localSorted.push_back({{u1, u2}, v});
        }
        
        std::sort(localSorted.begin(), localSorted.end());
        
        #pragma omp critical
        {
            sortedWedges.insert(sortedWedges.end(), 
                               localSorted.begin(), 
                               localSorted.end());
        }
    }
    
    std::sort(sortedWedges.begin(), sortedWedges.end());
    
    // Get unique endpoints and their frequencies
    std::vector<std::pair<int, int>> uniqueKeys;
    std::vector<int> frequencies(1, 0);
    
    if (!sortedWedges.empty()) {
        auto currentPair = sortedWedges[0].first;
        uniqueKeys.push_back(currentPair);
        int count = 1;
        
        for (size_t i = 1; i < sortedWedges.size(); i++) {
            if (sortedWedges[i].first == currentPair) {
                count++;
            } else {
                frequencies.push_back(frequencies.back() + count);
                uniqueKeys.push_back(sortedWedges[i].first);
                currentPair = sortedWedges[i].first;
                count = 1;
            }
        }
        
        frequencies.push_back(frequencies.back() + count);
    }
    
    return {uniqueKeys, frequencies};
}

// Count wedges and compute butterfly counts
std::vector<int> countVWedges(const BipartiteGraph& graph, 
                              const std::vector<std::tuple<int, int, int>>& wedges) {
    auto [keys, freq] = getFreq(wedges);
    
    // Maximum vertex ID
    int maxVertex = 0;
    for (const auto& [pair, _] : keys) {
        maxVertex = std::max(maxVertex, std::max(pair.first, pair.second));
    }
    
    // Initialize butterfly counts
    std::vector<int> butterflyCounts(maxVertex + 1, 0);
    
    // Count butterflies
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < keys.size(); i++) {
        auto [u1, u2] = keys[i];
        int start = (i == 0) ? 0 : freq[i-1];
        int end = freq[i];
        int wedgeCount = end - start;
        
        if (wedgeCount > 1) {
            // Each pair of wedges forms a butterfly
            int butterflies = (wedgeCount * (wedgeCount - 1)) / 2;
            
            // Update counts for endpoints
            #pragma omp atomic
            butterflyCounts[u1] += butterflies;
            
            #pragma omp atomic
            butterflyCounts[u2] += butterflies;
            
            // Update counts for centers
            for (int j = start; j < end; j++) {
                int v = std::get<2>(wedges[j]);
                
                #pragma omp atomic
                butterflyCounts[v] += (wedgeCount - 1);
            }
        }
    }
    
    return butterflyCounts;
}

// Update butterfly counts after peeling vertices
std::vector<int> updateV(const BipartiteGraph& graph, 
                        const std::vector<int>& butterflyCounts,
                        const std::vector<int>& peeledVertices) {
    // Initialize wedges array
    std::vector<std::tuple<int, int, int>> wedges;
    
    // Collect wedges from peeled vertices
    #pragma omp parallel
    {
        std::vector<std::tuple<int, int, int>> localWedges;
        
        #pragma omp for schedule(dynamic)
        for (auto u1 : peeledVertices) {
            for (int v : graph.getLeftNeighbors(u1)) {
                for (int u2 : graph.getRightNeighbors(v)) {
                    if (u2 != u1) {
                        localWedges.emplace_back(u1, u2, v);
                    }
                }
            }
        }
        
        #pragma omp critical
        {
            wedges.insert(wedges.end(), localWedges.begin(), localWedges.end());
        }
    }
    
    // Count wedges and compute butterfly counts to subtract
    std::vector<int> deltaButterflyCounts = countVWedges(graph, wedges);
    
    // Subtract from original counts
    std::vector<int> updatedCounts = butterflyCounts;
    for (size_t i = 0; i < updatedCounts.size(); i++) {
        if (i < deltaButterflyCounts.size()) {
            updatedCounts[i] -= deltaButterflyCounts[i];
        }
    }
    
    return updatedCounts;
}

// Vertex peeling (tip decomposition)
std::vector<int> peelV(const BipartiteGraph& graph, const std::vector<int>& initialButterflyCounts) {
    int numLeft = graph.getNumLeft();
    
    // Find maximum butterfly count
    int maxCount = 0;
    for (int count : initialButterflyCounts) {
        maxCount = std::max(maxCount, count);
    }
    
    // Initialize bucketing structure
    BucketingStructure buckets(numLeft, maxCount);
    for (int u = 0; u < numLeft; u++) {
        buckets.insert(u, initialButterflyCounts[u]);
    }
    
    // Initialize result array (tip numbers)
    std::vector<int> tipNumbers(numLeft, 0);
    
    // Variables for peeling process
    int peelingRound = 0;
    int processedCount = 0;
    std::vector<int> currentCounts = initialButterflyCounts;
    
    // Peeling process
    while (processedCount < numLeft) {
        // Get vertices with minimum butterfly count
        std::vector<int> toRemove = buckets.getNextBucket();
        if (toRemove.empty()) break;
        
        peelingRound++;
        
        // Set tip numbers for peeled vertices
        for (int v : toRemove) {
            tipNumbers[v] = peelingRound;
        }
        
        processedCount += toRemove.size();
        
        // Update butterfly counts
        currentCounts = updateV(graph, currentCounts, toRemove);
        
        // Update buckets with new counts
        for (int u = 0; u < numLeft; u++) {
            if (tipNumbers[u] == 0) { // Not yet processed
                buckets.update(u, currentCounts[u]);
            }
        }
    }
    
    return tipNumbers;
}
