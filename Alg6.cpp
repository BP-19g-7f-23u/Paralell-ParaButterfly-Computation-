// Algorithm 6: Parallel edge peeling (wing decomposition)
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <omp.h>
#include <queue>
#include <utility>
#include <tuple>
#include <set>

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
    
    bool hasEdge(int u, int v) const {
        return std::find(leftAdj[u].begin(), leftAdj[u].end(), v) != leftAdj[u].end();
    }
    
    int getNumEdges() const {
        int count = 0;
        for (const auto& adj : leftAdj) count += adj.size();
        return count;
    }
};

// Bucketing structure for edge peeling
class EdgeBucketingStructure {
private:
    std::unordered_map<std::pair<int, int>, int, pair_hash> position;
    std::vector<std::vector<std::pair<int, int>>> buckets;
    int maxValue;

public:
    EdgeBucketingStructure(int maxVal) : maxValue(maxVal + 1) {
        buckets.resize(maxVal + 1);
    }

    void insert(const std::pair<int, int>& edge, int value) {
        if (value >= 0 && value <= maxValue) {
            position[edge] = value;
            buckets[value].push_back(edge);
        }
    }

    void update(const std::pair<int, int>& edge, int newValue) {
        auto it = position.find(edge);
        if (it == position.end() || it->second == newValue || 
            newValue < 0 || newValue > maxValue) return;
        
        int oldValue = it->second;
        
        // Remove edge from old bucket
        auto& oldBucket = buckets[oldValue];
        oldBucket.erase(std::remove(oldBucket.begin(), oldBucket.end(), edge), oldBucket.end());
        
        // Add to new bucket
        position[edge] = newValue;
        buckets[newValue].push_back(edge);
    }

    std::vector<std::pair<int, int>> getNextBucket() {
        for (int i = 0; i <= maxValue; i++) {
            if (!buckets[i].empty()) {
                std::vector<std::pair<int, int>> result = std::move(buckets[i]);
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

// Compute intersection of two sorted lists
std::vector<int> intersect(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<int> result;
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

// Get frequency of element pairs
std::pair<std::vector<std::pair<int, int>>, std::vector<int>> 
getFreq(const std::vector<std::tuple<std::pair<int, int>, int, int>>& entries) {
    // Sort entries by key
    std::vector<std::pair<std::pair<int, int>, int>> sorted;
    for (const auto& entry : entries) {
        auto key = std::get<0>(entry);
        auto value = std::get<1>(entry);
        sorted.push_back({key, value});
    }
    
    std::sort(sorted.begin(), sorted.end());
    
    // Get unique keys and frequencies
    std::vector<std::pair<int, int>> uniqueKeys;
    std::vector<int> frequencies(1, 0);
    
    if (!sorted.empty()) {
        auto currentKey = sorted[0].first;
        uniqueKeys.push_back({currentKey.first, currentKey.second});
        int count = sorted[0].second;
        
        for (size_t i = 1; i < sorted.size(); i++) {
            if (sorted[i].first == currentKey) {
                count += sorted[i].second;
            } else {
                frequencies.push_back(frequencies.back() + count);
                uniqueKeys.push_back({sorted[i].first.first, sorted[i].first.second});
                currentKey = sorted[i].first;
                count = sorted[i].second;
            }
        }
        
        frequencies.push_back(frequencies.back() + count);
    }
    
    return {uniqueKeys, frequencies};
}

// Update butterfly counts after peeling edges
std::unordered_map<std::pair<int, int>, int, pair_hash> 
updateE(const BipartiteGraph& graph, 
        const std::unordered_map<std::pair<int, int>, int, pair_hash>& butterflyCounts,
        const std::vector<std::pair<int, int>>& peeledEdges) {
    
    // Initialize storage for butterfly count updates
    std::vector<std::tuple<std::pair<int, int>, int, int>> updates;
    
    // Process each peeled edge
    #pragma omp parallel
    {
        std::vector<std::tuple<std::pair<int, int>, int, int>> localUpdates;
        
        #pragma omp for schedule(dynamic)
        for (const auto& edge : peeledEdges) {
            int u1 = edge.first;
            int v1 = edge.second;
            
            // Process all neighbors of v1
            for (int u2 : graph.getRightNeighbors(v1)) {
                if (u2 != u1) {
                    // Find intersection of neighbors
                    std::vector<int> intersection = intersect(
                        graph.getLeftNeighbors(u1),
                        graph.getLeftNeighbors(u2)
                    );
                    
                    // Add update for edge (u2, v1)
                    if (intersection.size() > 1) {
                        localUpdates.emplace_back(
                            std::make_pair(u2, v1), 
                            intersection.size() - 1, 
                            0
                        );
                    }
                    
                    // Process each vertex in the intersection
                    for (int v2 : intersection) {
                        if (v2 != v1) {
                            // Add updates for edges (u1, v2) and (u2, v2)
                            localUpdates.emplace_back(
                                std::make_pair(u1, v2), 
                                1, 
                                0
                            );
                            
                            localUpdates.emplace_back(
                                std::make_pair(u2, v2), 
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
    auto [keys, freqs] = getFreq(updates);
    
    // Apply updates to butterfly counts
    std::unordered_map<std::pair<int, int>, int, pair_hash> updatedCounts = butterflyCounts;
    
    for (size_t i = 0; i < keys.size(); i++) {
        auto edge = std::make_pair(keys[i].first, keys[i].second);
        int start = (i == 0) ? 0 : freqs[i-1];
        int end = freqs[i];
        int updateCount = end - start;
        
        // Apply the update
        auto it = updatedCounts.find(edge);
        if (it != updatedCounts.end()) {
            it->second -= updateCount;
            if (it->second <= 0) {
                updatedCounts.erase(it);
            }
        }
    }
    
    return updatedCounts;
}

// Edge peeling (wing decomposition)
std::unordered_map<std::pair<int, int>, int, pair_hash> 
peelE(const BipartiteGraph& graph, 
      const std::unordered_map<std::pair<int, int>, int, pair_hash>& initialButterflyCounts) {
    
    // Find maximum butterfly count
    int maxCount = 0;
    for (const auto& pair : initialButterflyCounts) {
        maxCount = std::max(maxCount, pair.second);
    }
    
    // Initialize bucketing structure
    EdgeBucketingStructure buckets(maxCount);
    for (const auto& pair : initialButterflyCounts) {
        buckets.insert(pair.first, pair.second);
    }
    
    // Initialize result (wing numbers)
    std::unordered_map<std::pair<int, int>, int, pair_hash> wingNumbers;
    
    // Variables for peeling process
    int peelingRound = 0;
    int totalEdges = initialButterflyCounts.size();
    int processedCount = 0;
    std::unordered_map<std::pair<int, int>, int, pair_hash> currentCounts = initialButterflyCounts;
    
    // Peeling process
    while (processedCount < totalEdges) {
        // Get edges with minimum butterfly count
        std::vector<std::pair<int, int>> toRemove = buckets.getNextBucket();
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
