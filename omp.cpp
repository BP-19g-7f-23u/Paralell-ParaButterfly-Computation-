#include <omp.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <string>
#include <queue>
#include <set>
#include <tuple>
#include <numeric>
using namespace std;

struct pair_hash {
    template <class T1, class T2>
    size_t operator()(const pair<T1, T2>& p) const {
        return hash<T1>()(p.first) ^ (hash<T2>()(p.second) << 1);
    }
};

class BipartiteGraph {
public:
    unordered_map<int, vector<int>> authorAdj;
    unordered_map<int, vector<int>> paperAdj;

    BipartiteGraph(int authors, int papers) {}

    void addEdge(int author, int paper) {
        authorAdj[author].push_back(paper);
        paperAdj[paper].push_back(author);
    }

    int getNumAuthors() const { return authorAdj.size(); }
    int getNumPapers() const { return paperAdj.size(); }
    const vector<int>& getAuthorAdjList(int a) const { return authorAdj.at(a); }
    const vector<int>& getPaperAdjList(int p) const { return paperAdj.at(p); }

    void reorderVertices(const vector<int>& authorOrder, const vector<int>& paperOrder) {
        unordered_map<int, vector<int>> newAuthorAdj;
        unordered_map<int, vector<int>> newPaperAdj;
        unordered_map<int, int> authorMap, paperMap;

        for (int i = 0; i < authorOrder.size(); ++i) authorMap[authorOrder[i]] = i;
        for (int i = 0; i < paperOrder.size(); ++i) paperMap[paperOrder[i]] = i;

        for (const auto& entry : authorAdj) {
            int oldAuthor = entry.first;
            if (authorMap.find(oldAuthor) == authorMap.end()) continue;
            int newAuthor = authorMap[oldAuthor];
            for (int oldPaper : entry.second) {
                if (paperMap.find(oldPaper) == paperMap.end()) continue;
                int newPaper = paperMap[oldPaper];
                newAuthorAdj[newAuthor].push_back(newPaper);
                newPaperAdj[newPaper].push_back(newAuthor);
            }
        }

        authorAdj = move(newAuthorAdj);
        paperAdj = move(newPaperAdj);
    }

    BipartiteGraph extractSubgraph(const vector<int>& authorsSubset) {
        set<int> papersSubset;
        for (int a : authorsSubset) {
            if (authorAdj.find(a) == authorAdj.end()) continue;
            for (int p : authorAdj[a]) papersSubset.insert(p);
        }

        BipartiteGraph subgraph(0, 0);
        unordered_map<int, int> authorMap, paperMap;
        int idx = 0;
        for (int a : authorsSubset) authorMap[a] = idx++;
        idx = 0;
        for (int p : papersSubset) paperMap[p] = idx++;

        for (int a : authorsSubset) {
            if (authorAdj.find(a) == authorAdj.end()) continue;
            for (int p : authorAdj[a]) {
                if (papersSubset.count(p)) subgraph.addEdge(authorMap[a], paperMap[p]);
            }
        }

        return subgraph;
    }
};

BipartiteGraph loadGraph(const string& file, int authors, int papers) {
    ifstream fin(file);
    BipartiteGraph G(authors, papers);
    int u, v;
    while (fin >> u >> v) G.addEdge(u, v);
    return G;
}

long long countTotalButterflies(const BipartiteGraph& graph) {
    int numAuthors = graph.getNumAuthors();
    unordered_map<pair<int, int>, int, pair_hash> wedgeCounts;

    #pragma omp parallel
    {
        unordered_map<pair<int, int>, int, pair_hash> localCounts;

        #pragma omp for schedule(dynamic)
        for (int u = 0; u < numAuthors; ++u) {
            const auto& papers = graph.getAuthorAdjList(u);
            for (int i = 0; i < papers.size(); ++i) {
                for (int j = i + 1; j < papers.size(); ++j) {
                    int p1 = papers[i], p2 = papers[j];
                    const auto& authors1 = graph.getPaperAdjList(p1);
                    const auto& authors2 = graph.getPaperAdjList(p2);

                    for (int v1 : authors1) {
                        if (v1 == u) continue;
                        for (int v2 : authors2) {
                            if (v2 == u || v1 != v2) continue;
                            auto key = make_pair(min(u, v1), max(u, v1));
                            localCounts[key]++;
                        }
                    }
                }
            }
        }

        #pragma omp critical
        for (const auto& p : localCounts) {
            wedgeCounts[p.first] += p.second;
        }
    }

    long long total = 0;
    for (const auto& wc : wedgeCounts) {
        int c = wc.second;
        if (c > 1) total += (c * (c - 1)) / 2;
    }

    return total;
}

vector<int> countButterflyPerVertex(const BipartiteGraph& graph) {
    int n = graph.getNumAuthors();
    vector<int> butterflyCounts(n, 0);

    #pragma omp parallel for schedule(dynamic)
    for (int u = 0; u < n; ++u) {
        const auto& papers = graph.getAuthorAdjList(u);
        unordered_map<int, int> freq;
        for (int p : papers) {
            for (int v : graph.getPaperAdjList(p)) {
                if (v != u) freq[v]++;
            }
        }
        for (auto& kv : freq) {
            if (kv.second >= 2)
                butterflyCounts[u] += (kv.second * (kv.second - 1)) / 2;
        }
    }

    return butterflyCounts;
}

int tipDecomposition(const vector<int>& counts) {
    priority_queue<pair<int, int>> pq;
    for (int i = 0; i < counts.size(); ++i) pq.push({counts[i], i});
    return pq.top().first;
}

unordered_map<pair<int, int>, int, pair_hash> countButterflyPerEdge(const BipartiteGraph& graph) {
    unordered_map<pair<int, int>, int, pair_hash> edgeCounts;
    int n = graph.getNumAuthors();

    #pragma omp parallel
    {
        unordered_map<pair<int, int>, int, pair_hash> localCounts;

        #pragma omp for schedule(dynamic)
        for (int u = 0; u < n; ++u) {
            const auto& papers = graph.getAuthorAdjList(u);
            for (int i = 0; i < papers.size(); ++i) {
                for (int j = i + 1; j < papers.size(); ++j) {
                    int p1 = papers[i], p2 = papers[j];
                    const auto& authors1 = graph.getPaperAdjList(p1);
                    const auto& authors2 = graph.getPaperAdjList(p2);

                    unordered_set<int> authors2Set(authors2.begin(), authors2.end());
                    for (int v : authors1) {
                        if (v == u || !authors2Set.count(v)) continue;
                        auto key = make_pair(min(u, v), max(u, v));
                        localCounts[key]++;
                    }
                }
            }
        }

        #pragma omp critical
        for (const auto& kv : localCounts) edgeCounts[kv.first] += kv.second;
    }

    return edgeCounts;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cout << "Usage: " << argv[0] << " <edge_file> <num_authors> <num_papers>" << endl;
        return 1;
    }

    string edgeFile = argv[1];
    int numAuthors = stoi(argv[2]);
    int numPapers = stoi(argv[3]);

    BipartiteGraph graph = loadGraph(edgeFile, numAuthors, numPapers);

    cout << "Using " << omp_get_max_threads() << " threads\n";


    cout << "Choose method:\n1. Total Butterflies\n2. Vertex Butterfly Count & Tip Decomposition\n3. Edge Butterfly Count (Wing Info)\n4. Reorder Vertices\n5. Extract Subgraph\nEnter option: ";
    int choice;
    cin >> choice;

    auto start = chrono::high_resolution_clock::now();

    if (choice == 1) {
        long long total = countTotalButterflies(graph);
        auto end = chrono::high_resolution_clock::now();
        cout << "Total butterflies: " << total << " | Time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms\n";
    } else if (choice == 2) {
        auto vec = countButterflyPerVertex(graph);
        int maxTip = tipDecomposition(vec);
        auto end = chrono::high_resolution_clock::now();
        cout << "Tip value: " << maxTip << " | Time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms\n";
    } else if (choice == 3) {
        auto edgeCounts = countButterflyPerEdge(graph);
        int maxWings = 0;
        for (auto& kv : edgeCounts) maxWings = max(maxWings, kv.second);
        auto end = chrono::high_resolution_clock::now();
        cout << "Max wings: " << maxWings << " | Time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms\n";
    } else if (choice == 4) {
        vector<int> authorOrder(graph.getNumAuthors()), paperOrder(graph.getNumPapers());
        iota(authorOrder.begin(), authorOrder.end(), 0);
        iota(paperOrder.begin(), paperOrder.end(), 0);
        random_shuffle(authorOrder.begin(), authorOrder.end());
        random_shuffle(paperOrder.begin(), paperOrder.end());
        graph.reorderVertices(authorOrder, paperOrder);
        cout << "Graph reordered.\n";
    } else if (choice == 5) {
        int numSubset;
        cout << "Enter number of authors in subset: ";
        cin >> numSubset;
        vector<int> subset(numSubset);
        for (int i = 0; i < numSubset; ++i) cin >> subset[i];
        BipartiteGraph sub = graph.extractSubgraph(subset);
        cout << "Subgraph has " << sub.getNumAuthors() << " authors and " << sub.getNumPapers() << " papers.\n";
    } else {
        cout << "Invalid choice.\n";
    }

    return 0;
}

