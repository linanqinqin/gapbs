// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "benchmark.h"
#include "bitmap.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "platform_atomics.h"
#include "pvector.h"
#include "sliding_queue.h"
#include "timer.h"
#include "util.h"
#include "pthreadpp.h"


/*
GAP Benchmark Suite
Kernel: Betweenness Centrality (BC)
Author: Scott Beamer

Will return array of approx betweenness centrality scores for each vertex

This BC implementation makes use of the Brandes [1] algorithm with
implementation optimizations from Madduri et al. [2]. It is only approximate
because it does not compute the paths from every start vertex, but only a small
subset of them. Additionally, the scores are normalized to the range [0,1].

As an optimization to save memory, this implementation uses a Bitmap to hold
succ (list of successors) found during the BFS phase that are used in the back-
propagation phase.

[1] Ulrik Brandes. "A faster algorithm for betweenness centrality." Journal of
    Mathematical Sociology, 25(2):163–177, 2001.

[2] Kamesh Madduri, David Ediger, Karl Jiang, David A Bader, and Daniel
    Chavarria-Miranda. "A faster parallel algorithm and efficient multithreaded
    implementations for evaluating betweenness centrality on massive datasets."
    International Symposium on Parallel & Distributed Processing (IPDPS), 2009.
*/


using namespace std;
typedef float ScoreT;
typedef double CountT;


void PBFS(const Graph &g, NodeID source, pvector<CountT> &path_counts,
    Bitmap &succ, vector<SlidingQueue<NodeID>::iterator> &depth_index,
    SlidingQueue<NodeID> &queue) {
  pvector<NodeID> depths(g.num_nodes(), -1);
  depths[source] = 0;
  path_counts[source] = 1;
  queue.push_back(source);
  depth_index.push_back(queue.begin());
  queue.slide_window();
  const NodeID* g_out_start = g.out_neigh(0).begin();
  
  // Use a custom parallel region that properly handles the BFS algorithm
  // This is more complex than a simple parallel for, so we need to implement it carefully
  struct BFSData {
    const Graph* g;
    pvector<NodeID>* depths;
    pvector<CountT>* path_counts;
    Bitmap* succ;
    vector<SlidingQueue<NodeID>::iterator>* depth_index;
    SlidingQueue<NodeID>* queue;
    const NodeID* g_out_start;
  };
  
  BFSData data = {&g, &depths, &path_counts, &succ, &depth_index, &queue, g_out_start};
  
  // Use parallel region with proper synchronization
  int64_t result = 0;
  P3_PARALLEL_REGION([&](int thread_id, int num_threads) -> int64_t {
    NodeID depth = 0;
    QueueBuffer<NodeID> lqueue(queue);
    
    while (!queue.empty()) {
      depth++;
      
      // Process current level with dynamic scheduling
      size_t queue_size = queue.end() - queue.begin();
      size_t chunk_size = 64;
      size_t chunk_start = thread_id * chunk_size;
      
      while (chunk_start < queue_size) {
        size_t chunk_end = std::min(chunk_start + chunk_size, queue_size);
        
        for (size_t i = chunk_start; i < chunk_end; i++) {
          auto q_iter = queue.begin() + i;
          NodeID u = *q_iter;
          for (NodeID &v : g.out_neigh(u)) {
            if ((depths[v] == -1) &&
                (compare_and_swap(depths[v], static_cast<NodeID>(-1), depth))) {
              lqueue.push_back(v);
            }
            if (depths[v] == depth) {
              succ.set_bit_atomic(&v - g_out_start);
              P3_ATOMIC_ADD(path_counts[v], path_counts[u]);
            }
          }
        }
        
        // Get next chunk (dynamic scheduling)
        chunk_start += num_threads * chunk_size;
      }
      
      lqueue.flush();
      
      // Synchronize all threads before updating queue
      P3_BARRIER();
      
      // Only one thread updates the queue
      if (thread_id == 0) {
        depth_index.push_back(queue.begin());
        queue.slide_window();
      }
      
      P3_BARRIER();
    }
    
    return 0; // Return value not used
  }, result);
  
  depth_index.push_back(queue.begin());
}


pvector<ScoreT> Brandes(const Graph &g, SourcePicker<Graph> &sp,
                        NodeID num_iters, bool logging_enabled = false) {
  Timer t;
  t.Start();
  pvector<ScoreT> scores(g.num_nodes(), 0);
  pvector<CountT> path_counts(g.num_nodes());
  Bitmap succ(g.num_edges_directed());
  vector<SlidingQueue<NodeID>::iterator> depth_index;
  SlidingQueue<NodeID> queue(g.num_nodes());
  t.Stop();
  if (logging_enabled)
    PrintStep("a", t.Seconds());
  const NodeID* g_out_start = g.out_neigh(0).begin();
  for (NodeID iter=0; iter < num_iters; iter++) {
    NodeID source = sp.PickNext();
    if (logging_enabled)
      PrintStep("Source", static_cast<int64_t>(source));
    t.Start();
    path_counts.fill(0);
    depth_index.resize(0);
    queue.reset();
    succ.reset();
    PBFS(g, source, path_counts, succ, depth_index, queue);
    t.Stop();
    if (logging_enabled)
      PrintStep("b", t.Seconds());
    pvector<ScoreT> deltas(g.num_nodes(), 0);
    t.Start();
    for (int d=depth_index.size()-2; d >= 0; d--) {
      // Convert OpenMP parallel for schedule(dynamic, 64) to P3 dynamic scheduling
      P3_PARALLEL_FOR_DYNAMIC(depth_index[d+1] - depth_index[d], [&](size_t i) {
        auto it = depth_index[d] + i;
        NodeID u = *it;
        ScoreT delta_u = 0;
        for (NodeID &v : g.out_neigh(u)) {
          if (succ.get_bit(&v - g_out_start)) {
            delta_u += (path_counts[u] / path_counts[v]) * (1 + deltas[v]);
          }
        }
        deltas[u] = delta_u;
        scores[u] += delta_u;
      }, 64);
    }
    t.Stop();
    if (logging_enabled)
      PrintStep("p", t.Seconds());
  }
  // normalize scores
  ScoreT biggest_score = 0;
  // Convert OpenMP parallel for reduction(max : biggest_score) to P3 max reduction
  P3_PARALLEL_FOR_MAX_REDUCTION(g.num_nodes(), [&](size_t n) {
    return scores[n];
  }, biggest_score, biggest_score);
  
  // Convert OpenMP parallel for to P3 parallel for
  P3_PARALLEL_FOR(g.num_nodes(), [&](size_t n) {
    scores[n] = scores[n] / biggest_score;
  });
  return scores;
}


void PrintTopScores(const Graph &g, const pvector<ScoreT> &scores) {
  vector<pair<NodeID, ScoreT>> score_pairs(g.num_nodes());
  for (NodeID n : g.vertices())
    score_pairs[n] = make_pair(n, scores[n]);
  int k = 5;
  vector<pair<ScoreT, NodeID>> top_k = TopK(score_pairs, k);
  for (auto kvp : top_k)
    cout << kvp.second << ":" << kvp.first << endl;
}


// Still uses Brandes algorithm, but has the following differences:
// - serial (no need for atomics or dynamic scheduling)
// - uses vector for BFS queue
// - regenerates farthest to closest traversal order from depths
// - regenerates successors from depths
bool BCVerifier(const Graph &g, SourcePicker<Graph> &sp, NodeID num_iters,
                const pvector<ScoreT> &scores_to_test) {
  pvector<ScoreT> scores(g.num_nodes(), 0);
  for (int iter=0; iter < num_iters; iter++) {
    NodeID source = sp.PickNext();
    // BFS phase, only records depth & path_counts
    pvector<int> depths(g.num_nodes(), -1);
    depths[source] = 0;
    vector<CountT> path_counts(g.num_nodes(), 0);
    path_counts[source] = 1;
    vector<NodeID> to_visit;
    to_visit.reserve(g.num_nodes());
    to_visit.push_back(source);
    for (auto it = to_visit.begin(); it != to_visit.end(); it++) {
      NodeID u = *it;
      for (NodeID v : g.out_neigh(u)) {
        if (depths[v] == -1) {
          depths[v] = depths[u] + 1;
          to_visit.push_back(v);
        }
        if (depths[v] == depths[u] + 1)
          path_counts[v] += path_counts[u];
      }
    }
    // Get lists of vertices at each depth
    vector<vector<NodeID>> verts_at_depth;
    for (NodeID n : g.vertices()) {
      if (depths[n] != -1) {
        if (depths[n] >= static_cast<int>(verts_at_depth.size()))
          verts_at_depth.resize(depths[n] + 1);
        verts_at_depth[depths[n]].push_back(n);
      }
    }
    // Going from farthest to closest, compute "dependencies" (deltas)
    pvector<ScoreT> deltas(g.num_nodes(), 0);
    for (int depth=verts_at_depth.size()-1; depth >= 0; depth--) {
      for (NodeID u : verts_at_depth[depth]) {
        for (NodeID v : g.out_neigh(u)) {
          if (depths[v] == depths[u] + 1) {
            deltas[u] += (path_counts[u] / path_counts[v]) * (1 + deltas[v]);
          }
        }
        scores[u] += deltas[u];
      }
    }
  }
  // Normalize scores
  ScoreT biggest_score = *max_element(scores.begin(), scores.end());
  for (NodeID n : g.vertices())
    scores[n] = scores[n] / biggest_score;
  // Compare scores
  bool all_ok = true;
  for (NodeID n : g.vertices()) {
    ScoreT delta = abs(scores_to_test[n] - scores[n]);
    if (delta > std::numeric_limits<ScoreT>::epsilon()) {
      cout << n << ": " << scores[n] << " != " << scores_to_test[n];
      cout << "(" << delta << ")" << endl;
      all_ok = false;
    }
  }
  return all_ok;
}


int main(int argc, char* argv[]) {
  CLIterApp cli(argc, argv, "betweenness-centrality", 1);
  if (!cli.ParseArgs())
    return -1;
  if (cli.num_iters() > 1 && cli.start_vertex() != -1)
    cout << "Warning: iterating from same source (-r & -i)" << endl;
  
  // Set number of threads if specified via environment variable
  const char* p3_threads = getenv("P3_NUM_THREADS");
  if (p3_threads) {
    int threads = std::atoi(p3_threads);
    if (threads > 0) {
      p3_set_num_threads(threads);
      std::cout << "Using " << threads << " threads (from P3_NUM_THREADS)" << std::endl;
    }
  }
  
  Builder b(cli);
  Graph g = b.MakeGraph();
  
  // Barrier: wait for user input before starting BC processing
  std::cout << "Graph construction complete. Press Enter to start Betweenness Centrality benchmark..." << std::endl;
  std::string input;
  std::getline(std::cin, input);
  
  SourcePicker<Graph> sp(g, cli.start_vertex());
  auto BCBound = [&sp, &cli] (const Graph &g) {
    return Brandes(g, sp, cli.num_iters(), cli.logging_en());
  };
  SourcePicker<Graph> vsp(g, cli.start_vertex());
  auto VerifierBound = [&vsp, &cli] (const Graph &g,
                                     const pvector<ScoreT> &scores) {
    return BCVerifier(g, vsp, cli.num_iters(), scores);
  };
  BenchmarkKernel(cli, g, BCBound, PrintTopScores, VerifierBound);
  return 0;
}
