// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <algorithm>
#include <iostream>
#include <vector>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"
#include "pthreadpp.h"

/*
GAP Benchmark Suite
Kernel: PageRank (PR)
Author: Scott Beamer

Will return pagerank scores for all vertices once total change < epsilon

This PR implementation uses the traditional iterative approach. It performs
updates in the pull direction to remove the need for atomics, and it allows
new values to be immediately visible (like Gauss-Seidel method). The prior PR
implementation is still available in src/pr_spmv.cc.
*/


using namespace std;

typedef float ScoreT;
const float kDamp = 0.85;


pvector<ScoreT> PageRankPullGS(const Graph &g, int max_iters, double epsilon=0,
                               bool logging_enabled = false) {
  const ScoreT init_score = 1.0f / g.num_nodes();
  const ScoreT base_score = (1.0f - kDamp) / g.num_nodes();
  pvector<ScoreT> scores(g.num_nodes(), init_score);
  pvector<ScoreT> outgoing_contrib(g.num_nodes());
  P3_PARALLEL_FOR(g.num_nodes(),
    [&](NodeID n) {
      outgoing_contrib[n] = init_score / g.out_degree(n);
    });
  for (int iter=0; iter < max_iters; iter++) {
    double error = 0;
    P3_PARALLEL_REGION(
      [&](int thread_id, int num_threads) -> int64_t {
        double local_error = 0;
        
        // Distribute work among threads using static block distribution
        // This matches OpenMP's behavior for correctness
        NodeID start = (thread_id * g.num_nodes()) / num_threads;
        NodeID end = ((thread_id + 1) * g.num_nodes()) / num_threads;
        
        for (NodeID u = start; u < end; u++) {
          ScoreT incoming_total = 0;
          for (NodeID v : g.in_neigh(u))
            incoming_total += outgoing_contrib[v];
          ScoreT old_score = scores[u];
          scores[u] = base_score + kDamp * incoming_total;
          local_error += fabs(scores[u] - old_score);
          outgoing_contrib[u] = scores[u] / g.out_degree(u);
        }
        return static_cast<int64_t>(local_error * 1000000); // Scale to avoid precision loss
      }, error);
    error = static_cast<double>(error) / 1000000; // Scale back
    if (logging_enabled)
      PrintStep(iter, error);
    if (error < epsilon)
      break;
  }
  return scores;
}


void PrintTopScores(const Graph &g, const pvector<ScoreT> &scores) {
  vector<pair<NodeID, ScoreT>> score_pairs(g.num_nodes());
  for (NodeID n=0; n < g.num_nodes(); n++) {
    score_pairs[n] = make_pair(n, scores[n]);
  }
  int k = 5;
  vector<pair<ScoreT, NodeID>> top_k = TopK(score_pairs, k);
  for (auto kvp : top_k)
    cout << kvp.second << ":" << kvp.first << endl;
}


// Verifies by asserting a single serial iteration in push direction has
//   error < target_error
bool PRVerifier(const Graph &g, const pvector<ScoreT> &scores,
                        double target_error) {
  const ScoreT base_score = (1.0f - kDamp) / g.num_nodes();
  pvector<ScoreT> incoming_sums(g.num_nodes(), 0);
  double error = 0;
  for (NodeID u : g.vertices()) {
    ScoreT outgoing_contrib = scores[u] / g.out_degree(u);
    for (NodeID v : g.out_neigh(u))
      incoming_sums[v] += outgoing_contrib;
  }
  for (NodeID n : g.vertices()) {
    error += fabs(base_score + kDamp * incoming_sums[n] - scores[n]);
    incoming_sums[n] = 0;
  }
  PrintTime("Total Error", error);
  return error < target_error;
}


int main(int argc, char* argv[]) {
  CLPageRank cli(argc, argv, "pagerank", 1e-4, 20);
  if (!cli.ParseArgs())
    return -1;
  
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
  
  // Barrier: wait for user input before starting PageRank processing
  std::cout << "Graph construction complete. Press Enter to start PageRank benchmark..." << std::endl;
  std::string input;
  std::getline(std::cin, input);
  
  auto PRBound = [&cli] (const Graph &g) {
    return PageRankPullGS(g, cli.max_iters(), cli.tolerance(), cli.logging_en());
  };
  auto VerifierBound = [&cli] (const Graph &g, const pvector<ScoreT> &scores) {
    return PRVerifier(g, scores, cli.tolerance());
  };
  BenchmarkKernel(cli, g, PRBound, PrintTopScores, VerifierBound);
  return 0;
}
