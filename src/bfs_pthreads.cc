// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <iostream>
#include <vector>
#include <cstdlib>

#include "benchmark.h"
#include "bitmap.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"
#include "sliding_queue.h"
#include "timer.h"
#include "gapbs_pthreads.h"

// Atomic operations are now handled by platform_atomics.h


/*
GAP Benchmark Suite
Kernel: Breadth-First Search (BFS) - Pthreads Version
Author: Scott Beamer (Original), Custom Implementation (Pthreads)

Will return parent array for a BFS traversal from a source vertex

This BFS implementation makes use of the Direction-Optimizing approach [1].
It uses the alpha and beta parameters to determine whether to switch search
directions. For representing the frontier, it uses a SlidingQueue for the
top-down approach and a Bitmap for the bottom-up approach. To reduce
false-sharing for the top-down approach, thread-local QueueBuffer's are used.

To save time computing the number of edges exiting the frontier, this
implementation precomputes the degrees in bulk at the beginning by storing
them in the parent array as negative numbers. Thus, the encoding of parent is:
  parent[x] < 0 implies x is unvisited and parent[x] = -out_degree(x)
  parent[x] >= 0 implies x been visited

[1] Scott Beamer, Krste Asanović, and David Patterson. "Direction-Optimizing
    Breadth-First Search." International Conference on High Performance
    Computing, Networking, Storage and Analysis (SC), Salt Lake City, Utah,
    November 2012.
*/


using namespace std;

int64_t BUStep(const Graph &g, pvector<NodeID> &parent, Bitmap &front,
               Bitmap &next) {
  int64_t awake_count = 0;
  next.reset();
  
  // Replace: #pragma omp parallel for reduction(+ : awake_count) schedule(dynamic, 1024)
  GAPBS_PARALLEL_FOR_REDUCTION(g.num_nodes(), 
    [&](NodeID u) -> int64_t {
      int64_t local_count = 0;
      if (parent[u] < 0) {
        for (NodeID v : g.in_neigh(u)) {
          if (front.get_bit(v)) {
            parent[u] = v;
            local_count++;
            next.set_bit(u);
            break;
          }
        }
      }
      return local_count;
    }, awake_count);
  
  return awake_count;
}


int64_t TDStep(const Graph &g, pvector<NodeID> &parent,
               SlidingQueue<NodeID> &queue) {
  int64_t scout_count = 0;
  
  // Replace: #pragma omp parallel with thread-local QueueBuffer
  GAPBS_PARALLEL_REGION(
    [&](int thread_id, int num_threads) -> int64_t {
      int64_t local_count = 0;
      QueueBuffer<NodeID> lqueue(queue);
      
      // Distribute queue iterations among threads (like OpenMP for does)
      // Use round-robin distribution to match OpenMP behavior
      size_t queue_size = queue.size();
        for (size_t i = thread_id; i < queue_size; i += num_threads) {
          auto q_iter = queue.begin() + i;
          NodeID u = *q_iter;
          for (NodeID v : g.out_neigh(u)) {
            NodeID curr_val = parent[v];
            if (curr_val < 0) {
              if (compare_and_swap(parent[v], curr_val, u)) {
                lqueue.push_back(v);
                local_count += -curr_val;
              }
            }
          }
        }
      lqueue.flush();
      return local_count;
    }, scout_count);
  
  return scout_count;
}


void QueueToBitmap(const SlidingQueue<NodeID> &queue, Bitmap &bm) {
  // Replace: #pragma omp parallel for
  GAPBS_PARALLEL_FOR(queue.size(),
    [&](size_t i) {
      auto q_iter = queue.begin() + i;
      NodeID u = *q_iter;
      bm.set_bit_atomic(u);
    });
}

void BitmapToQueue(const Graph &g, const Bitmap &bm,
                   SlidingQueue<NodeID> &queue) {
  // Replace: #pragma omp parallel with thread-local QueueBuffer
  int64_t dummy_result; // Intentionally unused - required by macro
  (void)dummy_result; // Suppress unused variable warning
  GAPBS_PARALLEL_REGION(
    [&](int thread_id, int num_threads) -> int64_t {
      QueueBuffer<NodeID> lqueue(queue);
      
      // Distribute work among threads
      NodeID start = (thread_id * g.num_nodes()) / num_threads;
      NodeID end = ((thread_id + 1) * g.num_nodes()) / num_threads;
      
      for (NodeID n = start; n < end; n++) {
        if (bm.get_bit(n)) {
          lqueue.push_back(n);
        }
      }
      lqueue.flush();
      return 0; // No reduction needed
    }, dummy_result);
  
  queue.slide_window();
}

pvector<NodeID> InitParent(const Graph &g) {
  pvector<NodeID> parent(g.num_nodes());
  
  // Replace: #pragma omp parallel for
  GAPBS_PARALLEL_FOR(g.num_nodes(),
    [&](NodeID n) {
      parent[n] = g.out_degree(n) != 0 ? -g.out_degree(n) : -1;
    });
  
  return parent;
}

pvector<NodeID> DOBFS(const Graph &g, NodeID source, bool logging_enabled = false,
                      int alpha = 15, int beta = 18) {
  if (logging_enabled)
    PrintStep("Source", static_cast<int64_t>(source));
  Timer t;
  t.Start();
  pvector<NodeID> parent = InitParent(g);
  t.Stop();
  if (logging_enabled)
    PrintStep("i", t.Seconds());
  parent[source] = source;
  SlidingQueue<NodeID> queue(g.num_nodes());
  queue.push_back(source);
  queue.slide_window();
  Bitmap curr(g.num_nodes());
  curr.reset();
  Bitmap front(g.num_nodes());
  front.reset();
  int64_t edges_to_check = g.num_edges_directed();
  int64_t scout_count = g.out_degree(source);
  while (!queue.empty()) {
    if (scout_count > edges_to_check / alpha) {
      int64_t awake_count, old_awake_count;
      TIME_OP(t, QueueToBitmap(queue, front));
      if (logging_enabled)
        PrintStep("e", t.Seconds());
      awake_count = queue.size();
      queue.slide_window();
      do {
        t.Start();
        old_awake_count = awake_count;
        awake_count = BUStep(g, parent, front, curr);
        front.swap(curr);
        t.Stop();
        if (logging_enabled)
          PrintStep("bu", t.Seconds(), awake_count);
      } while ((awake_count >= old_awake_count) ||
               (awake_count > g.num_nodes() / beta));
      TIME_OP(t, BitmapToQueue(g, front, queue));
      if (logging_enabled)
        PrintStep("c", t.Seconds());
      scout_count = 1;
    } else {
      t.Start();
      edges_to_check -= scout_count;
      scout_count = TDStep(g, parent, queue);
      queue.slide_window();
      t.Stop();
      if (logging_enabled)
        PrintStep("td", t.Seconds(), queue.size());
    }
  }
  
  // Replace: #pragma omp parallel for
  GAPBS_PARALLEL_FOR(g.num_nodes(),
    [&](NodeID n) {
      if (parent[n] < -1)
        parent[n] = -1;
    });
  
  return parent;
}


void PrintBFSStats(const Graph &g, const pvector<NodeID> &bfs_tree) {
  int64_t tree_size = 0;
  int64_t n_edges = 0;
  for (NodeID n : g.vertices()) {
    if (bfs_tree[n] >= 0) {
      n_edges += g.out_degree(n);
      tree_size++;
    }
  }
  cout << "BFS Tree has " << tree_size << " nodes and ";
  cout << n_edges << " edges" << endl;
}


// BFS verifier does a serial BFS from same source and asserts:
// - parent[source] = source
// - parent[v] = u  =>  depth[v] = depth[u] + 1 (except for source)
// - parent[v] = u  => there is edge from u to v
// - all vertices reachable from source have a parent
bool BFSVerifier(const Graph &g, NodeID source,
                 const pvector<NodeID> &parent) {
  pvector<int> depth(g.num_nodes(), -1);
  depth[source] = 0;
  vector<NodeID> to_visit;
  to_visit.reserve(g.num_nodes());
  to_visit.push_back(source);
  for (auto it = to_visit.begin(); it != to_visit.end(); it++) {
    NodeID u = *it;
    for (NodeID v : g.out_neigh(u)) {
      if (depth[v] == -1) {
        depth[v] = depth[u] + 1;
        to_visit.push_back(v);
      }
    }
  }
  for (NodeID u : g.vertices()) {
    if ((depth[u] != -1) && (parent[u] != -1)) {
      if (u == source) {
        if (!((parent[u] == u) && (depth[u] == 0))) {
          cout << "Source wrong" << endl;
          return false;
        }
        continue;
      }
      bool parent_found = false;
      for (NodeID v : g.in_neigh(u)) {
        if (v == parent[u]) {
          if (depth[v] != depth[u] - 1) {
            cout << "Wrong depths for " << u << " & " << v << endl;
            return false;
          }
          parent_found = true;
          break;
        }
      }
      if (!parent_found) {
        cout << "Couldn't find edge from " << parent[u] << " to " << u << endl;
        return false;
      }
    } else if (depth[u] != parent[u]) {
      cout << "Reachability mismatch" << endl;
      return false;
    }
  }
  return true;
}


int main(int argc, char* argv[]) {
  CLApp cli(argc, argv, "breadth-first search");
  if (!cli.ParseArgs())
    return -1;
  
  // Set number of threads if specified via environment variable
  const char* env_threads = getenv("OMP_NUM_THREADS");
  if (env_threads) {
    int threads = std::atoi(env_threads);
    if (threads > 0) {
      gapbs_set_num_threads(threads);
      std::cout << "Using " << threads << " threads (from OMP_NUM_THREADS)" << std::endl;
    }
  }
  
  const char* gapbs_threads = getenv("GAPBS_NUM_THREADS");
  if (gapbs_threads) {
    int threads = std::atoi(gapbs_threads);
    if (threads > 0) {
      gapbs_set_num_threads(threads);
      std::cout << "Using " << threads << " threads (from GAPBS_NUM_THREADS)" << std::endl;
    }
  }
  
  Builder b(cli);
  Graph g = b.MakeGraph();
  SourcePicker<Graph> sp(g, cli.start_vertex());
  auto BFSBound = [&sp,&cli] (const Graph &g) {
    return DOBFS(g, sp.PickNext(), cli.logging_en());
  };
  SourcePicker<Graph> vsp(g, cli.start_vertex());
  auto VerifierBound = [&vsp] (const Graph &g, const pvector<NodeID> &parent) {
    return BFSVerifier(g, vsp.PickNext(), parent);
  };
  BenchmarkKernel(cli, g, BFSBound, PrintBFSStats, VerifierBound);
  return 0;
}
