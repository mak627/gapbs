// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <cinttypes>
#include <limits>
#include <iostream>
#include <queue>
#include <vector>
#include <algorithm>
#include <omp.h>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "platform_atomics.h"
#include "pvector.h"
#include "timer.h"
#include "util.h"

/*
GAP Benchmark Suite
Kernel: Single-source Shortest Paths (SSSP)
Author: Scott Beamer, Yunming Zhang

Returns array of distances for all vertices from given source vertex

This SSSP implementation makes use of the ∆-stepping algorithm [1]. The type
used for weights and distances (WeightT) is typedefined in benchmark.h. The
delta parameter (-d) should be set for each input graph. This implementation
incorporates a new bucket fusion optimization [2] that significantly reduces
the number of iterations (& barriers) needed.

The bins of width delta are actually all thread-local and of type std::vector
so they can grow but are otherwise capacity-proportional. Each iteration is
done in two phases separated by barriers. In the first phase, the current
shared bin is processed by all threads. As they find vertices whose distance
they are able to improve, they add them to their thread-local bins. During this
phase, each thread also votes on what the next bin should be (smallest
non-empty bin). In the next phase, each thread copies its selected
thread-local bin into the shared bin.

Once a vertex is added to a bin, it is not removed, even if its distance is
later updated and it now appears in a lower bin. We find ignoring vertices if
their distance is less than the min distance for the current bin removes
enough redundant work to be faster than removing the vertex from older bins.

The bucket fusion optimization [2] executes the next thread-local bin in
the same iteration if the vertices in the next thread-local bin have the
same priority as those in the current shared bin. This optimization greatly
reduces the number of iterations needed without violating the priority-based
execution order, leading to significant speedup on large diameter road networks.

[1] Ulrich Meyer and Peter Sanders. "δ-stepping: a parallelizable shortest path
    algorithm." Journal of Algorithms, 49(1):114–152, 2003.

[2] Yunming Zhang, Ajay Brahmakshatriya, Xinyi Chen, Laxman Dhulipala,
    Shoaib Kamil, Saman Amarasinghe, and Julian Shun. "Optimizing ordered graph
    algorithms with GraphIt." The 18th International Symposium on Code Generation
    and Optimization (CGO), pages 158-170, 2020.
*/


using namespace std;

const WeightT kDistInf = numeric_limits<WeightT>::max()/2;
const size_t kMaxBin = numeric_limits<size_t>::max()/2;
const size_t kBinSizeThreshold = 1000;
const size_t MAX_UPDATE_ITER = 5;
const size_t MAX_DEG_COUNT = 15;

inline
void RelaxEdges(const WGraph &g, NodeID u, WeightT delta,  WeightT threshold,
                pvector<WeightT> &dist, vector <vector<NodeID>> &local_bins,
                size_t& iter, size_t& curr_bin_index) {
  for (WNode wn : g.out_neigh(u)) {
    WeightT old_dist = dist[wn.v];
    WeightT new_dist = dist[u] + wn.w;
    while (new_dist < old_dist) {
      if (compare_and_swap(dist[wn.v], old_dist, new_dist)) {
        size_t dest_bin = curr_bin_index + (new_dist - threshold) / delta;
        if (iter < MAX_UPDATE_ITER - 1)
          dest_bin = min(dest_bin, curr_bin_index + 1);
        if (dest_bin >= local_bins.size())
          local_bins.resize(dest_bin+1);
        local_bins[dest_bin].push_back(wn.v);
        break;
      }
      old_dist = dist[wn.v];      // swap failed, recheck dist update & retry
    }
  }
}

inline WeightT medianWeight(const WGraph &g, NodeID u, bool returnMax = false){
    vector<WeightT> vec_wgt;
    for(WNode wn : g.out_neigh(u)){
        vec_wgt.push_back(wn.w);
        if (vec_wgt.size() == MAX_DEG_COUNT)
          break;
    }
    if (returnMax)
      return *max_element(vec_wgt.begin(), vec_wgt.end());
    int n = vec_wgt.size();
    // eliminate odd/even case since exact median is not needed
    nth_element(vec_wgt.begin(), vec_wgt.begin() + n/2, vec_wgt.end());
    return vec_wgt[n/2];
}

//cacheline size: 64B
//pvector size: 24B
pvector<WeightT> DeltaStep(const WGraph &g, NodeID source, WeightT delta) {
  Timer t;
  double t_tot = 0.;
  pvector<WeightT> dist(g.num_nodes(), kDistInf);
  dist[source] = 0;
  pvector<NodeID> frontier(g.num_edges_directed());
  // two element arrays for double buffering curr=iter&1, next=(iter+1)&1
  size_t shared_indexes[2] = {0, kMaxBin};
  size_t frontier_tails[2] = {1, 0};
  frontier[0] = source;
  //t.Start();
  int cnt = 1;
  int threshold = 0;
  //cout << "Startine Node--->" << source << endl;
  bool delta_update = true;
  #pragma omp parallel
  {
    vector<vector<NodeID>> local_bins(0);   //vector<NodeID> max_size() --> 4611686018427387903
    size_t iter = 0;
    int num_threads = omp_get_num_threads();
    while (shared_indexes[iter&1] != kMaxBin)
    {
      size_t &curr_bin_index = shared_indexes[iter&1];
      size_t &next_bin_index = shared_indexes[(iter+1)&1];
      size_t &curr_frontier_tail = frontier_tails[iter&1];
      size_t &next_frontier_tail = frontier_tails[(iter+1)&1];
      int delta_local = delta;

      if (delta_update)
      {
        #pragma omp single
        t.Start();
        #pragma omp for schedule(dynamic, 64)
        for (size_t i = 0; i < curr_frontier_tail; i++)
        {
          NodeID u = frontier[i];
          int out_degree = g.out_degree(u);
          if (dist[u] >= threshold &&  out_degree != 0)
          {
            delta_local = max(num_threads / out_degree, 1) * medianWeight(g, u, false);
            // Update delta based on thread local delta values
            fetch_and_add(delta, delta_local);
            fetch_and_add(cnt, 1);
	        }
        }
        #pragma omp single
        {
          if (cnt > 1)
          {
            delta /= cnt;
            cnt = 1;
          }
          t.Stop();
          t_tot += t.Seconds(); // overhead added by adaptive delta calculation part
        }
      }
      #pragma omp barrier
      #pragma omp for nowait schedule(dynamic, 64)
      for (size_t i = 0; i < curr_frontier_tail; i++)
      {
        NodeID u = frontier[i];
        if (dist[u] >= threshold && dist[u] < threshold + delta)
        {
            RelaxEdges(g, u, delta, threshold, dist, local_bins, iter, curr_bin_index);
        }
        else if (dist[u] >= threshold)
        {
	        size_t dest_bin = curr_bin_index;
	        dest_bin += (iter < MAX_UPDATE_ITER - 1) ? 1 : (dist[u] - threshold) / delta;
            if(dest_bin >= local_bins.size())
                local_bins.resize(dest_bin+1);
	        local_bins[dest_bin].push_back(u);
        }
      }

      // This part is bucket fusion optimization
      // If distances get updated such that some new nodes are again added to current local bin,
      // they're directly processed in current thread without having to go to next iteration.
      while (curr_bin_index < local_bins.size() &&
             !local_bins[curr_bin_index].empty() &&
             local_bins[curr_bin_index].size() < kBinSizeThreshold){
        vector<NodeID> curr_bin_copy = local_bins[curr_bin_index];
        local_bins[curr_bin_index].resize(0);
        for (NodeID u : curr_bin_copy)
          RelaxEdges(g, u, delta, threshold, dist, local_bins, iter, curr_bin_index);
      }
      for (size_t i=curr_bin_index; i < local_bins.size(); i++)
      {
        if (!local_bins[i].empty())
        {
          #pragma omp critical
          next_bin_index = min(next_bin_index, i);
          break;
        }
      }
      #pragma omp barrier
      #pragma omp single nowait
      {
        //t.Stop();
        //PrintStep(curr_bin_index, t.Millisecs(), curr_frontier_tail);
        //t.Start();
        // Set flag to change delta to true, if the next iteration is
        // within max update iterations and the bin index has changed
        if (iter < MAX_UPDATE_ITER - 1 && next_bin_index != curr_bin_index)
          delta_update = true;
        else
          delta_update = false;
        threshold += (next_bin_index - curr_bin_index) * delta;
        curr_bin_index = kMaxBin;
        curr_frontier_tail = 0;
      }
      if (next_bin_index < local_bins.size())
      {
        // Following step performs the action:
        // set copy_start = next_frontier_tail (1, if at first iteration, since only one source node in frontier)
        // atomically update next frontier tail value based on next bin size (cur frontier tail value for next step iterations)
        // since next_frontier tail is global, each thread updates this value to get the total frontier size for next step
        // next_frontier_tail = next_frontier_tail + local_bins[next_bin_index].size()
        size_t copy_start = fetch_and_add(next_frontier_tail,
                                          local_bins[next_bin_index].size());

        // Assign local next bin data to global frontier (because at each step, we process all nodes in current frontier)
        // *(frontier_address + copy_start) = local_bins[next_bin_index] values from start to end
        // Each threads copies local bin data to frontier (address offset by copy_start (next_frontier_tail) value)
        // previous frontier content get overwritten
        copy(local_bins[next_bin_index].begin(),
             local_bins[next_bin_index].end(), frontier.data() + copy_start);

        // resize the local_bins at next_bin_index to zero, since its nodes have been moved to frontier
        // if in the next step, further nodes are added due to updated distance, these will be new nodes, not present in frontier
        local_bins[next_bin_index].resize(0);
      }
      iter++;
      #pragma omp barrier
    }
    #pragma omp single
    {
        cout << "took " << iter << " iterations." << endl;
        //cout << "FINAL_DELTA_VALUE: " << delta << endl;
        PrintTime("Overhead Time", t_tot);
    }
  }
  return dist;
}


void PrintSSSPStats(const WGraph &g, const pvector<WeightT> &dist) {
  auto NotInf = [](WeightT d) { return d != kDistInf; };
  int64_t num_reached = count_if(dist.begin(), dist.end(), NotInf);
  cout << "SSSP Tree reaches " << num_reached << " nodes" << endl;
}


// Compares against simple serial implementation
bool SSSPVerifier(const WGraph &g, NodeID source,
                  const pvector<WeightT> &dist_to_test) {
  // Serial Dijkstra implementation to get oracle distances
  pvector<WeightT> oracle_dist(g.num_nodes(), kDistInf);
  oracle_dist[source] = 0;
  typedef pair<WeightT, NodeID> WN;
  priority_queue<WN, vector<WN>, greater<WN>> mq;
  mq.push(make_pair(0, source));
  while (!mq.empty()) {
    WeightT td = mq.top().first;
    NodeID u = mq.top().second;
    mq.pop();
    if (td == oracle_dist[u]) {
      for (WNode wn : g.out_neigh(u)) {
        if (td + wn.w < oracle_dist[wn.v]) {
          oracle_dist[wn.v] = td + wn.w;
          mq.push(make_pair(td + wn.w, wn.v));
        }
      }
    }
  }
  // Report any mismatches
  bool all_ok = true;
  for (NodeID n : g.vertices()) {
    if (dist_to_test[n] != oracle_dist[n]) {
      //cout << n << ": " << dist_to_test[n] << " != " << oracle_dist[n] << endl;
      all_ok = false;
    }
  }
  return all_ok;
}


int main(int argc, char* argv[]) {
  CLDelta<WeightT> cli(argc, argv, "single-source shortest-path");
  if (!cli.ParseArgs())
    return -1;
  WeightedBuilder b(cli);
  WGraph g = b.MakeGraph();
  SourcePicker<WGraph> sp(g, cli.start_vertex());
  auto SSSPBound = [&sp, &cli] (const WGraph &g) {
    return DeltaStep(g, sp.PickNext(), cli.delta());
  };
  SourcePicker<WGraph> vsp(g, cli.start_vertex());
  auto VerifierBound = [&vsp] (const WGraph &g, const pvector<WeightT> &dist) {
    return SSSPVerifier(g, vsp.PickNext(), dist);
  };
  BenchmarkKernel(cli, g, SSSPBound, PrintSSSPStats, VerifierBound);
  return 0;
}
