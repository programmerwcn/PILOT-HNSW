#pragma once

#include "visited_list_pool.h"
#include "hnswlib.h"
#include "adsampling.h"
#include "logging.h"
#include "rf_predictor.h"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <cstdlib>
#include <assert.h>
#include <unordered_set>
#include <unordered_map>
#include <list>
#include <deque>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <thread>
#include <cmath>
#include <limits>
#include <string>

using namespace std;
using namespace adsampling;

namespace hnswlib {
    inline std::mt19937& get_random_engine() {
        static thread_local std::random_device rd;
        static thread_local std::mt19937 gen(rd());
        return gen;
    }

    typedef unsigned int tableint;
    typedef unsigned int linklistsizeint;

    template<typename dist_t>
    class HierarchicalNSW : public AlgorithmInterface<dist_t> {
    public:
        static const tableint max_update_element_locks = 65536;

        /** Runtime toggle for all \c HNSW_LOG_* output (global logger). \c LogMode::RUN still disables logs. */
        static void setLoggingOutputEnabled(bool enabled) {
            setHnswLoggingOutputEnabled(enabled);
        }
        static bool isLoggingOutputEnabled() {
            return isHnswLoggingOutputEnabled();
        }

        HierarchicalNSW(SpaceInterface<dist_t> *s) {
        }

        HierarchicalNSW(SpaceInterface<dist_t> *s, const std::string &location, bool nmslib = false, size_t max_elements=0) {
            loadIndex(location, s, max_elements);
        }

        HierarchicalNSW(SpaceInterface<dist_t> *s, size_t max_elements, size_t M = 16, size_t ef_construction = 200, size_t random_seed = 100) :
                link_list_locks_(max_elements), link_list_update_locks_(max_update_element_locks), element_levels_(max_elements) {
            max_elements_ = max_elements;

            num_deleted_ = 0;
            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();
            M_ = M;
            maxM_ = M_;
            maxM0_ = M_ * 2;
            ef_construction_ = std::max(ef_construction,M_);
            ef_ = 10;

            level_generator_.seed(random_seed);
            update_probability_generator_.seed(random_seed + 1);

            size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
            size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
            offsetData_ = size_links_level0_;
            label_offset_ = size_links_level0_ + data_size_;
            offsetLevel0_ = 0;

            data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
            if (data_level0_memory_ == nullptr)
                throw std::runtime_error("Not enough memory");

            cur_element_count = 0;

            visited_list_pool_ = new VisitedListPool(1, max_elements);

            //initializations for special treatment of the first node
            enterpoint_node_ = -1;
            maxlevel_ = -1;

            linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
            if (linkLists_ == nullptr)
                throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
            size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
            mult_ = 1 / log(1.0 * M_);
            revSize_ = 1.0 / mult_;
        }

        struct CompareByFirst {
            constexpr bool operator()(std::pair<dist_t, tableint> const &a,
                                      std::pair<dist_t, tableint> const &b) const noexcept {
                return a.first < b.first;
            }
        };

        // Per-node counters for HNSW construction-time pruning events emitted
        // by getNeighborsByHeuristic2. Allocated only when
        // enablePruningInstrumentation(N) is called; otherwise the pointer
        // stays null and the heuristic's logging branch is skipped.
        struct PruningCounters {
            std::vector<int64_t> host_self_insert;  // R hosted, R's own insertion phase
            std::vector<int64_t> host_repruning;    // R hosted, later re-prune phase
            std::vector<int64_t> victim;            // R was the pruned candidate
        };

        ~HierarchicalNSW() {

            free(data_level0_memory_);
            for (tableint i = 0; i < cur_element_count; i++) {
                if (element_levels_[i] > 0)
                    free(linkLists_[i]);
            }
            free(linkLists_);
            delete visited_list_pool_;
            if (pruning_counters_ != nullptr) {
                delete pruning_counters_;
                pruning_counters_ = nullptr;
            }
        }

        size_t max_elements_;
        size_t cur_element_count;
        size_t size_data_per_element_;
        size_t size_links_per_element_;
        size_t num_deleted_;

        size_t M_;
        size_t maxM_;
        size_t maxM0_;
        size_t ef_construction_;

        double mult_, revSize_;
        int maxlevel_;


        VisitedListPool *visited_list_pool_;
        std::mutex cur_element_count_guard_;

        std::vector<std::mutex> link_list_locks_;

        // Locks to prevent race condition during update/insert of an element at same time.
        // Note: Locks for additions can also be used to prevent this race condition if the querying of KNN is not exposed along with update/inserts i.e multithread insert/update/query in parallel.
        std::vector<std::mutex> link_list_update_locks_;
        tableint enterpoint_node_;

        size_t size_links_level0_;
        size_t offsetData_, offsetLevel0_;

        char *data_level0_memory_;
        char **linkLists_;
        std::vector<int> element_levels_;

        // Pruning instrumentation: null by default. enablePruningInstrumentation
        // allocates the underlying counter vectors. getNeighborsByHeuristic2
        // increments counters when this pointer is non-null and phase >= 0.
        PruningCounters* pruning_counters_ = nullptr;

        size_t data_size_;

        size_t label_offset_;
        DISTFUNC<dist_t> fstdistfunc_;
        void *dist_func_param_;
        std::unordered_map<labeltype, tableint> label_lookup_;  // external label -> internal id

        std::default_random_engine level_generator_;
        std::default_random_engine update_probability_generator_;


        inline labeltype getExternalLabel(tableint internal_id) const {
            labeltype return_label;
            memcpy(&return_label,(data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
            return return_label;
        }

        inline tableint getInternalId(labeltype label) const {
            tableint label_c;
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
                throw std::runtime_error("Label not found");
            }
            label_c = search->second;
            return label_c;
        }

        inline void setExternalLabel(tableint internal_id, labeltype label) const {
            memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(labeltype));
        }

        inline labeltype *getExternalLabeLp(tableint internal_id) const {
            return (labeltype *) (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
        }

        inline char *getDataByInternalId(tableint internal_id) const {
            return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
        }

        int getRandomLevel(double reverse_size) {
            std::uniform_real_distribution<double> distribution(0.0, 1.0);
            double r = -log(distribution(get_random_engine())) * reverse_size;
            return (int) r;
        }

        // Reproducible random-level generator based on per-index seed.
        int getRandomLevelSeeded(double reverse_size) {
            std::uniform_real_distribution<double> distribution(0.0, 1.0);
            double r = -log(distribution(level_generator_)) * reverse_size;
            return (int) r;
        }


        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>> candidateSet;

            dist_t lowerBound;
            if (!isMarkedDeleted(ep_id)) {
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
                // HNSW_LOG_DEBUG("[searchBaseLayer] Entry point: ", ep_id, ", dist: ", dist);
                top_candidates.emplace(dist, ep_id);
                lowerBound = dist;
                candidateSet.emplace(-dist, ep_id);
            } else {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidateSet.emplace(-lowerBound, ep_id);
                // HNSW_LOG_DEBUG("[searchBaseLayer] Entry point marked deleted: ", ep_id);
            }
            visited_array[ep_id] = visited_array_tag;

            while (!candidateSet.empty()) {
                std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
                // HNSW_LOG_DEBUG("[searchBaseLayer] Pop candidate: node ", curr_el_pair.second, ", dist: ", -curr_el_pair.first);
                if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
                    // HNSW_LOG_DEBUG("[searchBaseLayer] Terminate: smallest in S > largest in R and R full.");
                    break;
                }
                candidateSet.pop();

                tableint curNodeNum = curr_el_pair.second;

                std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

                int *data;
                if (layer == 0) {
                    data = (int*)get_linklist0(curNodeNum);
                } else {
                    data = (int*)get_linklist(curNodeNum, layer);
                }
                size_t size = getListCount((linklistsizeint*)data);


                for (size_t j = 0; j < size; j++) {
                    tableint candidate_id = *(data + j + 1);
                    if (visited_array[candidate_id] == visited_array_tag) continue;
                    visited_array[candidate_id] = visited_array_tag;
                    char *currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                    // HNSW_LOG_DEBUG("[searchBaseLayer] Explore neighbor: ", candidate_id, ", dist: ", dist1);
                    if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                        candidateSet.emplace(-dist1, candidate_id);
                        // HNSW_LOG_DEBUG("[searchBaseLayer]   Add to candidateSet: ", candidate_id, ", dist: ", dist1);
                        if (!isMarkedDeleted(candidate_id)) {
                            top_candidates.emplace(dist1, candidate_id);
                            // HNSW_LOG_DEBUG("[searchBaseLayer]   Add to top_candidates: ", candidate_id, ", dist: ", dist1);
                        }
                        if (top_candidates.size() > ef_construction_) {
                            auto removed = top_candidates.top();
                            top_candidates.pop();
                            // HNSW_LOG_DEBUG("[searchBaseLayer]   Prune top_candidates: ", removed.second, ", dist: ", removed.first);
                        }
                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
            // HNSW_LOG_DEBUG("[searchBaseLayer] Final top_candidates:");
            auto copy = top_candidates;
            while (!copy.empty()) {
                auto p = copy.top();
                // HNSW_LOG_DEBUG("  node: ", p.second, ", dist: ", p.first);
                copy.pop();
            }
            visited_list_pool_->releaseVisitedList(vl);
            return top_candidates;
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayer_timed(tableint ep_id, const void *data_point, int layer) {
            auto search_start = std::chrono::high_resolution_clock::now();
            // HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED] Starting searchBaseLayer for ep_id=", ep_id, ", layer=", layer);
            
            HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED] Parameters: ef_construction_=", ef_construction_, ", layer=", layer, ", ep_id=", ep_id);
            
            auto visited_list_start = std::chrono::high_resolution_clock::now();
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;
            auto visited_list_end = std::chrono::high_resolution_clock::now();
            auto visited_list_duration = std::chrono::duration_cast<std::chrono::microseconds>(visited_list_end - visited_list_start);
            // HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED] Visited list setup: ", visited_list_duration.count(), " us");
            
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>> candidateSet;

            dist_t lowerBound;
            
            // Entry point processing
            auto entry_point_start = std::chrono::high_resolution_clock::now();
            if (!isMarkedDeleted(ep_id)) {
                auto dist_calc_start = std::chrono::high_resolution_clock::now();
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
                // auto dist_calc_end = std::chrono::high_resolution_clock::now();
                // auto dist_calc_duration = std::chrono::duration_cast<std::chrono::microseconds>(dist_calc_end - dist_calc_start);
                // HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED] Entry point distance calculation: ", dist_calc_duration.count(), " us, dist=", dist);
                
                top_candidates.emplace(dist, ep_id);
                lowerBound = dist;
                candidateSet.emplace(-dist, ep_id);
            } else {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidateSet.emplace(-lowerBound, ep_id);
                // HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED] Entry point marked deleted: ", ep_id);
            }
            auto entry_point_end = std::chrono::high_resolution_clock::now();
            auto entry_point_duration = std::chrono::duration_cast<std::chrono::microseconds>(entry_point_end - entry_point_start);
            HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED] Entry point processing: ", entry_point_duration.count(), " us");
            
            visited_array[ep_id] = visited_array_tag;

            int iteration_count = 0;
            int total_neighbors_explored = 0;
            int total_distance_calculations = 0;
            long total_lock_time = 0;
            long total_distance_time = 0;
            long total_linklist_access_time = 0;

            // Main search loop
            auto main_loop_start = std::chrono::high_resolution_clock::now();
            while (!candidateSet.empty()) {
                iteration_count++;
                auto iteration_start = std::chrono::high_resolution_clock::now();
                
                std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
                HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED] Iteration ", iteration_count, ": processing node ", curr_el_pair.second, 
                         ", dist=", -curr_el_pair.first, ", candidates_size=", candidateSet.size(), 
                         ", top_candidates_size=", top_candidates.size());
                
                if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
                    HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED] Terminating: smallest in S > largest in R and R full");
                    break;
                }
                candidateSet.pop();

                tableint curNodeNum = curr_el_pair.second;
                HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED]   About to acquire lock for node ", curNodeNum);
                
                // Check if curNodeNum is valid
                if (curNodeNum < 0 || curNodeNum >= max_elements_) {
                    HNSW_LOG_ERROR("[SEARCH_BASE_TIMED] ERROR: Invalid curNodeNum: ", curNodeNum, 
                             " (max_elements_: ", max_elements_, ")");
                    continue;
                }
                
                HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED]   curNodeNum is valid, accessing lock array");
                
                // Check lock array size
                HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED]   Lock array size: ", link_list_locks_.size());
                if (curNodeNum >= link_list_locks_.size()) {
                    HNSW_LOG_ERROR("[SEARCH_BASE_TIMED] ERROR: curNodeNum ", curNodeNum, 
                             " >= lock array size ", link_list_locks_.size());
                    continue;
                }

                // Lock acquisition timing with deadlock prevention
                auto lock_start = std::chrono::high_resolution_clock::now();
                HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED]   Attempting to lock node ", curNodeNum);
                
                // Use try_lock to prevent deadlock - if we can't get the lock, skip this iteration
                std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum], std::defer_lock);
                if (!lock.try_lock()) {
                    HNSW_LOG_WARN("[SEARCH_BASE_TIMED] WARNING: Failed to acquire lock for node ", curNodeNum, 
                             " - lock is already held, skipping this iteration to prevent deadlock");
                    continue;
                }
                
                HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED]   Successfully acquired lock for node ", curNodeNum);
                auto lock_end = std::chrono::high_resolution_clock::now();
                auto lock_duration = std::chrono::duration_cast<std::chrono::microseconds>(lock_end - lock_start);
                total_lock_time += lock_duration.count();

                // Link list access timing
                auto linklist_start = std::chrono::high_resolution_clock::now();
                HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED]   About to access linklist for node ", curNodeNum, ", layer=", layer);
                
                int *data;
                if (layer == 0) {
                    HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED]   Calling get_linklist0(", curNodeNum, ")");
                    data = (int*)get_linklist0(curNodeNum);
                    HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED]   get_linklist0 returned: ", (void*)data);
                } else {
                    if (curNodeNum < cur_element_count && layer > element_levels_[curNodeNum]) {
                        continue;
                    }
                    HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED]   Calling get_linklist(", curNodeNum, ", ", layer, ")");
                    data = (int*)get_linklist(curNodeNum, layer);
                    HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED]   get_linklist returned: ", (void*)data);
                }
                
                size_t size;
                if (data == nullptr) {
                    HNSW_LOG_ERROR("[SEARCH_BASE_TIMED] ERROR: Linklist data is null for node ", curNodeNum, "!");
                    size = 0;
                } else {
                    HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED]   About to call getListCount");
                    size = getListCount((linklistsizeint*)data);
                    HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED]   getListCount returned: ", size);
                }
                
                auto linklist_end = std::chrono::high_resolution_clock::now();
                auto linklist_duration = std::chrono::duration_cast<std::chrono::microseconds>(linklist_end - linklist_start);
                total_linklist_access_time += linklist_duration.count();
                
                HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED]   Node ", curNodeNum, " has ", size, " neighbors");

                // Neighbor exploration
                int neighbors_explored_this_iteration = 0;
                int distance_calcs_this_iteration = 0;
                auto neighbor_exploration_start = std::chrono::high_resolution_clock::now();
                
                for (size_t j = 0; j < size; j++) {
                    tableint candidate_id = *(data + j + 1);
                    if (candidate_id < cur_element_count && layer > element_levels_[candidate_id]) {
                        continue;
                    }
                    if (visited_array[candidate_id] == visited_array_tag) {
                        HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED]     Skipping already visited neighbor ", candidate_id);
                        continue;
                    }
                    visited_array[candidate_id] = visited_array_tag;
                    neighbors_explored_this_iteration++;
                    total_neighbors_explored++;
                    
                    HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED]     About to get data for candidate_id: ", candidate_id);
                    char *currObj1 = (getDataByInternalId(candidate_id));
                    HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED]     getDataByInternalId returned: ", (void*)currObj1);
                    
                    // Distance calculation timing
                    auto dist_calc_start = std::chrono::high_resolution_clock::now();
                    HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED]     About to calculate distance");
                    dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                    HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED]     Distance calculation completed: ", dist1);
                    auto dist_calc_end = std::chrono::high_resolution_clock::now();
                    auto dist_calc_duration = std::chrono::duration_cast<std::chrono::microseconds>(dist_calc_end - dist_calc_start);
                    total_distance_time += dist_calc_duration.count();
                    distance_calcs_this_iteration++;
                    total_distance_calculations++;
                    
                    HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED]     Explored neighbor ", candidate_id, ", dist=", dist1, 
                             " (calc_time=", dist_calc_duration.count(), " us)");
                    
                    if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                        candidateSet.emplace(-dist1, candidate_id);
                        HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED]       Added to candidateSet");

                        if (!isMarkedDeleted(candidate_id)) {
                            top_candidates.emplace(dist1, candidate_id);
                            HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED]       Added to top_candidates");
                        }

                        if (top_candidates.size() > ef_construction_) {
                            auto removed = top_candidates.top();
                            top_candidates.pop();
                            HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED]       Pruned top_candidates: removed node ", removed.second, 
                                     " with dist=", removed.first);
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    } else {
                        HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED]       Skipped: not better than current bound");
                    }
                }
                
                auto neighbor_exploration_end = std::chrono::high_resolution_clock::now();
                auto neighbor_exploration_duration = std::chrono::duration_cast<std::chrono::microseconds>(neighbor_exploration_end - neighbor_exploration_start);
                
                auto iteration_end = std::chrono::high_resolution_clock::now();
                auto iteration_duration = std::chrono::duration_cast<std::chrono::microseconds>(iteration_end - iteration_start);
                
                HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED]   Iteration ", iteration_count, " completed in ", iteration_duration.count(), " us");
                HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED]     - Lock time: ", lock_duration.count(), " us");
                HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED]     - Linklist access time: ", linklist_duration.count(), " us");
                HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED]     - Neighbor exploration time: ", neighbor_exploration_duration.count(), " us");
                HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED]     - Neighbors explored: ", neighbors_explored_this_iteration);
                HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED]     - Distance calculations: ", distance_calcs_this_iteration);
            }
            
            HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED] Search loop terminated after ", iteration_count, " iterations");
            HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED] Final state: candidateSet.size()=", candidateSet.size(), ", top_candidates.size()=", top_candidates.size());
            
            auto main_loop_end = std::chrono::high_resolution_clock::now();
            auto main_loop_duration = std::chrono::duration_cast<std::chrono::microseconds>(main_loop_end - main_loop_start);
            
            // Report performance issues
            if (main_loop_duration.count() > 1000000) { // If search took more than 1 second
                HNSW_LOG_WARN("[SEARCH_BASE_TIMED] PERFORMANCE WARNING: Search took ", main_loop_duration.count() / 1000.0, 
                         " ms - this is extremely slow! Consider reducing concurrency or optimizing lock strategy.");
            }
            
            // Final results processing
            auto final_processing_start = std::chrono::high_resolution_clock::now();
            // HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED] Final top_candidates:");
            auto copy = top_candidates;
            while (!copy.empty()) {
                auto p = copy.top();
                // HNSW_LOG_DEBUG("  node: ", p.second, ", dist: ", p.first);
                copy.pop();
            }
            
            auto visited_list_cleanup_start = std::chrono::high_resolution_clock::now();
            visited_list_pool_->releaseVisitedList(vl);
            auto visited_list_cleanup_end = std::chrono::high_resolution_clock::now();
            auto visited_list_cleanup_duration = std::chrono::duration_cast<std::chrono::microseconds>(visited_list_cleanup_end - visited_list_cleanup_start);
            
            auto final_processing_end = std::chrono::high_resolution_clock::now();
            auto final_processing_duration = std::chrono::duration_cast<std::chrono::microseconds>(final_processing_end - final_processing_start);
            
            auto search_end = std::chrono::high_resolution_clock::now();
            auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(search_end - search_start);
            
            // Summary statistics
            // HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED] ========== SEARCH BASE LAYER SUMMARY ==========");
            // HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED] Total search time: ", total_duration.count(), " us");
            // HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED] Breakdown:");
            // HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED]   - Visited list setup: ", visited_list_duration.count(), " us (", (100.0 * visited_list_duration.count() / total_duration.count()), "%)");
            // HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED]   - Entry point processing: ", entry_point_duration.count(), " us (", (100.0 * entry_point_duration.count() / total_duration.count()), "%)");
            // HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED]   - Main search loop: ", main_loop_duration.count(), " us (", (100.0 * main_loop_duration.count() / total_duration.count()), "%)");
            // HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED]   - Final processing: ", final_processing_duration.count(), " us (", (100.0 * final_processing_duration.count() / total_duration.count()), "%)");
            // HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED] Main loop breakdown:");
            // HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED]   - Total lock time: ", total_lock_time, " us (", (100.0 * total_lock_time / main_loop_duration.count()), "%)");
            // HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED]   - Total linklist access time: ", total_linklist_access_time, " us (", (100.0 * total_linklist_access_time / main_loop_duration.count()), "%)");
            // HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED]   - Total distance calculation time: ", total_distance_time, " us (", (100.0 * total_distance_time / main_loop_duration.count()), "%)");
            // HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED] Statistics:");
            // HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED]   - Total iterations: ", iteration_count);
            // HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED]   - Total neighbors explored: ", total_neighbors_explored);
            // HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED]   - Total distance calculations: ", total_distance_calculations);
            // HNSW_LOG_DEBUG("[SEARCH_BASE_TIMED]   - Average distance calc time: ", (total_distance_calculations > 0 ? total_distance_time / total_distance_calculations : 0), " us");
            // HNSW_LOG_DEBUG("================================================");

            return top_candidates;
        }

        mutable std::atomic<long> metric_distance_computations;
        mutable std::atomic<long> metric_hops;

        template <bool has_deletions, bool collect_metrics=false>
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>>
        searchBaseLayerST(tableint ep_id, const void *data_point, size_t ef, Metric* metric = nullptr) const {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            // top_candidates - the result set R
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>> top_candidates;
            // candidate_set  - the search set S
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

            dist_t lowerBound;
            // Insert the entry point to the result and search set with its exact distance as a key. 
            if (!has_deletions || !isMarkedDeleted(ep_id)) {
#ifdef COUNT_DIST_TIME
                StopW stopw = StopW();
#endif
#ifndef ED2IP
            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
#else
            dist_t dist = hnswlib::L2Sqr_by_IP(data_point, getDataByInternalId(ep_id), getExternalLabel(ep_id), dist_func_param_);
#endif
#ifdef COUNT_DIST_TIME
                adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif          
                adsampling::tot_dist_calculation++;
                adsampling::tot_full_dist ++;
                if(metric != nullptr)
                    metric->ndc++;
                lowerBound = dist;
                top_candidates.emplace(dist, ep_id);
                candidate_set.emplace(-dist, ep_id);
                HNSW_LOG_DEBUG("[searchBaseLayerST] Entry point: ", ep_id, ", dist: ", dist);
            } 
            else {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidate_set.emplace(-lowerBound, ep_id);
                HNSW_LOG_DEBUG("[searchBaseLayerST] Entry point marked deleted: ", ep_id);
            }

            visited_array[ep_id] = visited_array_tag;
            int cnt_visit = 0;

            // Iteratively generate candidates and conduct DCOs to maintain the result set R.
            while (!candidate_set.empty()) {
                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
                HNSW_LOG_DEBUG("[searchBaseLayerST] Pop candidate: node ", current_node_pair.second, ", dist: ", -current_node_pair.first);

                // When the smallest object in S has its distance larger than the largest in R, terminate the algorithm.
                if ((-current_node_pair.first) > lowerBound && (top_candidates.size() == ef || has_deletions == false)) {
                    HNSW_LOG_DEBUG("[searchBaseLayerST] Terminate: smallest in S > largest in R and R full.");
                    break;
                }
                candidate_set.pop();
                adsampling::tot_hops++;
                // Fetch the smallest object in S. 
                tableint current_node_id = current_node_pair.second;
                int *data = (int *) get_linklist0(current_node_id);
                size_t size = getListCount((linklistsizeint*)data);
                if(collect_metrics){
                    metric_hops++;
                    metric_distance_computations+=size;
                }
                metric->steps++;

                // Enumerate all the neighbors of the object and view them as candidates of KNNs. 
                for (size_t j = 1; j <= size; j++) {
                    int candidate_id = *(data + j);
                    if (!(visited_array[candidate_id] == visited_array_tag)) {
                        cnt_visit ++;
                        visited_array[candidate_id] = visited_array_tag;
                        char *currObj1 = (getDataByInternalId(candidate_id));
#ifdef COUNT_DIST_TIME
                        StopW stopw = StopW();
#endif
#ifndef ED2IP
            dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);
#else
            dist_t dist = hnswlib::L2Sqr_by_IP(data_point, currObj1, getExternalLabel(candidate_id), dist_func_param_);
#endif
#ifdef COUNT_DIST_TIME
                        adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif                  
                        adsampling::tot_full_dist++;
                        if(metric != nullptr)
                            metric->ndc++;
                        HNSW_LOG_DEBUG("[searchBaseLayerST] Explore neighbor: ", candidate_id, ", dist: ", dist);
                        if (top_candidates.size() < ef || lowerBound > dist) {                      
                            candidate_set.emplace(-dist, candidate_id);
                            HNSW_LOG_DEBUG("[searchBaseLayerST]   Add to candidate_set: ", candidate_id, ", dist: ", dist);
                            if (!has_deletions || !isMarkedDeleted(candidate_id)) {
                                top_candidates.emplace(dist, candidate_id);
                                HNSW_LOG_DEBUG("[searchBaseLayerST]   Add to top_candidates: ", candidate_id, ", dist: ", dist);
                            }
                            if (top_candidates.size() > ef) {
                                auto removed = top_candidates.top();
                                top_candidates.pop();
                                HNSW_LOG_DEBUG("[searchBaseLayerST]   Prune top_candidates: ", removed.second, ", dist: ", removed.first);
                            }
                            if (!top_candidates.empty())
                                lowerBound = top_candidates.top().first;
                        }
                    }
                }
            }
            adsampling::tot_dist_calculation += cnt_visit;
            HNSW_LOG_DEBUG("[searchBaseLayerST] Final top_candidates:");
            auto copy = top_candidates;
            while (!copy.empty()) {
                auto p = copy.top();
                HNSW_LOG_DEBUG("  node: ", p.second, ", dist: ", p.first);
                copy.pop();
            }
            visited_list_pool_->releaseVisitedList(vl);
            return top_candidates;
        }

        struct BridgeRescueParams {
            float min_parent_outside_reject_frac = 0.8f;
            float max_parent_push_frac = 0.2f;
            float max_boundary_ratio = 1.15f;
            size_t min_candidate_queue_size = 400;
            size_t rescue_every = 20;
            size_t max_rescue_expansions = 20;
            size_t rescue_buffer_size = 64;

            // RF-driven rescue (used by searchBaseLayerRFBridgeRescue). Leave
            // rf_predictor == nullptr to keep the legacy rule-based path.
            const RandomForestPredictor* rf_predictor = nullptr;
            float rf_threshold = 0.5f;
            size_t rf_max_rescues_per_query = 8;

            // Optional: when set, the RF search records a recall curve in the
            // returned stats. gt_set contains EXTERNAL labels of the top-K GT
            // for this query; recall_curve_k is the K used to compute recall.
            const std::unordered_set<labeltype>* gt_set = nullptr;
            size_t recall_curve_k = 10;
            float target_recall = 0.9f;

            // Search budget for searchBaseLayerRFBridgeRescue. When > 0, the
            // search runs past natural HNSW termination — it records the
            // termination step but keeps expanding until candidate_set is
            // empty or this hop budget is exhausted. 0 means use standard
            // HNSW termination (break at termination point).
            int search_budget = 0;

            // Per-node feature table for the v13 RF (30 features). Layout:
            //   node_features[external_label * per_node_stride + j]
            // The RF feature vector at predict time is [cheap[0..n_cheap-1],
            // node_features[label * stride .. label * stride + per_node_stride-1]].
            // When per_node_features == nullptr the predictor must be the
            // 11-feature cheap-only model.
            const float* per_node_features = nullptr;
            size_t per_node_stride = 0;        // = number of per-node features
            size_t per_node_medians_size = 0;  // = per_node_stride
            const float* per_node_medians = nullptr;
        };

        struct BridgeRescueStats {
            int steps = 0;
            int distance_computations = 0;
            int normal_expansions = 0;
            int rescue_expansions = 0;
            int blocked_parent_count = 0;
            int rescue_candidates_seen = 0;
            int rescue_candidates_added = 0;
            int rescue_candidates_dropped = 0;
            int parent_checked_total = 0;
            int parent_pushed_total = 0;
            int parent_outside_rejected_total = 0;

            // RF-driven rescue counters
            int rf_rescue_count = 0;          // # bridges fired
            int rf_neighbors_expanded = 0;    // total neighbor edges explored
            int rf_neighbors_inserted = 0;    // edges inserted into top_candidates
            // Per-fire log: external label of fired node + its RF score + the
            // search step at which it fired. Populated only when the search
            // wants attribution (callers can leave empty).
            std::vector<std::tuple<labeltype, float, int>> rf_fires;

            // recall_curve: (step, recall, candidate_rank) tuples emitted
            // whenever recall@k strictly increases.
            std::vector<std::tuple<int, float, int>> recall_curve;
            int total_distance_calcs = 0;     // mirror of distance_computations for convenience

            // The first step at which the standard HNSW termination condition
            // fires. With search_budget > 0 the loop continues past this
            // point so we can observe extra recall events.
            int early_termination_step = -1;
            int step_target_recall_achieved = -1;
            bool target_recall_achieved = false;
            float final_recall = 0.0f;
        };

        struct BridgeRescueSearchResult {
            std::priority_queue<std::pair<dist_t, labeltype>> neighbors;
            BridgeRescueStats stats;
        };

        struct BridgeRescueBaseLayerResult {
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>> top_candidates;
            BridgeRescueStats stats;
        };

        struct BridgeRescueCandidate {
            dist_t distance = 0;
            tableint node_id = 0;
            tableint parent_id = 0;
            float boundary_ratio = 0.0f;
            float score = 0.0f;
        };

        struct CompareBridgeRescueCandidate {
            bool operator()(const BridgeRescueCandidate& a, const BridgeRescueCandidate& b) const {
                if (a.score == b.score) return a.node_id > b.node_id;
                return a.score > b.score;
            }
        };

        template <bool has_deletions, bool collect_metrics=false>
        BridgeRescueBaseLayerResult searchBaseLayerBridgeRescue(
            tableint ep_id,
            const void *data_point,
            size_t ef,
            const BridgeRescueParams& params,
            Metric* metric = nullptr) const {
            BridgeRescueBaseLayerResult result;

            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;
            std::priority_queue<BridgeRescueCandidate, std::vector<BridgeRescueCandidate>, CompareBridgeRescueCandidate> rescue_buffer;
            std::vector<unsigned char> expanded(cur_element_count, 0);

            auto add_top_candidate = [&](tableint candidate_id, dist_t dist, dist_t& lowerBound) {
                if (!has_deletions || !isMarkedDeleted(candidate_id)) {
                    top_candidates.emplace(dist, candidate_id);
                }
                if (top_candidates.size() > ef) {
                    top_candidates.pop();
                }
                if (!top_candidates.empty()) {
                    lowerBound = top_candidates.top().first;
                }
            };

            dist_t lowerBound;
            if (!has_deletions || !isMarkedDeleted(ep_id)) {
#ifndef ED2IP
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
#else
                dist_t dist = hnswlib::L2Sqr_by_IP(data_point, getDataByInternalId(ep_id), getExternalLabel(ep_id), dist_func_param_);
#endif
                adsampling::tot_dist_calculation++;
                adsampling::tot_full_dist++;
                if (metric != nullptr) metric->ndc++;
                result.stats.distance_computations++;
                lowerBound = dist;
                top_candidates.emplace(dist, ep_id);
                candidate_set.emplace(-dist, ep_id);
            } else {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidate_set.emplace(-lowerBound, ep_id);
            }
            visited_array[ep_id] = visited_array_tag;

            auto expand_node = [&](tableint current_node_id, dist_t current_dist, bool rescue_expansion) {
                if (current_node_id >= expanded.size() || expanded[current_node_id]) return;
                expanded[current_node_id] = 1;
                result.stats.steps++;
                if (metric != nullptr) metric->steps++;
                if (rescue_expansion) {
                    result.stats.rescue_expansions++;
                } else {
                    result.stats.normal_expansions++;
                }
                adsampling::tot_hops++;

                int *data = (int *) get_linklist0(current_node_id);
                size_t size = getListCount((linklistsizeint*)data);
                if (collect_metrics) {
                    metric_hops++;
                    metric_distance_computations += size;
                }

                struct LocalReject {
                    tableint node_id;
                    dist_t distance;
                    float boundary_ratio;
                };
                std::vector<LocalReject> local_rejects;
                int checked = 0;
                int pushed = 0;
                int outside_rejected = 0;

                for (size_t j = 1; j <= size; j++) {
                    tableint candidate_id = *(data + j);
                    if (visited_array[candidate_id] == visited_array_tag) {
                        continue;
                    }

                    visited_array[candidate_id] = visited_array_tag;
                    checked++;

                    char *currObj1 = getDataByInternalId(candidate_id);
#ifndef ED2IP
                    dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);
#else
                    dist_t dist = hnswlib::L2Sqr_by_IP(data_point, currObj1, getExternalLabel(candidate_id), dist_func_param_);
#endif
                    adsampling::tot_full_dist++;
                    if (metric != nullptr) metric->ndc++;
                    result.stats.distance_computations++;

                    if (top_candidates.size() < ef || lowerBound > dist) {
                        candidate_set.emplace(-dist, candidate_id);
                        pushed++;
                        add_top_candidate(candidate_id, dist, lowerBound);
                    } else {
                        outside_rejected++;
                        result.stats.rescue_candidates_seen++;
                        float ratio = lowerBound > 0
                            ? static_cast<float>(dist / lowerBound)
                            : std::numeric_limits<float>::infinity();
                        local_rejects.push_back({candidate_id, dist, ratio});
                    }
                }

                result.stats.parent_checked_total += checked;
                result.stats.parent_pushed_total += pushed;
                result.stats.parent_outside_rejected_total += outside_rejected;

                if (checked == 0) return;
                float outside_frac = static_cast<float>(outside_rejected) / static_cast<float>(checked);
                float push_frac = static_cast<float>(pushed) / static_cast<float>(checked);
                bool blocked_parent =
                    candidate_set.size() >= params.min_candidate_queue_size &&
                    outside_frac >= params.min_parent_outside_reject_frac &&
                    push_frac <= params.max_parent_push_frac;
                if (!blocked_parent) return;

                result.stats.blocked_parent_count++;
                for (const auto& rejected : local_rejects) {
                    if (rejected.boundary_ratio > params.max_boundary_ratio) continue;
                    if (rejected.node_id >= expanded.size() || expanded[rejected.node_id]) continue;
                    if (params.rescue_buffer_size == 0 || rescue_buffer.size() >= params.rescue_buffer_size) {
                        result.stats.rescue_candidates_dropped++;
                        continue;
                    }
                    BridgeRescueCandidate rescue;
                    rescue.distance = rejected.distance;
                    rescue.node_id = rejected.node_id;
                    rescue.parent_id = current_node_id;
                    rescue.boundary_ratio = rejected.boundary_ratio;
                    rescue.score = rejected.boundary_ratio;
                    rescue_buffer.push(rescue);
                    result.stats.rescue_candidates_added++;
                }
            };

            while (!candidate_set.empty()) {
                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
                dist_t current_dist = -current_node_pair.first;

                bool should_terminate =
                    current_dist > lowerBound &&
                    (top_candidates.size() == ef || has_deletions == false);
                bool rescue_ready =
                    !rescue_buffer.empty() &&
                    result.stats.rescue_expansions < static_cast<int>(params.max_rescue_expansions) &&
                    params.rescue_every > 0 &&
                    result.stats.normal_expansions > 0 &&
                    (should_terminate ||
                     result.stats.normal_expansions % static_cast<int>(params.rescue_every) == 0);

                if (should_terminate && !rescue_ready) {
                    break;
                }

                if (rescue_ready) {
                    BridgeRescueCandidate rescue = rescue_buffer.top();
                    rescue_buffer.pop();
                    if (rescue.node_id < expanded.size() && !expanded[rescue.node_id]) {
                        expand_node(rescue.node_id, rescue.distance, true);
                    }
                    continue;
                }

                candidate_set.pop();
                tableint current_node_id = current_node_pair.second;
                expand_node(current_node_id, current_dist, false);
            }

            adsampling::tot_dist_calculation += result.stats.distance_computations;
            result.top_candidates = top_candidates;
            visited_list_pool_->releaseVisitedList(vl);
            return result;
        }

        // RF-driven bridge rescue. At the end of each parent's neighbor loop,
        // every locally rejected candidate is scored with the trained Random
        // Forest using 11 cheap features (all O(1) at score time). If a
        // candidate's score >= rf_threshold and the per-query rescue budget is
        // not exhausted, the candidate is *immediately* expanded: every one of
        // its unvisited HNSW neighbors gets a query-distance computation and an
        // insertion attempt into top_candidates / candidate_set.
        template <bool has_deletions, bool collect_metrics=false>
        BridgeRescueBaseLayerResult searchBaseLayerRFBridgeRescue(
            tableint ep_id,
            const void *data_point,
            size_t ef,
            const BridgeRescueParams& params,
            Metric* metric = nullptr) const {
            BridgeRescueBaseLayerResult result;
            if (params.rf_predictor == nullptr || params.rf_predictor->n_trees() == 0) {
                // No model loaded; fall back to legacy rule-based rescue so the
                // caller still gets a sensible result.
                return searchBaseLayerBridgeRescue<has_deletions, collect_metrics>(
                    ep_id, data_point, ef, params, metric);
            }
            const RandomForestPredictor& rf = *params.rf_predictor;
            const size_t kCheapFeatures = 11;
            const size_t kPerNodeFeatures = params.per_node_stride;  // 0 = cheap-only model
            const size_t kTotalFeatures = kCheapFeatures + kPerNodeFeatures;
            const bool use_per_node = (params.per_node_features != nullptr && kPerNodeFeatures > 0);
            // Sanity: model n_features must match what we're about to build.
            if (rf.n_features() != kTotalFeatures) {
                HNSW_LOG_ERROR("RF n_features mismatch: model=", rf.n_features(),
                               " expected=", kTotalFeatures,
                               " (cheap=", kCheapFeatures, " per_node=", kPerNodeFeatures, ")");
                return result;
            }

            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;
            std::vector<unsigned char> expanded(cur_element_count, 0);

            // Top-K-by-distance shadow heap (max-heap of size <= recall_curve_k)
            // and per-curve bookkeeping. Only populated when gt_set is set.
            const std::unordered_set<labeltype>* gt_set = params.gt_set;
            const size_t recall_k = (params.recall_curve_k > 0 ? params.recall_curve_k : 1);
            std::priority_queue<std::pair<dist_t, tableint>> top_k_heap;
            int gt_in_topk = 0;
            float last_recorded_recall = 0.0f;

            auto record_recall = [&](tableint nid, dist_t nd) {
                if (!gt_set || gt_set->empty()) return;
                bool inserted_is_gt = (gt_set->count(getExternalLabel(nid)) > 0);
                top_k_heap.emplace(nd, nid);
                if (inserted_is_gt) ++gt_in_topk;
                if (top_k_heap.size() > recall_k) {
                    auto worst = top_k_heap.top();
                    if (gt_set->count(getExternalLabel(worst.second)) > 0) --gt_in_topk;
                    top_k_heap.pop();
                }
                float current_recall =
                    static_cast<float>(gt_in_topk) / static_cast<float>(recall_k);
                if (current_recall > last_recorded_recall + 1e-9f) {
                    // candidate_rank: rank of nd among the current top_candidates
                    // entries when sorted ascending by distance (matches the
                    // semantics used by analyzeRecallProgress).
                    int candidate_rank = 1;
                    {
                        auto tmp = top_candidates;
                        while (!tmp.empty()) {
                            if (tmp.top().first <= nd) ++candidate_rank;
                            tmp.pop();
                        }
                    }
                    result.stats.recall_curve.emplace_back(
                        result.stats.steps, current_recall, candidate_rank);
                    last_recorded_recall = current_recall;
                }
            };

            // Running min/max of all visited query-distances — used as a cheap
            // O(1) approximation of dist_rank_frac in feature vectors.
            dist_t min_visited_dist = std::numeric_limits<dist_t>::max();
            dist_t max_visited_dist = std::numeric_limits<dist_t>::lowest();
            auto update_minmax = [&](dist_t d) {
                if (d < min_visited_dist) min_visited_dist = d;
                if (d > max_visited_dist) max_visited_dist = d;
            };

            auto add_top_candidate = [&](tableint candidate_id, dist_t dist, dist_t& lowerBound) {
                if (!has_deletions || !isMarkedDeleted(candidate_id)) {
                    top_candidates.emplace(dist, candidate_id);
                    record_recall(candidate_id, dist);
                }
                if (top_candidates.size() > ef) {
                    top_candidates.pop();
                }
                if (!top_candidates.empty()) {
                    lowerBound = top_candidates.top().first;
                }
            };

            dist_t lowerBound;
            if (!has_deletions || !isMarkedDeleted(ep_id)) {
#ifndef ED2IP
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
#else
                dist_t dist = hnswlib::L2Sqr_by_IP(data_point, getDataByInternalId(ep_id), getExternalLabel(ep_id), dist_func_param_);
#endif
                adsampling::tot_dist_calculation++;
                adsampling::tot_full_dist++;
                if (metric != nullptr) metric->ndc++;
                result.stats.distance_computations++;
                update_minmax(dist);
                lowerBound = dist;
                top_candidates.emplace(dist, ep_id);
                record_recall(ep_id, dist);
                candidate_set.emplace(-dist, ep_id);
            } else {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidate_set.emplace(-lowerBound, ep_id);
            }
            visited_array[ep_id] = visited_array_tag;

            // Expand one HNSW node — distance-compute its unvisited neighbors,
            // push pass-the-boundary nodes, and record rejects locally.
            struct LocalReject {
                tableint node_id;
                dist_t distance;
                dist_t lower_bound_at_reject;
                size_t queue_size_at_reject;
            };

            auto expand_neighbors_for_rescue = [&](tableint bridge_id) {
                // Iterate v's adjacency list; compute query-distance for each
                // unvisited neighbor and try to slot it into the result heap.
                int *bdata = (int *) get_linklist0(bridge_id);
                size_t bsize = getListCount((linklistsizeint*)bdata);
                if (collect_metrics) {
                    metric_distance_computations += bsize;
                }
                for (size_t j = 1; j <= bsize; j++) {
                    tableint n_id = *(bdata + j);
                    if (n_id >= cur_element_count) continue;
                    if (visited_array[n_id] == visited_array_tag) continue;
                    visited_array[n_id] = visited_array_tag;
                    result.stats.rf_neighbors_expanded++;

                    char *n_obj = getDataByInternalId(n_id);
#ifndef ED2IP
                    dist_t nd = fstdistfunc_(data_point, n_obj, dist_func_param_);
#else
                    dist_t nd = hnswlib::L2Sqr_by_IP(data_point, n_obj, getExternalLabel(n_id), dist_func_param_);
#endif
                    adsampling::tot_full_dist++;
                    if (metric != nullptr) metric->ndc++;
                    result.stats.distance_computations++;
                    update_minmax(nd);

                    if (top_candidates.size() < ef || lowerBound > nd) {
                        candidate_set.emplace(-nd, n_id);
                        add_top_candidate(n_id, nd, lowerBound);
                        result.stats.rf_neighbors_inserted++;
                    }
                }
            };

            auto expand_node = [&](tableint current_node_id, dist_t current_dist, bool rescue_expansion) {
                if (current_node_id >= expanded.size() || expanded[current_node_id]) return;
                expanded[current_node_id] = 1;
                result.stats.steps++;
                if (metric != nullptr) metric->steps++;
                if (rescue_expansion) {
                    result.stats.rescue_expansions++;
                } else {
                    result.stats.normal_expansions++;
                }
                adsampling::tot_hops++;

                int *data = (int *) get_linklist0(current_node_id);
                size_t size = getListCount((linklistsizeint*)data);
                if (collect_metrics) {
                    metric_hops++;
                    metric_distance_computations += size;
                }

                std::vector<LocalReject> local_rejects;
                int checked = 0;
                int pushed = 0;
                int outside_rejected = 0;

                for (size_t j = 1; j <= size; j++) {
                    tableint candidate_id = *(data + j);
                    if (visited_array[candidate_id] == visited_array_tag) {
                        continue;
                    }
                    visited_array[candidate_id] = visited_array_tag;
                    checked++;

                    char *currObj1 = getDataByInternalId(candidate_id);
#ifndef ED2IP
                    dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);
#else
                    dist_t dist = hnswlib::L2Sqr_by_IP(data_point, currObj1, getExternalLabel(candidate_id), dist_func_param_);
#endif
                    adsampling::tot_full_dist++;
                    if (metric != nullptr) metric->ndc++;
                    result.stats.distance_computations++;
                    update_minmax(dist);

                    if (top_candidates.size() < ef || lowerBound > dist) {
                        candidate_set.emplace(-dist, candidate_id);
                        pushed++;
                        add_top_candidate(candidate_id, dist, lowerBound);
                    } else {
                        outside_rejected++;
                        result.stats.rescue_candidates_seen++;
                        local_rejects.push_back({candidate_id, dist, lowerBound, candidate_set.size()});
                    }
                }

                result.stats.parent_checked_total += checked;
                result.stats.parent_pushed_total += pushed;
                result.stats.parent_outside_rejected_total += outside_rejected;

                if (checked == 0 || local_rejects.empty()) return;
                if (static_cast<size_t>(result.stats.rf_rescue_count) >= params.rf_max_rescues_per_query) return;

                // Build the per-parent counts now that the loop is finished;
                // these match the end-of-expansion semantics used during RF
                // training in analyze_bridge_feature_signals.py.
                const float parent_checked_f = static_cast<float>(checked);
                const float parent_pushed_f = static_cast<float>(pushed);
                const float parent_rejected_f = static_cast<float>(checked - pushed);
                const float parent_outside_rejected_f = static_cast<float>(outside_rejected);
                const float parent_push_frac =
                    checked > 0 ? parent_pushed_f / parent_checked_f : 0.0f;
                const float parent_outside_reject_frac =
                    checked > 0 ? parent_outside_rejected_f / parent_checked_f : 0.0f;

                const float minmax_span =
                    static_cast<float>(max_visited_dist - min_visited_dist);
                const float minmax_span_safe =
                    minmax_span > 1e-12f ? minmax_span : 1e-12f;

                // Score each rejected candidate. Stop firing once the budget
                // is exhausted.
                for (const auto& r : local_rejects) {
                    if (static_cast<size_t>(result.stats.rf_rescue_count) >= params.rf_max_rescues_per_query) break;
                    if (r.node_id >= expanded.size() || expanded[r.node_id]) continue;

                    const float dist_f = static_cast<float>(r.distance);
                    const float lb_f = static_cast<float>(r.lower_bound_at_reject);
                    const float boundary_ratio_sq =
                        lb_f > 0.0f ? dist_f / lb_f : std::numeric_limits<float>::infinity();
                    const float boundary_margin_sq = dist_f - lb_f;
                    const float dist_rank_frac =
                        (dist_f - static_cast<float>(min_visited_dist)) / minmax_span_safe;
                    // Note: feature vector ordering must match CHEAP_FEATURES
                    // in bridge/train_rf_cheap.py and the exported JSON.
                    const float qv_minus_qparent_l2 = dist_f - static_cast<float>(current_dist);
                    // Build the full feature vector: 11 cheap + (optionally) per-node.
                    // Order MUST match CHEAP_FEATURES + PER_NODE_FEATURES in
                    // bridge/train_rf_v13.py.
                    float feats[64];  // ample headroom for kTotalFeatures (<= 30 today)
                    feats[0] = dist_rank_frac;
                    feats[1] = boundary_ratio_sq;
                    feats[2] = boundary_margin_sq;
                    feats[3] = qv_minus_qparent_l2;
                    feats[4] = static_cast<float>(r.queue_size_at_reject);
                    feats[5] = parent_checked_f;
                    feats[6] = parent_pushed_f;
                    feats[7] = parent_rejected_f;
                    feats[8] = parent_outside_rejected_f;
                    feats[9] = parent_outside_reject_frac;
                    feats[10] = parent_push_frac;
                    if (use_per_node) {
                        // Index by EXTERNAL label (the per-node table was built
                        // by build_node_features_bin.py keyed on node_id == label).
                        labeltype el = getExternalLabel(r.node_id);
                        size_t base = static_cast<size_t>(el) * kPerNodeFeatures;
                        // Defensive: if external label is somehow out of range,
                        // fall back to medians for those features.
                        if (el < cur_element_count) {
                            for (size_t j = 0; j < kPerNodeFeatures; ++j) {
                                feats[kCheapFeatures + j] = params.per_node_features[base + j];
                            }
                        } else {
                            for (size_t j = 0; j < kPerNodeFeatures; ++j) {
                                feats[kCheapFeatures + j] =
                                    (params.per_node_medians != nullptr) ? params.per_node_medians[j] : 0.0f;
                            }
                        }
                    }
                    rf.apply_medians(feats);
                    float score = rf.predict_proba(feats);
                    if (score < params.rf_threshold) continue;

                    // Fire rescue: directly expand the bridge's neighbors into
                    // the search frontier. Do NOT mark the bridge itself as
                    // expanded — that flag is reserved for actually-popped
                    // nodes in expand_node.
                    result.stats.rf_rescue_count++;
                    result.stats.rf_fires.emplace_back(
                        getExternalLabel(r.node_id), score, result.stats.steps);
                    expand_neighbors_for_rescue(r.node_id);
                }
            };

            // Termination policy. When params.search_budget == 0 we break at
            // HNSW's natural termination point (same as searchBaseLayerST).
            // When > 0 we mirror analyzeRecallProgress / searchKNNTestCollection
            // (mode 22): record the termination step the first time it fires
            // but keep popping until candidate_set drains or the budget is
            // exhausted. This lets the RF rescue accumulate extra recall
            // events past the natural stopping point for direct comparison
            // against the mode-22 baseline file.
            const bool keep_going = params.search_budget > 0;
            const int budget = keep_going ? params.search_budget
                                          : std::numeric_limits<int>::max();
            bool termination_triggered = false;
            int current_hop = 0;

            while (!candidate_set.empty() && current_hop < budget) {
                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
                dist_t current_dist = -current_node_pair.first;

                bool natural_stop =
                    current_dist > lowerBound &&
                    (top_candidates.size() == ef || has_deletions == false);

                if (natural_stop) {
                    if (!termination_triggered) {
                        termination_triggered = true;
                        result.stats.early_termination_step = result.stats.steps + 1;
                        if (!result.stats.target_recall_achieved &&
                            last_recorded_recall + 1e-9f >= params.target_recall &&
                            gt_set && !gt_set->empty()) {
                            result.stats.target_recall_achieved = true;
                            result.stats.step_target_recall_achieved = result.stats.steps;
                        }
                    }
                    if (!keep_going) break;
                }

                candidate_set.pop();
                tableint current_node_id = current_node_pair.second;
                expand_node(current_node_id, current_dist, false);
                ++current_hop;

                // After expansion, recall may have crossed target; record once.
                if (!result.stats.target_recall_achieved &&
                    last_recorded_recall + 1e-9f >= params.target_recall &&
                    gt_set && !gt_set->empty()) {
                    result.stats.target_recall_achieved = true;
                    result.stats.step_target_recall_achieved = result.stats.steps;
                }
            }

            result.stats.final_recall = last_recorded_recall;
            adsampling::tot_dist_calculation += result.stats.distance_computations;
            result.top_candidates = top_candidates;
            visited_list_pool_->releaseVisitedList(vl);
            return result;
        }
        // Run a normal base-layer search for one query and, for every candidate
        // whose distance gets computed (i.e. every "neighbor_checked" event),
        // emit a CSV row indicating whether *that* candidate's outgoing HNSW
        // neighbor list contains at least one ground-truth node that has not
        // yet been visited at the moment of the check. Used by mode 28
        // (relabelBridgeStrict) to produce a stricter v6 label.
        //
        // Output columns (one row per candidate checked):
        //   query_index, node_id (external label), has_unvisited_gt (0|1)
        //
        // Notes on visited-state semantics: visited_array is marked for the
        // candidate AFTER it is taken off the parent's neighbor list and
        // BEFORE the strict-bridge probe — so when we walk the candidate's
        // outgoing neighbors here, the candidate itself is already "visited"
        // (which is correct: we are checking whether the *other* outgoing
        // neighbors of this candidate include unvisited GT). Sibling neighbors
        // of the same parent that come later in this parent's neighbor list
        // are NOT yet visited and therefore eligible to count as
        // "unvisited GT" — which matches the v5 trace's "first_visited_step"
        // ordering exactly.
        void labelBridgeStrictForQuery(
            tableint ep_id,
            const void* query_data,
            size_t k,
            size_t ef,
            const std::unordered_set<labeltype>& gt_set,
            size_t query_index,
            std::ostream& csv_out,
            std::ostream* dynamics_out = nullptr,
            const std::vector<int>* node_cluster_ids = nullptr,
            std::ostream* heap_out = nullptr,
            size_t heap_k = 10,
            std::ostream* phi_out = nullptr) const {
            (void)k; // k is not needed for the per-candidate strict probe; the
                     // GT set already encodes top-k membership.

            if (cur_element_count == 0) return;
            const bool has_gt = !gt_set.empty();
            const bool emit_dyn = (dynamics_out != nullptr);
            const bool have_clusters = (node_cluster_ids != nullptr && !node_cluster_ids->empty());
            // Inline phi-feature emission (large-training-set experiment). Buffer
            // per-candidate state, accumulate per-parent stats, and write at the
            // end (dist_rank_frac needs the final visited-distance ranking).
            const bool emit_phi = (phi_out != nullptr);
            struct PhiRow {
                labeltype node; int has_gt; int accepted; float dist; float lower; float parent_dist;
                int qsize; int parent_idx;
            };
            struct ParentStat { int checked = 0, pushed = 0, rejected = 0, outside_rejected = 0; };
            std::vector<PhiRow> phi_rows;
            std::vector<ParentStat> parent_stats;
            std::vector<float> all_dists;  // every visited candidate's dist, for dist_rank_frac
            auto compute_has_gt = [&](tableint cid, vl_type* va, vl_type tag) -> int {
                if (!has_gt) return 0;
                int* nbr = (int*) get_linklist0(cid);
                size_t nbr_size = getListCount((linklistsizeint*)nbr);
                for (size_t j = 1; j <= nbr_size; ++j) {
                    tableint n = *(nbr + j);
                    if (n >= cur_element_count) continue;
                    if (va[n] == tag) continue;
                    if (gt_set.count(getExternalLabel(n))) return 1;
                }
                return 0;
            };
            // Heap-snapshot capture (Stage 0 of the teacher-student GNN plan):
            // for each emitted candidate, write the heap_k CLOSEST node-ids in
            // top_candidates at that moment. Binary record (little-endian):
            //   uint64 query_index, uint64 node_id (external label of candidate),
            //   int64  ids[heap_k]   (external labels, closest first, -1 padded)
            const bool emit_heap = (heap_out != nullptr);

            // Rolling-window state for recent-search dynamics features (mode 28
            // extension). All updated *after* emit_row, so each emitted row
            // reflects the search state *before* this candidate's accept/reject
            // decision.
            std::deque<int> accept_window;       // 1/0 for last WIN_ACCEPT distance-evaluated candidates
            std::deque<int> cluster_window;      // cluster_id of last WIN_CLUSTER accepted candidates
            const size_t WIN_ACCEPT = 50;
            const size_t WIN_CLUSTER = 20;
            int step_counter = 0;                // distance-evaluation step
            int last_topk_improve_step = 0;      // step at which lowerBound last *decreased*
            int last_gt_push_step = -1;          // step at which a GT label was last pushed; -1 = never

            VisitedList* vl = visited_list_pool_->getFreeVisitedList();
            vl_type* visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

            auto emit_row = [&](tableint cid) {
                // visited_array[cid] == visited_array_tag at this point.
                bool has_unvisited_gt = false;
                if (has_gt) {
                    int* nbr = (int*) get_linklist0(cid);
                    size_t nbr_size = getListCount((linklistsizeint*)nbr);
                    for (size_t j = 1; j <= nbr_size; ++j) {
                        tableint n = *(nbr + j);
                        if (n >= cur_element_count) continue;
                        if (visited_array[n] == visited_array_tag) continue;
                        if (gt_set.count(getExternalLabel(n))) {
                            has_unvisited_gt = true;
                            break;
                        }
                    }
                }
                csv_out << query_index << ','
                        << static_cast<unsigned long long>(getExternalLabel(cid)) << ','
                        << (has_unvisited_gt ? 1 : 0) << '\n';

                if (emit_dyn) {
                    float accept_rate = accept_window.empty()
                        ? 0.0f
                        : static_cast<float>(std::accumulate(accept_window.begin(), accept_window.end(), 0))
                          / static_cast<float>(accept_window.size());
                    int unique_clusters = 0;
                    if (have_clusters && !cluster_window.empty()) {
                        std::unordered_set<int> s(cluster_window.begin(), cluster_window.end());
                        unique_clusters = static_cast<int>(s.size());
                    }
                    int dt_topk = step_counter - last_topk_improve_step;
                    int dt_gt = (last_gt_push_step >= 0) ? (step_counter - last_gt_push_step) : -1;
                    (*dynamics_out)
                        << query_index << ','
                        << static_cast<unsigned long long>(getExternalLabel(cid)) << ','
                        << accept_rate << ','
                        << dt_topk << ','
                        << dt_gt << ','
                        << unique_clusters << '\n';
                }

                if (emit_heap) {
                    // Extract the heap_k closest entries from top_candidates
                    // (a max-heap on distance) without disturbing it.
                    std::vector<std::pair<dist_t, tableint>> tmp;
                    tmp.reserve(top_candidates.size());
                    {
                        auto copy = top_candidates;
                        while (!copy.empty()) { tmp.push_back(copy.top()); copy.pop(); }
                    }
                    size_t kk = std::min(heap_k, tmp.size());
                    std::partial_sort(tmp.begin(), tmp.begin() + kk, tmp.end(),
                                      [](const std::pair<dist_t, tableint>& a,
                                         const std::pair<dist_t, tableint>& b) {
                                          return a.first < b.first;  // closest first
                                      });
                    uint64_t qi64 = static_cast<uint64_t>(query_index);
                    uint64_t nid64 = static_cast<uint64_t>(getExternalLabel(cid));
                    heap_out->write(reinterpret_cast<const char*>(&qi64), sizeof(uint64_t));
                    heap_out->write(reinterpret_cast<const char*>(&nid64), sizeof(uint64_t));
                    for (size_t i = 0; i < heap_k; ++i) {
                        int64_t id = (i < kk)
                            ? static_cast<int64_t>(getExternalLabel(tmp[i].second))
                            : -1;
                        heap_out->write(reinterpret_cast<const char*>(&id), sizeof(int64_t));
                    }
                }
            };

            dist_t ep_dist;
#ifndef ED2IP
            ep_dist = fstdistfunc_(query_data, getDataByInternalId(ep_id), dist_func_param_);
#else
            ep_dist = hnswlib::L2Sqr_by_IP(query_data, getDataByInternalId(ep_id), getExternalLabel(ep_id), dist_func_param_);
#endif
            visited_array[ep_id] = visited_array_tag;
            dist_t lowerBound = ep_dist;
            top_candidates.emplace(ep_dist, ep_id);
            candidate_set.emplace(-ep_dist, ep_id);
            // The entry point itself is also "checked" — emit a row so any
            // CSV row keyed on it can still join.
            emit_row(ep_id);
            if (emit_phi) {
                int hg = compute_has_gt(ep_id, visited_array, visited_array_tag);
                phi_rows.push_back({getExternalLabel(ep_id), hg, /*accepted=*/1, (float)ep_dist,
                                    (float)lowerBound, (float)ep_dist,
                                    (int)candidate_set.size(), -1});
                all_dists.push_back((float)ep_dist);
            }

            // Treat ep_id as the first "accepted" candidate for rolling state.
            if (emit_dyn) {
                accept_window.push_back(1);
                if (have_clusters) {
                    labeltype el = getExternalLabel(ep_id);
                    if (static_cast<size_t>(el) < node_cluster_ids->size()) {
                        cluster_window.push_back((*node_cluster_ids)[el]);
                    }
                }
                if (gt_set.count(getExternalLabel(ep_id))) {
                    last_gt_push_step = 0;
                }
                ++step_counter;
            }

            // Run past natural HNSW termination to match the v5 trace coverage
            // (searchBaseLayerTraceCollection runs until target_recall=1.0 or
            // candidate_set drains). We exhaust candidate_set so every rejected
            // candidate that the v5 trace recorded gets a strict-label row.
            while (!candidate_set.empty()) {
                std::pair<dist_t, tableint> cur = candidate_set.top();
                dist_t current_dist = -cur.first;
                (void)current_dist;
                candidate_set.pop();
                tableint current_node_id = cur.second;

                int p_idx = -1;
                if (emit_phi) { parent_stats.push_back(ParentStat{}); p_idx = (int)parent_stats.size() - 1; }

                int* data = (int*) get_linklist0(current_node_id);
                size_t size = getListCount((linklistsizeint*)data);
                for (size_t j = 1; j <= size; ++j) {
                    tableint candidate_id = *(data + j);
                    if (visited_array[candidate_id] == visited_array_tag) continue;
                    visited_array[candidate_id] = visited_array_tag;

#ifndef ED2IP
                    dist_t dist = fstdistfunc_(query_data, getDataByInternalId(candidate_id), dist_func_param_);
#else
                    dist_t dist = hnswlib::L2Sqr_by_IP(query_data, getDataByInternalId(candidate_id), getExternalLabel(candidate_id), dist_func_param_);
#endif
                    emit_row(candidate_id);
                    int phi_idx = -1;
                    if (emit_phi) {
                        int hg = compute_has_gt(candidate_id, visited_array, visited_array_tag);
                        phi_rows.push_back({getExternalLabel(candidate_id), hg, /*accepted=*/0, (float)dist,
                                            (float)lowerBound, (float)current_dist,
                                            (int)candidate_set.size(), p_idx});
                        phi_idx = (int)phi_rows.size() - 1;
                        all_dists.push_back((float)dist);
                        parent_stats[p_idx].checked++;
                    }

                    const dist_t prev_lower = lowerBound;
                    const bool accepted = (top_candidates.size() < ef || lowerBound > dist);
                    if (emit_phi) {
                        phi_rows[phi_idx].accepted = accepted ? 1 : 0;
                        if (accepted) parent_stats[p_idx].pushed++;
                        else { parent_stats[p_idx].rejected++; parent_stats[p_idx].outside_rejected++; }
                    }
                    if (accepted) {
                        candidate_set.emplace(-dist, candidate_id);
                        top_candidates.emplace(dist, candidate_id);
                        if (top_candidates.size() > ef) top_candidates.pop();
                        if (!top_candidates.empty()) lowerBound = top_candidates.top().first;
                    }

                    if (emit_dyn) {
                        accept_window.push_back(accepted ? 1 : 0);
                        if (accept_window.size() > WIN_ACCEPT) accept_window.pop_front();
                        if (accepted) {
                            if (have_clusters) {
                                labeltype el = getExternalLabel(candidate_id);
                                if (static_cast<size_t>(el) < node_cluster_ids->size()) {
                                    cluster_window.push_back((*node_cluster_ids)[el]);
                                    if (cluster_window.size() > WIN_CLUSTER) cluster_window.pop_front();
                                }
                            }
                            if (lowerBound < prev_lower) {
                                last_topk_improve_step = step_counter;
                            }
                            if (gt_set.count(getExternalLabel(candidate_id))) {
                                last_gt_push_step = step_counter;
                            }
                        }
                        ++step_counter;
                    }
                }
            }

            // Write buffered phi rows (dist_rank_frac needs the final ranking).
            if (emit_phi && !phi_rows.empty()) {
                std::vector<float> sorted_d = all_dists;
                std::sort(sorted_d.begin(), sorted_d.end());
                const float total = static_cast<float>(sorted_d.size());
                for (const auto& r : phi_rows) {
                    // Bridge universe = REJECTED candidates only. Keep all positives;
                    // deterministically subsample negatives ~1/40 (hash on node id)
                    // to keep ~4% positive rate and a trainable file size.
                    if (r.accepted) continue;
                    if (r.has_gt == 0) {
                        uint64_t h = (static_cast<uint64_t>(r.node) * 2654435761ull) ^ (query_index * 40503ull);
                        if (h % 40ull != 0ull) continue;
                    }
                    float lower = r.lower;
                    float boundary_ratio = (lower > 0.0f) ? (r.dist / lower) : 0.0f;
                    float boundary_margin = r.dist - lower;
                    float qv_minus_qparent = r.dist - r.parent_dist;
                    size_t lo = std::lower_bound(sorted_d.begin(), sorted_d.end(), r.dist) - sorted_d.begin();
                    float dist_rank_frac = static_cast<float>(lo) / total;
                    int checked = 0, pushed = 0, rejected = 0, outside = 0;
                    if (r.parent_idx >= 0 && r.parent_idx < (int)parent_stats.size()) {
                        const auto& ps = parent_stats[r.parent_idx];
                        checked = ps.checked; pushed = ps.pushed; rejected = ps.rejected; outside = ps.outside_rejected;
                    }
                    float ck = static_cast<float>(checked);
                    float push_frac = (ck > 0.0f) ? (pushed / ck) : 0.0f;
                    float outside_frac = (ck > 0.0f) ? (outside / ck) : 0.0f;
                    (*phi_out)
                        << query_index << ',' << static_cast<unsigned long long>(r.node) << ','
                        << r.has_gt << ','
                        << dist_rank_frac << ',' << boundary_ratio << ',' << boundary_margin << ','
                        << qv_minus_qparent << ',' << r.qsize << ','
                        << checked << ',' << pushed << ',' << rejected << ','
                        << outside << ',' << outside_frac << ',' << push_frac << '\n';
                }
            }

            visited_list_pool_->releaseVisitedList(vl);
        }

        // Struct for probe search statistics (local navigability)
        struct ProbeSearchStats {
            std::vector<float> best_dist_curve;  // Best distance at each step
            std::vector<float> recall_at_step;   // Recall@k at each step
            std::vector<int> frontier_sizes;      // Frontier size at each step
            std::vector<float> frontier_entropy;  // Frontier entropy at each step
            std::vector<std::pair<dist_t, labeltype>> visited_nodes; // All visited nodes with distances
            std::vector<std::pair<dist_t, labeltype>> top_k_nodes;  // Final top-k nodes with distances
            std::vector<std::pair<dist_t, labeltype>> groundtruth_nodes;  // Groundtruth nodes with distances
            int stagnation_steps;                // Steps where distance didn't improve
            int total_steps;                     // Total steps taken
            int early_termination_step;          // Step where early termination occurred (-1 if not terminated early)
            float early_termination_recall;      // Recall value at early termination point
        };

        struct KNNTestSearchStats {
            std::vector<std::pair<dist_t, labeltype>> visited_nodes;              // Detailed visit records
            std::vector<std::pair<dist_t, labeltype>> top_k_nodes;      // Final top-k nodes with distances
            std::vector<std::pair<dist_t, labeltype>> groundtruth_nodes;// Groundtruth nodes with distances

            // Graph feature collection (optional)
            struct GraphNodeFeature {
                labeltype external_id;
                dist_t distance_to_query;
                uint32_t degree;
                float avg_distance_with_neighbors;
            };
            std::vector<GraphNodeFeature> graph_nodes;                  // Collected node features
            std::vector<std::pair<labeltype, labeltype>> graph_edges;   // Directed edges (src -> dst) using external IDs

            int total_steps = 0;                                        // Total steps processed
            int early_termination_step = -1;                            // Step where early termination first triggered
            float early_termination_recall = 0.0f;                      // Recall when termination triggered
            bool target_recall_achieved = false;                        // Whether target recall was achieved
            int step_target_recall_achieved = -1;                       // Step when target recall was achieved

            int hops_after_termination = 0;                             // Additional hops after termination to reach target recall
            int distance_calcs_after_termination = 0;                   // Distance evaluations after termination until target recall met
            int total_distance_calcs = 0;                               // Total distance evaluations during search

            float final_recall = 0.0f;                                  // Final recall@k
            // Recall improvement curve: (step, recall, candidate_rank) triples when recall increases.
            // candidate_rank = position of the triggering neighbor in result_set sorted by distance (1-based).
            // ef must be >= candidate_rank to keep that neighbor in the beam.
            std::vector<std::tuple<int, float, int>> recall_curve;
            // NDC (base-layer distance evaluations) at each recall_curve event, same order/length.
            // Kept as a parallel vector so the recall_curve tuple type stays unchanged for
            // existing consumers (bridge-rescue, trace collection).
            std::vector<int> recall_curve_ndc;

            // First ground truth node encountered during search
            int first_gt_node_step = -1;                               // Step number where first GT node was encountered (-1 if not found)
            labeltype first_gt_node_id = 0;                             // ID of the first GT node encountered
            dist_t first_gt_node_distance = 0.0f;                      // Distance of the first GT node encountered
        };

        struct BaseLayerNodeTrace {
            labeltype external_id = 0;
            dist_t distance_to_query = std::numeric_limits<dist_t>::max();
            int first_distance_computed_step = -1;
            int first_pushed_step = -1;
            int first_visited_step = -1;
            int first_popped_step = -1;
            int first_expanded_step = -1;
            int first_result_insert_step = -1;
            int first_result_pruned_step = -1;
            int first_distance_computed_search_step = -1;
            int first_pushed_search_step = -1;
            int first_visited_search_step = -1;
            int first_popped_search_step = -1;
            int first_expanded_search_step = -1;
            bool finally_returned = false;
            long long parent_at_first_discovery = -1;
            int gt_rank = -1;
            std::string reject_or_prune_reason;
        };

        struct BaseLayerSearchEvent {
            int step = 0;
            std::string event_type;
            long long node_id = -1;
            long long parent_id = -1;
            dist_t distance_to_query = std::numeric_limits<dist_t>::max();
            size_t candidate_queue_size = 0;
            size_t result_heap_size = 0;
            size_t visited_size = 0;
            dist_t current_kth_distance = std::numeric_limits<dist_t>::max();
            dist_t current_worst_result_distance = std::numeric_limits<dist_t>::max();
            int gt_rank = -1;
            std::string reason;
        };

        struct BaseLayerProgressSnapshot {
            int step = 0;
            int distance_computations = 0;
            int visited_count = 0;
            int pushed_count = 0;
            int popped_count = 0;
            int expanded_count = 0;
            dist_t best_result_distance = std::numeric_limits<dist_t>::max();
            dist_t kth_result_distance = std::numeric_limits<dist_t>::max();
            dist_t worst_result_distance = std::numeric_limits<dist_t>::max();
            int gt_k_pushed = 0;
            int gt_k_visited = 0;
            int gt_k_popped = 0;
            int gt_k_expanded = 0;
            int gt_k_in_result = 0;
        };

        struct BaseLayerFrontierNode {
            labeltype external_id = 0;
            dist_t distance_to_query = std::numeric_limits<dist_t>::max();
            long long parent_at_first_discovery = -1;
            bool pushed = false;
            bool popped = false;
            bool expanded = false;
            bool pruned = false;
            int first_seen_step = -1;
            int gt_rank = -1;
            std::string reason;
        };

        struct BaseLayerSearchTraceStats {
            size_t k = 0;
            size_t ef = 0;
            labeltype base_entry_label = 0;
            int distance_computations = 0;
            int visited_count = 0;
            int pushed_count = 0;
            int popped_count = 0;
            int expanded_count = 0;
            int pruned_count = 0;
            std::string termination_reason = "candidate_queue_empty";
            std::vector<std::pair<dist_t, labeltype>> final_results;
            std::vector<std::pair<dist_t, labeltype>> groundtruth_nodes;
            std::vector<BaseLayerNodeTrace> nodes;
            std::vector<BaseLayerSearchEvent> events;
            std::vector<BaseLayerProgressSnapshot> progress;
            std::vector<BaseLayerFrontierNode> frontier;
        };

        template <bool has_deletions>
        BaseLayerSearchTraceStats searchBaseLayerTraceCollection(
            tableint ep_id,
            const void* query_data,
            size_t k,
            size_t ef,
            const std::vector<labeltype>* gt_labels = nullptr,
            float target_recall = 1.0f,
            size_t progress_interval = 10,
            size_t max_events = 20000,
            size_t max_frontier = 512
        ) const {
            BaseLayerSearchTraceStats stats;
            stats.k = k;
            stats.ef = ef;
            stats.base_entry_label = getExternalLabel(ep_id);

            std::unordered_map<labeltype, int> gt_rank_by_label;
            if (gt_labels != nullptr) {
                for (size_t i = 0; i < gt_labels->size(); ++i) {
                    gt_rank_by_label[(*gt_labels)[i]] = static_cast<int>(i + 1);
                    try {
                        tableint gt_internal = getInternalId((*gt_labels)[i]);
                        if (gt_internal < cur_element_count) {
#ifndef ED2IP
                            dist_t gt_dist = fstdistfunc_(query_data, getDataByInternalId(gt_internal), dist_func_param_);
#else
                            dist_t gt_dist = hnswlib::L2Sqr_by_IP(query_data, getDataByInternalId(gt_internal), getExternalLabel(gt_internal), dist_func_param_);
#endif
                            stats.groundtruth_nodes.emplace_back(gt_dist, (*gt_labels)[i]);
                        }
                    } catch (...) {
                        continue;
                    }
                }
                std::sort(stats.groundtruth_nodes.begin(), stats.groundtruth_nodes.end(),
                          [](const std::pair<dist_t, labeltype>& a, const std::pair<dist_t, labeltype>& b) {
                              return a.first < b.first;
                          });
            }

            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

            std::unordered_map<tableint, size_t> node_index;
            std::unordered_set<tableint> result_members;
            int event_step = 0;
            dist_t lowerBound = std::numeric_limits<dist_t>::max();
            float recall_threshold = target_recall;
            if (recall_threshold < 0.0f) recall_threshold = 0.0f;
            if (recall_threshold > 1.0f) recall_threshold = 1.0f;

            auto external_or_minus_one = [&](tableint internal_id) -> long long {
                if (internal_id >= cur_element_count) return -1;
                return static_cast<long long>(getExternalLabel(internal_id));
            };

            auto gt_rank_for_label = [&](labeltype label) -> int {
                auto it = gt_rank_by_label.find(label);
                return it == gt_rank_by_label.end() ? -1 : it->second;
            };

            auto get_node_trace = [&](tableint internal_id, dist_t dist) -> BaseLayerNodeTrace& {
                auto it = node_index.find(internal_id);
                if (it == node_index.end()) {
                    BaseLayerNodeTrace node;
                    node.external_id = getExternalLabel(internal_id);
                    node.distance_to_query = dist;
                    node.gt_rank = gt_rank_for_label(node.external_id);
                    stats.nodes.push_back(node);
                    size_t pos = stats.nodes.size() - 1;
                    node_index[internal_id] = pos;
                    return stats.nodes[pos];
                }
                BaseLayerNodeTrace& node = stats.nodes[it->second];
                if (dist < node.distance_to_query) node.distance_to_query = dist;
                return node;
            };

            auto heap_best_distance = [&](const std::priority_queue<std::pair<dist_t, tableint>>& heap) -> dist_t {
                if (heap.empty()) return std::numeric_limits<dist_t>::max();
                auto tmp = heap;
                dist_t best = std::numeric_limits<dist_t>::max();
                while (!tmp.empty()) {
                    if (tmp.top().first < best) best = tmp.top().first;
                    tmp.pop();
                }
                return best;
            };

            auto kth_distance = [&](const std::priority_queue<std::pair<dist_t, tableint>>& heap) -> dist_t {
                if (heap.empty()) return std::numeric_limits<dist_t>::max();
                if (heap.size() < k) return heap.top().first;
                std::vector<std::pair<dist_t, tableint>> tmp_vec;
                auto tmp = heap;
                while (!tmp.empty()) {
                    tmp_vec.push_back(tmp.top());
                    tmp.pop();
                }
                std::sort(tmp_vec.begin(), tmp_vec.end(),
                          [](const std::pair<dist_t, tableint>& a, const std::pair<dist_t, tableint>& b) {
                              return a.first < b.first;
                          });
                return tmp_vec[std::min(k, tmp_vec.size()) - 1].first;
            };

            auto log_event = [&](const std::string& event_type,
                                 tableint node_id,
                                 tableint parent_id,
                                 dist_t dist,
                                 const std::string& reason = std::string()) {
                ++event_step;
                if (stats.events.size() >= max_events) return;
                BaseLayerSearchEvent event;
                event.step = event_step;
                event.event_type = event_type;
                event.node_id = external_or_minus_one(node_id);
                event.parent_id = external_or_minus_one(parent_id);
                event.distance_to_query = dist;
                event.candidate_queue_size = candidate_set.size();
                event.result_heap_size = top_candidates.size();
                event.visited_size = static_cast<size_t>(stats.visited_count);
                event.current_kth_distance = kth_distance(top_candidates);
                event.current_worst_result_distance = top_candidates.empty() ? std::numeric_limits<dist_t>::max() : top_candidates.top().first;
                event.gt_rank = (node_id < cur_element_count) ? gt_rank_for_label(getExternalLabel(node_id)) : -1;
                event.reason = reason;
                stats.events.push_back(event);
            };

            auto count_gt_k = [&](int BaseLayerNodeTrace::*step_field) -> int {
                int count = 0;
                std::unordered_set<labeltype> seen;
                for (const auto& node : stats.nodes) {
                    if (node.gt_rank > 0 && node.gt_rank <= static_cast<int>(k) && node.*step_field >= 0) {
                        seen.insert(node.external_id);
                    }
                }
                count = static_cast<int>(seen.size());
                return count;
            };

            auto count_gt_k_in_result = [&]() -> int {
                int count = 0;
                auto tmp = top_candidates;
                while (!tmp.empty()) {
                    labeltype label = getExternalLabel(tmp.top().second);
                    int rank = gt_rank_for_label(label);
                    if (rank > 0 && rank <= static_cast<int>(k)) ++count;
                    tmp.pop();
                }
                return count;
            };

            auto recall_satisfied = [&]() -> bool {
                if (gt_labels == nullptr || gt_labels->empty() || k == 0) return false;
                std::vector<std::pair<dist_t, tableint>> tmp_vec;
                auto tmp = top_candidates;
                while (!tmp.empty()) {
                    tmp_vec.push_back(tmp.top());
                    tmp.pop();
                }
                std::sort(tmp_vec.begin(), tmp_vec.end(),
                          [](const std::pair<dist_t, tableint>& a, const std::pair<dist_t, tableint>& b) {
                              return a.first < b.first;
                          });
                size_t limit = std::min(k, tmp_vec.size());
                int found = 0;
                for (size_t i = 0; i < limit; ++i) {
                    int rank = gt_rank_for_label(getExternalLabel(tmp_vec[i].second));
                    if (rank > 0 && rank <= static_cast<int>(k)) ++found;
                }
                float recall = static_cast<float>(found) / static_cast<float>(k);
                return recall >= recall_threshold;
            };

            auto record_progress = [&]() {
                if (progress_interval == 0) return;
                if (stats.expanded_count % static_cast<int>(progress_interval) != 0) return;
                BaseLayerProgressSnapshot snap;
                snap.step = event_step;
                snap.distance_computations = stats.distance_computations;
                snap.visited_count = stats.visited_count;
                snap.pushed_count = stats.pushed_count;
                snap.popped_count = stats.popped_count;
                snap.expanded_count = stats.expanded_count;
                snap.best_result_distance = heap_best_distance(top_candidates);
                snap.kth_result_distance = kth_distance(top_candidates);
                snap.worst_result_distance = top_candidates.empty() ? std::numeric_limits<dist_t>::max() : top_candidates.top().first;
                snap.gt_k_pushed = count_gt_k(&BaseLayerNodeTrace::first_pushed_step);
                snap.gt_k_visited = count_gt_k(&BaseLayerNodeTrace::first_visited_step);
                snap.gt_k_popped = count_gt_k(&BaseLayerNodeTrace::first_popped_step);
                snap.gt_k_expanded = count_gt_k(&BaseLayerNodeTrace::first_expanded_step);
                snap.gt_k_in_result = count_gt_k_in_result();
                stats.progress.push_back(snap);
            };

            tableint no_parent = std::numeric_limits<tableint>::max();
            if (!has_deletions || !isMarkedDeleted(ep_id)) {
#ifndef ED2IP
                dist_t dist = fstdistfunc_(query_data, getDataByInternalId(ep_id), dist_func_param_);
#else
                dist_t dist = hnswlib::L2Sqr_by_IP(query_data, getDataByInternalId(ep_id), getExternalLabel(ep_id), dist_func_param_);
#endif
                stats.distance_computations++;
                lowerBound = dist;
                BaseLayerNodeTrace& node = get_node_trace(ep_id, dist);
                node.first_distance_computed_step = event_step + 1;
                node.first_visited_step = event_step + 1;
                node.first_pushed_step = event_step + 1;
                node.first_result_insert_step = event_step + 1;
                node.first_distance_computed_search_step = 0;
                node.first_visited_search_step = 0;
                node.first_pushed_search_step = 0;
                node.parent_at_first_discovery = -1;
                visited_array[ep_id] = visited_array_tag;
                stats.visited_count++;
                top_candidates.emplace(dist, ep_id);
                result_members.insert(ep_id);
                candidate_set.emplace(-dist, ep_id);
                stats.pushed_count++;
                log_event("base_entry_received", ep_id, no_parent, dist);
                log_event("candidate_pushed", ep_id, no_parent, dist);
                log_event("result_inserted", ep_id, no_parent, dist);
            } else {
                visited_array[ep_id] = visited_array_tag;
                stats.visited_count++;
                candidate_set.emplace(-lowerBound, ep_id);
                stats.pushed_count++;
                BaseLayerNodeTrace& node = get_node_trace(ep_id, lowerBound);
                node.first_visited_step = event_step + 1;
                node.first_pushed_step = event_step + 1;
                node.first_visited_search_step = 0;
                node.first_pushed_search_step = 0;
                node.reject_or_prune_reason = "entry_deleted";
                log_event("base_entry_received", ep_id, no_parent, lowerBound, "entry_deleted");
                log_event("candidate_pushed", ep_id, no_parent, lowerBound, "entry_deleted");
            }

            while (!candidate_set.empty()) {
                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
                dist_t current_dist = -current_node_pair.first;
                tableint current_node_id = current_node_pair.second;

                candidate_set.pop();
                stats.popped_count++;
                BaseLayerNodeTrace& current_trace = get_node_trace(current_node_id, current_dist);
                if (current_trace.first_popped_step < 0) current_trace.first_popped_step = event_step + 1;
                log_event("candidate_popped", current_node_id, no_parent, current_dist);

                stats.expanded_count++;
                if (current_trace.first_popped_search_step < 0) current_trace.first_popped_search_step = stats.expanded_count;
                if (current_trace.first_expanded_step < 0) current_trace.first_expanded_step = event_step + 1;
                if (current_trace.first_expanded_search_step < 0) current_trace.first_expanded_search_step = stats.expanded_count;
                log_event("node_expanded", current_node_id, no_parent, current_dist);

                int *data = (int *) get_linklist0(current_node_id);
                size_t size = getListCount((linklistsizeint*)data);
                for (size_t j = 1; j <= size; j++) {
                    tableint candidate_id = *(data + j);
                    if (visited_array[candidate_id] == visited_array_tag) {
                        log_event("already_visited", candidate_id, current_node_id, std::numeric_limits<dist_t>::max());
                        continue;
                    }

                    visited_array[candidate_id] = visited_array_tag;
                    stats.visited_count++;
                    char *currObj1 = (getDataByInternalId(candidate_id));
#ifndef ED2IP
                    dist_t dist = fstdistfunc_(query_data, currObj1, dist_func_param_);
#else
                    dist_t dist = hnswlib::L2Sqr_by_IP(query_data, currObj1, getExternalLabel(candidate_id), dist_func_param_);
#endif
                    stats.distance_computations++;
                    BaseLayerNodeTrace& candidate_trace = get_node_trace(candidate_id, dist);
                    if (candidate_trace.first_distance_computed_step < 0) candidate_trace.first_distance_computed_step = event_step + 1;
                    if (candidate_trace.first_visited_step < 0) candidate_trace.first_visited_step = event_step + 1;
                    if (candidate_trace.first_distance_computed_search_step < 0) candidate_trace.first_distance_computed_search_step = stats.expanded_count;
                    if (candidate_trace.first_visited_search_step < 0) candidate_trace.first_visited_search_step = stats.expanded_count;
                    if (candidate_trace.parent_at_first_discovery < 0) {
                        candidate_trace.parent_at_first_discovery = external_or_minus_one(current_node_id);
                    }
                    log_event("neighbor_checked", candidate_id, current_node_id, dist);

                    if (top_candidates.size() < ef || lowerBound > dist) {
                        candidate_set.emplace(-dist, candidate_id);
                        stats.pushed_count++;
                        if (candidate_trace.first_pushed_step < 0) candidate_trace.first_pushed_step = event_step + 1;
                        if (candidate_trace.first_pushed_search_step < 0) candidate_trace.first_pushed_search_step = stats.expanded_count;
                        log_event("candidate_pushed", candidate_id, current_node_id, dist);

                        if (!has_deletions || !isMarkedDeleted(candidate_id)) {
                            top_candidates.emplace(dist, candidate_id);
                            result_members.insert(candidate_id);
                            if (candidate_trace.first_result_insert_step < 0) candidate_trace.first_result_insert_step = event_step + 1;
                            log_event("result_inserted", candidate_id, current_node_id, dist);
                        } else {
                            candidate_trace.reject_or_prune_reason = "deleted";
                            log_event("candidate_rejected", candidate_id, current_node_id, dist, "deleted");
                        }

                        if (top_candidates.size() > ef) {
                            auto removed = top_candidates.top();
                            top_candidates.pop();
                            result_members.erase(removed.second);
                            stats.pruned_count++;
                            BaseLayerNodeTrace& removed_trace = get_node_trace(removed.second, removed.first);
                            if (removed_trace.first_result_pruned_step < 0) removed_trace.first_result_pruned_step = event_step + 1;
                            removed_trace.reject_or_prune_reason = "result_heap_over_ef";
                            log_event("result_pruned", removed.second, current_node_id, removed.first, "result_heap_over_ef");
                        }
                        if (!top_candidates.empty()) lowerBound = top_candidates.top().first;
                    } else {
                        candidate_trace.reject_or_prune_reason = "outside_result_boundary";
                        log_event("candidate_rejected", candidate_id, current_node_id, dist, "outside_result_boundary");
                    }
                }
                record_progress();

                if (recall_satisfied()) {
                    stats.termination_reason = "target_recall_satisfied";
                    log_event("search_terminated", current_node_id, no_parent, current_dist, stats.termination_reason);
                    break;
                }
            }

            if (candidate_set.empty()) {
                stats.termination_reason = "candidate_queue_empty";
                log_event("search_terminated", no_parent, no_parent, lowerBound, stats.termination_reason);
            }

            std::vector<std::pair<dist_t, tableint>> final_internal;
            auto copy = top_candidates;
            while (!copy.empty()) {
                final_internal.push_back(copy.top());
                copy.pop();
            }
            std::sort(final_internal.begin(), final_internal.end(),
                      [](const std::pair<dist_t, tableint>& a, const std::pair<dist_t, tableint>& b) {
                          return a.first < b.first;
                      });
            for (size_t i = 0; i < final_internal.size() && i < k; ++i) {
                stats.final_results.emplace_back(final_internal[i].first, getExternalLabel(final_internal[i].second));
                BaseLayerNodeTrace& node = get_node_trace(final_internal[i].second, final_internal[i].first);
                node.finally_returned = true;
            }

            for (const auto& kv : node_index) {
                const BaseLayerNodeTrace& node = stats.nodes[kv.second];
                bool is_frontier = false;
                std::string reason;
                if (node.first_pushed_step >= 0 && node.first_popped_step < 0) {
                    is_frontier = true;
                    reason = "pushed_not_popped";
                } else if (node.first_popped_step >= 0 && node.first_expanded_step < 0) {
                    is_frontier = true;
                    reason = "popped_not_expanded";
                } else if (node.first_result_pruned_step >= 0) {
                    is_frontier = true;
                    reason = "result_pruned";
                } else if (node.gt_rank > 0 && node.gt_rank <= static_cast<int>(std::max(k, static_cast<size_t>(1))) && !node.finally_returned) {
                    is_frontier = true;
                    reason = "gt_k_not_returned";
                }
                if (is_frontier && stats.frontier.size() < max_frontier) {
                    BaseLayerFrontierNode frontier;
                    frontier.external_id = node.external_id;
                    frontier.distance_to_query = node.distance_to_query;
                    frontier.parent_at_first_discovery = node.parent_at_first_discovery;
                    frontier.pushed = node.first_pushed_step >= 0;
                    frontier.popped = node.first_popped_step >= 0;
                    frontier.expanded = node.first_expanded_step >= 0;
                    frontier.pruned = node.first_result_pruned_step >= 0;
                    frontier.first_seen_step = node.first_visited_step;
                    frontier.gt_rank = node.gt_rank;
                    frontier.reason = reason;
                    stats.frontier.push_back(frontier);
                }
            }

            visited_list_pool_->releaseVisitedList(vl);
            return stats;
        }

        /**
         * @brief Perform base layer KNN search with budget (max hops) and collect probe statistics.
         * 
         * This function performs a KNN search in the base layer while collecting statistics
         * for local navigability analysis, including early-recall@T, best-dist curve slope,
         * stagnation ratio, and frontier entropy.
         * 
         * @tparam has_deletions Indicates whether the graph contains deleted nodes
         * @param ep_id Entry point node ID (anchor's internal ID)
         * @param query_data Pointer to the query vector (anchor's vector)
         * @param k Target number of nearest neighbors
         * @param budget Maximum number of hops allowed
         * @param gt_set Ground truth set for recall calculation (nullptr if not available)
         * @return ProbeSearchStats containing search trajectory statistics
         */
        template <bool has_deletions>
        ProbeSearchStats searchBaseLayerKNNAnchors(
            tableint ep_id,
            const void* query_data,
            size_t k,
            int budget,
            const std::unordered_set<unsigned>* gt_set = nullptr
        ) const {
            ProbeSearchStats stats;
            stats.stagnation_steps = 0;
            stats.total_steps = 0;
            stats.early_termination_step = -1;  // -1 means no early termination
            stats.early_termination_recall = 0.0f;
            
            if (cur_element_count == 0) {
                return stats;
            }
            
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;
            
            // Priority queue for candidates: (distance, node_id)
            std::priority_queue<std::pair<dist_t, tableint>, 
                               std::vector<std::pair<dist_t, tableint>>, 
                               CompareByFirst> candidate_set;
            
            // Result set: top-k closest nodes
            std::priority_queue<std::pair<dist_t, tableint>> result_set;
            
            // Start from entry point
            dist_t lowerBound;
            if (!has_deletions || !isMarkedDeleted(ep_id)) {
#ifndef ED2IP
                dist_t anchor_dist = fstdistfunc_(query_data, getDataByInternalId(ep_id), dist_func_param_);
#else
                dist_t anchor_dist = hnswlib::L2Sqr_by_IP(query_data, getDataByInternalId(ep_id), getExternalLabel(ep_id), dist_func_param_);
#endif
                lowerBound = anchor_dist;
                result_set.emplace(anchor_dist, ep_id);
                candidate_set.emplace(-anchor_dist, ep_id);
            } else {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidate_set.emplace(-lowerBound, ep_id);
            }
            
            visited_array[ep_id] = visited_array_tag;
            
            float current_best_dist = (lowerBound == std::numeric_limits<dist_t>::max()) ? 
                                      std::numeric_limits<float>::max() : lowerBound;
            float previous_best_dist = current_best_dist;
            int current_hop = 0;
            
            // Track visited nodes for recall calculation
            std::unordered_set<unsigned> visited_labels;
            if (!has_deletions || !isMarkedDeleted(ep_id)) {
                visited_labels.insert(getExternalLabel(ep_id));
                stats.visited_nodes.emplace_back(current_best_dist, getExternalLabel(ep_id));
            }
            
            // Calculate initial recall
            float initial_recall = 0.0f;
            if (gt_set && !result_set.empty()) {
                unsigned label = getExternalLabel(result_set.top().second);
                if (gt_set->count(label)) initial_recall = 1.0f / k;
            }
            stats.recall_at_step.push_back(initial_recall);
            stats.best_dist_curve.push_back(current_best_dist);
            stats.frontier_sizes.push_back(candidate_set.size());
            
            // Calculate initial frontier entropy
            if (!candidate_set.empty()) {
                std::vector<float> frontier_dists;
                auto temp_candidate = candidate_set;
                while (!temp_candidate.empty()) {
                    frontier_dists.push_back(-temp_candidate.top().first);
                    temp_candidate.pop();
                }
                stats.frontier_entropy.push_back(calculateEntropy(frontier_dists));
            } else {
                stats.frontier_entropy.push_back(0.0f);
            }
            
            stats.total_steps = 1;
            
            // Output initial step
            HNSW_LOG_INFO("[KNNAnchors] Step 0: Visiting entry point node=", getExternalLabel(ep_id), 
                         ", distance=", current_best_dist);
            if (!has_deletions || !isMarkedDeleted(ep_id)) {
                HNSW_LOG_INFO("[KNNAnchors] Step 0: Initial result set size=1, best distance=", current_best_dist);
            }
            
            // Early-recall checkpoints
            std::vector<int> recall_checkpoints = {5, 10, 15, 20};
            std::vector<bool> recall_recorded(recall_checkpoints.size(), false);
            
            // Main search loop with hop limit
            while (!candidate_set.empty()) {
                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
                dist_t current_dist = -current_node_pair.first;
                tableint current_node_id = current_node_pair.second;
                candidate_set.pop();
                
                // Check termination condition
                // Early termination: if the best candidate is farther than the worst result,
                // we cannot find any closer nodes, so stop searching
                if (current_dist > lowerBound && (result_set.size() >= k || !has_deletions)) {
                    // Record early termination step and recall
                    stats.early_termination_step = stats.total_steps + 1;
                    // Calculate current recall before termination
                    float termination_recall = 0.0f;
                    if (gt_set && !result_set.empty()) {
                        int found = 0;
                        std::priority_queue<std::pair<dist_t, tableint>> temp_result = result_set;
                        int count = 0;
                        while (!temp_result.empty() && count < k) {
                            unsigned label = getExternalLabel(temp_result.top().second);
                            if (gt_set->count(label)) found++;
                            temp_result.pop();
                            count++;
                        }
                        termination_recall = float(found) / k;
                    }
                    stats.early_termination_recall = termination_recall;
                    HNSW_LOG_INFO("[KNNAnchors] Early termination at step ", stats.total_steps + 1, 
                                 ": current_dist=", current_dist, " > lowerBound=", lowerBound, 
                                 ", result_set.size()=", result_set.size(), " >= k=", k,
                                 ", recall=", termination_recall);
                }
                
                current_hop++;
                stats.total_steps++;
                
                // Mark as visited
                unsigned node_label = getExternalLabel(current_node_id);
                visited_labels.insert(node_label);
                stats.visited_nodes.emplace_back(current_dist, node_label);
                
                // Output step information
                HNSW_LOG_INFO("[KNNAnchors] Step ", stats.total_steps, " (hop ", current_hop, "): Visiting node=", 
                             node_label, ", distance=", current_dist);
                
                // Update result set
                result_set.emplace(current_dist, current_node_id);
                if (result_set.size() > ef_) {
                    result_set.pop();
                }
                
                // Update lower bound
                if (!result_set.empty()) {
                    lowerBound = result_set.top().first;
                }
                
                // Check if distance improved
                bool improved = false;
                if (current_dist < current_best_dist) {
                    current_best_dist = current_dist;
                    improved = true;
                    HNSW_LOG_INFO("[KNNAnchors] Step ", stats.total_steps, ": Distance improved! New best=", current_best_dist);
                } else {
                    stats.stagnation_steps++;
                }
                previous_best_dist = current_best_dist;
                
                // Record statistics at this step
                stats.best_dist_curve.push_back(current_best_dist);
                
                // Calculate recall from result_set
                float step_recall = 0.0f;
                if (gt_set) {
                    int found = 0;
                    std::priority_queue<std::pair<dist_t, tableint>> temp_result = result_set;
                    int count = 0;
                    while (!temp_result.empty() && count < k) {
                        unsigned label = getExternalLabel(temp_result.top().second);
                        if (gt_set->count(label)) found++;
                        temp_result.pop();
                        count++;
                    }
                    step_recall = float(found) / k;
                }
                stats.recall_at_step.push_back(step_recall);
                
                // Check early-recall checkpoints
                for (size_t i = 0; i < recall_checkpoints.size(); ++i) {
                    if (!recall_recorded[i] && stats.total_steps >= recall_checkpoints[i]) {
                        recall_recorded[i] = true;
                        // Recall is already recorded in stats.recall_at_step
                    }
                }
                
                // Explore neighbors
                int *data = (int *) get_linklist0(current_node_id);
                size_t size = getListCount((linklistsizeint*)data);
                
                HNSW_LOG_INFO("[KNNAnchors] Step ", stats.total_steps, ": Node ", node_label, " has ", size, " neighbors");
                
                std::vector<float> frontier_dists_step;
                int neighbors_added = 0;
                int neighbors_skipped = 0;
                for (size_t j = 1; j <= size; j++) {
                    tableint neighbor_id = *(data + j);
                    if (neighbor_id < 0 || neighbor_id >= cur_element_count) {
                        neighbors_skipped++;
                        continue;
                    }
                    
                    if (visited_array[neighbor_id] != visited_array_tag) {
                        visited_array[neighbor_id] = visited_array_tag;
#ifndef ED2IP
                        dist_t neighbor_dist = fstdistfunc_(query_data, getDataByInternalId(neighbor_id), dist_func_param_);
#else
                        dist_t neighbor_dist = hnswlib::L2Sqr_by_IP(query_data, getDataByInternalId(neighbor_id), getExternalLabel(neighbor_id), dist_func_param_);
#endif
                        unsigned neighbor_label = getExternalLabel(neighbor_id);
                        
                        if (result_set.size() < ef_ || lowerBound > neighbor_dist) {
                            candidate_set.emplace(-neighbor_dist, neighbor_id);
                            if (!has_deletions || !isMarkedDeleted(neighbor_id)) {
                                result_set.emplace(neighbor_dist, neighbor_id);
                                if (result_set.size() > ef_) {
                                    result_set.pop();
                                }
                                if (!result_set.empty()) {
                                    lowerBound = result_set.top().first;
                                }
                                // Update top-k queue
                            // top_k_queue.emplace(neighbor_dist, neighbor_id);
                            // if (top_k_queue.size() > k) {
                            //     top_k_queue.pop();  // Remove the farthest node
                            // }
                            
                            // // Record recall improvement in recall_curve
                            // if (gt_set && !gt_set->empty()) {
                            //     float current_recall = compute_top_k_recall();
                            //     if (current_recall > last_recorded_recall) {
                            //         stats.recall_curve.emplace_back(stats.total_steps, current_recall);
                            //         last_recorded_recall = current_recall;
                            //     }
                            // }
                        }
                            frontier_dists_step.push_back(neighbor_dist);
                            neighbors_added++;
                            // HNSW_LOG_INFO("[KNNAnchors] Step ", stats.total_steps, ": Added neighbor node=", neighbor_label, 
                            //              ", distance=", neighbor_dist);
                        } else {
                            neighbors_skipped++;
                            // HNSW_LOG_INFO("[KNNAnchors] Step ", stats.total_steps, ": Skipped neighbor node=", neighbor_label, 
                            //              ", distance=", neighbor_dist, " (too far, lowerBound=", lowerBound, ")");
                        }
                    } else {
                        neighbors_skipped++;
                        unsigned neighbor_label = getExternalLabel(neighbor_id);
                        // HNSW_LOG_INFO("[KNNAnchors] Step ", stats.total_steps, ": Skipped neighbor node=", neighbor_label, " (already visited)");
                    }
                }
                
                HNSW_LOG_INFO("[KNNAnchors] Step ", stats.total_steps, ": Neighbors update - added=", neighbors_added, 
                             ", skipped=", neighbors_skipped, ", total=", size);
                HNSW_LOG_INFO("[KNNAnchors] Step ", stats.total_steps, ": Result set size=", result_set.size(), 
                             ", candidate set size=", candidate_set.size(), ", lowerBound=", lowerBound);
                
                // Record frontier size and entropy
                int frontier_size = candidate_set.size();
                stats.frontier_sizes.push_back(frontier_size);
                float step_entropy = 0.0f;
                if (!frontier_dists_step.empty()) {
                    step_entropy = calculateEntropy(frontier_dists_step);
                    stats.frontier_entropy.push_back(step_entropy);
                } else {
                    stats.frontier_entropy.push_back(0.0f);
                }
            }
            
            // Extract final top-k nodes from result_set (in order from closest to furthest)
            // Exclude the query point itself (ep_id) from the results
            std::vector<std::pair<dist_t, tableint>> top_k_temp;
            while (!result_set.empty()) {
                top_k_temp.push_back(result_set.top());
                result_set.pop();
            }
            // Reverse to get closest to furthest order
            std::reverse(top_k_temp.begin(), top_k_temp.end());
            // Filter out the query point itself
            for (const auto& [dist, node_id] : top_k_temp) {
                if (node_id != ep_id) {  // Exclude the query point itself
                    stats.top_k_nodes.emplace_back(dist, getExternalLabel(node_id));
                    if (stats.top_k_nodes.size() >= k) {
                        break;  // Stop once we have k neighbors
                    }
                }
            }
            
            // Output final top-k nodes & distances
            HNSW_LOG_INFO("[KNNAnchors] Final top-", k, " nodes & distances (", stats.top_k_nodes.size(), " nodes):");
            for (size_t i = 0; i < stats.top_k_nodes.size(); ++i) {
                HNSW_LOG_INFO("[KNNAnchors]   Top-", (i+1), ": node=", stats.top_k_nodes[i].second, 
                             ", distance=", stats.top_k_nodes[i].first);
            }
            
            // Extract groundtruth nodes with distances if available
            if (gt_set && !gt_set->empty()) {
                for (unsigned gt_label : *gt_set) {
                    try {
                        tableint gt_internal_id = getInternalId(gt_label);
                        if (gt_internal_id >= 0 && gt_internal_id < cur_element_count) {
#ifndef ED2IP
                            dist_t gt_dist = fstdistfunc_(query_data, getDataByInternalId(gt_internal_id), dist_func_param_);
#else
                            dist_t gt_dist = hnswlib::L2Sqr_by_IP(query_data, getDataByInternalId(gt_internal_id), getExternalLabel(gt_internal_id), dist_func_param_);
#endif
                            stats.groundtruth_nodes.emplace_back(gt_dist, gt_label);
                        }
                    } catch (...) {
                        // Skip if internal ID not found
                        continue;
                    }
                }
                // Sort groundtruth by distance (closest to furthest)
                std::sort(stats.groundtruth_nodes.begin(), stats.groundtruth_nodes.end(),
                         [](const std::pair<dist_t, labeltype>& a, const std::pair<dist_t, labeltype>& b) {
                             return a.first < b.first;
                         });
                
                // Output groundtruth nodes & distances
                HNSW_LOG_INFO("[KNNAnchors] Groundtruth nodes & distances (", stats.groundtruth_nodes.size(), " nodes):");
                for (size_t i = 0; i < stats.groundtruth_nodes.size(); ++i) {
                    HNSW_LOG_INFO("[KNNAnchors]   GT-", (i+1), ": node=", stats.groundtruth_nodes[i].second, 
                                 ", distance=", stats.groundtruth_nodes[i].first);
                }
            } else {
                HNSW_LOG_INFO("[KNNAnchors] No groundtruth nodes available");
            }
            
            HNSW_LOG_INFO("[KNNAnchors] Search completed: total steps=", stats.total_steps, ", hops=", current_hop, 
                         ", budget=", budget, ", stagnation_steps=", stats.stagnation_steps);
            
            visited_list_pool_->releaseVisitedList(vl);
            return stats;
        }

        template <bool has_deletions>
        KNNTestSearchStats searchBaseLayerKNNTestCollection(
            tableint ep_id,
            const void* query_data,
            size_t k,
            int budget,
            float target_recall,
            const std::unordered_set<unsigned>* gt_set = nullptr,
            bool collect_graph = false
        ) const {
            KNNTestSearchStats stats;
            if (cur_element_count == 0) {
                return stats;
            }

            float recall_threshold = target_recall;
            if (recall_threshold < 0.0f) recall_threshold = 0.0f;
            if (recall_threshold > 1.0f) recall_threshold = 1.0f;

            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>,
                                std::vector<std::pair<dist_t, tableint>>,
                                CompareByFirst> candidate_set;
            std::priority_queue<std::pair<dist_t, tableint>> result_set;
            std::priority_queue<std::pair<dist_t, tableint>> top_k_queue;  // 独立的 top-k queue

            // Graph feature helpers
            std::unordered_set<labeltype> recorded_nodes;
            auto record_node_feature = [&](tableint internal_id, dist_t dist_to_query) {
                if (!collect_graph) return;
                labeltype ext_id = getExternalLabel(internal_id);
                if (!recorded_nodes.insert(ext_id).second) return;

                linklistsizeint* ll = get_linklist0(internal_id);
                uint32_t degree = getListCount(ll);

                float avg_dist = 0.0f;
                size_t cnt = 0;
                int* data_nb = (int*)ll;
                size_t deg_size = getListCount((linklistsizeint*)data_nb);
                for (size_t t = 1; t <= deg_size; t++) {
                    tableint nb = *(data_nb + t);
                    if (nb < 0 || nb >= cur_element_count) continue;
#ifndef ED2IP
                    dist_t nb_dist = fstdistfunc_(getDataByInternalId(internal_id), getDataByInternalId(nb), dist_func_param_);
#else
                    dist_t nb_dist = hnswlib::L2Sqr_by_IP(getDataByInternalId(internal_id), getDataByInternalId(nb), getExternalLabel(nb), dist_func_param_);
#endif
                    avg_dist += nb_dist;
                    cnt++;
                }
                if (cnt > 0) avg_dist /= static_cast<float>(cnt);
                 // hnswalg.h, record_node_feature 内
                typename KNNTestSearchStats::GraphNodeFeature node_feat{
                    ext_id, dist_to_query, degree, avg_dist
                };
                stats.graph_nodes.push_back(node_feat);
                // stats.graph_nodes.push_back(KNNTestSearchStats::GraphNodeFeature{
                //     ext_id, dist_to_query, degree, avg_dist
                // });
            };

            auto compute_recall = [&](const std::priority_queue<std::pair<dist_t, tableint>>& heap) -> float {
                if (!gt_set || gt_set->empty() || k == 0) {
                    return 0.0f;
                }
                
                // result_set is a max-heap (largest distance at top)
                // We need to get the k smallest distances (bottom of the heap)
                std::vector<std::pair<dist_t, tableint>> temp_vec;
                auto temp = heap;
                while (!temp.empty()) {
                    temp_vec.push_back(temp.top());
                    temp.pop();
                }
                
                // Sort by distance (ascending: smallest first)
                std::sort(temp_vec.begin(), temp_vec.end(),
                          [](const std::pair<dist_t, tableint>& a, const std::pair<dist_t, tableint>& b) {
                              return a.first < b.first;
                          });
                
                // Count how many of the k smallest are in ground truth
                int found = 0;
                size_t top_k_size = std::min(temp_vec.size(), static_cast<size_t>(k));
                for (size_t i = 0; i < top_k_size; ++i) {
                    unsigned label = getExternalLabel(temp_vec[i].second);
                    if (gt_set->count(label)) {
                        found++;
                    }
                }
                
                return static_cast<float>(found) / static_cast<float>(k);
            };

            auto compute_top_k_recall = [&]() -> float {
                if (!gt_set || gt_set->empty() || k == 0) {
                    return 0.0f;
                }
                
                // Directly check all elements in top_k_queue against groundtruth
                int found = 0;
                auto temp = top_k_queue;
                while (!temp.empty()) {
                    unsigned label = getExternalLabel(temp.top().second);
                    if (gt_set->count(label)) {
                        found++;
                    }
                    temp.pop();
                }
                
                return static_cast<float>(found) / static_cast<float>(k);
            };

            dist_t lowerBound;
            if (!has_deletions || !isMarkedDeleted(ep_id)) {
#ifndef ED2IP
                dist_t entry_dist = fstdistfunc_(query_data, getDataByInternalId(ep_id), dist_func_param_);
#else
                dist_t entry_dist = hnswlib::L2Sqr_by_IP(query_data, getDataByInternalId(ep_id), getExternalLabel(ep_id), dist_func_param_);
#endif
                // Standard searchBaseLayer logic: add entry point to both result_set and candidate_set
                result_set.emplace(entry_dist, ep_id);
                lowerBound = entry_dist;
                candidate_set.emplace(-entry_dist, ep_id);
                top_k_queue.emplace(entry_dist, ep_id);
                stats.total_distance_calcs++; // entry point distance
            } else {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidate_set.emplace(-lowerBound, ep_id);
            }

            visited_array[ep_id] = visited_array_tag;
            bool termination_triggered = false;
            float latest_recall = 0.0f;
            int current_hop = 0;
            float last_recorded_recall = 0.0f;  // Track last recorded recall for recall_curve
            bool first_gt_found = false;  // Track if first ground truth node has been encountered

            while (!candidate_set.empty() && current_hop < budget) {
                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
                
                // Check termination condition (searchBaseLayerST logic)
                dist_t current_dist = -current_node_pair.first;
                // if (current_dist > lowerBound && (result_set.size() == ef_ || !has_deletions)) {
                //     break;
                // }
                
                candidate_set.pop();
                tableint current_node_id = current_node_pair.second;

                // float recall_before_step = compute_recall(result_set);

                if (!termination_triggered &&
                    current_dist > lowerBound &&
                    (result_set.size() == ef_ || !has_deletions)) {
                    termination_triggered = true;
                    stats.early_termination_step = stats.total_steps + 1;
                    stats.early_termination_recall = compute_recall(result_set);
                    if (!stats.target_recall_achieved && stats.early_termination_recall >= recall_threshold && gt_set && !gt_set->empty()) {
                        stats.target_recall_achieved = true;
                        stats.step_target_recall_achieved = stats.total_steps;
                    }
                }

                stats.total_steps++;
                current_hop++;

                bool count_post_termination = termination_triggered && !stats.target_recall_achieved;
                if (count_post_termination) {
                    stats.hops_after_termination++;
                }

                stats.visited_nodes.emplace_back(current_dist, getExternalLabel(current_node_id));
                record_node_feature(current_node_id, current_dist);

                // Check if current node is the first ground truth node encountered
                // if (stats.total_steps == 10) {
                //         // labeltype current_label = getExternalLabel(current_node_id);
                      
                //             stats.first_gt_node_step = stats.total_steps;
                //             stats.first_gt_node_id = getExternalLabel(current_node_id);
                //             stats.first_gt_node_distance = current_dist;
                //             first_gt_found = true;
                //             break;
                        
                //     }
                

                // Explore neighbors (standard searchBaseLayer logic)
                int *data = (int *) get_linklist0(current_node_id);
                size_t size = getListCount((linklistsizeint*)data);

                for (size_t j = 1; j <= size; j++) {
                    tableint neighbor_id = *(data + j);
                    if (neighbor_id < 0 || neighbor_id >= cur_element_count) {
                        continue;
                    }
                    if (visited_array[neighbor_id] == visited_array_tag) {
                        continue;
                    }
                    visited_array[neighbor_id] = visited_array_tag;
#ifndef ED2IP
                    dist_t neighbor_dist = fstdistfunc_(query_data, getDataByInternalId(neighbor_id), dist_func_param_);
#else
                    dist_t neighbor_dist = hnswlib::L2Sqr_by_IP(query_data, getDataByInternalId(neighbor_id), getExternalLabel(neighbor_id), dist_func_param_);
#endif
                    stats.total_distance_calcs++;
                    if (termination_triggered) {
                        stats.distance_calcs_after_termination++;
                    }
                    
                    // searchBaseLayerST logic: check condition, add to candidate_set, then update result_set
                    if (result_set.size() < ef_ || lowerBound > neighbor_dist) {
#ifdef ED2IP
                        labeltype neighbor_ext = getExternalLabel(neighbor_id);
                        labeltype current_ext = getExternalLabel(current_node_id);
#else
                        labeltype neighbor_ext = getExternalLabel(neighbor_id);
                        labeltype current_ext = getExternalLabel(current_node_id);
#endif
                        candidate_set.emplace(-neighbor_dist, neighbor_id);
                        if (!has_deletions || !isMarkedDeleted(neighbor_id)) {
                            result_set.emplace(neighbor_dist, neighbor_id);
                            if (result_set.size() > ef_) {
                                result_set.pop();
                            }
                            if (!result_set.empty()) {
                                lowerBound = result_set.top().first;
                            }
                            
                            // Update top-k queue
                            top_k_queue.emplace(neighbor_dist, neighbor_id);
                            if (top_k_queue.size() > k) {
                                top_k_queue.pop();  // Remove the farthest node
                            }

                            if (collect_graph) {
                                stats.graph_edges.emplace_back(current_ext, neighbor_ext);
                                record_node_feature(neighbor_id, neighbor_dist);
                            }
                            
                            // Record recall improvement in recall_curve
                            if (gt_set && !gt_set->empty()) {
                                float current_recall = compute_top_k_recall();
                                if (current_recall > last_recorded_recall) {
                                    // Compute rank of neighbor_dist in result_set (sorted ascending).
                                    // rank = number of nodes in result_set with dist <= neighbor_dist (1-based).
                                    // This is the minimum ef needed to retain this neighbor in the beam.
                                    int candidate_rank = 1;
                                    {
                                        auto tmp = result_set; // result_set is a max-heap
                                        while (!tmp.empty()) {
                                            if (tmp.top().first <= neighbor_dist) {
                                                candidate_rank++;
                                            }
                                            tmp.pop();
                                        }
                                    }
                                    stats.recall_curve.emplace_back(stats.total_steps, current_recall, candidate_rank);
                                    // total_distance_calcs was already incremented for this neighbor above,
                                    // so it is exactly the NDC consumed to reach this recall level.
                                    stats.recall_curve_ndc.push_back(stats.total_distance_calcs);
                                    last_recorded_recall = current_recall;
                                }
                            }
                        }
                    }

                    // if (!first_gt_found && gt_set && !gt_set->empty()) {
                    //     // labeltype current_label = getExternalLabel(current_node_id);
                    //     if (gt_set->count(neighbor_id)) {
                    //         stats.first_gt_node_step = stats.total_steps;
                    //         stats.first_gt_node_id = neighbor_id;
                    //         stats.first_gt_node_distance = neighbor_dist;
                    //         first_gt_found = true;
                    //     }
                    // }
                }

                 // Check recall at each step
                 if (!stats.target_recall_achieved && gt_set && !gt_set->empty()) {
                    float current_recall = compute_top_k_recall();
                    if (current_recall >= recall_threshold) {
                        stats.target_recall_achieved = true;
                        stats.step_target_recall_achieved = stats.total_steps;
                    }
                }

                // latest_recall = compute_recall(result_set);
                // stats.final_recall = latest_recall;

                if (!stats.target_recall_achieved && termination_triggered && latest_recall >= recall_threshold && gt_set && !gt_set->empty()) {
                    stats.target_recall_achieved = true;
                    stats.step_target_recall_achieved = stats.total_steps;
                }

                // if (stats.target_recall_achieved && termination_triggered) {
                //     break;
                // }
            }

            // Extract final top-k nodes
            std::vector<std::pair<dist_t, tableint>> top_k_temp;
            while (!result_set.empty()) {
                top_k_temp.push_back(result_set.top());
                result_set.pop();
            }
            std::reverse(top_k_temp.begin(), top_k_temp.end());
            for (const auto& [dist, node_id] : top_k_temp) {
                if (node_id != ep_id) {
                    stats.top_k_nodes.emplace_back(dist, getExternalLabel(node_id));
                }
                if (stats.top_k_nodes.size() >= k) {
                    break;
                }
            }

            if (!gt_set || gt_set->empty()) {
                stats.early_termination_recall = 0.0f;
                stats.final_recall = 0.0f;
            } else {
                for (unsigned gt_label : *gt_set) {
                    try {
                        tableint gt_internal_id = getInternalId(gt_label);
                        if (gt_internal_id >= 0 && gt_internal_id < cur_element_count) {
#ifndef ED2IP
                            dist_t gt_dist = fstdistfunc_(query_data, getDataByInternalId(gt_internal_id), dist_func_param_);
#else
                            dist_t gt_dist = hnswlib::L2Sqr_by_IP(query_data, getDataByInternalId(gt_internal_id), getExternalLabel(gt_internal_id), dist_func_param_);
#endif
                            stats.groundtruth_nodes.emplace_back(gt_dist, gt_label);
                        }
                    } catch (...) {
                        continue;
                    }
                }
                std::sort(stats.groundtruth_nodes.begin(), stats.groundtruth_nodes.end(),
                          [](const std::pair<dist_t, labeltype>& a, const std::pair<dist_t, labeltype>& b) {
                              return a.first < b.first;
                          });
                
                // Calculate final recall based on top_k_nodes and groundtruth
                int found = 0;
                size_t top_k_size = std::min(stats.top_k_nodes.size(), k);
                for (size_t i = 0; i < top_k_size; ++i) {
                    if (gt_set->count(stats.top_k_nodes[i].second)) {
                        found++;
                    }
                }
                stats.final_recall = (k > 0) ? static_cast<float>(found) / static_cast<float>(k) : 0.0f;
            }

            // Update step_target_recall_achieved based on recall_curve
            if (!stats.recall_curve.empty()) {
                // First, try to find the step where target_recall is achieved
                bool found_target = false;
                for (const auto& [step, recall, rank] : stats.recall_curve) {
                    if (recall >= recall_threshold) {
                        stats.step_target_recall_achieved = step;
                        found_target = true;
                        break;
                    }
                }
                
                // If target_recall not found in curve, use the step with maximum recall
                if (!found_target) {
                    int max_recall_step = std::get<0>(stats.recall_curve[0]);
                    float max_recall = std::get<1>(stats.recall_curve[0]);
                    for (const auto& [step, recall, rank] : stats.recall_curve) {
                        if (recall > max_recall) {
                            max_recall = recall;
                            max_recall_step = step;
                        }
                    }
                    stats.step_target_recall_achieved = max_recall_step;
                }
            } else {
                // If recall_curve is empty, use original logic
                if (!stats.target_recall_achieved) {
                    stats.step_target_recall_achieved = stats.total_steps;
                }
            }

            visited_list_pool_->releaseVisitedList(vl);
            return stats;
        }

        BaseLayerSearchTraceStats searchBaseLayerTrace(
            const void* query_data,
            size_t k,
            size_t ef,
            const std::vector<labeltype>* gt_labels = nullptr,
            float target_recall = 1.0f,
            size_t progress_interval = 10,
            size_t max_events = 20000,
            size_t max_frontier = 512
        ) const {
            BaseLayerSearchTraceStats stats;
            if (cur_element_count == 0) {
                return stats;
            }

            tableint currObj = enterpoint_node_;
#ifndef ED2IP
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);
#else
            dist_t curdist = hnswlib::L2Sqr_by_IP(query_data, getDataByInternalId(enterpoint_node_), getExternalLabel(enterpoint_node_), dist_func_param_);
#endif
            for (int level = maxlevel_; level > 0; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data = (unsigned int *) get_linklist(currObj, level);
                    int size = getListCount(data);
                    tableint *datal = (tableint *) (data + 1);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_) {
                            throw std::runtime_error("cand error");
                        }
#ifndef ED2IP
                        dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);
#else
                        dist_t d = hnswlib::L2Sqr_by_IP(query_data, getDataByInternalId(cand), getExternalLabel(cand), dist_func_param_);
#endif
                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }

            if (num_deleted_) {
                stats = searchBaseLayerTraceCollection<true>(
                    currObj, query_data, k, std::max(ef, k), gt_labels,
                    target_recall, progress_interval, max_events, max_frontier);
            } else {
                stats = searchBaseLayerTraceCollection<false>(
                    currObj, query_data, k, std::max(ef, k), gt_labels,
                    target_recall, progress_interval, max_events, max_frontier);
            }
            return stats;
        }

        /**
         * @brief Perform KNN search from an anchor point with fixed budget and collect probe statistics.
         * 
         * This is a wrapper function that calls searchBaseLayerKNNAnchors with appropriate template parameters.
         * 
         * @param anchor_id Internal ID of the anchor node (starting point)
         * @param query_data Pointer to the query vector (anchor's vector)
         * @param k Number of neighbors for recall calculation
         * @param budget Maximum number of hops allowed
         * @param gt_set Ground truth set for recall calculation (nullptr if not available)
         * @return ProbeSearchStats containing search trajectory statistics
         */
        ProbeSearchStats searchKNNAnchors(
            // tableint anchor_id,
            const void* query_data,
            size_t k,
            int budget,
            const std::unordered_set<unsigned>* gt_set = nullptr
        ) const {
            ProbeSearchStats stats;
            stats.stagnation_steps = 0;
            stats.total_steps = 0;
            stats.early_termination_step = -1;
            stats.early_termination_recall = 0.0f;
            // stats.skipped_nodes = 0;
            // stats.entry_dist = 0.0f;
            // std::priority_queue<std::pair<dist_t, labeltype>> &result = stats.neighbors;
            if (cur_element_count == 0) {
                HNSW_LOG_INFO("Graph is empty. Returning empty result.");
                return stats;
            }
            tableint currObj = enterpoint_node_;
#ifndef ED2IP
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);
#else
            dist_t curdist = hnswlib::L2Sqr_by_IP(query_data, getDataByInternalId(enterpoint_node_), getExternalLabel(enterpoint_node_), dist_func_param_);
#endif
            // stats.entry_dist = curdist;
            // HNSW_LOG_INFO("Starting search from entry point: ", currObj, ", distance: ", curdist);
            adsampling::tot_dist_calculation ++;
            adsampling::tot_full_dist++;

            /* Traverse the graph from the top level to the base layer */ 
            for (int level = maxlevel_; level > 0; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data;
                    data = (unsigned int *) get_linklist(currObj, level);
                    int size = getListCount(data);
                    metric_hops++;
                    metric_distance_computations+=size;
                    adsampling::tot_hops++;
                    tableint *datal = (tableint *) (data + 1);
                    HNSW_LOG_INFO("Exploring level ", level, ", current node: ", currObj, ", neighbors: ", size);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                        else {
#ifndef ED2IP
                            dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);
#else
                            dist_t d = hnswlib::L2Sqr_by_IP(query_data, getDataByInternalId(cand), getExternalLabel(cand), dist_func_param_);
#endif
                            adsampling::tot_full_dist++;
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                                HNSW_LOG_INFO("Found closer node: ", cand, ", distance: ", d);
                            }
                        }
                    }
                }
            }
            // Use the updated searchBaseLayerDataCollection function to collect neighbors and skipped node count
            if (num_deleted_) {
                stats = searchBaseLayerKNNAnchors<true>(currObj, query_data, k, budget, gt_set);
            } else {
                stats = searchBaseLayerKNNAnchors<false>(currObj, query_data, k, budget, gt_set);
            }
            // HNSW_LOG_INFO("Collected ", neighbors.size(), " neighbors from base layer.");
            // for (const auto &neighbor : neighbors) {
            //     dist_t distance = neighbor.first;
            //     int hop_count = neighbor.second;
            //     // Note: neighbor.second is hop_count, not internal_id. You may want to return internal_id in the tuple if needed.
            //     // Here, we just use hop_count as a placeholder for labeltype if needed.
            //     result.emplace(distance, hop_count);
            // }
            // HNSW_LOG_INFO("Search completed. Returning top ", k, " neighbors.");
            return stats;
        }

        KNNTestSearchStats searchKNNTestCollection(
            const void* query_data,
            size_t k,
            int budget,
            float target_recall,
            const std::unordered_set<unsigned>* gt_set = nullptr,
            bool collect_graph = false
        ) const {
            KNNTestSearchStats stats;
            if (cur_element_count == 0) {
                HNSW_LOG_INFO("Graph is empty. Returning empty KNN test stats.");
                return stats;
            }

            tableint currObj = enterpoint_node_;
#ifndef ED2IP
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);
#else
            dist_t curdist = hnswlib::L2Sqr_by_IP(query_data, getDataByInternalId(enterpoint_node_), getExternalLabel(enterpoint_node_), dist_func_param_);
#endif
            adsampling::tot_dist_calculation++;
            adsampling::tot_full_dist++;

            for (int level = maxlevel_; level > 0; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data = (unsigned int *) get_linklist(currObj, level);
                    int size = getListCount(data);
                    metric_hops++;
                    metric_distance_computations += size;
                    adsampling::tot_hops++;
                    tableint *datal = (tableint *) (data + 1);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                        else {
#ifndef ED2IP
                            dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);
#else
                            dist_t d = hnswlib::L2Sqr_by_IP(query_data, getDataByInternalId(cand), getExternalLabel(cand), dist_func_param_);
#endif
                            adsampling::tot_full_dist++;
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }
            // HNSW_LOG_INFO("Searching from current object ", currObj, " with k=", k, " and budget=", budget, " and target_recall=", target_recall);
            stats = searchBaseLayerKNNTestCollection<false>(currObj, query_data, k, budget, target_recall, gt_set, collect_graph);

            // if (num_deleted_) {
            //     stats = searchBaseLayerKNNTestCollection<true>(currObj, query_data, k, budget, target_recall, gt_set);
            // } else {
            //     stats = searchBaseLayerKNNTestCollection<false>(currObj, query_data, k, budget, target_recall, gt_set);
            // }
            return stats;
        }

/**
 * @brief Structure to hold effective branching factor calculation results
 */
struct EffectiveBranchingResult {
    std::vector<std::pair<dist_t, int>> neighbors;  // Original neighbor results
    float effective_branching_factor;                // Effective branching factor b_eff
    float novelty_rate;                             // Novelty rate ν̄(q)
    float local_clustering_coeff;                   // Local clustering coefficient LCC_local(q)
    // float prior_branching_factor;                   // Prior estimate b_eff^prior
    std::vector<float> node_novelty_rates;          // Individual node novelty rates
    std::vector<float> node_clustering_coeffs;      // Individual node clustering coefficients

    int visited_node_num;
    int average_degree;
    
    // Path length estimation components
    float estimated_path_hops;                      // Estimated path length path_hops(q)
    float basin_size_estimate;                     // Estimated basin size N_basin(q)
    float replacement_rate;                         // Replacement rate p_rep
    float gap12_normalized;                        // Normalized gap12 for alignment correction
    float alignment_correction;                     // Alignment correction factor φ(gap12_norm)
    
    // LID-volume method components
    float lid_estimate;                             // Estimated LID for the query
    float current_best_distance;                    // Current best distance r_*
    float distance_decay_rate;                      // Distance decay rate dr_*/dexp
    float basin_size_lid_method;                   // Basin size estimate using LID method
    float volume_expansion_factor;                  // Volume expansion factor from LID
    
    // Top-k distances for gap calculation
    std::vector<dist_t> top_distances;             // Top-k distances for gap analysis
    std::vector<int> replacement_events;           // Replacement events during search
    
    // Routing difficulty features
    float cohesion;                                 // Cohesion measure: mean pairwise distance / median query distance
    float contrast;                                 // Contrast measure: separation between top-m and background
    float cone_concentration;                       // Direction cone concentration (when vector coordinates available)
    float routing_difficulty_score;                // Overall routing difficulty score (0-1, higher = more difficult)
    bool is_converged_to_knn_region;               // Whether the search has converged to a kNN-like region
    
    // Structural consistency features
    float mutual_nn_rate;                           // Mutual-NN rate: fraction of top-m pairs that are neighbors
    float induced_subgraph_lcc;                     // Local clustering coefficient on induced subgraph of top-m
    float anchor_consistency;                       // Anchor consistency: max vote ratio for top-m mapping to same anchor
    
    // Trajectory convergence features
    float replacement_rate_decay;                   // Replacement rate decay slope over sliding window
    float improvement_slope;                        // Median improvement rate of best distance
    float improvement_curvature;                    // Curvature of improvement trend (log fitting residuals)
    float topk_half_life;                          // Median survival steps of new nodes in top-k
    
    // Target radius hit features
    float radius_hit_rate;                         // Fraction of top-m nodes within typical kNN radius
    float lid_topm_vs_frontier;                    // LID difference between top-m and background
    
    // Early detour/redundancy features (E. "是否在浪费步？")
    float entry_detour_steps;                       // E1: Steps to reach threshold stability
    float entry_detour_ratio;                       // E1: Ratio of detour steps to total steps
    float novelty_rate_ratio;                       // E2: Novelty rate ratio (new discoveries)
    float duplication_rate;                         // E2: Duplication rate (redundant expansions)
    float frontier_overlap_ratio;                   // E3: Ratio of new discoveries already in queue/visited
    
    // NN Statistics
    int ndis;                                       // Number of distance calculations
    int ninserts;                                   // Number of updates in NN result set
    float first_NN_distance;                        // Distance of first NN found
    float closest_NN_distance;                      // Distance of current closest NN
    float furthest_NN_distance;                     // Distance of current furthest (k-th) NN
    float avg;                                      // Average of distances of the NN
    float var;                                      // Variance of distances of NN
    float med;                                      // Median of distances of NN
    float perc25;                                   // 25th percentile of distances of NN
    float perc75;                                   // 75th percentile of distances of NN
};

/**
 * @brief Per-step statistics collected during base layer search
 * 
 * Contains comprehensive statistics for each step of the search process,
 * organized into five categories: Geometry/Distance, Bounds/Certainty,
 * Search State, Graph/Structure, and Cost/Time features.
 */
struct StepStatistics {
    // Step metadata
    int step_number;
    tableint current_node_id;
    
    // A. Geometry/Distance features
    float sim_abs;              // <q, u_t> or ||q - u_t||
    float dist_rel_k;           // dist_abs / UB_k(t)
    float delta_k;              // UB_k(t-1) - UB_k(t)
    float gap_top;              // UB_k(t) - UB_1(t)
    float gap_delta;            // gap_top(t-1) - gap_top(t)
    float dir_cos;              // cos angle with previous step direction
    float dir_var_w;            // direction variance over window W
    float dist;
    float delta_distance;       // Distance change from previous step
    float best_distance_so_far; // Current best distance (UB_1)
    
    // B. Bounds/Certainty features
    float lb_min;               // min lower bound estimate
    float lb_gap;               // UB_k(t) - lb_min(t)
    float lb_tight_ratio;       // lb_gap(t) / (lb_gap(0) + eps)
    float lb_delta;             // lb_gap(t-1) - lb_gap(t)
    
    // C. Search State features
    int heap_size;
    float visited_ratio;
    float cand_lb_q25, cand_lb_q50, cand_lb_q75;
    float cand_sim_q25, cand_sim_q50, cand_sim_q75;
    float topk_spread;
    bool improve_flag;
    int stale_len;
    
    // D. Graph/Structure features
    int deg;                    // current node degree
    float nn_sim_q25, nn_sim_q50, nn_sim_q75;
    float local_cc;             // local clustering coefficient
    bool revisit;
    float frontier_overlap;
    
    // E. Cost/Time features
    int cost_dist;              // distance evaluations this step
    int cost_heap;              // heap operations this step
    int cost_total;             // total cost this step
    int cost_cum;               // cumulative cost
    float gain_per_cost;        // delta UB_k / cost

    int hop_count;
    
    // Neighbors info for current node
    struct {
        int updates_in_result_set;   // Number of result set insertions this step
        float updates_ratio;          // updates / neighbors_explored
        float dist_std;               // Std dev of neighbor distances to query
        // int degree;                   // Degree of current node (duplicate of deg)
        float min_dist;               // Minimum distance among neighbors
    } neighbors_info;
    
    // Fourier cost encoding (for positional encoding based on cost)
    std::vector<float> fourier_cost;  // sin/cos features of cumulative cost
};

/**
 * @brief Result structure for search with step-by-step statistics
 */
struct SearchWithStatsResult {
    std::priority_queue<std::pair<dist_t, tableint>> top_candidates;
    std::vector<StepStatistics> step_stats;
};

/**
 * @brief Perform early probe routing analysis to assess if the search has converged to correct kNN region
 * 
 * This function performs a lightweight probe (50-100 unique nodes, ≤200 distance calculations)
 * to determine if the index has routed the query to the correct kNN region and compute
 * routing difficulty features.
 * 
 * @tparam has_deletions Indicates whether the graph contains deleted nodes
 * @param ep_id Entry point node ID
 * @param data_point Pointer to the query vector
 * @param m Maximum number of distinct neighbors to visit (early probe limit)
 * @param ef Exploration factor for the probe
 * @param k Target k for kNN search
 * @param metric Optional parameter for collecting metrics
 * @return EffectiveBranchingResult containing routing analysis and difficulty features
 */
template <bool has_deletions, bool collect_metrics = false>
EffectiveBranchingResult searchBaseLayerRouting(
    tableint ep_id, 
    const void *data_point, 
    size_t m, 
    size_t ef, 
    size_t k,
    Metric* metric = nullptr) const {
    
    // Start overall timing
    auto overall_start = std::chrono::high_resolution_clock::now();
    
    HNSW_LOG_DEBUG("[searchBaseLayerRouting] Starting early probe routing analysis with m=", m, ", ef=", ef, ", k=", k);
    
    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    VisitedList *cl = visited_list_pool_->getFreeVisitedList();
    vl_type *calculated_array = cl->mass;
    vl_type calculated_array_tag = cl->curV;

    EffectiveBranchingResult result;
    
    // Initialize routing difficulty features
    result.cohesion = 0.0f;
    result.contrast = 0.0f;
    result.cone_concentration = 0.0f;
    result.routing_difficulty_score = 0.0f;
    result.is_converged_to_knn_region = false;
    
    // Initialize structural consistency features
    result.mutual_nn_rate = 0.0f;
    result.induced_subgraph_lcc = 0.0f;
    result.anchor_consistency = 0.0f;
    
    // Initialize trajectory convergence features
    result.replacement_rate_decay = 0.0f;
    result.improvement_slope = 0.0f;
    result.improvement_curvature = 0.0f;
    result.topk_half_life = 0.0f;
    
    // Initialize target radius hit features
    result.radius_hit_rate = 0.0f;
    result.lid_topm_vs_frontier = 0.0f;
    
    // Initialize early detour/redundancy features
    result.entry_detour_steps = 0.0f;
    result.entry_detour_ratio = 0.0f;
    result.novelty_rate_ratio = 0.0f;
    result.duplication_rate = 0.0f;
    result.frontier_overlap_ratio = 0.0f;
    
    // Initialize other fields
    result.effective_branching_factor = 0.0f;
    result.novelty_rate = 0.0f;
    result.local_clustering_coeff = 0.0f;
    result.estimated_path_hops = 0.0f;
    result.basin_size_estimate = 0.0f;
    result.replacement_rate = 0.0f;
    result.gap12_normalized = 0.0f;
    result.alignment_correction = 1.0f;
    result.lid_estimate = 0.0f;
    result.current_best_distance = 0.0f;
    result.distance_decay_rate = 0.0f;
    result.basin_size_lid_method = 0.0f;
    result.volume_expansion_factor = 1.0f;
    result.visited_node_num = 0;
    result.average_degree = 0;
    
    // Initialize NN Statistics
    result.ndis = 0;
    result.ninserts = 0;
    result.first_NN_distance = 0.0f;
    result.closest_NN_distance = 0.0f;
    result.furthest_NN_distance = 0.0f;
    result.avg = 0.0f;
    result.var = 0.0f;
    result.med = 0.0f;
    result.perc25 = 0.0f;
    result.perc75 = 0.0f;
    
    using Candidate = std::tuple<dist_t, tableint, int>;
    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<>> candidate_set;
    
    // Maintain top-k distances for analysis
    std::priority_queue<std::pair<dist_t, tableint>> top_k_distances;
    std::vector<dist_t> top_distances;
    top_distances.reserve(k);
    
    // Track visited nodes and their distances for routing analysis
    std::vector<std::pair<tableint, dist_t>> visited_nodes_with_distances;
    visited_nodes_with_distances.reserve(m);
    
    // Track trajectory convergence features
    const size_t SLIDING_WINDOW_SIZE = 32;  // Sliding window for replacement rate decay
    std::vector<bool> replacement_events;   // Track replacement events over time
    std::vector<dist_t> best_distances;     // Track best distance improvements over time
    std::vector<std::pair<tableint, int>> topk_node_lifetimes; // Track node survival in top-k
    
    // Track early detour/redundancy features
    std::vector<dist_t> distance_history;   // Track distance improvements over time
    std::vector<bool> is_novel_discovery;   // Track whether each discovery is novel
    std::vector<bool> is_in_queue;          // Track whether discoveries are already in queue
    std::vector<int> discovery_steps;       // Track at which step each discovery was made
    
    replacement_events.reserve(m);
    best_distances.reserve(m);
    topk_node_lifetimes.reserve(m);
    distance_history.reserve(m);
    is_novel_discovery.reserve(m);
    is_in_queue.reserve(m);
    discovery_steps.reserve(m);
    
    dist_t lowerBound;
    
    if (!has_deletions || !isMarkedDeleted(ep_id)) {
        dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
        result.ndis++; // Count distance calculation
        candidate_set.emplace(dist, ep_id, 0);
        lowerBound = dist;
        
        // Initialize top-k with entry point
        top_k_distances.emplace(dist, ep_id);
        top_distances.push_back(dist);
        
        // Record entry point
        visited_nodes_with_distances.emplace_back(ep_id, dist);
        
        // Set first NN distance (entry point is the first NN found)
        result.first_NN_distance = dist;
    } else {
        lowerBound = std::numeric_limits<dist_t>::max();
        candidate_set.emplace(lowerBound, ep_id, 0);
        HNSW_LOG_WARN("[searchBaseLayerRouting] Entry point is marked as deleted.");
    }
    
    visited_array[ep_id] = visited_array_tag;
    int cnt_visit = 0;
    int hop_count = 0;
    size_t distinct_neighbors_visited = 0;
    
    // Track expanded nodes for analysis
    std::vector<tableint> expanded_nodes;
    expanded_nodes.reserve(m);
    
    // Start timing for early probe search loop
    // auto probe_loop_start = std::chrono::high_resolution_clock::now();
    // HNSW_LOG_DEBUG("[searchBaseLayerRouting] Starting early probe loop...");
    
    // Early probe loop - limit to m distinct nodes
    while (!candidate_set.empty() && result.visited_node_num < m) {
        auto [current_dist, current_node_id, current_hop_count] = candidate_set.top();
        candidate_set.pop();
        
        // If the node is already visited, skip it
        if (visited_array[current_node_id] == visited_array_tag && current_hop_count > 0) {
            continue;
        }
        
        // Mark the node as visited
        visited_array[current_node_id] = visited_array_tag;
        
        // Track expanded nodes for analysis
        if (current_hop_count > 0) { // Don't count entry point
            expanded_nodes.push_back(current_node_id);
            result.visited_node_num++;
        }

        // Explore neighbors of the current node
        int *data = (int *) get_linklist0(current_node_id);
        size_t size = getListCount((linklistsizeint*)data);
        result.average_degree += size;

        // Calculate Novelty count
        float novelty = calculateNodeNoveltyRate(current_node_id, calculated_array, calculated_array_tag);
        result.novelty_rate += novelty;
        
        if(collect_metrics){
            metric_hops++;
            metric_distance_computations+=size;
        }
        
        for (size_t j = 1; j <= size; j++) {
            int candidate_id = *(data + j);
            
            // If the neighbor is already visited, skip it
            if (visited_array[candidate_id] == visited_array_tag) {
                continue;
            }

            // Calculate distance to the neighbor
            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(candidate_id), dist_func_param_);
            result.ndis++; // Count distance calculation
            calculated_array[candidate_id] = calculated_array_tag;
            
            candidate_set.emplace(dist, candidate_id, current_hop_count + 1);
            
            // Record visited node with distance for routing analysis
            visited_nodes_with_distances.emplace_back(candidate_id, dist);
            
            // Check if this distance improves top-k
            bool replacement_occurred = false;
            bool is_novel = true;
            bool already_in_queue = false;
            
            // Check if this is a novel discovery (not already in top-k)
            if (top_k_distances.size() >= k) {
                // Check if candidate_id is already in top-k
                auto temp_queue = top_k_distances;
                while (!temp_queue.empty()) {
                    if (temp_queue.top().second == candidate_id) {
                        is_novel = false;
                        break;
                    }
                    temp_queue.pop();
                }
            }
            
            // Check if candidate is already in the candidate queue
            auto temp_candidate_set = candidate_set;
            while (!temp_candidate_set.empty()) {
                auto [temp_dist, temp_id, temp_hop] = temp_candidate_set.top();
                if (temp_id == candidate_id) {
                    already_in_queue = true;
                    break;
                }
                temp_candidate_set.pop();
            }
            
            if (top_k_distances.size() < k) {
                top_k_distances.emplace(dist, candidate_id);
                top_distances.push_back(dist);
                std::sort(top_distances.begin(), top_distances.end());
                result.ninserts++; // Count insertion to NN result set
                
                // Track new node entry to top-k
                topk_node_lifetimes.emplace_back(candidate_id, hop_count);
                
                // Track novel discovery
                is_novel_discovery.push_back(true);
                is_in_queue.push_back(already_in_queue);
                discovery_steps.push_back(hop_count);
                distance_history.push_back(dist);
            } else if (dist < top_k_distances.top().first) {
                // Replacement event occurred
                replacement_occurred = true;
                replacement_events.push_back(true);
                
                // Remove replaced node from lifetime tracking
                tableint replaced_node = top_k_distances.top().second;
                for (auto it = topk_node_lifetimes.begin(); it != topk_node_lifetimes.end(); ++it) {
                    if (it->first == replaced_node) {
                        topk_node_lifetimes.erase(it);
                        break;
                    }
                }
                
                top_k_distances.pop();
                top_k_distances.emplace(dist, candidate_id);
                result.ninserts++; // Count replacement in NN result set
                
                // Update top_distances vector
                top_k_distances = std::priority_queue<std::pair<dist_t, tableint>>();
                auto temp_queue = top_k_distances;
                while (!temp_queue.empty()) {
                    top_distances.push_back(temp_queue.top().first);
                    temp_queue.pop();
                }
                std::sort(top_distances.begin(), top_distances.end());
                
                // Track new node entry to top-k
                topk_node_lifetimes.emplace_back(candidate_id, hop_count);
                
                // Track novel discovery
                is_novel_discovery.push_back(true);
                is_in_queue.push_back(already_in_queue);
                discovery_steps.push_back(hop_count);
                distance_history.push_back(dist);
            } else {
                // No replacement
                replacement_events.push_back(false);
                
                // Track non-novel discovery
                is_novel_discovery.push_back(false);
                is_in_queue.push_back(already_in_queue);
                discovery_steps.push_back(hop_count);
                distance_history.push_back(dist);
            }
            
            // Track best distance improvement
            if (!top_distances.empty()) {
                best_distances.push_back(top_distances[0]);
            }
            
            if (distinct_neighbors_visited >= m) {
                break;
            }
        }
        hop_count++;
    }
    
    // End timing for early probe loop
    // auto probe_loop_end = std::chrono::high_resolution_clock::now();
    // auto probe_loop_duration = std::chrono::duration_cast<std::chrono::microseconds>(probe_loop_end - probe_loop_start);
    // HNSW_LOG_INFO("[searchBaseLayerRouting] [TIMING] Early probe loop: ", probe_loop_duration.count() / 1000.0, " ms");
    // HNSW_LOG_INFO("[searchBaseLayerRouting] [STATS] Visited ", result.visited_node_num, " nodes, ", hop_count, " hops");
    
    // Store top distances
    result.top_distances = top_distances;
    
    // Start timing for NN statistics calculation
    // auto nn_stats_start = std::chrono::high_resolution_clock::now();
    
    // Calculate NN Statistics
    if (!top_distances.empty()) {
        // Sort distances for statistics calculation
        std::vector<dist_t> sorted_distances = top_distances;
        std::sort(sorted_distances.begin(), sorted_distances.end());
        
        // Set closest and furthest NN distances
        result.closest_NN_distance = sorted_distances[0]; // Closest (smallest distance)
        result.furthest_NN_distance = sorted_distances.back(); // Furthest (largest distance)
        
        // Calculate average
        float sum = 0.0f;
        for (dist_t dist : sorted_distances) {
            sum += dist;
        }
        result.avg = sum / sorted_distances.size();
        
        // Calculate variance
        float variance_sum = 0.0f;
        for (dist_t dist : sorted_distances) {
            float diff = dist - result.avg;
            variance_sum += diff * diff;
        }
        result.var = variance_sum / sorted_distances.size();
        
        // Calculate median
        size_t n = sorted_distances.size();
        if (n % 2 == 0) {
            result.med = (sorted_distances[n/2 - 1] + sorted_distances[n/2]) / 2.0f;
        } else {
            result.med = sorted_distances[n/2];
        }
        
        // Calculate percentiles
        result.perc25 = sorted_distances[static_cast<size_t>(n * 0.25)];
        result.perc75 = sorted_distances[static_cast<size_t>(n * 0.75)];
    }
    
    // auto nn_stats_end = std::chrono::high_resolution_clock::now();
    // auto nn_stats_duration = std::chrono::duration_cast<std::chrono::microseconds>(nn_stats_end - nn_stats_start);
    // HNSW_LOG_INFO("[searchBaseLayerRouting] [TIMING] NN Statistics: ", nn_stats_duration.count() / 1000.0, " ms");
    
    // Calculate routing difficulty features with detailed timing
    if (!expanded_nodes.empty() && !top_distances.empty()) {
        // HNSW_LOG_INFO("[searchBaseLayerRouting] ========== Computing Routing Difficulty Features ==========");
        
        // === Category A: Target Clustering/Separation ===
        // auto cat_a_start = std::chrono::high_resolution_clock::now();
        
        // A1. Calculate cohesion
        // auto cohesion_start = std::chrono::high_resolution_clock::now();
        // result.cohesion = calculateCohesion(visited_nodes_with_distances, top_distances);
        result.cohesion = calculateCohesion2(visited_nodes_with_distances, top_distances, k);
        // auto cohesion_end = std::chrono::high_resolution_clock::now();
        // auto cohesion_duration = std::chrono::duration_cast<std::chrono::microseconds>(cohesion_end - cohesion_start);
        // HNSW_LOG_INFO("[searchBaseLayerRouting] [A1] Cohesion = ", result.cohesion, " | Time: ", cohesion_duration.count() / 1000.0, " ms");
        
        // A2. Calculate contrast
        // auto contrast_start = std::chrono::high_resolution_clock::now();
        result.contrast = calculateContrast(visited_nodes_with_distances, top_distances, m, k);
        // auto contrast_end = std::chrono::high_resolution_clock::now();
        // auto contrast_duration = std::chrono::duration_cast<std::chrono::microseconds>(contrast_end - contrast_start);
        // HNSW_LOG_INFO("[searchBaseLayerRouting] [A2] Contrast = ", result.contrast, " | Time: ", contrast_duration.count() / 1000.0, " ms");
        
        // A3. Calculate cone concentration
        // auto cone_start = std::chrono::high_resolution_clock::now();
        result.cone_concentration = calculateConeConcentration(data_point, visited_nodes_with_distances, top_distances, k);
        // auto cone_end = std::chrono::high_resolution_clock::now();
        // auto cone_duration = std::chrono::duration_cast<std::chrono::microseconds>(cone_end - cone_start);
        // HNSW_LOG_INFO("[searchBaseLayerRouting] [A3] Cone Concentration = ", result.cone_concentration, " | Time: ", cone_duration.count() / 1000.0, " ms");
        
        // auto cat_a_end = std::chrono::high_resolution_clock::now();
        // auto cat_a_duration = std::chrono::duration_cast<std::chrono::microseconds>(cat_a_end - cat_a_start);
        // HNSW_LOG_INFO("[searchBaseLayerRouting] [TIMING] Category A Total: ", cat_a_duration.count() / 1000.0, " ms");
        
        // === Category B: Structural Consistency ===
        // auto cat_b_start = std::chrono::high_resolution_clock::now();
        
        // B1. Calculate mutual-NN rate
        // auto mutual_nn_start = std::chrono::high_resolution_clock::now();
        result.mutual_nn_rate = calculateMutualNNRate(visited_nodes_with_distances, top_distances, m);
        // auto mutual_nn_end = std::chrono::high_resolution_clock::now();
        // auto mutual_nn_duration = std::chrono::duration_cast<std::chrono::microseconds>(mutual_nn_end - mutual_nn_start);
        // HNSW_LOG_INFO("[searchBaseLayerRouting] [B1] Mutual NN Rate = ", result.mutual_nn_rate, " | Time: ", mutual_nn_duration.count() / 1000.0, " ms");
        
        // B2. Calculate induced subgraph LCC
        // auto lcc_start = std::chrono::high_resolution_clock::now();
        result.induced_subgraph_lcc = calculateInducedSubgraphLCC(visited_nodes_with_distances, top_distances, m);
        // auto lcc_end = std::chrono::high_resolution_clock::now();
        // auto lcc_duration = std::chrono::duration_cast<std::chrono::microseconds>(lcc_end - lcc_start);
        // HNSW_LOG_INFO("[searchBaseLayerRouting] [B2] Induced Subgraph LCC = ", result.induced_subgraph_lcc, " | Time: ", lcc_duration.count() / 1000.0, " ms");
        
        // B3. Calculate anchor consistency
        // auto anchor_start = std::chrono::high_resolution_clock::now();
        result.anchor_consistency = calculateAnchorConsistency(visited_nodes_with_distances, top_distances, m);
        // auto anchor_end = std::chrono::high_resolution_clock::now();
        // auto anchor_duration = std::chrono::duration_cast<std::chrono::microseconds>(anchor_end - anchor_start);
        // HNSW_LOG_INFO("[searchBaseLayerRouting] [B3] Anchor Consistency = ", result.anchor_consistency, " | Time: ", anchor_duration.count() / 1000.0, " ms");
        
        // auto cat_b_end = std::chrono::high_resolution_clock::now();
        // auto cat_b_duration = std::chrono::duration_cast<std::chrono::microseconds>(cat_b_end - cat_b_start);
        // HNSW_LOG_INFO("[searchBaseLayerRouting] [TIMING] Category B Total: ", cat_b_duration.count() / 1000.0, " ms");
        
        // // === Category C: Trajectory Convergence ===
        // auto cat_c_start = std::chrono::high_resolution_clock::now();
        
        // C1. Calculate replacement rate decay
        // auto decay_start = std::chrono::high_resolution_clock::now();
        result.replacement_rate_decay = calculateReplacementRateDecay(replacement_events, SLIDING_WINDOW_SIZE);
        // auto decay_end = std::chrono::high_resolution_clock::now();
        // auto decay_duration = std::chrono::duration_cast<std::chrono::microseconds>(decay_end - decay_start);
        // HNSW_LOG_INFO("[searchBaseLayerRouting] [C1] Replacement Rate Decay = ", result.replacement_rate_decay, " | Time: ", decay_duration.count() / 1000.0, " ms");
        
        // C2. Calculate improvement slope and curvature
        // auto improvement_start = std::chrono::high_resolution_clock::now();
        auto [slope, curvature] = calculateImprovementTrend(best_distances);
        result.improvement_slope = slope;
        result.improvement_curvature = curvature;
        // auto improvement_end = std::chrono::high_resolution_clock::now();
        // auto improvement_duration = std::chrono::duration_cast<std::chrono::microseconds>(improvement_end - improvement_start);
        // HNSW_LOG_INFO("[searchBaseLayerRouting] [C2] Improvement Slope = ", result.improvement_slope, ", Curvature = ", result.improvement_curvature, " | Time: ", improvement_duration.count() / 1000.0, " ms");
        
        // C3. Calculate top-k half-life
        // auto halflife_start = std::chrono::high_resolution_clock::now();
        result.topk_half_life = calculateTopKHalfLife(topk_node_lifetimes, hop_count);
        // auto halflife_end = std::chrono::high_resolution_clock::now();
        // auto halflife_duration = std::chrono::duration_cast<std::chrono::microseconds>(halflife_end - halflife_start);
        // HNSW_LOG_INFO("[searchBaseLayerRouting] [C3] Top-k Half Life = ", result.topk_half_life, " | Time: ", halflife_duration.count() / 1000.0, " ms");
        
        // auto cat_c_end = std::chrono::high_resolution_clock::now();
        // auto cat_c_duration = std::chrono::duration_cast<std::chrono::microseconds>(cat_c_end - cat_c_start);
        // HNSW_LOG_INFO("[searchBaseLayerRouting] [TIMING] Category C Total: ", cat_c_duration.count() / 1000.0, " ms");
        
        // === Category D: Target Radius Hit ===
        // auto cat_d_start = std::chrono::high_resolution_clock::now();
        
        // D1. Calculate radius hit rate
        // auto radius_start = std::chrono::high_resolution_clock::now();
        result.radius_hit_rate = calculateRadiusHitRate(visited_nodes_with_distances, top_distances, m, k);
        // auto radius_end = std::chrono::high_resolution_clock::now();
        // auto radius_duration = std::chrono::duration_cast<std::chrono::microseconds>(radius_end - radius_start);
        // HNSW_LOG_INFO("[searchBaseLayerRouting] [D1] Radius Hit Rate = ", result.radius_hit_rate, " | Time: ", radius_duration.count() / 1000.0, " ms");
        
        // D2. Calculate LID difference between top-m and background
        // auto lid_diff_start = std::chrono::high_resolution_clock::now();
        result.lid_topm_vs_frontier = calculateLIDTopmVsFrontier(visited_nodes_with_distances, top_distances, m, k);
        // auto lid_diff_end = std::chrono::high_resolution_clock::now();
        // auto lid_diff_duration = std::chrono::duration_cast<std::chrono::microseconds>(lid_diff_end - lid_diff_start);
        // HNSW_LOG_INFO("[searchBaseLayerRouting] [D2] LID Top-m vs Frontier = ", result.lid_topm_vs_frontier, " | Time: ", lid_diff_duration.count() / 1000.0, " ms");
        
        // auto cat_d_end = std::chrono::high_resolution_clock::now();
        // auto cat_d_duration = std::chrono::duration_cast<std::chrono::microseconds>(cat_d_end - cat_d_start);
        // HNSW_LOG_INFO("[searchBaseLayerRouting] [TIMING] Category D Total: ", cat_d_duration.count() / 1000.0, " ms");
        
        // // === Category E: Early Detour/Redundancy ===
        // auto cat_e_start = std::chrono::high_resolution_clock::now();
        
        // E1. Calculate entry detour features
        // auto detour_start = std::chrono::high_resolution_clock::now();
        result.entry_detour_steps = calculateEntryDetourSteps(distance_history, discovery_steps, k);
        result.entry_detour_ratio = calculateEntryDetourRatio(result.entry_detour_steps, hop_count);
        // auto detour_end = std::chrono::high_resolution_clock::now();
        // auto detour_duration = std::chrono::duration_cast<std::chrono::microseconds>(detour_end - detour_start);
        // HNSW_LOG_INFO("[searchBaseLayerRouting] [E1] Entry Detour Steps = ", result.entry_detour_steps, ", Ratio = ", result.entry_detour_ratio, " | Time: ", detour_duration.count() / 1000.0, " ms");
        
        // E2. Calculate novelty and duplication rates
        // auto novelty_start = std::chrono::high_resolution_clock::now();
        result.novelty_rate_ratio = calculateNoveltyRateRatio(is_novel_discovery);
        result.duplication_rate = calculateDuplicationRate(is_novel_discovery);
        // auto novelty_end = std::chrono::high_resolution_clock::now();
        // auto novelty_duration = std::chrono::duration_cast<std::chrono::microseconds>(novelty_end - novelty_start);
        // HNSW_LOG_INFO("[searchBaseLayerRouting] [E2] Novelty Rate = ", result.novelty_rate_ratio, ", Duplication = ", result.duplication_rate, " | Time: ", novelty_duration.count() / 1000.0, " ms");
        
        // E3. Calculate frontier overlap
        // auto overlap_start = std::chrono::high_resolution_clock::now();
        result.frontier_overlap_ratio = calculateFrontierOverlapRatio(is_in_queue, is_novel_discovery);
        // auto overlap_end = std::chrono::high_resolution_clock::now();
        // auto overlap_duration = std::chrono::duration_cast<std::chrono::microseconds>(overlap_end - overlap_start);
        // HNSW_LOG_INFO("[searchBaseLayerRouting] [E3] Frontier Overlap Ratio = ", result.frontier_overlap_ratio, " | Time: ", overlap_duration.count() / 1000.0, " ms");
        
        // auto cat_e_end = std::chrono::high_resolution_clock::now();
        // auto cat_e_duration = std::chrono::duration_cast<std::chrono::microseconds>(cat_e_end - cat_e_start);
        // HNSW_LOG_INFO("[searchBaseLayerRouting] [TIMING] Category E Total: ", cat_e_duration.count() / 1000.0, " ms");
        
        // // === Final Scoring ===
        // auto scoring_start = std::chrono::high_resolution_clock::now();
        
        // Calculate overall routing difficulty score
        result.routing_difficulty_score = calculateRoutingDifficultyScoreComplete(
            result.cohesion, result.contrast, result.cone_concentration,
            result.mutual_nn_rate, result.induced_subgraph_lcc, result.anchor_consistency,
            result.replacement_rate_decay, result.improvement_slope, result.improvement_curvature,
            result.topk_half_life, result.radius_hit_rate, result.lid_topm_vs_frontier,
            result.entry_detour_steps, result.entry_detour_ratio, result.novelty_rate_ratio,
            result.duplication_rate, result.frontier_overlap_ratio);
        
        // Determine if converged to kNN region
        result.is_converged_to_knn_region = assessConvergenceToKNNRegionComplete(
            result.cohesion, result.contrast, result.cone_concentration, 
            result.mutual_nn_rate, result.induced_subgraph_lcc, result.anchor_consistency,
            result.replacement_rate_decay, result.improvement_slope, result.improvement_curvature,
            result.topk_half_life, result.radius_hit_rate, result.lid_topm_vs_frontier,
            result.entry_detour_steps, result.entry_detour_ratio, result.novelty_rate_ratio,
            result.duplication_rate, result.frontier_overlap_ratio, result.routing_difficulty_score);
        
        // auto scoring_end = std::chrono::high_resolution_clock::now();
        // auto scoring_duration = std::chrono::duration_cast<std::chrono::microseconds>(scoring_end - scoring_start);
        // HNSW_LOG_INFO("[searchBaseLayerRouting] [SCORING] Routing Difficulty = ", result.routing_difficulty_score, " | Converged = ", (result.is_converged_to_knn_region ? "YES" : "NO"), " | Time: ", scoring_duration.count() / 1000.0, " ms");
        
        // Calculate other metrics
        result.novelty_rate = result.novelty_rate / result.visited_node_num;
        result.average_degree = result.average_degree / result.visited_node_num;
        
        // === Overall Feature Computation Summary ===
        // auto features_total_duration = cat_a_duration + cat_b_duration + cat_c_duration + cat_d_duration + cat_e_duration + scoring_duration;
        // HNSW_LOG_INFO("[searchBaseLayerRouting] ========== Feature Computation Summary ==========");
        // HNSW_LOG_INFO("[searchBaseLayerRouting] [TIMING] Total features time: ", features_total_duration.count() / 1000.0, " ms");
        // HNSW_LOG_INFO("[searchBaseLayerRouting] [TIMING]   Category A (Clustering/Separation): ", cat_a_duration.count() / 1000.0, " ms (", 
        //              std::fixed, std::setprecision(1), 100.0 * cat_a_duration.count() / features_total_duration.count(), "%)");
        // HNSW_LOG_INFO("[searchBaseLayerRouting] [TIMING]   Category B (Structural): ", cat_b_duration.count() / 1000.0, " ms (", 
        //              std::fixed, std::setprecision(1), 100.0 * cat_b_duration.count() / features_total_duration.count(), "%)");
        // HNSW_LOG_INFO("[searchBaseLayerRouting] [TIMING]   Category C (Trajectory): ", cat_c_duration.count() / 1000.0, " ms (", 
        //              std::fixed, std::setprecision(1), 100.0 * cat_c_duration.count() / features_total_duration.count(), "%)");
        // HNSW_LOG_INFO("[searchBaseLayerRouting] [TIMING]   Category D (Radius): ", cat_d_duration.count() / 1000.0, " ms (", 
        //              std::fixed, std::setprecision(1), 100.0 * cat_d_duration.count() / features_total_duration.count(), "%)");
        // HNSW_LOG_INFO("[searchBaseLayerRouting] [TIMING]   Category E (Detour): ", cat_e_duration.count() / 1000.0, " ms (", 
        //              std::fixed, std::setprecision(1), 100.0 * cat_e_duration.count() / features_total_duration.count(), "%)");
        // HNSW_LOG_INFO("[searchBaseLayerRouting] [TIMING]   Scoring: ", scoring_duration.count() / 1000.0, " ms");
        
        // Detailed breakdown of most expensive operations
        // HNSW_LOG_INFO("[searchBaseLayerRouting] ========== Expensive Operations ==========");
        // HNSW_LOG_INFO("[searchBaseLayerRouting]   1. Cohesion (A1): ", cohesion_duration.count() / 1000.0, " ms - O(m²×d)");
        // HNSW_LOG_INFO("[searchBaseLayerRouting]   2. Mutual NN (B1): ", mutual_nn_duration.count() / 1000.0, " ms - O(m²×deg)");
        // HNSW_LOG_INFO("[searchBaseLayerRouting]   3. Induced LCC (B2): ", lcc_duration.count() / 1000.0, " ms - O(m²×deg + m³)");
    }
    
    // End overall timing
    // auto overall_end = std::chrono::high_resolution_clock::now();
    // auto overall_duration = std::chrono::duration_cast<std::chrono::microseconds>(overall_end - overall_start);
    
    // HNSW_LOG_INFO("[searchBaseLayerRouting] ========== Overall Timing Summary ==========");
    // HNSW_LOG_INFO("[searchBaseLayerRouting] [TIMING] Total time: ", overall_duration.count() / 1000.0, " ms");
    // HNSW_LOG_INFO("[searchBaseLayerRouting] [TIMING]   Early probe loop: ", probe_loop_duration.count() / 1000.0, " ms (", 
    //              std::fixed, std::setprecision(1), 100.0 * probe_loop_duration.count() / overall_duration.count(), "%)");
    
    // if (!expanded_nodes.empty() && !top_distances.empty()) {
    //     auto features_total_duration = cat_a_duration + cat_b_duration + cat_c_duration + cat_d_duration + cat_e_duration + scoring_duration;
    //     HNSW_LOG_INFO("[searchBaseLayerRouting] [TIMING]   Feature computation: ", features_total_duration.count() / 1000.0, " ms (", 
    //                  std::fixed, std::setprecision(1), 100.0 * features_total_duration.count() / overall_duration.count(), "%)");
    //     HNSW_LOG_INFO("[searchBaseLayerRouting] [TIMING]   NN statistics: ", nn_stats_duration.count() / 1000.0, " ms (", 
    //                  std::fixed, std::setprecision(1), 100.0 * nn_stats_duration.count() / overall_duration.count(), "%)");
    // }
    
    visited_list_pool_->releaseVisitedList(vl);
    visited_list_pool_->releaseVisitedList(cl);
    
    return result;
}

/**
 * @brief Calculate cohesion measure: mean pairwise distance / median query distance
 * 
 * @param visited_nodes Vector of (node_id, distance) pairs
 * @param top_distances Vector of top-k distances
 * @return Cohesion value (lower = more clustered)
 */
float calculateCohesion(const std::vector<std::pair<tableint, dist_t>>& visited_nodes, 
                       const std::vector<dist_t>& top_distances) const {
    if (visited_nodes.size() < 2 || top_distances.empty()) {
        return 0.0f;
    }
    
    // Calculate median query distance
    std::vector<dist_t> sorted_distances = top_distances;
    std::sort(sorted_distances.begin(), sorted_distances.end());
    dist_t median_query_dist = sorted_distances[sorted_distances.size() / 2];
    
    if (median_query_dist <= 0) {
        return 0.0f;
    }
    
    // Calculate mean pairwise distance between visited nodes
    float total_pairwise_dist = 0.0f;
    int pair_count = 0;
    
    for (size_t i = 0; i < visited_nodes.size(); ++i) {
        for (size_t j = i + 1; j < visited_nodes.size(); ++j) {
            dist_t pairwise_dist = fstdistfunc_(
                getDataByInternalId(visited_nodes[i].first), 
                getDataByInternalId(visited_nodes[j].first), 
                dist_func_param_);
            total_pairwise_dist += pairwise_dist;
            pair_count++;
        }
    }
    
    if (pair_count == 0) {
        return 0.0f;
    }
    
    float mean_pairwise_dist = total_pairwise_dist / pair_count;
    return mean_pairwise_dist / median_query_dist;
}

/**
 * @brief Calculate cohesion measure using only top-k nodes (optimized version)
 * 
 * This function computes cohesion by only considering the k nearest nodes,
 * significantly reducing computational complexity from O(m²×d) to O(k²×d).
 * 
 * @param visited_nodes Vector of (node_id, distance) pairs
 * @param top_distances Vector of top-k distances
 * @param k Number of top nodes to consider for cohesion calculation
 * @return Cohesion value (lower = more clustered)
 */
float calculateCohesion2(const std::vector<std::pair<tableint, dist_t>>& visited_nodes, 
                        const std::vector<dist_t>& top_distances,
                        size_t k) const {
    if (visited_nodes.size() < 2 || top_distances.empty() || k < 2) {
        return 0.0f;
    }
    
    // Calculate median query distance from top-k distances
    std::vector<dist_t> sorted_distances = top_distances;
    std::sort(sorted_distances.begin(), sorted_distances.end());
    dist_t median_query_dist = sorted_distances[sorted_distances.size() / 2];
    
    if (median_query_dist <= 0) {
        return 0.0f;
    }
    
    // Sort visited nodes by distance to get top-k
    std::vector<std::pair<tableint, dist_t>> sorted_nodes = visited_nodes;
    std::sort(sorted_nodes.begin(), sorted_nodes.end(), 
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    // Only use top-k nodes for cohesion calculation
    size_t top_k_count = std::min(k, sorted_nodes.size());
    
    if (top_k_count < 2) {
        return 0.0f;
    }
    
    // Calculate mean pairwise distance between top-k nodes only
    float total_pairwise_dist = 0.0f;
    int pair_count = 0;
    
    for (size_t i = 0; i < top_k_count; ++i) {
        for (size_t j = i + 1; j < top_k_count; ++j) {
            dist_t pairwise_dist = fstdistfunc_(
                getDataByInternalId(sorted_nodes[i].first), 
                getDataByInternalId(sorted_nodes[j].first), 
                dist_func_param_);
            total_pairwise_dist += pairwise_dist;
            pair_count++;
        }
    }
    
    if (pair_count == 0) {
        return 0.0f;
    }
    
    float mean_pairwise_dist = total_pairwise_dist / pair_count;
    return mean_pairwise_dist / median_query_dist;
}

/**
 * @brief Calculate contrast measure: separation between top-m and background
 * 
 * @param visited_nodes Vector of (node_id, distance) pairs
 * @param top_distances Vector of top-k distances
 * @param m Number of top nodes to consider
 * @param k Target k for kNN
 * @return Contrast value (higher = better separation)
 */
float calculateContrast(const std::vector<std::pair<tableint, dist_t>>& visited_nodes,
                       const std::vector<dist_t>& top_distances,
                       size_t m, size_t k) const {
    if (visited_nodes.size() < m || top_distances.empty()) {
        return 0.0f;
    }
    
    // Sort visited nodes by distance to get top-m
    std::vector<std::pair<tableint, dist_t>> sorted_nodes = visited_nodes;
    std::sort(sorted_nodes.begin(), sorted_nodes.end(), 
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    // Calculate mean distance to top-m nodes
    float top_m_mean_dist = 0.0f;
    size_t top_m_count = std::min(m, sorted_nodes.size());
    for (size_t i = 0; i < top_m_count; ++i) {
        top_m_mean_dist += sorted_nodes[i].second;
    }
    top_m_mean_dist /= top_m_count;
    
    // Sample background nodes (non-top-k from frontier)
    float background_mean_dist = 0.0f;
    int background_count = 0;
    
    // Use nodes beyond top-k as background
    for (size_t i = k; i < sorted_nodes.size() && background_count < m; ++i) {
        background_mean_dist += sorted_nodes[i].second;
        background_count++;
    }
    
    if (background_count == 0) {
        return 0.0f;
    }
    
    background_mean_dist /= background_count;
    
    // Calculate contrast: (background_mean - top_m_mean) / max(background_mean, top_m_mean)
    float numerator = background_mean_dist - top_m_mean_dist;
    float denominator = std::max(background_mean_dist, top_m_mean_dist);
    
    return numerator / denominator;
}

/**
 * @brief Calculate cone concentration (direction cone density)
 * 
 * @param query_data Pointer to query vector
 * @param visited_nodes Vector of (node_id, distance) pairs
 * @param top_distances Vector of top-k distances
 * @param k Target k for kNN
 * @return Cone concentration value (higher = more directional)
 */
float calculateConeConcentration(const void* query_data,
                                const std::vector<std::pair<tableint, dist_t>>& visited_nodes,
                                const std::vector<dist_t>& top_distances,
                                size_t k) const {
    if (visited_nodes.size() < 2 || top_distances.empty()) {
        return 0.0f;
    }
    
    // Sort visited nodes by distance to get top-k
    std::vector<std::pair<tableint, dist_t>> sorted_nodes = visited_nodes;
    std::sort(sorted_nodes.begin(), sorted_nodes.end(), 
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    size_t top_k_count = std::min(k, sorted_nodes.size());
    
    // Calculate direction vectors from query to top-k nodes
    std::vector<std::vector<float>> direction_vectors;
    direction_vectors.reserve(top_k_count);
    
    // Get data dimension from dist_func_param_
    size_t dim = *((size_t *) dist_func_param_);
    
            for (size_t i = 0; i < top_k_count; ++i) {
            tableint node_id = sorted_nodes[i].first;
            char* node_data = getDataByInternalId(node_id);
            
            // Calculate direction vector: (x_i - q) / ||x_i - q||
            std::vector<float> direction(dim);
            float norm = 0.0f;
            
            for (size_t d = 0; d < dim; ++d) {
                float query_val = *((float*)query_data + d);
                float node_val = *((float*)node_data + d);
                direction[d] = node_val - query_val;
                norm += direction[d] * direction[d];
            }
        
        // Normalize the direction vector
        if (norm > 0) {
            norm = std::sqrt(norm);
            for (size_t i = 0; i < dim; ++i) {
                direction[i] /= norm;
            }
            direction_vectors.push_back(direction);
        }
    }
    
    if (direction_vectors.size() < 2) {
        return 0.0f;
    }
    
    // Calculate mean pairwise cosine similarity between direction vectors
    float total_similarity = 0.0f;
    int pair_count = 0;
    
    for (size_t i = 0; i < direction_vectors.size(); ++i) {
        for (size_t j = i + 1; j < direction_vectors.size(); ++j) {
            float dot_product = 0.0f;
            for (size_t d = 0; d < dim; ++d) {
                dot_product += direction_vectors[i][d] * direction_vectors[j][d];
            }
            total_similarity += dot_product;
            pair_count++;
        }
    }
    
    if (pair_count == 0) {
        return 0.0f;
    }
    
    return total_similarity / pair_count;
}

/**
 * @brief Calculate overall routing difficulty score
 * 
 * @param cohesion Cohesion measure
 * @param contrast Contrast measure
 * @param cone_concentration Cone concentration measure
 * @return Routing difficulty score (0-1, higher = more difficult)
 */
float calculateRoutingDifficultyScore(float cohesion, float contrast, float cone_concentration) const {
    // Normalize and combine the three measures
    // Lower cohesion (more clustered) = easier routing
    // Higher contrast (better separation) = easier routing  
    // Higher cone concentration (more directional) = easier routing
    
    // Normalize cohesion: 0 = very clustered, 1 = very scattered
    float normalized_cohesion = std::min(1.0f, cohesion / 10.0f); // Assume cohesion > 10 is very scattered
    
    // Normalize contrast: 0 = no separation, 1 = strong separation
    float normalized_contrast = std::min(1.0f, std::max(0.0f, contrast));
    
    // Normalize cone concentration: 0 = no direction, 1 = strong direction
    float normalized_cone = std::min(1.0f, std::max(0.0f, cone_concentration));
    
    // Calculate difficulty score: higher values = more difficult routing
    float difficulty_score = normalized_cohesion * 0.4f + 
                           (1.0f - normalized_contrast) * 0.3f + 
                           (1.0f - normalized_cone) * 0.3f;
    
    return std::min(1.0f, std::max(0.0f, difficulty_score));
}

/**
 * @brief Assess whether the search has converged to a kNN-like region
 * 
 * @param cohesion Cohesion measure
 * @param contrast Contrast measure
 * @param cone_concentration Cone concentration measure
 * @param routing_difficulty Routing difficulty score
 * @return True if converged to kNN region
 */
bool assessConvergenceToKNNRegion(float cohesion, float contrast, float cone_concentration, 
                                 float routing_difficulty) const {
    // Define thresholds for convergence assessment
    const float COHESION_THRESHOLD = 5.0f;      // Top-m nodes should be relatively close
    const float CONTRAST_THRESHOLD = 0.3f;      // Should have some separation from background
    const float CONE_THRESHOLD = 0.5f;          // Should have some directional consistency
    const float DIFFICULTY_THRESHOLD = 0.6f;    // Routing should not be too difficult
    
    bool good_cohesion = cohesion < COHESION_THRESHOLD;
    bool good_contrast = contrast > CONTRAST_THRESHOLD;
    bool good_cone = cone_concentration > CONE_THRESHOLD;
    bool manageable_difficulty = routing_difficulty < DIFFICULTY_THRESHOLD;
    
    // Require at least 3 out of 4 criteria to be met
    int criteria_met = (good_cohesion ? 1 : 0) + 
                      (good_contrast ? 1 : 0) + 
                      (good_cone ? 1 : 0) + 
                      (manageable_difficulty ? 1 : 0);
    
    return criteria_met >= 3;
}

/**
 * @brief Calculate mutual-NN rate: fraction of top-m pairs that are neighbors
 * 
 * @param visited_nodes Vector of (node_id, distance) pairs
 * @param top_distances Vector of top-k distances
 * @param m Number of top nodes to consider
 * @return Mutual-NN rate (0-1, higher = more connected)
 */
float calculateMutualNNRate(const std::vector<std::pair<tableint, dist_t>>& visited_nodes,
                           const std::vector<dist_t>& top_distances,
                           size_t m) const {
    if (visited_nodes.size() < 2 || m < 2) {
        return 0.0f;
    }
    
    // Sort visited nodes by distance to get top-m
    std::vector<std::pair<tableint, dist_t>> sorted_nodes = visited_nodes;
    std::sort(sorted_nodes.begin(), sorted_nodes.end(), 
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    size_t top_m_count = std::min(m, sorted_nodes.size());
    if (top_m_count < 2) {
        return 0.0f;
    }
    
    // Count mutual neighbor pairs
    int mutual_neighbor_pairs = 0;
    int total_pairs = 0;
    
    for (size_t i = 0; i < top_m_count; ++i) {
        for (size_t j = i + 1; j < top_m_count; ++j) {
            total_pairs++;
            
            tableint node_i = sorted_nodes[i].first;
            tableint node_j = sorted_nodes[j].first;
            
            // Check if node_i is in node_j's neighbor list
            bool i_in_j_neighbors = false;
            bool j_in_i_neighbors = false;
            
            // Check node_i's neighbors
            int *data_i = (int *) get_linklist0(node_i);
            size_t size_i = getListCount((linklistsizeint*)data_i);
            for (size_t k = 1; k <= size_i; ++k) {
                if (*(data_i + k) == node_j) {
                    j_in_i_neighbors = true;
                    break;
                }
            }
            
            // Check node_j's neighbors
            int *data_j = (int *) get_linklist0(node_j);
            size_t size_j = getListCount((linklistsizeint*)data_j);
            for (size_t k = 1; k <= size_j; ++k) {
                if (*(data_j + k) == node_i) {
                    i_in_j_neighbors = true;
                    break;
                }
            }
            
            // If either direction exists, count as mutual neighbor
            if (i_in_j_neighbors || j_in_i_neighbors) {
                mutual_neighbor_pairs++;
            }
        }
    }
    
    if (total_pairs == 0) {
        return 0.0f;
    }
    
    return static_cast<float>(mutual_neighbor_pairs) / total_pairs;
}

/**
 * @brief Calculate induced subgraph LCC: clustering coefficient on top-m induced subgraph
 * 
 * @param visited_nodes Vector of (node_id, distance) pairs
 * @param top_distances Vector of top-k distances
 * @param m Number of top nodes to consider
 * @return Induced subgraph LCC (0-1, higher = more clustered)
 */
float calculateInducedSubgraphLCC(const std::vector<std::pair<tableint, dist_t>>& visited_nodes,
                                 const std::vector<dist_t>& top_distances,
                                 size_t m) const {
    if (visited_nodes.size() < 3 || m < 3) {
        return 0.0f;
    }
    
    // Sort visited nodes by distance to get top-m
    std::vector<std::pair<tableint, dist_t>> sorted_nodes = visited_nodes;
    std::sort(sorted_nodes.begin(), sorted_nodes.end(), 
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    size_t top_m_count = std::min(m, sorted_nodes.size());
    if (top_m_count < 3) {
        return 0.0f;
    }
    
    // Build adjacency matrix for top-m induced subgraph
    std::vector<std::vector<bool>> adjacency(top_m_count, std::vector<bool>(top_m_count, false));
    
    for (size_t i = 0; i < top_m_count; ++i) {
        tableint node_i = sorted_nodes[i].first;
        int *data_i = (int *) get_linklist0(node_i);
        size_t size_i = getListCount((linklistsizeint*)data_i);
        
        for (size_t k = 1; k <= size_i; ++k) {
            tableint neighbor = *(data_i + k);
            
            // Find if this neighbor is in top-m
            for (size_t j = 0; j < top_m_count; ++j) {
                if (sorted_nodes[j].first == neighbor) {
                    adjacency[i][j] = true;
                    adjacency[j][i] = true; // Undirected graph
                    break;
                }
            }
        }
    }
    
    // Calculate LCC for each node in the induced subgraph
    float total_lcc = 0.0f;
    int valid_nodes = 0;
    
    for (size_t i = 0; i < top_m_count; ++i) {
        // Count neighbors of node i in the induced subgraph
        int neighbor_count = 0;
        for (size_t j = 0; j < top_m_count; ++j) {
            if (i != j && adjacency[i][j]) {
                neighbor_count++;
            }
        }
        
        if (neighbor_count < 2) {
            continue; // Need at least 2 neighbors to form triangles
        }
        
        // Count triangles involving node i
        int triangles = 0;
        for (size_t j = 0; j < top_m_count; ++j) {
            if (i != j && adjacency[i][j]) {
                for (size_t k = j + 1; k < top_m_count; ++k) {
                    if (i != k && adjacency[i][k] && adjacency[j][k]) {
                        triangles++;
                    }
                }
            }
        }
        
        // Calculate LCC for this node: triangles / possible_triangles
        int possible_triangles = neighbor_count * (neighbor_count - 1) / 2;
        if (possible_triangles > 0) {
            total_lcc += static_cast<float>(triangles) / possible_triangles;
            valid_nodes++;
        }
    }
    
    if (valid_nodes == 0) {
        return 0.0f;
    }
    
    return total_lcc / valid_nodes;
}

/**
 * @brief Calculate anchor consistency: mapping consistency to anchor points
 * 
 * @param visited_nodes Vector of (node_id, distance) pairs
 * @param top_distances Vector of top-k distances
 * @param m Number of top nodes to consider
 * @return Anchor consistency (0-1, higher = more consistent)
 */
float calculateAnchorConsistency(const std::vector<std::pair<tableint, dist_t>>& visited_nodes,
                               const std::vector<dist_t>& top_distances,
                               size_t m) const {
    if (visited_nodes.size() < 2 || m < 2) {
        return 0.0f;
    }
    
    // Sort visited nodes by distance to get top-m
    std::vector<std::pair<tableint, dist_t>> sorted_nodes = visited_nodes;
    std::sort(sorted_nodes.begin(), sorted_nodes.end(), 
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    size_t top_m_count = std::min(m, sorted_nodes.size());
    
    // For HNSW, we can use the level assignment as a form of "anchor" or "representative"
    // Higher level nodes serve as representatives for their regions
    std::unordered_map<int, int> level_votes;
    
    for (size_t i = 0; i < top_m_count; ++i) {
        tableint node_id = sorted_nodes[i].first;
        int level = element_levels_[node_id];
        level_votes[level]++;
    }
    
    if (level_votes.empty()) {
        return 0.0f;
    }
    
    // Find the level with maximum votes
    int max_votes = 0;
    for (const auto& vote : level_votes) {
        max_votes = std::max(max_votes, vote.second);
    }
    
    return static_cast<float>(max_votes) / top_m_count;
}

/**
 * @brief Calculate routing difficulty score including structural consistency features
 * 
 * @param cohesion Cohesion measure
 * @param contrast Contrast measure
 * @param cone_concentration Cone concentration measure
 * @param mutual_nn_rate Mutual-NN rate
 * @param induced_subgraph_lcc Induced subgraph LCC
 * @param anchor_consistency Anchor consistency
 * @return Updated routing difficulty score (0-1, higher = more difficult)
 */
float calculateRoutingDifficultyScoreWithStructure(
    float cohesion, float contrast, float cone_concentration,
    float mutual_nn_rate, float induced_subgraph_lcc, float anchor_consistency) const {
    
    // Original difficulty components (40% weight)
    float original_difficulty = calculateRoutingDifficultyScore(cohesion, contrast, cone_concentration);
    
    // Structural consistency components (60% weight)
    // Higher structural consistency = easier routing
    
    // Normalize structural features
    float normalized_mutual_nn = std::min(1.0f, std::max(0.0f, mutual_nn_rate));
    float normalized_lcc = std::min(1.0f, std::max(0.0f, induced_subgraph_lcc));
    float normalized_anchor = std::min(1.0f, std::max(0.0f, anchor_consistency));
    
    // Calculate structural difficulty: lower values = easier routing
    float structural_difficulty = (1.0f - normalized_mutual_nn) * 0.3f + 
                                 (1.0f - normalized_lcc) * 0.2f + 
                                 (1.0f - normalized_anchor) * 0.1f;
    
    // Combine original and structural difficulty
    float combined_difficulty = original_difficulty * 0.4f + structural_difficulty * 0.6f;
    
    return std::min(1.0f, std::max(0.0f, combined_difficulty));
}

/**
 * @brief Assess convergence to kNN region including structural consistency
 * 
 * @param cohesion Cohesion measure
 * @param contrast Contrast measure
 * @param cone_concentration Cone concentration measure
 * @param mutual_nn_rate Mutual-NN rate
 * @param induced_subgraph_lcc Induced subgraph LCC
 * @param anchor_consistency Anchor consistency
 * @param routing_difficulty Routing difficulty score
 * @return True if converged to kNN region
 */
bool assessConvergenceToKNNRegionWithStructure(
    float cohesion, float contrast, float cone_concentration,
    float mutual_nn_rate, float induced_subgraph_lcc, float anchor_consistency,
    float routing_difficulty) const {
    
    // Original convergence criteria
    const float COHESION_THRESHOLD = 5.0f;
    const float CONTRAST_THRESHOLD = 0.3f;
    const float CONE_THRESHOLD = 0.5f;
    const float DIFFICULTY_THRESHOLD = 0.6f;
    
    // New structural consistency thresholds
    const float MUTUAL_NN_THRESHOLD = 0.3f;        // At least 30% of top-m pairs should be neighbors
    const float INDUCED_LCC_THRESHOLD = 0.2f;      // Induced subgraph should have some clustering
    const float ANCHOR_CONSISTENCY_THRESHOLD = 0.5f; // At least 50% should map to same anchor/level
    
    bool good_cohesion = cohesion < COHESION_THRESHOLD;
    bool good_contrast = contrast > CONTRAST_THRESHOLD;
    bool good_cone = cone_concentration > CONE_THRESHOLD;
    bool manageable_difficulty = routing_difficulty < DIFFICULTY_THRESHOLD;
    bool good_mutual_nn = mutual_nn_rate > MUTUAL_NN_THRESHOLD;
    bool good_induced_lcc = induced_subgraph_lcc > INDUCED_LCC_THRESHOLD;
    bool good_anchor_consistency = anchor_consistency > ANCHOR_CONSISTENCY_THRESHOLD;
    
    // Require at least 5 out of 7 criteria to be met for convergence
    int criteria_met = (good_cohesion ? 1 : 0) + 
                      (good_contrast ? 1 : 0) + 
                      (good_cone ? 1 : 0) + 
                      (manageable_difficulty ? 1 : 0) +
                      (good_mutual_nn ? 1 : 0) +
                      (good_induced_lcc ? 1 : 0) +
                      (good_anchor_consistency ? 1 : 0);
    
    return criteria_met >= 5;
}

/**
 * @brief Calculate replacement rate decay slope over sliding window
 * 
 * @param replacement_events Vector of boolean replacement events over time
 * @param window_size Size of sliding window for decay calculation
 * @return Replacement rate decay slope (negative = stabilizing)
 */
float calculateReplacementRateDecay(const std::vector<bool>& replacement_events, size_t window_size) const {
    if (replacement_events.size() < window_size * 2) {
        return 0.0f;
    }
    
    // Calculate replacement rates over sliding windows
    std::vector<float> replacement_rates;
    for (size_t i = 0; i <= replacement_events.size() - window_size; ++i) {
        int replacements_in_window = 0;
        for (size_t j = 0; j < window_size; ++j) {
            if (replacement_events[i + j]) {
                replacements_in_window++;
            }
        }
        float rate = static_cast<float>(replacements_in_window) / window_size;
        replacement_rates.push_back(rate);
    }
    
    if (replacement_rates.size() < 2) {
        return 0.0f;
    }
    
    // Calculate linear regression slope
    float sum_x = 0.0f, sum_y = 0.0f, sum_xy = 0.0f, sum_x2 = 0.0f;
    int n = replacement_rates.size();
    
    for (int i = 0; i < n; ++i) {
        float x = static_cast<float>(i);
        float y = replacement_rates[i];
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_x2 += x * x;
    }
    
    float denominator = n * sum_x2 - sum_x * sum_x;
    if (std::abs(denominator) < 1e-6f) {
        return 0.0f;
    }
    
    float slope = (n * sum_xy - sum_x * sum_y) / denominator;
    return slope;
}

/**
 * @brief Calculate improvement trend slope and curvature
 * 
 * @param best_distances Vector of best distances over time
 * @return Pair of (slope, curvature)
 */
std::pair<float, float> calculateImprovementTrend(const std::vector<dist_t>& best_distances) const {
    if (best_distances.size() < 3) {
        return {0.0f, 0.0f};
    }
    
    // Calculate improvement deltas: Δ_t = best_{t-1} - best_t
    std::vector<float> deltas;
    for (size_t i = 1; i < best_distances.size(); ++i) {
        float delta = best_distances[i-1] - best_distances[i];
        if (delta > 0) { // Only positive improvements
            deltas.push_back(delta);
        }
    }
    
    if (deltas.empty()) {
        return {0.0f, 0.0f};
    }
    
    // Calculate median slope
    std::sort(deltas.begin(), deltas.end());
    float median_slope = deltas[deltas.size() / 2];
    
    // Calculate curvature using log trend fitting residuals
    float curvature = 0.0f;
    if (deltas.size() >= 3) {
        // Fit log trend: log(y) = a + b*log(x)
        std::vector<float> log_x, log_y;
        for (size_t i = 0; i < deltas.size(); ++i) {
            if (deltas[i] > 0) {
                log_x.push_back(std::log(static_cast<float>(i + 1)));
                log_y.push_back(std::log(deltas[i]));
            }
        }
        
        if (log_x.size() >= 3) {
            // Simple linear regression on log-log data
            float sum_x = 0.0f, sum_y = 0.0f, sum_xy = 0.0f, sum_x2 = 0.0f;
            int n = log_x.size();
            
            for (int i = 0; i < n; ++i) {
                sum_x += log_x[i];
                sum_y += log_y[i];
                sum_xy += log_x[i] * log_y[i];
                sum_x2 += log_x[i] * log_x[i];
            }
            
            float denominator = n * sum_x2 - sum_x * sum_x;
            if (std::abs(denominator) > 1e-6f) {
                float b = (n * sum_xy - sum_x * sum_y) / denominator;
                float a = (sum_y - b * sum_x) / n;
                
                // Calculate residuals and their mean
                float total_residuals = 0.0f;
                int valid_residuals = 0;
                
                for (int i = 0; i < n; ++i) {
                    float predicted = a + b * log_x[i];
                    float residual = std::abs(log_y[i] - predicted);
                    if (residual > 0) {
                        total_residuals += residual;
                        valid_residuals++;
                    }
                }
                
                if (valid_residuals > 0) {
                    curvature = total_residuals / valid_residuals;
                }
            }
        }
    }
    
    return {median_slope, curvature};
}

/**
 * @brief Calculate top-k half-life: median survival steps of new nodes
 * 
 * @param node_lifetimes Vector of (node_id, entry_step) pairs
 * @param current_step Current search step
 * @return Median survival steps (longer = more stable)
 */
float calculateTopKHalfLife(const std::vector<std::pair<tableint, int>>& node_lifetimes, int current_step) const {
    if (node_lifetimes.empty()) {
        return 0.0f;
    }
    
    // Calculate survival steps for each node
    std::vector<int> survival_steps;
    for (const auto& [node_id, entry_step] : node_lifetimes) {
        int survival = current_step - entry_step;
        if (survival > 0) {
            survival_steps.push_back(survival);
        }
    }
    
    if (survival_steps.empty()) {
        return 0.0f;
    }
    
    // Calculate median survival
    std::sort(survival_steps.begin(), survival_steps.end());
    float median_survival = static_cast<float>(survival_steps[survival_steps.size() / 2]);
    
    return median_survival;
}

/**
 * @brief Calculate complete routing difficulty score including all features
 * 
 * @param cohesion Cohesion measure
 * @param contrast Contrast measure
 * @param cone_concentration Cone concentration measure
 * @param mutual_nn_rate Mutual-NN rate
 * @param induced_subgraph_lcc Induced subgraph LCC
 * @param anchor_consistency Anchor consistency
 * @param replacement_rate_decay Replacement rate decay slope
 * @param improvement_slope Improvement slope
 * @param improvement_curvature Improvement curvature
 * @param topk_half_life Top-k half-life
 * @param radius_hit_rate Radius hit rate
 * @param lid_topm_vs_frontier LID difference between top-m and background
 * @param entry_detour_steps Entry detour steps
 * @param entry_detour_ratio Entry detour ratio
 * @param novelty_rate_ratio Novelty rate ratio
 * @param duplication_rate Duplication rate
 * @param frontier_overlap_ratio Frontier overlap ratio
 * @return Complete routing difficulty score (0-1, higher = more difficult)
 */
float calculateRoutingDifficultyScoreComplete(
    float cohesion, float contrast, float cone_concentration,
    float mutual_nn_rate, float induced_subgraph_lcc, float anchor_consistency,
    float replacement_rate_decay, float improvement_slope, float improvement_curvature,
    float topk_half_life, float radius_hit_rate, float lid_topm_vs_frontier,
    float entry_detour_steps, float entry_detour_ratio, float novelty_rate_ratio,
    float duplication_rate, float frontier_overlap_ratio) const {
    
    // Original features (25% weight)
    float original_difficulty = calculateRoutingDifficultyScore(cohesion, contrast, cone_concentration);
    
    // Structural consistency features (35% weight)
    float structural_difficulty = calculateRoutingDifficultyScoreWithStructure(
        cohesion, contrast, cone_concentration,
        mutual_nn_rate, induced_subgraph_lcc, anchor_consistency) - original_difficulty;
    
    // Trajectory convergence features (35% weight)
    // Normalize trajectory features
    float normalized_decay = std::min(1.0f, std::max(0.0f, -replacement_rate_decay)); // Negative decay = good
    float normalized_slope = std::min(1.0f, std::max(0.0f, 1.0f - improvement_slope / 0.1f)); // Higher slope = good
    float normalized_curvature = std::min(1.0f, std::max(0.0f, improvement_curvature / 2.0f)); // Lower curvature = good
    float normalized_half_life = std::min(1.0f, std::max(0.0f, 1.0f - topk_half_life / 10.0f)); // Longer half-life = good
    
    float trajectory_difficulty = normalized_decay * 0.12f + 
                                 normalized_slope * 0.08f + 
                                 normalized_curvature * 0.08f + 
                                 normalized_half_life * 0.07f;
    
    // Target radius hit features (5% weight)
    float normalized_radius_hit = std::min(1.0f, std::max(0.0f, 1.0f - radius_hit_rate)); // Higher hit rate = easier routing
    float normalized_lid_diff = std::min(1.0f, std::max(0.0f, (lid_topm_vs_frontier + 2.0f) / 4.0f)); // Negative diff = easier routing
    
    float radius_difficulty = normalized_radius_hit * 0.03f + normalized_lid_diff * 0.02f;
    
    // Early detour/redundancy features (5% weight)
    // Normalize early detour features
    float normalized_detour_steps = std::min(1.0f, std::max(0.0f, entry_detour_steps / 20.0f)); // More detour steps = harder routing
    float normalized_detour_ratio = std::min(1.0f, std::max(0.0f, entry_detour_ratio)); // Higher ratio = harder routing
    float normalized_novelty = std::min(1.0f, std::max(0.0f, 1.0f - novelty_rate_ratio)); // Lower novelty = harder routing
    float normalized_duplication = std::min(1.0f, std::max(0.0f, duplication_rate)); // Higher duplication = harder routing
    float normalized_overlap = std::min(1.0f, std::max(0.0f, frontier_overlap_ratio)); // Higher overlap = harder routing
    
    float detour_difficulty = normalized_detour_steps * 0.015f + 
                             normalized_detour_ratio * 0.01f + 
                             normalized_novelty * 0.01f + 
                             normalized_duplication * 0.01f + 
                             normalized_overlap * 0.005f;
    
    // Combine all difficulties (adjust weights to accommodate new features)
    float combined_difficulty = original_difficulty * 0.20f + 
                               structural_difficulty * 0.30f + 
                               trajectory_difficulty * 0.30f + 
                               radius_difficulty * 0.05f + 
                               detour_difficulty * 0.15f;
    
    return std::min(1.0f, std::max(0.0f, combined_difficulty));
}

/**
 * @brief Assess convergence to kNN region including all features
 * 
 * @param cohesion Cohesion measure
 * @param contrast Contrast measure
 * @param cone_concentration Cone concentration measure
 * @param mutual_nn_rate Mutual-NN rate
 * @param induced_subgraph_lcc Induced subgraph LCC
 * @param anchor_consistency Anchor consistency
 * @param replacement_rate_decay Replacement rate decay slope
 * @param improvement_slope Improvement slope
 * @param improvement_curvature Improvement curvature
 * @param topk_half_life Top-k half-life
 * @param radius_hit_rate Radius hit rate
 * @param lid_topm_vs_frontier LID difference between top-m and background
 * @param entry_detour_steps Entry detour steps
 * @param entry_detour_ratio Entry detour ratio
 * @param novelty_rate_ratio Novelty rate ratio
 * @param duplication_rate Duplication rate
 * @param frontier_overlap_ratio Frontier overlap ratio
 * @param routing_difficulty Routing difficulty score
 * @return True if converged to kNN region
 */
bool assessConvergenceToKNNRegionComplete(
    float cohesion, float contrast, float cone_concentration,
    float mutual_nn_rate, float induced_subgraph_lcc, float anchor_consistency,
    float replacement_rate_decay, float improvement_slope, float improvement_curvature,
    float topk_half_life, float radius_hit_rate, float lid_topm_vs_frontier,
    float entry_detour_steps, float entry_detour_ratio, float novelty_rate_ratio,
    float duplication_rate, float frontier_overlap_ratio, float routing_difficulty) const {
    
    // Original convergence criteria
    const float COHESION_THRESHOLD = 5.0f;
    const float CONTRAST_THRESHOLD = 0.3f;
    const float CONE_THRESHOLD = 0.5f;
    const float DIFFICULTY_THRESHOLD = 0.6f;
    
    // Structural consistency thresholds
    const float MUTUAL_NN_THRESHOLD = 0.3f;
    const float INDUCED_LCC_THRESHOLD = 0.2f;
    const float ANCHOR_CONSISTENCY_THRESHOLD = 0.5f;
    
    // New trajectory convergence thresholds
    const float REPLACEMENT_DECAY_THRESHOLD = -0.01f;  // Negative slope = stabilizing
    const float IMPROVEMENT_SLOPE_THRESHOLD = 0.001f;  // Positive slope = improving
    const float IMPROVEMENT_CURVATURE_THRESHOLD = 0.5f; // Low curvature = smooth
    const float TOPK_HALF_LIFE_THRESHOLD = 3.0f;       // Longer survival = stable
    
    // New target radius hit thresholds
    const float RADIUS_HIT_RATE_THRESHOLD = 0.6f;      // At least 60% should be within typical radius
    const float LID_DIFF_THRESHOLD = -0.5f;            // Top-m should have lower LID than background
    
    // New early detour/redundancy thresholds
    const float ENTRY_DETOUR_STEPS_THRESHOLD = 15.0f;  // Should reach stability within reasonable steps
    const float ENTRY_DETOUR_RATIO_THRESHOLD = 0.7f;   // Detour should not dominate total steps
    const float NOVELTY_RATE_THRESHOLD = 0.6f;         // At least 60% should be novel discoveries
    const float DUPLICATION_RATE_THRESHOLD = 0.4f;     // Duplication should not exceed 40%
    const float FRONTIER_OVERLAP_THRESHOLD = 0.3f;     // Overlap should not exceed 30%
    
    bool good_cohesion = cohesion < COHESION_THRESHOLD;
    bool good_contrast = contrast > CONTRAST_THRESHOLD;
    bool good_cone = cone_concentration > CONE_THRESHOLD;
    bool manageable_difficulty = routing_difficulty < DIFFICULTY_THRESHOLD;
    bool good_mutual_nn = mutual_nn_rate > MUTUAL_NN_THRESHOLD;
    bool good_induced_lcc = induced_subgraph_lcc > INDUCED_LCC_THRESHOLD;
    bool good_anchor_consistency = anchor_consistency > ANCHOR_CONSISTENCY_THRESHOLD;
    bool good_replacement_decay = replacement_rate_decay < REPLACEMENT_DECAY_THRESHOLD;
    bool good_improvement_slope = improvement_slope > IMPROVEMENT_SLOPE_THRESHOLD;
    bool good_improvement_curvature = improvement_curvature < IMPROVEMENT_CURVATURE_THRESHOLD;
    bool good_topk_half_life = topk_half_life > TOPK_HALF_LIFE_THRESHOLD;
    
    // New target radius hit criteria
    bool good_radius_hit_rate = radius_hit_rate > RADIUS_HIT_RATE_THRESHOLD;
    bool good_lid_diff = lid_topm_vs_frontier < LID_DIFF_THRESHOLD;
    
    // New early detour/redundancy criteria
    bool good_entry_detour_steps = entry_detour_steps < ENTRY_DETOUR_STEPS_THRESHOLD;
    bool good_entry_detour_ratio = entry_detour_ratio < ENTRY_DETOUR_RATIO_THRESHOLD;
    bool good_novelty_rate = novelty_rate_ratio > NOVELTY_RATE_THRESHOLD;
    bool good_duplication_rate = duplication_rate < DUPLICATION_RATE_THRESHOLD;
    bool good_frontier_overlap = frontier_overlap_ratio < FRONTIER_OVERLAP_THRESHOLD;
    
    // Require at least 15 out of 18 criteria to be met for convergence
    int criteria_met = (good_cohesion ? 1 : 0) + 
                      (good_contrast ? 1 : 0) + 
                      (good_cone ? 1 : 0) + 
                      (manageable_difficulty ? 1 : 0) +
                      (good_mutual_nn ? 1 : 0) +
                      (good_induced_lcc ? 1 : 0) +
                      (good_anchor_consistency ? 1 : 0) +
                      (good_replacement_decay ? 1 : 0) +
                      (good_improvement_slope ? 1 : 0) +
                      (good_improvement_curvature ? 1 : 0) +
                      (good_topk_half_life ? 1 : 0) +
                      (good_radius_hit_rate ? 1 : 0) +
                      (good_lid_diff ? 1 : 0) +
                      (good_entry_detour_steps ? 1 : 0) +
                      (good_entry_detour_ratio ? 1 : 0) +
                      (good_novelty_rate ? 1 : 0) +
                      (good_duplication_rate ? 1 : 0) +
                      (good_frontier_overlap ? 1 : 0);
    
    return criteria_met >= 15;
}

/**
 * @brief Calculate radius hit rate: fraction of top-m nodes within typical kNN radius
 * 
 * @param visited_nodes Vector of (node_id, distance) pairs
 * @param top_distances Vector of top-k distances
 * @param m Number of top nodes to consider
 * @param k Target k for kNN
 * @return Radius hit rate (0-1, higher = better alignment)
 */
float calculateRadiusHitRate(const std::vector<std::pair<tableint, dist_t>>& visited_nodes,
                            const std::vector<dist_t>& top_distances,
                            size_t m, size_t k) const {
    if (visited_nodes.size() < m || top_distances.empty()) {
        return 0.0f;
    }
    
    // Sort visited nodes by distance to get top-m
    std::vector<std::pair<tableint, dist_t>> sorted_nodes = visited_nodes;
    std::sort(sorted_nodes.begin(), sorted_nodes.end(), 
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    size_t top_m_count = std::min(m, sorted_nodes.size());
    
    // Estimate typical kNN radius r̃_k from top-k distances
    // Use the median of top-k distances as a proxy for typical kNN radius
    std::vector<dist_t> sorted_top_k = top_distances;
    std::sort(sorted_top_k.begin(), sorted_top_k.end());
    dist_t typical_radius = sorted_top_k[sorted_top_k.size() / 2]; // median
    
    // Count how many top-m nodes are within the typical radius
    int nodes_within_radius = 0;
    for (size_t i = 0; i < top_m_count; ++i) {
        if (sorted_nodes[i].second <= typical_radius) {
            nodes_within_radius++;
        }
    }
    
    return static_cast<float>(nodes_within_radius) / top_m_count;
}

/**
 * @brief Calculate LID difference between top-m and background frontier
 * 
 * @param visited_nodes Vector of (node_id, distance) pairs
 * @param top_distances Vector of top-k distances
 * @param m Number of top nodes to consider
 * @param k Target k for kNN
 * @return LID difference (negative = top-m more regular than background)
 */
float calculateLIDTopmVsFrontier(const std::vector<std::pair<tableint, dist_t>>& visited_nodes,
                                 const std::vector<dist_t>& top_distances,
                                 size_t m, size_t k) const {
    if (visited_nodes.size() < m || top_distances.empty()) {
        return 0.0f;
    }
    
    // Sort visited nodes by distance
    std::vector<std::pair<tableint, dist_t>> sorted_nodes = visited_nodes;
    std::sort(sorted_nodes.begin(), sorted_nodes.end(), 
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    size_t top_m_count = std::min(m, sorted_nodes.size());
    
    // Extract top-m distances for LID estimation
    std::vector<dist_t> top_m_distances;
    top_m_distances.reserve(top_m_count);
    for (size_t i = 0; i < top_m_count; ++i) {
        top_m_distances.push_back(sorted_nodes[i].second);
    }
    
    // Extract background frontier distances (nodes beyond top-k)
    std::vector<dist_t> background_distances;
    for (size_t i = k; i < sorted_nodes.size() && background_distances.size() < m; ++i) {
        background_distances.push_back(sorted_nodes[i].second);
    }
    
    // Calculate LID for top-m
    float lid_topm = calculateLIDEstimate(top_m_distances, std::min(10, (int)top_m_distances.size()));
    
    // Calculate LID for background frontier
    float lid_frontier = 0.0f;
    if (!background_distances.empty()) {
        lid_frontier = calculateLIDEstimate(background_distances, std::min(10, (int)background_distances.size()));
    }
    
    // Return difference: top-m LID - frontier LID
    // Negative values indicate top-m is more regular (lower LID) than background
    return lid_topm - lid_frontier;
}

/**
 * @brief Calculate entry detour steps: steps to reach threshold stability
 * 
 * @param distance_history Vector of distances discovered over time
 * @param discovery_steps Vector of steps when each discovery was made
 * @param k Target k for kNN
 * @return Number of steps to reach threshold stability
 */
float calculateEntryDetourSteps(const std::vector<dist_t>& distance_history, 
                               const std::vector<int>& discovery_steps, 
                               size_t k) const {
    if (distance_history.size() < k || discovery_steps.size() < k) {
        return 0.0f;
    }
    
    // Find the step when the k-th best distance was discovered
    // This represents when we have a stable top-k set
    std::vector<std::pair<dist_t, int>> distance_step_pairs;
    for (size_t i = 0; i < distance_history.size(); ++i) {
        distance_step_pairs.emplace_back(distance_history[i], discovery_steps[i]);
    }
    
    // Sort by distance (ascending) to find top-k
    std::sort(distance_step_pairs.begin(), distance_step_pairs.end());
    
    // The k-th best distance discovery step
    int kth_discovery_step = distance_step_pairs[k-1].second;
    
    return static_cast<float>(kth_discovery_step);
}

/**
 * @brief Calculate entry detour ratio: ratio of detour steps to total steps
 * 
 * @param entry_detour_steps Steps to reach threshold stability
 * @param total_steps Total steps taken in search
 * @return Ratio of detour steps to total steps
 */
float calculateEntryDetourRatio(float entry_detour_steps, int total_steps) const {
    if (total_steps <= 0) {
        return 0.0f;
    }
    
    return entry_detour_steps / total_steps;
}

/**
 * @brief Calculate novelty rate ratio: ratio of novel discoveries
 * 
 * @param is_novel_discovery Vector indicating whether each discovery was novel
 * @return Ratio of novel discoveries to total discoveries
 */
float calculateNoveltyRateRatio(const std::vector<bool>& is_novel_discovery) const {
    if (is_novel_discovery.empty()) {
        return 0.0f;
    }
    
    int novel_count = 0;
    for (bool is_novel : is_novel_discovery) {
        if (is_novel) {
            novel_count++;
        }
    }
    
    return static_cast<float>(novel_count) / is_novel_discovery.size();
}

/**
 * @brief Calculate duplication rate: ratio of redundant expansions
 * 
 * @param is_novel_discovery Vector indicating whether each discovery was novel
 * @return Ratio of redundant expansions (non-novel discoveries)
 */
float calculateDuplicationRate(const std::vector<bool>& is_novel_discovery) const {
    if (is_novel_discovery.empty()) {
        return 0.0f;
    }
    
    int duplicate_count = 0;
    for (bool is_novel : is_novel_discovery) {
        if (!is_novel) {
            duplicate_count++;
        }
    }
    
    return static_cast<float>(duplicate_count) / is_novel_discovery.size();
}

/**
 * @brief Calculate frontier overlap ratio: ratio of discoveries already in queue/visited
 * 
 * @param is_in_queue Vector indicating whether each discovery was already in queue
 * @param is_novel_discovery Vector indicating whether each discovery was novel
 * @return Ratio of discoveries that were already in queue/visited
 */
float calculateFrontierOverlapRatio(const std::vector<bool>& is_in_queue, 
                                   const std::vector<bool>& is_novel_discovery) const {
    if (is_in_queue.empty() || is_novel_discovery.empty()) {
        return 0.0f;
    }
    
    int overlap_count = 0;
    int total_novel = 0;
    
    for (size_t i = 0; i < is_in_queue.size() && i < is_novel_discovery.size(); ++i) {
        if (is_novel_discovery[i]) {
            total_novel++;
            if (is_in_queue[i]) {
                overlap_count++;
            }
        }
    }
    
    if (total_novel == 0) {
        return 0.0f;
    }
    
    return static_cast<float>(overlap_count) / total_novel;
}

/**
 * @brief Calculate local clustering coefficient for a node using wedge sampling
 * 
 * @param node_id The node ID to calculate LCC for
 * @param num_samples Number of wedge samples (default: 10)
 * @return Local clustering coefficient estimate
 */
float calculateLocalClusteringCoefficient(tableint node_id, int num_samples = 10) const {
    // Get neighbors from baselayer
    linklistsizeint *ll_cur = get_linklist0(node_id);
    int size = getListCount(ll_cur);
    
    if (size < 2) return 0.0f; // Need at least 2 neighbors for triangles
    
    tableint *data = (tableint *)(ll_cur + 1);
    int triangles = 0;
    int valid_samples = 0;
    
    // Sample random neighbor pairs
    for (int s = 0; s < num_samples; ++s) {
        // Randomly select two different neighbors
        int idx1 = rand() % size;
        int idx2 = rand() % size;
        
        if (idx1 != idx2) {
            tableint n1 = data[idx1];
            tableint n2 = data[idx2];
            
            // Check if n1 and n2 are connected in baselayer
            linklistsizeint *ll_n1 = get_linklist0(n1);
            int size_n1 = getListCount(ll_n1);
            tableint *data_n1 = (tableint *)(ll_n1 + 1);
            
            bool connected = false;
            for (int k = 0; k < size_n1; ++k) {
                if (data_n1[k] == n2) {
                    connected = true;
                    break;
                }
            }
            
            if (connected) {
                triangles++;
            }
            valid_samples++;
        }
    }
    
    return valid_samples > 0 ? static_cast<float>(triangles) / valid_samples : 0.0f;
}

/**
 * @brief Calculate quantiles (p25, p50, p75) from a vector of values
 * 
 * @param values Vector of values to calculate quantiles from
 * @param q25 Reference to store 25th percentile
 * @param q50 Reference to store 50th percentile (median)
 * @param q75 Reference to store 75th percentile
 */
void calculateQuantiles(const std::vector<float>& values, float& q25, float& q50, float& q75) const {
    if (values.empty()) {
        q25 = q50 = q75 = 0.0f;
        return;
    }
    
    std::vector<float> sorted_values = values;
    std::sort(sorted_values.begin(), sorted_values.end());
    
    size_t n = sorted_values.size();
    if (n == 1) {
        q25 = q50 = q75 = sorted_values[0];
        return;
    }
    
    // Calculate indices for quantiles
    size_t idx25 = static_cast<size_t>(0.25 * (n - 1));
    size_t idx50 = static_cast<size_t>(0.5 * (n - 1));
    size_t idx75 = static_cast<size_t>(0.75 * (n - 1));
    
    q25 = sorted_values[idx25];
    q50 = sorted_values[idx50];
    q75 = sorted_values[idx75];
}

/**
 * @brief Calculate cosine of angle between two direction vectors
 * 
 * @param prev_vector Previous step direction vector
 * @param curr_vector Current step direction vector
 * @param dim Vector dimension
 * @return Cosine of angle between vectors
 */
float calculateDirectionCosine(const float* prev_vector, const float* curr_vector, int dim) const {
    if (!prev_vector || !curr_vector || dim <= 0) {
        return 0.0f;
    }
    
    float dot_product = 0.0f;
    float norm_prev = 0.0f;
    float norm_curr = 0.0f;
    
    for (int i = 0; i < dim; ++i) {
        dot_product += prev_vector[i] * curr_vector[i];
        norm_prev += prev_vector[i] * prev_vector[i];
        norm_curr += curr_vector[i] * curr_vector[i];
    }
    
    norm_prev = std::sqrt(norm_prev);
    norm_curr = std::sqrt(norm_curr);
    
    if (norm_prev < 1e-8f || norm_curr < 1e-8f) {
        return 0.0f;
    }
    
    return dot_product / (norm_prev * norm_curr);
}

/**
 * @brief Calculate variance of direction vectors over a sliding window
 * 
 * @param direction_vectors Vector of direction vectors (each as float array)
 * @param window_size Size of sliding window
 * @param dim Vector dimension
 * @return Variance of direction cosines over the window
 */
float calculateDirectionVariance(const std::vector<std::vector<float>>& direction_vectors, 
                                int window_size, int dim) const {
    if (direction_vectors.size() < 2 || window_size <= 1) {
        return 0.0f;
    }
    
    int start_idx = std::max(0, static_cast<int>(direction_vectors.size()) - window_size);
    std::vector<float> cosines;
    
    for (size_t i = start_idx + 1; i < direction_vectors.size(); ++i) {
        if (i < direction_vectors.size() && i - 1 < direction_vectors.size()) {
            float cos_val = calculateDirectionCosine(
                direction_vectors[i-1].data(), 
                direction_vectors[i].data(), 
                dim
            );
            cosines.push_back(cos_val);
        }
    }
    
    if (cosines.empty()) {
        return 0.0f;
    }
    
    // Calculate variance
    float mean = 0.0f;
    for (float cos_val : cosines) {
        mean += cos_val;
    }
    mean /= cosines.size();
    
    float variance = 0.0f;
    for (float cos_val : cosines) {
        float diff = cos_val - mean;
        variance += diff * diff;
    }
    variance /= cosines.size();
    
    return variance;
}

/**
 * @brief Calculate Fourier features for cumulative cost (positional encoding)
 * 
 * @param cumulative_cost Current cumulative cost
 * @param num_frequencies Number of Fourier frequencies to use
 * @return Vector of sin/cos features
 */
std::vector<float> calculateFourierCostFeatures(int cumulative_cost, int num_frequencies = 4) const {
    std::vector<float> fourier_features;
    fourier_features.reserve(2 * num_frequencies);
    
    for (int f = 1; f <= num_frequencies; ++f) {
        float angle = 2.0f * M_PI * f * cumulative_cost / 1000.0f; // Scale by 1000 for reasonable range
        fourier_features.push_back(std::sin(angle));
        fourier_features.push_back(std::cos(angle));
    }
    
    return fourier_features;
}

/**
 * @brief Calculate Jaccard overlap between new neighbors and visited set
 * 
 * @param new_neighbors Vector of new neighbor node IDs
 * @param visited_set Set of visited node IDs
 * @return Jaccard overlap ratio
 */
float calculateFrontierOverlap(const std::vector<tableint>& new_neighbors, 
                              const std::unordered_set<tableint>& visited_set) const {
    if (new_neighbors.empty()) {
        return 0.0f;
    }
    
    int intersection = 0;
    for (tableint neighbor : new_neighbors) {
        if (visited_set.find(neighbor) != visited_set.end()) {
            intersection++;
        }
    }
    
    int union_size = new_neighbors.size() + visited_set.size() - intersection;
    return union_size > 0 ? static_cast<float>(intersection) / union_size : 0.0f;
}

/**
 * @brief Calculate novelty rate for a node during search
 * 
 * @param node_id The node ID to calculate novelty for
 * @param visited_array Array tracking visited nodes
 * @param visited_tag Current visit tag
 * @return Novelty rate (fraction of unvisited neighbors)
 */
float calculateNodeNoveltyRate(tableint node_id, vl_type* visited_array, vl_type visited_tag) const {
    linklistsizeint *ll_cur = get_linklist0(node_id);
    int size = getListCount(ll_cur);
    
    if (size == 0) return 0.0f;
    
    tableint *data = (tableint *)(ll_cur + 1);
    int unvisited_neighbors = 0;
    
    for (int j = 0; j < size; ++j) {
        tableint neighbor_id = data[j];
        if (visited_array[neighbor_id] != visited_tag) {
            unvisited_neighbors++;
        }
    }
    
    return static_cast<float>(unvisited_neighbors) / size;
}

/**
 * @brief Calculate LID estimate using anchor distances
 * 
 * @param anchor_distances Vector of distances to anchor points (sorted in ascending order)
 * @param k Number of nearest anchors to use for LID estimation
 * @return Estimated LID value
 */
float calculateLIDEstimate(const std::vector<dist_t>& anchor_distances, int k = 10) const {
    if (anchor_distances.size() < k || k < 2) {
        return 0.0f;
    }
    
    float epsilon = 1e-6f;
    float sum_log_ratios = 0.0f;
    int valid_samples = 0;
    
    // Use the formula: LID_k = -1 / (1/k * sum(log(r_i/r_k)))
    // where r_1 <= r_2 <= ... <= r_k are the k smallest distances
    float r_k = anchor_distances[k-1] + epsilon;
    
    for (int i = 0; i < k; ++i) {
        float r_i = anchor_distances[i] + epsilon;
        if (r_i > 0 && r_k > 0) {
            sum_log_ratios += std::log(r_i / r_k);
            valid_samples++;
        }
    }
    
    if (valid_samples == k && sum_log_ratios != 0) {
        return -1.0f / (sum_log_ratios / k);
    } else {
        return 0.0f; // Return 0 for invalid cases
    }
}

/**
 * @brief Perform a search in the base layer with effective branching factor calculation
 *
 * @tparam has_deletions Indicates whether the graph contains deleted nodes.
 * @tparam collect_metrics Indicates whether to collect metrics during the search.
 *
 * @param ep_id Entry point node ID at the base layer where the search begins.
 * @param data_point Pointer to the query vector.
 * @param m Maximum number of distinct neighbors to visit.
 * @param ef Exploration factor.
 * @param metric Optional parameter for collecting metrics during the search.
 * @param skipped_nodes Optional pointer to count skipped nodes.
 * @param prior_avg_degree Prior average degree from index signature (default: 16.0).
 * @param prior_avg_clustering Prior average clustering coefficient from index signature (default: 0.3).
 * @param k Number of top distances to track for replacement rate calculation (default: 10).
 * @param recall_target Target recall for basin size estimation (default: 0.9).
 * @param alpha Dataset-level scale constant for path length estimation (default: 1.0).
 * @param gamma Parameter for alignment correction function (default: 0.1).
 * @param lambda Parameter for alignment correction function (default: 1.0).
 * @param anchor_distances Pointer to vector of anchor distances for LID estimation (default: nullptr).
 * @param lid_k Number of nearest anchors to use for LID estimation (default: 10).
 * @param global_r_k_percentile Global k-NN distance percentile from offline signature (default: 0.5).
 * @param use_lid_method Whether to use LID-volume method for basin size estimation (default: false).
 * @return EffectiveBranchingResult containing neighbors and branching factor analysis.
 * 
 * @note IMPORTANT: Neighbors are marked as visited immediately after calculating their distances.
 * This ensures that novelty rate calculation considers both expanded nodes and computed neighbors
 * as visited, providing a more accurate assessment of search progress.
 * 
 * @note LID-Volume Method: Uses the formula N_basin(q) ≈ N_seen * (r_k_tilde / r_star)^LID
 * where N_seen is the number of expanded nodes, r_k_tilde is the global reference distance,
 * r_star is the current best distance, and LID is the estimated local intrinsic dimensionality.
 */
template <bool has_deletions, bool collect_metrics = false>
EffectiveBranchingResult searchBaseLayerDataCollectionWithBranching(
    tableint ep_id, 
    const void *data_point, 
    size_t m, 
    size_t ef, 
    Metric* metric = nullptr, 
    int* skipped_nodes = nullptr,
    float prior_avg_degree = 16.0f,
    float prior_avg_clustering = 0.3f,
    size_t k = 10,
    float recall_target = 0.9f,
    float alpha = 1.0f,
    float gamma = 0.1f,
    float lambda = 1.0f,
    // LID-volume method parameters
    // const std::vector<float>* anchor_distances = nullptr,
    int lid_estimate = 0,
    float lid_r_k = 0,
    float global_r_k_percentile = 0.5f,
    bool use_lid_method = false,
    float prior_branching_factor = 0.0f) const {
    
    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    VisitedList *cl = visited_list_pool_->getFreeVisitedList();
    vl_type *calculated_array = cl->mass;
    vl_type calculated_array_tag = cl->curV;

    
    
    EffectiveBranchingResult result;
    result.effective_branching_factor = 0.0f;
    result.novelty_rate = 0.0f;
    result.local_clustering_coeff = 0.0f;
    // result.prior_branching_factor = prior_avg_degree * (1.0f - prior_avg_clustering);
    result.estimated_path_hops = 0.0f;
    result.basin_size_estimate = 0.0f;
    result.replacement_rate = 0.0f;
    result.gap12_normalized = 0.0f;
    result.alignment_correction = 1.0f;
    
    // Initialize LID-related fields
    result.lid_estimate = 0.0f;
    result.current_best_distance = 0.0f;
    result.distance_decay_rate = 0.0f;
    result.basin_size_lid_method = 0.0f;
    result.volume_expansion_factor = 1.0f;
    result.visited_node_num = 0;
    result.average_degree = 0;
    
    using Candidate = std::tuple<dist_t, tableint, int>;
    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<>> candidate_set;
    
    // Maintain top-k distances for replacement rate calculation
    std::priority_queue<std::pair<dist_t, tableint>> top_k_distances;
    std::vector<dist_t> top_distances;
    top_distances.reserve(k);
    
    // Track replacement events
    // std::vector<int> replacement_events;
    // int total_expansions = 0;
    int replacement_events = 0;
    
    dist_t lowerBound;
    
    if (!has_deletions || !isMarkedDeleted(ep_id)) {
        dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
        candidate_set.emplace(dist, ep_id, 0);
        lowerBound = dist;
        
        // Initialize top-k with entry point
        top_k_distances.emplace(dist, ep_id);
        top_distances.push_back(dist);
    } else {
        lowerBound = std::numeric_limits<dist_t>::max();
        candidate_set.emplace(lowerBound, ep_id, 0);
        HNSW_LOG_WARN("[searchBaseLayerDataCollectionWithBranching] Entry point is marked as deleted.");
    }
    
    visited_array[ep_id] = visited_array_tag;
    int cnt_visit = 0;
    int hop_count = 0;
    size_t distinct_neighbors_visited = 0;
    
    // Track expanded nodes for LCC calculation
    std::vector<tableint> expanded_nodes;
    expanded_nodes.reserve(m);
    
    while (!candidate_set.empty() && result.visited_node_num < m) {
        auto [current_dist, current_node_id, current_hop_count] = candidate_set.top();
        candidate_set.pop();
        
        // result.neighbors.emplace_back(current_dist, current_hop_count);

        // HNSW_LOG_INFO("[searchBaseLayerDataCollectionWithBranching] Processing node: ", current_node_id, 
                    //  ", distance: ", current_dist, ", hop_count: ", current_hop_count);

        // If the node is already visited, skip it
        if (visited_array[current_node_id] == visited_array_tag && current_hop_count > 0) {
            // HNSW_LOG_INFO("[searchBaseLayerDataCollectionWithBranching] Node ", current_node_id, " already visited. Skipping.");
            // if (skipped_nodes) (*skipped_nodes)++;
            continue;
        }
        
        // Mark the node as visited
        visited_array[current_node_id] = visited_array_tag;
        
        // Track expanded nodes for LCC calculation
        if (current_hop_count > 0) { // Don't count entry point
            expanded_nodes.push_back(current_node_id);
            result.visited_node_num++;
            // total_expansions++;
        }

        // Explore neighbors of the current node
        int *data = (int *) get_linklist0(current_node_id);
        size_t size = getListCount((linklistsizeint*)data);
        result.average_degree += size;

        // Calculate Novelty count
        float novelty = calculateNodeNoveltyRate(current_node_id, calculated_array, calculated_array_tag);
        result.novelty_rate += novelty;
        
        if(collect_metrics){
            metric_hops++;
            metric_distance_computations+=size;
        }
        
        for (size_t j = 1; j <= size; j++) {
            int candidate_id = *(data + j);
            
            // If the neighbor is already visited, skip it
            if (visited_array[candidate_id] == visited_array_tag) {
                // HNSW_LOG_INFO("[searchBaseLayerDataCollectionWithBranching] Neighbor ", candidate_id, " already visited. Skipping.");
                continue;
            }

            // Calculate distance to the neighbor
            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(candidate_id), dist_func_param_);
            calculated_array[candidate_id] = calculated_array_tag;
            
            candidate_set.emplace(dist, candidate_id, current_hop_count + 1);
            
            // Check if this distance improves top-k
            if (top_k_distances.size() < k) {
                top_k_distances.emplace(dist, candidate_id);
                top_distances.push_back(dist);
                std::sort(top_distances.begin(), top_distances.end());
                // HNSW_LOG_INFO("[searchBaseLayerDataCollectionWithBranching] Top-k distances: ", top_k_distances.top().first);
            } else if (dist < top_k_distances.top().first) {
                // Replacement event occurred
                top_k_distances.pop();
                top_k_distances.emplace(dist, candidate_id);
                
                // Update top_distances vector
                top_distances.clear();
                auto temp_queue = top_k_distances;
                while (!temp_queue.empty()) {
                    top_distances.push_back(temp_queue.top().first);
                    temp_queue.pop();
                }
                std::sort(top_distances.begin(), top_distances.end());
                
                // Record replacement event
                replacement_events++;
                // HNSW_LOG_INFO("[searchBaseLayerDataCollectionWithBranching] Replacement event: ", replacement_events);
            }
            
            if (distinct_neighbors_visited >= m) {
                break;
            }
        }
        hop_count++;
    }
    
    // Store top distances and replacement events
    result.top_distances = top_distances;
    // result.replacement_events = replacement_events;
    
    // Calculate effective branching factor components
    if (!expanded_nodes.empty()) {
        // Calculate novelty rates for expanded nodes
        result.node_novelty_rates.reserve(expanded_nodes.size());
        result.node_clustering_coeffs.reserve(expanded_nodes.size());
        
        float total_novelty = 0.0f;
        float total_clustering = 0.0f;
        
        for (tableint node_id : expanded_nodes) {
            // Calculate novelty rate at the time of expansion
            float novelty = calculateNodeNoveltyRate(node_id, visited_array, visited_array_tag);
            result.node_novelty_rates.push_back(novelty);
            total_novelty += novelty;
            
            // Calculate local clustering coefficient
            float lcc = calculateLocalClusteringCoefficient(node_id, 10); // 10 samples per node
            result.node_clustering_coeffs.push_back(lcc);
            total_clustering += lcc;
        }
        
        // Calculate effective branching factor: log(b_eff) = (1-eta) * log(b_eff^prior) + eta * log(nu_bar * d_bar)
        result.novelty_rate = result.novelty_rate / result.visited_node_num;
        result.average_degree = result.average_degree / result.visited_node_num;
        float eta = std::min(1.0f, (float)m / 50);    
        result.effective_branching_factor = (1.0f - eta) * std::log(prior_branching_factor) + eta * std::log(result.novelty_rate * result.average_degree);
        HNSW_LOG_INFO("[searchBaseLayerDataCollectionWithBranching] effective branching factor: ", result.effective_branching_factor, "prior_branching_factor", prior_branching_factor, "novelty rate: ", result.novelty_rate, " average_degree: ", result.average_degree);

        result.local_clustering_coeff = total_clustering / expanded_nodes.size();
        
        // Calculate effective branching factor: b_eff = E[deg] * (1 - LCC_local) * ν̄
        // result.effective_branching_factor = prior_avg_degree * 
        //                                   (1.0f - result.local_clustering_coeff) * 
        //                                   result.novelty_rate;
        
        // Calculate replacement rate (Method A: Insertion/Replacement rate)
        if (result.visited_node_num > 0) {
            result.replacement_rate = static_cast<float>(replacement_events) / result.visited_node_num;
            HNSW_LOG_INFO("[searchBaseLayerDataCollectionWithBranching] Replacement rate: ", result.replacement_rate, "replacement_events: ", replacement_events, "visited_node_num: ", result.visited_node_num);
        } else {
            result.replacement_rate = 0.0f;
        }
        
        // Calculate basin size estimate: N_basin ≈ (R_t * k) / (p_rep + ε)
        float epsilon = 1e-6f;
        result.basin_size_estimate = (recall_target * k) / (result.replacement_rate + epsilon);
        HNSW_LOG_INFO("[searchBaseLayerDataCollectionWithBranching] Basin size estimate: ", result.basin_size_estimate, "replacement_rate: ", result.replacement_rate, "epsilon: ", epsilon, "recall_target: ", recall_target, "k: ", k);
        
        // LID-Volume Method (Method B) for basin size estimation
        // if (use_lid_method && anchor_distances != nullptr && !anchor_distances->empty()) {
        if (use_lid_method) {
            // compute online lid_estimate: lid  = -(1/(k-1) * \sum_{i=1}^{k-1} \ln(d_i/d_k))^{-1}
            float lid_estimate_online = 0.0f;
            for (size_t i = 0; i < top_distances.size() - 1; i++) {
                lid_estimate_online += std::log(top_distances[i] / top_distances[top_distances.size() - 1]);
                // HNSW_LOG_INFO("[searchBaseLayerDataCollectionWithBranching] lid_estimate_online: ", lid_estimate_online, "top_distances[i]: ", top_distances[i], "top_distances[top_distances.size() - 1]: ", top_distances[top_distances.size() - 1]);
            }
            lid_estimate_online = -1.0f / (lid_estimate_online / (top_distances.size() - 1));
            float eta = min(1.0f, (float)m / 50);
            result.lid_estimate = (1.0f - eta) * lid_estimate + eta * lid_estimate_online;
            HNSW_LOG_INFO("[searchBaseLayerDataCollectionWithBranching] lid_estimate: ", result.lid_estimate, "lid_estimate_online: ", lid_estimate_online, "eta: ", eta, "lid_estimate_offline: ", lid_estimate);
            
            // Get current best distance r_* (minimum distance found so far)
            if (!top_distances.empty()) {
                result.current_best_distance = top_distances[0];
                HNSW_LOG_INFO("[searchBaseLayerDataCollectionWithBranching] current_best_distance: ", result.current_best_distance);
            }
            
            // Calculate distance decay rate: dr_*/dexp
            // This represents how fast the best distance decreases with each expansion
            if (result.visited_node_num > 0 && result.current_best_distance > 0) {
                // Simple approximation: assume linear decay
                // In practice, this could be refined with more sophisticated curve fitting
                float initial_distance = (ep_id >= 0) ? fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_) : 0.0f;
                if (initial_distance > result.current_best_distance) {
                    result.distance_decay_rate = (initial_distance - result.current_best_distance) / result.visited_node_num;
                } else {
                    result.distance_decay_rate = 0.0f;
                }
            }
            
            // Calculate basin size estimate using LID method
            // Formula: N_basin(q) ≈ N_seen * (r_k_tilde / r_star)^LID
            // where:
            // - N_seen: number of nodes seen/expanded so far
            // - r_k_tilde: typical k-NN distance from offline signature (global reference)
            // - r_star: current best distance found
            // - LID: estimated local intrinsic dimensionality
            if (result.lid_estimate > 0 && result.current_best_distance > 0) {
                // Estimate r_k_tilde from offline signature
                // For now, use a heuristic based on current progress
                // In practice, this would come from the index signature
                // float r_k_tilde = result.current_best_distance * (1.0f + global_r_k_percentile);
                
                // Calculate radius ratio: r_k_tilde / r_star
                float radius_ratio = lid_r_k / result.current_best_distance;
                HNSW_LOG_INFO("[searchBaseLayerDataCollectionWithBranching] radius_ratio: ", radius_ratio, "lid_r_k: ", lid_r_k, "result.current_best_distance: ", result.current_best_distance);                
                // Calculate volume expansion factor: (r_k_tilde / r_star)^LID
                result.volume_expansion_factor = std::pow(radius_ratio, result.lid_estimate);
                HNSW_LOG_INFO("[searchBaseLayerDataCollectionWithBranching] volume_expansion_factor: ", result.volume_expansion_factor, "radius_ratio: ", radius_ratio, "lid_estimate: ", result.lid_estimate);
                
                // Calculate N_basin using the LID formula
                int N_seen = result.visited_node_num + 1; // +1 for entry point
                result.basin_size_lid_method = N_seen * result.volume_expansion_factor;
                HNSW_LOG_INFO("[searchBaseLayerDataCollectionWithBranching] N_seen: ", N_seen, "volume_expansion_factor: ", result.volume_expansion_factor, "radius_ratio: ", radius_ratio, "lid_basin_method: ", result.basin_size_lid_method);
                
                // Use ensemble approach: combine both methods
                // Method A (replacement rate): 70% weight
                // Method B (LID-volume): 30% weight
                result.basin_size_estimate = 0.0f * result.basin_size_estimate + 1.0f * result.basin_size_lid_method;
                
                HNSW_LOG_INFO("[searchBaseLayerDataCollectionWithBranching] LID method: N_seen=", N_seen, 
                             ", r_k_tilde=", lid_r_k, ", r_star=", result.current_best_distance,
                             ", radius_ratio=", radius_ratio, ", LID=", result.lid_estimate,
                             ", volume_expansion=", result.volume_expansion_factor,
                             ", N_basin_LID=", result.basin_size_lid_method);
            }
        }
        
        // Calculate gap12_normalized for alignment correction
        if (top_distances.size() >= 2) {
            float d1 = top_distances[0];
            float d2 = top_distances[1];
            result.gap12_normalized = (d2 - d1) / (d1 + epsilon);
        } else {
            result.gap12_normalized = 0.0f;
        }
        
        // Calculate alignment correction factor: φ(gap12_norm) = 1/(1 + γ*e^(-λ*gap12_norm))
        if (result.gap12_normalized > 0) {
            result.alignment_correction = 1.0f / (1.0f + gamma * std::exp(-lambda * result.gap12_normalized));
        } else {
            result.alignment_correction = 1.0f;
        }
        
        // Calculate estimated path hops: path_hops(q) ≈ α * log_b_eff(N_basin) * φ(gap12_norm)
        if (result.effective_branching_factor > 1.0f && result.basin_size_estimate > 1.0f) {
            result.estimated_path_hops = alpha * 
                                       std::log(result.basin_size_estimate) / result.effective_branching_factor * 
                                       result.alignment_correction;
            HNSW_LOG_INFO("[searchBaseLayerDataCollectionWithBranching] Estimated path hops: ", result.estimated_path_hops, "alpha: ", alpha, "result.basin_size_estimate: ", result.basin_size_estimate, "result.effective_branching_factor: ", result.effective_branching_factor, "result.alignment_correction: ", result.alignment_correction);
        } else {
            result.estimated_path_hops = 0.0f;
        }
        
        HNSW_LOG_INFO("[searchBaseLayerDataCollectionWithBranching] Effective branching factor: ", 
                     result.effective_branching_factor);
        HNSW_LOG_INFO("[searchBaseLayerDataCollectionWithBranching] Novelty rate: ", result.novelty_rate);
        HNSW_LOG_INFO("[searchBaseLayerDataCollectionWithBranching] Local clustering coefficient: ", 
                     result.local_clustering_coeff);
        HNSW_LOG_INFO("[searchBaseLayerDataCollectionWithBranching] Replacement rate: ", result.replacement_rate);
        HNSW_LOG_INFO("[searchBaseLayerDataCollectionWithBranching] Basin size estimate: ", result.basin_size_estimate);
        HNSW_LOG_INFO("[searchBaseLayerDataCollectionWithBranching] Gap12 normalized: ", result.gap12_normalized);
        HNSW_LOG_INFO("[searchBaseLayerDataCollectionWithBranching] Alignment correction: ", result.alignment_correction);
        HNSW_LOG_INFO("[searchBaseLayerDataCollectionWithBranching] Estimated path hops: ", result.estimated_path_hops);
        
        // Log LID-related information if LID method is used
        if (use_lid_method && result.lid_estimate > 0) {
            HNSW_LOG_INFO("[searchBaseLayerDataCollectionWithBranching] LID estimate: ", result.lid_estimate);
            HNSW_LOG_INFO("[searchBaseLayerDataCollectionWithBranching] Current best distance: ", result.current_best_distance);
            HNSW_LOG_INFO("[searchBaseLayerDataCollectionWithBranching] Distance decay rate: ", result.distance_decay_rate);
            HNSW_LOG_INFO("[searchBaseLayerDataCollectionWithBranching] Basin size (LID method): ", result.basin_size_lid_method);
            HNSW_LOG_INFO("[searchBaseLayerDataCollectionWithBranching] Volume expansion factor: ", result.volume_expansion_factor);
        }
    }
    
    visited_list_pool_->releaseVisitedList(vl);
    return result;
}

/**
 * @brief Perform a search in the base layer of the HNSW graph, stopping when `m` distinct neighbors are visited.
 *
 * @tparam has_deletions Indicates whether the graph contains deleted nodes.
 * @tparam collect_metrics Indicates whether to collect metrics during the search.
 *
 * @param ep_id Entry point node ID at the base layer where the search begins.
 * @param data_point Pointer to the query vector.
 * @param m Maximum number of distinct neighbors to visit.
 * @param metric Optional parameter for collecting metrics during the search.
 * @return A vector of pairs containing distance (`r_i`) and hop count (`h_i`) for each visited neighbor.
 */
template <bool has_deletions, bool collect_metrics = false>
std::vector<std::pair<dist_t, int>> searchBaseLayerDataCollection(tableint ep_id, const void *data_point, size_t m, size_t ef, Metric* metric = nullptr, int* skipped_nodes = nullptr) const {
    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;
    std::vector<std::pair<dist_t, int>> result;
    using Candidate = std::tuple<dist_t, tableint, int>;
    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<>> candidate_set;
    dist_t lowerBound;
    if (!has_deletions || !isMarkedDeleted(ep_id)) {
        dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
        candidate_set.emplace(dist, ep_id, 0);
        lowerBound = dist;
    } else {
        lowerBound = std::numeric_limits<dist_t>::max();
        candidate_set.emplace(lowerBound, ep_id, 0);
        HNSW_LOG_DEBUG("[searchBaseLayerDataCollection] Entry point is marked as deleted.");
    }
    visited_array[ep_id] = visited_array_tag;
    int cnt_visit = 0;
    int hop_count = 0;
    size_t distinct_neighbors_visited = 0;
    while (!candidate_set.empty()  && result.size() < m) {
        auto [current_dist, current_node_id, current_hop_count] = candidate_set.top();
        candidate_set.pop();
        result.emplace_back(current_dist, current_hop_count);

        // HNSW_LOG_INFO("[searchBaseLayerDataCollection] Processing node: ", current_node_id, ", distance: ", current_dist, ", hop_count: ", current_hop_count);


        // If the node is already visited, skip it
        if (visited_array[current_node_id] == visited_array_tag && current_hop_count > 0) {
            // HNSW_LOG_INFO("[searchBaseLayerDataCollection] Node ", current_node_id, " already visited. Skipping.");
            if (skipped_nodes) (*skipped_nodes)++;
            continue;
        }
        // Mark the node as visited
        visited_array[current_node_id] = visited_array_tag;

        // Explore neighbors of the current node
        int *data = (int *) get_linklist0(current_node_id);
        size_t size = getListCount((linklistsizeint*)data);
        if(collect_metrics){
            metric_hops++;
            metric_distance_computations+=size;
        }
        for (size_t j = 1; j <= size; j++) {
            int candidate_id = *(data + j);
            // If the neighbor is already visited, skip it
            if (visited_array[candidate_id] == visited_array_tag) {
                // HNSW_LOG_INFO("[searchBaseLayerDataCollection] Neighbor ", candidate_id, " already visited. Skipping.");
                continue;
            }

            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(candidate_id), dist_func_param_);
            // if (top_candidates.size() < ef || lowerBound > dist) {                      
            candidate_set.emplace(dist, candidate_id, current_hop_count + 1);
            // }
            if (distinct_neighbors_visited >= m) {
                break;
            }
        }
        hop_count++;
    }
    visited_list_pool_->releaseVisitedList(vl);
    return result;
}




        template <bool has_deletions, bool collect_metrics=false>
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>>
        searchBaseLayernStep(tableint ep_id, const void *data_point, size_t nstep, size_t k, Metric* metric = nullptr) const {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            // top_candidates - the result set R
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>> top_candidates;
            // candidate_set  - the search set S
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

            dist_t lowerBound;
            // Insert the entry point to the result and search set with its exact distance as a key. 
            if (!has_deletions || !isMarkedDeleted(ep_id)) {
#ifdef COUNT_DIST_TIME
                StopW stopw = StopW();
#endif
#ifndef ED2IP
            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
#else
            dist_t dist = hnswlib::L2Sqr_by_IP(data_point, getDataByInternalId(ep_id), getExternalLabel(ep_id), dist_func_param_);
#endif

#ifdef COUNT_DIST_TIME
                adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif          
                adsampling::tot_dist_calculation++;
                adsampling::tot_full_dist ++;
                metric->ndc++;
                lowerBound = dist;
                top_candidates.emplace(dist, ep_id);
                candidate_set.emplace(-dist, ep_id);
            } 
            else {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidate_set.emplace(-lowerBound, ep_id);
            }

            visited_array[ep_id] = visited_array_tag;
            int cnt_visit = 0;
            int nhop = 0;

            // Iteratively generate candidates and conduct DCOs to maintain the result set R.
            while (!candidate_set.empty()) {
                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();

                if(nhop > nstep){
                    break;
                }

                // When the smallest object in S has its distance larger than the largest in R, terminate the algorithm.
                if ((-current_node_pair.first) > lowerBound && (top_candidates.size() == k || has_deletions == false)) {
                    break;
                }
                candidate_set.pop();
                adsampling::tot_hops++;
                nhop++;
                // Fetch the smallest object in S. 
                tableint current_node_id = current_node_pair.second;
                int *data = (int *) get_linklist0(current_node_id);
                size_t size = getListCount((linklistsizeint*)data);
                if(collect_metrics){
                    metric_hops++;
                    metric_distance_computations+=size;
                }

                // Enumerate all the neighbors of the object and view them as candidates of KNNs. 
                for (size_t j = 1; j <= size; j++) {
                    int candidate_id = *(data + j);
                    if (!(visited_array[candidate_id] == visited_array_tag)) {
                        cnt_visit ++;
                        visited_array[candidate_id] = visited_array_tag;

                        // Conduct DCO with FDScanning wrt the N_ef th NN: 
                        // (1) calculate its exact distance 
                        // (2) compare it with the N_ef th distance (i.e., lowerBound)
                        char *currObj1 = (getDataByInternalId(candidate_id));
#ifdef COUNT_DIST_TIME
                        StopW stopw = StopW();
#endif
#ifndef ED2IP
            dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);
#else
            dist_t dist = hnswlib::L2Sqr_by_IP(data_point, currObj1, getExternalLabel(candidate_id), dist_func_param_);
#endif
#ifdef COUNT_DIST_TIME
                        adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif                  
                        adsampling::tot_full_dist++;
                        metric->ndc++;
                        if (top_candidates.size() < k || lowerBound > dist) {                      
                            candidate_set.emplace(-dist, candidate_id);
                            if (!has_deletions || !isMarkedDeleted(candidate_id))
                                top_candidates.emplace(dist, candidate_id);

                            if (top_candidates.size() > k)
                                top_candidates.pop();

                            if (!top_candidates.empty())
                                lowerBound = top_candidates.top().first;
                        }
                    }
                }
            }
            adsampling::tot_dist_calculation += cnt_visit;
            visited_list_pool_->releaseVisitedList(vl);
            return top_candidates;
        }



        /**
         * @brief Perform base layer search with comprehensive step-by-step statistics collection
         * 
         * This function performs a search in the base layer of the HNSW graph while collecting
         * detailed statistics at each step, including geometry/distance features, bounds/certainty
         * measures, search state metrics, graph structure properties, and cost/time tracking.
         * 
         * @tparam has_deletions Indicates whether the graph contains deleted nodes
         * @tparam collect_metrics Indicates whether to collect basic metrics during the search
         * 
         * @param ep_id Entry point node ID at the base layer where the search begins
         * @param data_point Pointer to the query vector
         * @param k Target number of nearest neighbors to find
         * @param m Maximum number of distinct neighbors to visit
         * @param direction_window Window size for directional variance calculation
         * @param fourier_freqs Number of Fourier frequencies for cost encoding
         * @param metric Optional parameter for collecting basic metrics during the search
         * @return SearchWithStatsResult containing search results and step statistics
         */
        template <bool has_deletions, bool collect_metrics=false>
        SearchWithStatsResult searchBaseLayerWithStepStats(
            tableint ep_id, 
            const void *data_point, 
            size_t k,
            size_t m,
            int direction_window = 5,
            int fourier_freqs = 4,
            Metric* metric = nullptr) const {
            using Candidate = std::tuple<dist_t, tableint, int>;
            
            SearchWithStatsResult result;
            
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;
            
            // Search state tracking
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>> top_candidates;
            std::priority_queue<Candidate, std::vector<Candidate>, std::greater<>> candidate_set;
            // std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;
            
            // Statistics tracking
            std::vector<std::vector<float>> direction_vectors; // For directional features
            std::unordered_set<tableint> calculated_set; // For frontier overlap
            std::vector<float> prev_ub_k_history; // For delta calculations
            std::vector<float> prev_gap_top_history; // For gap delta calculations
            std::vector<float> prev_lb_gap_history; // For lb delta calculations
            
            // Previous step state for delta calculations
            float prev_ub_k = std::numeric_limits<float>::max();
            float prev_gap_top = 0.0f;
            float prev_lb_gap = 0.0f;
            float initial_lb_gap = 0.0f;
            
            // Cost tracking
            int cumulative_cost = 0;
            int stale_length = 0;
            
            // New tracking variables for additional statistics
            float prev_step_distance = 0;
            int updates_in_result_set_this_step = 0;
            
            dist_t lowerBound;
            int step_number = 0;
            
            // Initialize entry point
            if (!has_deletions || !isMarkedDeleted(ep_id)) {
#ifndef ED2IP
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
#else
                dist_t dist = hnswlib::L2Sqr_by_IP(data_point, getDataByInternalId(ep_id), getExternalLabel(ep_id), dist_func_param_);
#endif
                if (collect_metrics) {
                    metric->ndc++;
                }
                lowerBound = dist;
                top_candidates.emplace(dist, ep_id);
                candidate_set.emplace(dist, ep_id, 0);
                calculated_set.insert(ep_id);
            } else {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidate_set.emplace(lowerBound, ep_id, 0);
            }
            
            visited_array[ep_id] = visited_array_tag;
            
            // Main search loop
            while (!candidate_set.empty() && step_number < m) {
                auto [dist, current_node_id, hop_count] = candidate_set.top();
                // HNSW_LOG_INFO("[searchBaseLayerWithStepStats]Step ", step_number, ": Current node pair: ", dist, ", current_node_id: ", current_node_id, ", hop_count: ", hop_count);
                // Termination condition: no improvement possible
                // if ((-current_node_pair.first) > lowerBound && (top_candidates.size() == k || has_deletions == false)) {
                //     HNSW_LOG_INFO("[searchBaseLayerWithStepStats]Terminate: smallest in S > largest in R and R full.");
                //     break;
                // }
                
                candidate_set.pop();
                
                // Check if already visited (revisit detection)
                bool is_revisit = (visited_array[current_node_id] == visited_array_tag && step_number > 0);
                if (is_revisit) {
                    // HNSW_LOG_INFO("[searchBaseLayerWithStepStats]Revisit: ", current_node_id);
                    continue;
                }
                
                visited_array[current_node_id] = visited_array_tag;
                // visited_set.insert(current_node_id);
                
                // Start collecting statistics for this step
                StepStatistics step_stats;
                step_stats.step_number = step_number;
                step_stats.current_node_id = current_node_id;
                step_stats.revisit = is_revisit;
                
                // Cost tracking for this step
                int step_cost_dist = 0;
                int step_cost_heap = 0;
                
                // Get current node's neighbors
                int *data = (int *) get_linklist0(current_node_id);
                size_t size = getListCount((linklistsizeint*)data);
                step_stats.deg = size;
                
                if (collect_metrics) {
                    metric_hops++;
                    metric_distance_computations += size;
                }
                
                // Collect candidate distances and similarities for quantile calculation
                std::vector<float> candidate_distances;
                std::vector<float> candidate_similarities;
                std::vector<tableint> new_neighbors;
                
                // Process neighbors
                for (size_t j = 1; j <= size; j++) {
                    int candidate_id = *(data + j);
                    if (!(visited_array[candidate_id] == visited_array_tag)) {
                        new_neighbors.push_back(candidate_id);
                        // visited_array[candidate_id] = visited_array_tag;
                        calculated_set.insert(candidate_id);
                        
                        // Calculate distance
                        char *currObj1 = (getDataByInternalId(candidate_id));
#ifndef ED2IP
                        dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);
#else
                        dist_t dist = hnswlib::L2Sqr_by_IP(data_point, currObj1, getExternalLabel(candidate_id), dist_func_param_);
#endif
                        step_cost_dist++;
                        if (collect_metrics) {
                            metric->ndc++;
                        }
                        
                        candidate_distances.push_back(dist);
                        candidate_similarities.push_back(1.0f / (1.0f + dist)); // Convert distance to similarity
                        
                        // Add to candidate set if promising
                        if (top_candidates.size() < k || lowerBound > dist) {
                            // HNSW_LOG_INFO("[searchBaseLayerWithStepStats]Add to candidate set: ", candidate_id, ", dist: ", dist);
                            candidate_set.emplace(dist, candidate_id, hop_count + 1);
                            step_cost_heap++;
                            
                            if (!has_deletions || !isMarkedDeleted(candidate_id)) {
                                top_candidates.emplace(dist, candidate_id);
                                updates_in_result_set_this_step++;
                                step_cost_heap++;
                                
                                if (top_candidates.size() > k) {
                                    top_candidates.pop();
                                    step_cost_heap++;
                                }
                                
                                if (!top_candidates.empty()) {
                                    lowerBound = top_candidates.top().first;
                                }
                            }
                        }
                    else{
                        // HNSW_LOG_INFO("[searchBaseLayerWithStepStats]Skip: ", candidate_id);
                    }
                    }
                    else{
                        // HNSW_LOG_INFO("[searchBaseLayerWithStepStats]Already visited: ", candidate_id);
                    }
                }
                // HNSW_LOG_INFO("[searchBaseLayerWithStepStats]Candidate set size: ", candidate_set.size());
                
                // Calculate quantiles for candidates
                calculateQuantiles(candidate_distances, step_stats.cand_lb_q25, step_stats.cand_lb_q50, step_stats.cand_lb_q75);
                calculateQuantiles(candidate_similarities, step_stats.cand_sim_q25, step_stats.cand_sim_q50, step_stats.cand_sim_q75);
                
                // Calculate neighbor similarity quantiles
                std::vector<float> neighbor_similarities;
                for (size_t j = 1; j <= size; j++) {
                    int neighbor_id = *(data + j);
                    char *neighbor_data = getDataByInternalId(neighbor_id);
                    char *current_data = getDataByInternalId(current_node_id);
                    
                    // Calculate similarity between current node and its neighbor
                    float neighbor_sim = 1.0f / (1.0f + fstdistfunc_(current_data, neighbor_data, dist_func_param_));
                    neighbor_similarities.push_back(neighbor_sim);
                }
                calculateQuantiles(neighbor_similarities, step_stats.nn_sim_q25, step_stats.nn_sim_q50, step_stats.nn_sim_q75);
                
                // Neighbors info
                // step_stats.neighbors_info.degree = step_stats.deg;
                step_stats.neighbors_info.updates_in_result_set = updates_in_result_set_this_step;
                step_stats.neighbors_info.updates_ratio = (size > 0) ? 
                    static_cast<float>(updates_in_result_set_this_step) / size : 0.0f;

                // Calculate std dev of neighbor distances to query
                if (!candidate_distances.empty()) {
                    float mean_dist = 0.0f;
                    for (float d : candidate_distances) mean_dist += d;
                    mean_dist /= candidate_distances.size();
                    
                    float variance = 0.0f;
                    for (float d : candidate_distances) {
                        float diff = d - mean_dist;
                        variance += diff * diff;
                    }
                    step_stats.neighbors_info.dist_std = std::sqrt(variance / candidate_distances.size());
                    
                    // Min distance
                    step_stats.neighbors_info.min_dist = *std::min_element(
                        candidate_distances.begin(), candidate_distances.end());
                } else {
                    step_stats.neighbors_info.dist_std = 0.0f;
                    step_stats.neighbors_info.min_dist = std::numeric_limits<float>::max();
                }

                // Reset counter for next step
                updates_in_result_set_this_step = 0;
                
                // A. Geometry/Distance features
                step_stats.sim_abs = 1.0f / (1.0f + (dist)); // Convert distance to similarity
                step_stats.dist_rel_k = (dist) / (top_candidates.empty() ? 1.0f : top_candidates.top().first);
                step_stats.dist = (dist);
                
                // Calculate delta_distance (distance change from previous step)
                if (step_number > 0) {
                    step_stats.delta_distance = prev_step_distance - dist;
                } else {
                    step_stats.delta_distance = 0;
                }
                prev_step_distance = dist;
                
                // Calculate deltas
                float current_ub_k = top_candidates.empty() ? std::numeric_limits<float>::max() : top_candidates.top().first;
                step_stats.delta_k = prev_ub_k - current_ub_k;
                prev_ub_k = current_ub_k;
                
                // Calculate best_distance_so_far (minimum distance in top_candidates)
                if (!top_candidates.empty()) {
                    auto temp_pq = top_candidates;
                    float best_dist = std::numeric_limits<float>::max();
                    while (!temp_pq.empty()) {
                        best_dist = std::min(best_dist, temp_pq.top().first);
                        temp_pq.pop();
                    }
                    step_stats.best_distance_so_far = best_dist;
                } else {
                    step_stats.best_distance_so_far = std::numeric_limits<float>::max();
                }
                
                // Calculate gap_top (UB_k - UB_1)
                if (top_candidates.size() >= 2) {
                    // Need to extract both top and bottom from priority queue
                    std::vector<std::pair<dist_t, tableint>> temp_candidates;
                    while (!top_candidates.empty()) {
                        temp_candidates.push_back(top_candidates.top());
                        top_candidates.pop();
                    }
                    step_stats.gap_top = temp_candidates[0].first - temp_candidates.back().first;
                    // Restore priority queue
                    for (const auto& pair : temp_candidates) {
                        top_candidates.push(pair);
                    }
                } else {
                    step_stats.gap_top = 0.0f;
                }
                
                step_stats.gap_delta = prev_gap_top - step_stats.gap_top;
                prev_gap_top = step_stats.gap_top;
                
                // Directional features (if we have vector data)
                if (step_number > 0 && !direction_vectors.empty()) {
                    // Calculate direction vector from previous to current node
                    std::vector<float> current_direction;
                    char *prev_data = getDataByInternalId(direction_vectors.back().data()[0]); // Previous node data
                    char *curr_data = getDataByInternalId(current_node_id);
                    
                    // Calculate direction vector (simplified - using first few dimensions)
                    int dim = std::min(static_cast<int>(*((size_t *) dist_func_param_)), 128); // Limit to first 128 dimensions for efficiency
                    float* prev_float_data = reinterpret_cast<float*>(prev_data);
                    float* curr_float_data = reinterpret_cast<float*>(curr_data);
                    for (int i = 0; i < dim; ++i) {
                        current_direction.push_back(curr_float_data[i] - prev_float_data[i]);
                    }
                    current_direction.push_back(static_cast<float>(current_node_id)); // Store node ID for reference
                    direction_vectors.push_back(current_direction);
                    
                    // Calculate direction cosine
                    if (direction_vectors.size() >= 2) {
                        step_stats.dir_cos = calculateDirectionCosine(
                            direction_vectors[direction_vectors.size()-2].data(),
                            direction_vectors.back().data(),
                            dim
                        );
                    }
                    
                    // Calculate direction variance
                    step_stats.dir_var_w = calculateDirectionVariance(direction_vectors, direction_window, dim);
                } else {
                    step_stats.dir_cos = 0.0f;
                    step_stats.dir_var_w = 0.0f;
                    // Initialize direction vector
                    std::vector<float> initial_direction;
                    char *curr_data = getDataByInternalId(current_node_id);
                    int dim = std::min(static_cast<int>(*((size_t *) dist_func_param_)), 128);
                    float* curr_float_data = reinterpret_cast<float*>(curr_data);
                    for (int i = 0; i < dim; ++i) {
                        initial_direction.push_back(curr_float_data[i]);
                    }
                    initial_direction.push_back(static_cast<float>(current_node_id));
                    direction_vectors.push_back(initial_direction);
                }
                
                // B. Bounds/Certainty features
                step_stats.lb_min = candidate_set.empty() ? current_ub_k : std::get<0>(candidate_set.top());
                step_stats.lb_gap = current_ub_k - step_stats.lb_min;
                
                if (step_number == 0) {
                    initial_lb_gap = step_stats.lb_gap;
                }
                step_stats.lb_tight_ratio = initial_lb_gap > 1e-8f ? step_stats.lb_gap / initial_lb_gap : 0.0f;
                step_stats.lb_delta = prev_lb_gap - step_stats.lb_gap;
                prev_lb_gap = step_stats.lb_gap;
                
                // C. Search State features
                step_stats.heap_size = candidate_set.size();
                step_stats.visited_ratio = static_cast<float>(calculated_set.size()) / cur_element_count;
                
                // Calculate top-k spread
                if (top_candidates.size() >= 2) {
                    std::vector<dist_t> topk_distances;
                    std::vector<std::pair<dist_t, tableint>> temp_candidates;
                    while (!top_candidates.empty()) {
                        temp_candidates.push_back(top_candidates.top());
                        top_candidates.pop();
                    }
                    for (const auto& pair : temp_candidates) {
                        topk_distances.push_back(pair.first);
                    }
                    // Restore priority queue
                    for (const auto& pair : temp_candidates) {
                        top_candidates.push(pair);
                    }
                    
                    // Calculate standard deviation
                    float mean = 0.0f;
                    for (float dist : topk_distances) {
                        mean += dist;
                    }
                    mean /= topk_distances.size();
                    
                    float variance = 0.0f;
                    for (float dist : topk_distances) {
                        float diff = dist - mean;
                        variance += diff * diff;
                    }
                    step_stats.topk_spread = std::sqrt(variance / topk_distances.size());
                } else {
                    step_stats.topk_spread = 0.0f;
                }
                
                // Check for improvement
                step_stats.improve_flag = (step_stats.delta_k > 1e-8f);
                if (step_stats.improve_flag) {
                    stale_length = 0;
                } else {
                    stale_length++;
                }
                step_stats.stale_len = stale_length;
                
                // D. Graph/Structure features
                step_stats.local_cc = calculateLocalClusteringCoefficient(current_node_id, 10);
                step_stats.frontier_overlap = calculateFrontierOverlap(new_neighbors, calculated_set);
                
                // E. Cost/Time features
                step_stats.cost_dist = step_cost_dist;
                step_stats.cost_heap = step_cost_heap;
                step_stats.cost_total = step_cost_dist + step_cost_heap;
                cumulative_cost += step_stats.cost_total;
                step_stats.cost_cum = cumulative_cost;
                step_stats.gain_per_cost = step_stats.delta_k / (step_stats.cost_total + 1e-8f);
                
                // Fourier cost encoding
                step_stats.fourier_cost = calculateFourierCostFeatures(cumulative_cost, fourier_freqs);

                step_stats.hop_count = hop_count;
                
                // Store step statistics
                result.step_stats.push_back(step_stats);
                
                step_number++;
            }
            
            // Copy final results
            result.top_candidates = top_candidates;
            
            visited_list_pool_->releaseVisitedList(vl);
            return result;
        }


        
        // host_node and phase are used only for pruning instrumentation
        // (Batch 2a-3). When phase >= 0 and pruning_counters_ is non-null, the
        // heuristic increments per-node counters on every good=false event:
        //   phase=0: host_self_insert[host_node]++ (R's own insertion)
        //   phase=1: host_repruning[host_node]++   (later re-prune)
        // and victim[candidate]++ in both phases. Default args = no logging.
        void getNeighborsByHeuristic2(
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        const size_t M,
        tableint host_node = std::numeric_limits<tableint>::max(),
        int phase = -1) {
            if (top_candidates.size() < M) {
                return;
            }

            const bool log_prune = (phase >= 0 && pruning_counters_ != nullptr);

            std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
            std::vector<std::pair<dist_t, tableint>> return_list;
            while (top_candidates.size() > 0) {
                queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
                top_candidates.pop();
            }

            while (queue_closest.size()) {
                if (return_list.size() >= M)
                    break;
                std::pair<dist_t, tableint> curent_pair = queue_closest.top();
                dist_t dist_to_query = -curent_pair.first;
                queue_closest.pop();
                bool good = true;

                for (std::pair<dist_t, tableint> second_pair : return_list) {
                    dist_t curdist =
                            fstdistfunc_(getDataByInternalId(second_pair.second),
                                         getDataByInternalId(curent_pair.second),
                                         dist_func_param_);;
                    if (curdist < dist_to_query) {
                        good = false;
                        break;
                    }
                }
                if (good) {
                    return_list.push_back(curent_pair);
                } else if (log_prune) {
                    // Log: host pruned this victim. Phase 0 = self-insert, 1 = repruning.
                    tableint victim = curent_pair.second;
                    if (phase == 0 && host_node < pruning_counters_->host_self_insert.size()) {
                        pruning_counters_->host_self_insert[host_node]++;
                    } else if (phase == 1 && host_node < pruning_counters_->host_repruning.size()) {
                        pruning_counters_->host_repruning[host_node]++;
                    }
                    if (victim < pruning_counters_->victim.size()) {
                        pruning_counters_->victim[victim]++;
                    }
                }
            }

            for (std::pair<dist_t, tableint> curent_pair : return_list) {
                top_candidates.emplace(-curent_pair.first, curent_pair.second);
            }
        }


        linklistsizeint *get_linklist0(tableint internal_id) const {
            return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
        };

        linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const {
            return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
        };

        linklistsizeint *get_linklist(tableint internal_id, int level) const {
            return (linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_);
        };

        linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const {
            return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
        };

        std::vector<labeltype> getBaseLayerNeighborLabels(labeltype label) const {
            std::vector<labeltype> labels;
            tableint internal_id = getInternalId(label);
            if (internal_id >= cur_element_count) return labels;

            linklistsizeint *ll = get_linklist0(internal_id);
            size_t size = getListCount(ll);
            tableint *data = (tableint *) (ll + 1);
            labels.reserve(size);
            for (size_t i = 0; i < size; ++i) {
                tableint neighbor_id = data[i];
                if (neighbor_id < cur_element_count) {
                    labels.push_back(getExternalLabel(neighbor_id));
                }
            }
            return labels;
        }

        tableint mutuallyConnectNewElement_timed(const void *data_point, tableint cur_c,
                                       std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        int level, bool isUpdate, bool caller_holds_cur_lock = true) {
            auto connect_start = std::chrono::high_resolution_clock::now();
            HNSW_LOG_DEBUG("[MUTUAL_CONNECT_TIMED] Starting mutual connection for cur_c=", cur_c, ", level=", level);
            
            size_t Mcurmax = level ? maxM_ : maxM0_;
            
            // Heuristic application timing
            auto heuristic_start = std::chrono::high_resolution_clock::now();
            // Phase 0 = self-insert: host is the newly-inserted node cur_c.
            getNeighborsByHeuristic2(top_candidates, M_, cur_c, /*phase=*/0);
            auto heuristic_end = std::chrono::high_resolution_clock::now();
            auto heuristic_duration = std::chrono::duration_cast<std::chrono::microseconds>(heuristic_end - heuristic_start);
            HNSW_LOG_DEBUG("[MUTUAL_CONNECT_TIMED] Heuristic application: ", heuristic_duration.count(), " us");
            
            if (top_candidates.size() > M_)
                throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

            std::vector<tableint> selectedNeighbors;
            selectedNeighbors.reserve(M_);
            while (top_candidates.size() > 0) {
                selectedNeighbors.push_back(top_candidates.top().second);
                top_candidates.pop();
            }

            tableint next_closest_entry_point = selectedNeighbors.back();
            HNSW_LOG_DEBUG("[MUTUAL_CONNECT_TIMED] Selected ", selectedNeighbors.size(), " neighbors, next_closest=", next_closest_entry_point);

            // Setting up connections from cur_c to neighbors
            auto setup_connections_start = std::chrono::high_resolution_clock::now();
            {
                // Parallel construction must not hold cur_c while acquiring
                // neighbor locks later in this function.  Publish cur_c's own
                // list under a short, standalone lock to prevent lock cycles.
                std::unique_lock<std::mutex> cur_lock(link_list_locks_[cur_c], std::defer_lock);
                if (!caller_holds_cur_lock)
                    cur_lock.lock();
                linklistsizeint *ll_cur;
                if (level == 0)
                    ll_cur = get_linklist0(cur_c);
                else
                    ll_cur = get_linklist(cur_c, level);

                tableint *data = (tableint *) (ll_cur + 1);
                const size_t existing_count = getListCount(ll_cur);
                if (!caller_holds_cur_lock && existing_count > 0 && !isUpdate) {
                    // cur_c can become reachable after its first reciprocal edge,
                    // allowing another insertion to publish an inbound edge before
                    // this thread reaches the next layer. Merge those edges instead
                    // of assuming an empty list or overwriting concurrent work.
                    std::priority_queue<std::pair<dist_t, tableint>,
                                        std::vector<std::pair<dist_t, tableint>>, CompareByFirst> merged;
                    std::unordered_set<tableint> seen;
                    auto add_merged = [&](tableint node) {
                        if (node != cur_c && seen.insert(node).second) {
                            merged.emplace(fstdistfunc_(data_point, getDataByInternalId(node),
                                                       dist_func_param_), node);
                        }
                    };
                    for (size_t idx = 0; idx < existing_count; ++idx)
                        add_merged(data[idx]);
                    for (tableint node : selectedNeighbors)
                        add_merged(node);
                    getNeighborsByHeuristic2(merged, Mcurmax, cur_c, /*phase=*/1);
                    size_t out = 0;
                    while (!merged.empty()) {
                        data[out++] = merged.top().second;
                        merged.pop();
                    }
                    setListCount(ll_cur, out);
                } else {
                    if (existing_count && !isUpdate)
                        throw std::runtime_error("The newly inserted element should have blank link list");
                    setListCount(ll_cur, selectedNeighbors.size());
                    for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                        if (level > element_levels_[selectedNeighbors[idx]])
                            throw std::runtime_error("Trying to make a link on a non-existent level");
                        data[idx] = selectedNeighbors[idx];
                    }
                }
            }
            auto setup_connections_end = std::chrono::high_resolution_clock::now();
            auto setup_connections_duration = std::chrono::duration_cast<std::chrono::microseconds>(setup_connections_end - setup_connections_start);
            HNSW_LOG_DEBUG("[MUTUAL_CONNECT_TIMED] Setup connections from cur_c: ", setup_connections_duration.count(), " us");

            // Setting up connections from neighbors to cur_c
            auto mutual_setup_start = std::chrono::high_resolution_clock::now();
            int total_distance_calculations = 0;
            long total_distance_time = 0;
            long total_lock_time = 0;
            long total_heuristic_time = 0;
            
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                auto neighbor_start = std::chrono::high_resolution_clock::now();
                HNSW_LOG_DEBUG("[MUTUAL_CONNECT_TIMED] Processing neighbor ", idx, "/", selectedNeighbors.size(), 
                         ": ", selectedNeighbors[idx]);

                // Lock acquisition timing
                auto lock_start = std::chrono::high_resolution_clock::now();
                std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);
                auto lock_end = std::chrono::high_resolution_clock::now();
                auto lock_duration = std::chrono::duration_cast<std::chrono::microseconds>(lock_end - lock_start);
                total_lock_time += lock_duration.count();

                linklistsizeint *ll_other;
                if (level == 0)
                    ll_other = get_linklist0(selectedNeighbors[idx]);
                else
                    ll_other = get_linklist(selectedNeighbors[idx], level);

                size_t sz_link_list_other = getListCount(ll_other);
                HNSW_LOG_DEBUG("[MUTUAL_CONNECT_TIMED]   Neighbor ", selectedNeighbors[idx], " has ", sz_link_list_other, " existing connections");

                if (sz_link_list_other > Mcurmax)
                    throw std::runtime_error("Bad value of sz_link_list_other");
                if (selectedNeighbors[idx] == cur_c)
                    throw std::runtime_error("Trying to connect an element to itself");
                if (level > element_levels_[selectedNeighbors[idx]]) {
                    throw std::runtime_error("Trying to make a link on a non-existent level");
                }

                tableint *data = (tableint *) (ll_other + 1);

                bool is_cur_c_present = false;
                // Parallel insertions can race to publish the same reciprocal
                // edge even when this is not an update operation.
                for (size_t j = 0; j < sz_link_list_other; j++) {
                    if (data[j] == cur_c) {
                        is_cur_c_present = true;
                        break;
                    }
                }

                // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
                if (!is_cur_c_present) {
                    if (sz_link_list_other < Mcurmax) {
                        data[sz_link_list_other] = cur_c;
                        setListCount(ll_other, sz_link_list_other + 1);
                        HNSW_LOG_DEBUG("[MUTUAL_CONNECT_TIMED]   Added cur_c to neighbor's list (simple case)");
                    } else {
                        HNSW_LOG_DEBUG("[MUTUAL_CONNECT_TIMED]   Neighbor's list is full, running heuristic");
                        
                        // finding the "weakest" element to replace it with the new one
                        auto dist_calc_start = std::chrono::high_resolution_clock::now();
                        dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                                    dist_func_param_);
                        auto dist_calc_end = std::chrono::high_resolution_clock::now();
                        auto dist_calc_duration = std::chrono::duration_cast<std::chrono::microseconds>(dist_calc_end - dist_calc_start);
                        total_distance_time += dist_calc_duration.count();
                        total_distance_calculations++;
                        
                        // Heuristic:
                        auto neighbor_heuristic_start = std::chrono::high_resolution_clock::now();
                        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                        candidates.emplace(d_max, cur_c);

                        for (size_t j = 0; j < sz_link_list_other; j++) {
                            auto inner_dist_calc_start = std::chrono::high_resolution_clock::now();
                            candidates.emplace(
                                    fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),
                                                 dist_func_param_), data[j]);
                            auto inner_dist_calc_end = std::chrono::high_resolution_clock::now();
                            auto inner_dist_calc_duration = std::chrono::duration_cast<std::chrono::microseconds>(inner_dist_calc_end - inner_dist_calc_start);
                            total_distance_time += inner_dist_calc_duration.count();
                            total_distance_calculations++;
                        }

                        // Phase 1 = re-prune triggered by new inbound edge: host
                        // is the neighbor whose list overflowed.
                        getNeighborsByHeuristic2(candidates, Mcurmax,
                                                 selectedNeighbors[idx], /*phase=*/1);
                        auto neighbor_heuristic_end = std::chrono::high_resolution_clock::now();
                        auto neighbor_heuristic_duration = std::chrono::duration_cast<std::chrono::microseconds>(neighbor_heuristic_end - neighbor_heuristic_start);
                        total_heuristic_time += neighbor_heuristic_duration.count();

                        int indx = 0;
                        while (candidates.size() > 0) {
                            data[indx] = candidates.top().second;
                            candidates.pop();
                            indx++;
                        }

                        setListCount(ll_other, indx);
                        HNSW_LOG_DEBUG("[MUTUAL_CONNECT_TIMED]   Heuristic completed, kept ", indx, " connections");
                    }
                } else {
                    HNSW_LOG_DEBUG("[MUTUAL_CONNECT_TIMED]   cur_c already present in neighbor's list, skipping");
                }
                
                auto neighbor_end = std::chrono::high_resolution_clock::now();
                auto neighbor_duration = std::chrono::duration_cast<std::chrono::microseconds>(neighbor_end - neighbor_start);
                HNSW_LOG_DEBUG("[MUTUAL_CONNECT_TIMED]   Neighbor ", selectedNeighbors[idx], " processed in ", neighbor_duration.count(), " us");
            }
            
            auto mutual_setup_end = std::chrono::high_resolution_clock::now();
            auto mutual_setup_duration = std::chrono::duration_cast<std::chrono::microseconds>(mutual_setup_end - mutual_setup_start);
            
            auto connect_end = std::chrono::high_resolution_clock::now();
            auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(connect_end - connect_start);
            
            // Summary statistics
            HNSW_LOG_DEBUG("[MUTUAL_CONNECT_TIMED] ========== MUTUAL CONNECTION SUMMARY ==========");
            HNSW_LOG_DEBUG("[MUTUAL_CONNECT_TIMED] Total connection time: ", total_duration.count(), " us");
            HNSW_LOG_DEBUG("[MUTUAL_CONNECT_TIMED] Breakdown:");
            HNSW_LOG_DEBUG("[MUTUAL_CONNECT_TIMED]   - Initial heuristic: ", heuristic_duration.count(), " us (", 
                     (100.0 * heuristic_duration.count() / total_duration.count()), "%)");
            HNSW_LOG_DEBUG("[MUTUAL_CONNECT_TIMED]   - Setup connections from cur_c: ", setup_connections_duration.count(), " us (", 
                     (100.0 * setup_connections_duration.count() / total_duration.count()), "%)");
            HNSW_LOG_DEBUG("[MUTUAL_CONNECT_TIMED]   - Mutual setup: ", mutual_setup_duration.count(), " us (", 
                     (100.0 * mutual_setup_duration.count() / total_duration.count()), "%)");
            HNSW_LOG_DEBUG("[MUTUAL_CONNECT_TIMED] Mutual setup breakdown:");
            HNSW_LOG_DEBUG("[MUTUAL_CONNECT_TIMED]   - Total lock time: ", total_lock_time, " us (", 
                     (100.0 * total_lock_time / mutual_setup_duration.count()), "%)");
            HNSW_LOG_DEBUG("[MUTUAL_CONNECT_TIMED]   - Total distance calculation time: ", total_distance_time, " us (", 
                     (100.0 * total_distance_time / mutual_setup_duration.count()), "%)");
            HNSW_LOG_DEBUG("[MUTUAL_CONNECT_TIMED]   - Total heuristic time: ", total_heuristic_time, " us (", 
                     (100.0 * total_heuristic_time / mutual_setup_duration.count()), "%)");
            HNSW_LOG_DEBUG("[MUTUAL_CONNECT_TIMED] Statistics:");
            HNSW_LOG_DEBUG("[MUTUAL_CONNECT_TIMED]   - Total neighbors processed: ", selectedNeighbors.size());
            HNSW_LOG_DEBUG("[MUTUAL_CONNECT_TIMED]   - Total distance calculations: ", total_distance_calculations);
            HNSW_LOG_DEBUG("[MUTUAL_CONNECT_TIMED]   - Average distance calc time: ", (total_distance_calculations > 0 ? total_distance_time / total_distance_calculations : 0), " us");
            HNSW_LOG_DEBUG("[MUTUAL_CONNECT_TIMED] ================================================");

            return next_closest_entry_point;
        }

        tableint mutuallyConnectNewElement(const void *data_point, tableint cur_c,
                                       std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        int level, bool isUpdate) {
            size_t Mcurmax = level ? maxM_ : maxM0_;
            getNeighborsByHeuristic2(top_candidates, M_);
            if (top_candidates.size() > M_)
                throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

            std::vector<tableint> selectedNeighbors;
            selectedNeighbors.reserve(M_);
            while (top_candidates.size() > 0) {
                selectedNeighbors.push_back(top_candidates.top().second);
                top_candidates.pop();
            }

            tableint next_closest_entry_point = selectedNeighbors.back();

            {
                linklistsizeint *ll_cur;
                if (level == 0)
                    ll_cur = get_linklist0(cur_c);
                else
                    ll_cur = get_linklist(cur_c, level);

                if (*ll_cur && !isUpdate) {
                    throw std::runtime_error("The newly inserted element should have blank link list");
                }
                setListCount(ll_cur,selectedNeighbors.size());
                tableint *data = (tableint *) (ll_cur + 1);
                for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                    if (data[idx] && !isUpdate)
                        throw std::runtime_error("Possible memory corruption");
                    if (level > element_levels_[selectedNeighbors[idx]])
                        throw std::runtime_error("Trying to make a link on a non-existent level");

                    data[idx] = selectedNeighbors[idx];

                }
            }

            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {

                std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

                linklistsizeint *ll_other;
                if (level == 0)
                    ll_other = get_linklist0(selectedNeighbors[idx]);
                else
                    ll_other = get_linklist(selectedNeighbors[idx], level);

                size_t sz_link_list_other = getListCount(ll_other);

                if (sz_link_list_other > Mcurmax)
                    throw std::runtime_error("Bad value of sz_link_list_other");
                if (selectedNeighbors[idx] == cur_c)
                    throw std::runtime_error("Trying to connect an element to itself");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                tableint *data = (tableint *) (ll_other + 1);

                bool is_cur_c_present = false;
                if (isUpdate) {
                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        if (data[j] == cur_c) {
                            is_cur_c_present = true;
                            break;
                        }
                    }
                }

                // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
                if (!is_cur_c_present) {
                    if (sz_link_list_other < Mcurmax) {
                        data[sz_link_list_other] = cur_c;
                        setListCount(ll_other, sz_link_list_other + 1);
                    } else {
                        // finding the "weakest" element to replace it with the new one
                        dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                                    dist_func_param_);
                        // Heuristic:
                        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                        candidates.emplace(d_max, cur_c);

                        for (size_t j = 0; j < sz_link_list_other; j++) {
                            candidates.emplace(
                                    fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),
                                                 dist_func_param_), data[j]);
                        }

                        getNeighborsByHeuristic2(candidates, Mcurmax);

                        int indx = 0;
                        while (candidates.size() > 0) {
                            data[indx] = candidates.top().second;
                            candidates.pop();
                            indx++;
                        }

                        setListCount(ll_other, indx);
                        // Nearest K:
                        /*int indx = -1;
                        for (int j = 0; j < sz_link_list_other; j++) {
                            dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                            if (d > d_max) {
                                indx = j;
                                d_max = d;
                            }
                        }
                        if (indx >= 0) {
                            data[indx] = cur_c;
                        } */
                    }
                }
            }

            return next_closest_entry_point;
        }

        std::mutex global;
        size_t ef_;

        void setEf(size_t ef) {
            ef_ = ef;
        }

        // Pruning instrumentation (Batch 2a-3). Allocate per-node counters; the
        // heuristic will increment them on every good=false event.
        // Must be called BEFORE any addPoint*.
        void enablePruningInstrumentation(size_t N) {
            if (pruning_counters_ != nullptr) {
                delete pruning_counters_;
            }
            pruning_counters_ = new PruningCounters();
            pruning_counters_->host_self_insert.assign(N, 0);
            pruning_counters_->host_repruning.assign(N, 0);
            pruning_counters_->victim.assign(N, 0);
        }

        bool isPruningInstrumentationEnabled() const {
            return pruning_counters_ != nullptr;
        }

        // Dump per-node pruning counts to CSV. Caller must have called
        // enablePruningInstrumentation before building.
        void dumpPruningCounts(const char* path) const {
            if (pruning_counters_ == nullptr) return;
            std::ofstream f(path);
            if (!f.is_open()) return;
            f << "node_id,pruned_as_host_self_insert,pruned_as_host_repruning,"
                 "pruned_as_host_total,pruned_as_victim_total\n";
            const auto& si = pruning_counters_->host_self_insert;
            const auto& rp = pruning_counters_->host_repruning;
            const auto& vt = pruning_counters_->victim;
            size_t n = cur_element_count;
            for (size_t i = 0; i < n; ++i) {
                int64_t self_ins = (i < si.size()) ? si[i] : 0;
                int64_t repr    = (i < rp.size()) ? rp[i] : 0;
                int64_t vic     = (i < vt.size()) ? vt[i] : 0;
                f << static_cast<unsigned long long>(getExternalLabel(static_cast<tableint>(i))) << ','
                  << self_ins << ',' << repr << ',' << (self_ins + repr) << ','
                  << vic << '\n';
            }
            f.close();
        }


        std::priority_queue<std::pair<dist_t, tableint>> searchKnnInternal(void *query_data, int k) {
            std::priority_queue<std::pair<dist_t, tableint  >> top_candidates;
            if (cur_element_count == 0) return top_candidates;
            tableint currObj = enterpoint_node_;
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

            for (size_t level = maxlevel_; level > 0; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    int *data;
                    data = (int *) get_linklist(currObj,level);
                    int size = getListCount(data);
                    tableint *datal = (tableint *) (data + 1);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                        dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }

            if (num_deleted_) {
                std::priority_queue<std::pair<dist_t, tableint  >> top_candidates1=searchBaseLayerST<true>(currObj, query_data,
                                                                                                           ef_);
                top_candidates.swap(top_candidates1);
            }
            else{
                std::priority_queue<std::pair<dist_t, tableint  >> top_candidates1=searchBaseLayerST<false>(currObj, query_data,
                                                                                                            ef_);
                top_candidates.swap(top_candidates1);
            }

            while (top_candidates.size() > k) {
                top_candidates.pop();
            }
            return top_candidates;
        };

        void resizeIndex(size_t new_max_elements){
            if (new_max_elements<cur_element_count)
                throw std::runtime_error("Cannot resize, max element is less than the current number of elements");


            delete visited_list_pool_;
            visited_list_pool_ = new VisitedListPool(1, new_max_elements);


            element_levels_.resize(new_max_elements);

            std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

            // Reallocate base layer
            char * data_level0_memory_new = (char *) realloc(data_level0_memory_, new_max_elements * size_data_per_element_);
            if (data_level0_memory_new == nullptr)
                throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
            data_level0_memory_ = data_level0_memory_new;

            // Reallocate all other layers
            char ** linkLists_new = (char **) realloc(linkLists_, sizeof(void *) * new_max_elements);
            if (linkLists_new == nullptr)
                throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
            linkLists_ = linkLists_new;

            max_elements_ = new_max_elements;
        }

        void saveIndex(const std::string &location) {
            std::ofstream output(location, std::ios::binary);
            std::streampos position;

            writeBinaryPOD(output, offsetLevel0_);
            writeBinaryPOD(output, max_elements_);
            writeBinaryPOD(output, cur_element_count);
            writeBinaryPOD(output, size_data_per_element_);
            writeBinaryPOD(output, label_offset_);
            writeBinaryPOD(output, offsetData_);
            writeBinaryPOD(output, maxlevel_);
            writeBinaryPOD(output, enterpoint_node_);
            writeBinaryPOD(output, maxM_);

            writeBinaryPOD(output, maxM0_);
            writeBinaryPOD(output, M_);
            writeBinaryPOD(output, mult_);
            writeBinaryPOD(output, ef_construction_);

            output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

            for (size_t i = 0; i < cur_element_count; i++) {
                unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
                writeBinaryPOD(output, linkListSize);
                if (linkListSize)
                    output.write(linkLists_[i], linkListSize);
            }
            output.close();
        }

        void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i=0) {
            std::ifstream input(location, std::ios::binary);

            if (!input.is_open())
                throw std::runtime_error("Cannot open file");

            // get file size:
            input.seekg(0,input.end);
            std::streampos total_filesize=input.tellg();
            input.seekg(0,input.beg);

            readBinaryPOD(input, offsetLevel0_);
            readBinaryPOD(input, max_elements_);
            readBinaryPOD(input, cur_element_count);

            size_t max_elements = max_elements_i;
            if(max_elements < cur_element_count)
                max_elements = max_elements_;
            max_elements_ = max_elements;
            readBinaryPOD(input, size_data_per_element_);
            readBinaryPOD(input, label_offset_);
            readBinaryPOD(input, offsetData_);
            readBinaryPOD(input, maxlevel_);
            readBinaryPOD(input, enterpoint_node_);

            readBinaryPOD(input, maxM_);
            readBinaryPOD(input, maxM0_);
            readBinaryPOD(input, M_);
            readBinaryPOD(input, mult_);
            readBinaryPOD(input, ef_construction_);


            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();

            auto pos=input.tellg();


            /// Optional - check if index is ok:

            input.seekg(cur_element_count * size_data_per_element_,input.cur);
            for (size_t i = 0; i < cur_element_count; i++) {
                if(input.tellg() < 0 || input.tellg()>=total_filesize){
                    throw std::runtime_error("Index seems to be corrupted or unsupported");
                }

                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize != 0) {
                    input.seekg(linkListSize,input.cur);
                }
            }

            // throw exception if it either corrupted or old index
            if(input.tellg()!=total_filesize)
                throw std::runtime_error("Index seems to be corrupted or unsupported");

            input.clear();

            /// Optional check end

            input.seekg(pos,input.beg);

            data_level0_memory_ = (char *) malloc(max_elements * size_data_per_element_);
            if (data_level0_memory_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
            input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

            size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

            size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
            std::vector<std::mutex>(max_elements).swap(link_list_locks_);
            std::vector<std::mutex>(max_update_element_locks).swap(link_list_update_locks_);

            visited_list_pool_ = new VisitedListPool(1, max_elements);

            linkLists_ = (char **) malloc(sizeof(void *) * max_elements);
            if (linkLists_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
            element_levels_ = std::vector<int>(max_elements);
            revSize_ = 1.0 / mult_;
            ef_ = 10;
            for (size_t i = 0; i < cur_element_count; i++) {
                label_lookup_[getExternalLabel(i)]=i;
                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize == 0) {
                    element_levels_[i] = 0;

                    linkLists_[i] = nullptr;
                } else {
                    element_levels_[i] = linkListSize / size_links_per_element_;
                    linkLists_[i] = (char *) malloc(linkListSize);
                    if (linkLists_[i] == nullptr)
                        throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                    input.read(linkLists_[i], linkListSize);
                }
            }

            for (size_t i = 0; i < cur_element_count; i++) {
                if(isMarkedDeleted(i))
                    num_deleted_ += 1;
            }

            input.close();

            return;
        }

        template<typename data_t>
        std::vector<data_t> getDataByLabel(labeltype label) const
        {
            tableint label_c;
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
                throw std::runtime_error("Label not found");
            }
            label_c = search->second;

            char* data_ptrv = getDataByInternalId(label_c);
            size_t dim = *((size_t *) dist_func_param_);
            std::vector<data_t> data;
            data_t* data_ptr = (data_t*) data_ptrv;
            for (int i = 0; i < dim; i++) {
                data.push_back(*data_ptr);
                data_ptr += 1;
            }
            return data;
        }

        static const unsigned char DELETE_MARK = 0x01;
        // static const unsigned char REUSE_MARK = 0x10;
        /**
         * Marks an element with the given label deleted, does NOT really change the current graph.
         * @param label
         */
        void markDelete(labeltype label)
        {
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end()) {
                throw std::runtime_error("Label not found");
            }
            tableint internalId = search->second;
            markDeletedInternal(internalId);
        }

        /**
         * Uses the first 8 bits of the memory for the linked list to store the mark,
         * whereas maxM0_ has to be limited to the lower 24 bits, however, still large enough in almost all cases.
         * @param internalId
         */
        void markDeletedInternal(tableint internalId) {
            assert(internalId < cur_element_count);
            if (!isMarkedDeleted(internalId))
            {
                unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
                *ll_cur |= DELETE_MARK;
                num_deleted_ += 1;
            }
            else
            {
                throw std::runtime_error("The requested to delete element is already deleted");
            }
        }

        /**
         * Remove the deleted mark of the node, does NOT really change the current graph.
         * @param label
         */
        void unmarkDelete(labeltype label)
        {
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end()) {
                throw std::runtime_error("Label not found");
            }
            tableint internalId = search->second;
            unmarkDeletedInternal(internalId);
        }

        /**
         * Remove the deleted mark of the node.
         * @param internalId
         */
        void unmarkDeletedInternal(tableint internalId) {
            assert(internalId < cur_element_count);
            if (isMarkedDeleted(internalId))
            {
                unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
                *ll_cur &= ~DELETE_MARK;
                num_deleted_ -= 1;
            }
            else
            {
                throw std::runtime_error("The requested to undelete element is not deleted");
            }
        }

        /**
         * Checks the first 8 bits of the memory to see if the element is marked deleted.
         * @param internalId
         * @return
         */
        bool isMarkedDeleted(tableint internalId) const {
            unsigned char *ll_cur = ((unsigned char*)get_linklist0(internalId))+2;
            return *ll_cur & DELETE_MARK;
        }

        unsigned short int getListCount(linklistsizeint * ptr) const {
            return *((unsigned short int *)ptr);
        }

        void setListCount(linklistsizeint * ptr, unsigned short int size) const {
            *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
        }

        void addPoint(const void *data_point, labeltype label) {
            addPoint(data_point, label,-1);
        }

        void updatePoint(const void *dataPoint, tableint internalId, float updateNeighborProbability) {
            // update the feature vector associated with existing point with new vector
            memcpy(getDataByInternalId(internalId), dataPoint, data_size_);

            int maxLevelCopy = maxlevel_;
            tableint entryPointCopy = enterpoint_node_;
            // If point to be updated is entry point and graph just contains single element then just return.
            if (entryPointCopy == internalId && cur_element_count == 1)
                return;

            int elemLevel = element_levels_[internalId];
            std::uniform_real_distribution<float> distribution(0.0, 1.0);
            for (int layer = 0; layer <= elemLevel; layer++) {
                std::unordered_set<tableint> sCand;
                std::unordered_set<tableint> sNeigh;
                std::vector<tableint> listOneHop = getConnectionsWithLock(internalId, layer);
                if (listOneHop.size() == 0)
                    continue;

                sCand.insert(internalId);

                for (auto&& elOneHop : listOneHop) {
                    sCand.insert(elOneHop);

                    if (distribution(update_probability_generator_) > updateNeighborProbability)
                        continue;

                    sNeigh.insert(elOneHop);

                    std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
                    for (auto&& elTwoHop : listTwoHop) {
                        sCand.insert(elTwoHop);
                    }
                }

                for (auto&& neigh : sNeigh) {
                    // if (neigh == internalId)
                    //     continue;

                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    size_t size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() - 1; // sCand guaranteed to have size >= 1
                    size_t elementsToKeep = std::min(ef_construction_, size);
                    for (auto&& cand : sCand) {
                        if (cand == neigh)
                            continue;

                        dist_t distance = fstdistfunc_(getDataByInternalId(neigh), getDataByInternalId(cand), dist_func_param_);
                        if (candidates.size() < elementsToKeep) {
                            candidates.emplace(distance, cand);
                        } else {
                            if (distance < candidates.top().first) {
                                candidates.pop();
                                candidates.emplace(distance, cand);
                            }
                        }
                    }

                    // Retrieve neighbours using heuristic and set connections.
                    getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);

                    {
                        std::unique_lock <std::mutex> lock(link_list_locks_[neigh]);
                        linklistsizeint *ll_cur;
                        ll_cur = get_linklist_at_level(neigh, layer);
                        size_t candSize = candidates.size();
                        setListCount(ll_cur, candSize);
                        tableint *data = (tableint *) (ll_cur + 1);
                        for (size_t idx = 0; idx < candSize; idx++) {
                            data[idx] = candidates.top().second;
                            candidates.pop();
                        }
                    }
                }
            }

            repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
        };

        void repairConnectionsForUpdate(const void *dataPoint, tableint entryPointInternalId, tableint dataPointInternalId, int dataPointLevel, int maxLevel) {
            tableint currObj = entryPointInternalId;
            if (dataPointLevel < maxLevel) {
                dist_t curdist = fstdistfunc_(dataPoint, getDataByInternalId(currObj), dist_func_param_);
                for (int level = maxLevel; level > dataPointLevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int *data;
                        std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                        data = get_linklist_at_level(currObj,level);
                        int size = getListCount(data);
                        tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
#endif
                        for (int i = 0; i < size; i++) {
#ifdef USE_SSE
                            _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
#endif
                            tableint cand = datal[i];
                            dist_t d = fstdistfunc_(dataPoint, getDataByInternalId(cand), dist_func_param_);
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            if (dataPointLevel > maxLevel)
                throw std::runtime_error("Level of item to be updated cannot be bigger than max level");

            for (int level = dataPointLevel; level >= 0; level--) {
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> topCandidates = searchBaseLayer(
                        currObj, dataPoint, level);

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> filteredTopCandidates;
                while (topCandidates.size() > 0) {
                    if (topCandidates.top().second != dataPointInternalId)
                        filteredTopCandidates.push(topCandidates.top());

                    topCandidates.pop();
                }

                // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where `topCandidates` could just contains entry point itself.
                // To prevent self loops, the `topCandidates` is filtered and thus can be empty.
                if (filteredTopCandidates.size() > 0) {
                    bool epDeleted = isMarkedDeleted(entryPointInternalId);
                    if (epDeleted) {
                        filteredTopCandidates.emplace(fstdistfunc_(dataPoint, getDataByInternalId(entryPointInternalId), dist_func_param_), entryPointInternalId);
                        if (filteredTopCandidates.size() > ef_construction_)
                            filteredTopCandidates.pop();
                    }

                    currObj = mutuallyConnectNewElement_timed(dataPoint, dataPointInternalId, filteredTopCandidates, level, true);
                }
            }
        }

        std::vector<tableint> getConnectionsWithLock(tableint internalId, int level) {
            std::unique_lock <std::mutex> lock(link_list_locks_[internalId]);
            unsigned int *data = get_linklist_at_level(internalId, level);
            int size = getListCount(data);
            std::vector<tableint> result(size);
            tableint *ll = (tableint *) (data + 1);
            memcpy(result.data(), ll,size * sizeof(tableint));
            return result;
        };

        tableint addPoint(const void *data_point, labeltype label, int level) {

            tableint cur_c = 0;
            {
                // Checking if the element with the same label already exists
                // if so, updating it *instead* of creating a new element.
                std::unique_lock <std::mutex> templock_curr(cur_element_count_guard_);
                auto search = label_lookup_.find(label);
                if (search != label_lookup_.end()) {
                    tableint existingInternalId = search->second;
                    templock_curr.unlock();

                    std::unique_lock <std::mutex> lock_el_update(link_list_update_locks_[(existingInternalId & (max_update_element_locks - 1))]);

                    if (isMarkedDeleted(existingInternalId)) {
                        unmarkDeletedInternal(existingInternalId);
                    }
                    updatePoint(data_point, existingInternalId, 1.0);
                    
                    return existingInternalId;
                }

                if (cur_element_count >= max_elements_) {
                    throw std::runtime_error("The number of elements exceeds the specified limit");
                };

                cur_c = cur_element_count;
                cur_element_count++;
                label_lookup_[label] = cur_c;
            }

            // Take update lock to prevent race conditions on an element with insertion/update at the same time.
            std::unique_lock <std::mutex> lock_el_update(link_list_update_locks_[(cur_c & (max_update_element_locks - 1))]);
            std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
            int curlevel = getRandomLevel(mult_);
            if (level > 0)
                curlevel = level;

            element_levels_[cur_c] = curlevel;


            std::unique_lock <std::mutex> templock(global);
            int maxlevelcopy = maxlevel_;
            if (curlevel <= maxlevelcopy)
                templock.unlock();
            tableint currObj = enterpoint_node_;
            tableint enterpoint_copy = enterpoint_node_;


            memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

            // Initialisation of the dat and labela
            memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
            memcpy(getDataByInternalId(cur_c), data_point, data_size_);


            if (curlevel) {
                linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
                if (linkLists_[cur_c] == nullptr)
                    throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
                memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
            }

            if ((signed)currObj != -1) { // not the first object

                if (curlevel < maxlevelcopy) {

                    dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                    for (int level = maxlevelcopy; level > curlevel; level--) {


                        bool changed = true;
                        while (changed) {
                            changed = false;
                            unsigned int *data;
                            std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                            data = get_linklist(currObj,level);
                            int size = getListCount(data);

                            tableint *datal = (tableint *) (data + 1);
                            for (int i = 0; i < size; i++) {
                                tableint cand = datal[i];
                                if (cand < 0 || cand > max_elements_)
                                    throw std::runtime_error("cand error");
                                dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                                if (d < curdist) {
                                    curdist = d;
                                    currObj = cand;
                                    changed = true;
                                }
                            }
                        }
                    }
                }

                bool epDeleted = isMarkedDeleted(enterpoint_copy);
                for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                    if (level > maxlevelcopy || level < 0)  // possible?
                        throw std::runtime_error("Level error");

                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                            currObj, data_point, level);
                    if (epDeleted) {
                        top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();
                    }
                    currObj = mutuallyConnectNewElement_timed(data_point, cur_c, top_candidates, level, false);
                }


            } else {
                // Do nothing for the first element
                enterpoint_node_ = 0;
                maxlevel_ = curlevel;

            }

            //Releasing lock for the maximum level
            if (curlevel > maxlevelcopy) {
                enterpoint_node_ = cur_c;
                maxlevel_ = curlevel;
            }
            return cur_c;
        };


        tableint addPointPlain(const void *data_point, labeltype label, int level = -1) {

            tableint cur_c = 0;
            {
                // Checking if the element with the same label already exists
                // if so, updating it *instead* of creating a new element.
                std::unique_lock <std::mutex> templock_curr(cur_element_count_guard_);
                auto search = label_lookup_.find(label);
                if (search != label_lookup_.end()) {
                    tableint existingInternalId = search->second;
                    templock_curr.unlock();

                    std::unique_lock <std::mutex> lock_el_update(link_list_update_locks_[(existingInternalId & (max_update_element_locks - 1))]);

                    if (isMarkedDeleted(existingInternalId)) {
                        unmarkDeletedInternal(existingInternalId);
                    }
                    updatePoint(data_point, existingInternalId, 1.0);
                    
                    return existingInternalId;
                }

                if (cur_element_count >= max_elements_) {
                    throw std::runtime_error("The number of elements exceeds the specified limit");
                };

                cur_c = cur_element_count;
                cur_element_count++;
                label_lookup_[label] = cur_c;
            }

            // Take update lock to prevent race conditions on an element with insertion/update at the same time.
            std::unique_lock <std::mutex> lock_el_update(link_list_update_locks_[(cur_c & (max_update_element_locks - 1))]);
            std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
            // int curlevel = getRandomLevel(mult_);
            int curlevel = 0;
            if (level > 0)
                curlevel = level;

            element_levels_[cur_c] = curlevel;


            std::unique_lock <std::mutex> templock(global);
            int maxlevelcopy = maxlevel_;
            if (curlevel <= maxlevelcopy)
                templock.unlock();
            tableint currObj = enterpoint_node_;
            tableint enterpoint_copy = enterpoint_node_;


            memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

            // Initialisation of the dat and labela
            memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
            memcpy(getDataByInternalId(cur_c), data_point, data_size_);
            // if(cur_c != label){
            //     std::cout << cur_c << " ," << label << std::endl;
            // }


            if (curlevel) {
                linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
                if (linkLists_[cur_c] == nullptr)
                    throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
                memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
            }

            if ((signed)currObj != -1) { // not the first object

                if (curlevel < maxlevelcopy) {

                    dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                    for (int level = maxlevelcopy; level > curlevel; level--) {


                        bool changed = true;
                        while (changed) {
                            changed = false;
                            unsigned int *data;
                            std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                            data = get_linklist(currObj,level);
                            int size = getListCount(data);

                            tableint *datal = (tableint *) (data + 1);
                            for (int i = 0; i < size; i++) {
                                tableint cand = datal[i];
                                if (cand < 0 || cand > max_elements_)
                                    throw std::runtime_error("cand error");
                                dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                                if (d < curdist) {
                                    curdist = d;
                                    currObj = cand;
                                    changed = true;
                                }
                            }
                        }
                    }
                }

                bool epDeleted = isMarkedDeleted(enterpoint_copy);
                for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                    if (level > maxlevelcopy || level < 0)  // possible?
                        throw std::runtime_error("Level error");

                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                            currObj, data_point, level);
                    if (epDeleted) {
                        top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();
                    }
                    currObj = mutuallyConnectNewElement_timed(data_point, cur_c, top_candidates, level, false);
                }


            } else {
                // Do nothing for the first element
                enterpoint_node_ = 0;
                maxlevel_ = curlevel;

            }

            //Releasing lock for the maximum level
            if (curlevel > maxlevelcopy) {
                enterpoint_node_ = cur_c;
                maxlevel_ = curlevel;
            }
            return cur_c;
        };

        tableint addPointPlain_with_id(const void *data_point, labeltype label, int internal_id) {
            // std::cerr << "[addPointPlain_with_id] Starting with label=" << label << ", internal_id=" << internal_id << std::endl;

            tableint cur_c = internal_id;
            {
                // Checking if the element with the same label already exists
                // if so, updating it *instead* of creating a new element.
                std::unique_lock <std::mutex> templock_curr(cur_element_count_guard_);
                // std::cerr << "[addPointPlain_with_id] Current element count: " << cur_element_count << ", max_elements: " << max_elements_ << std::endl;
                
                if (cur_element_count >= max_elements_) {
                    // std::cerr << "[addPointPlain_with_id] ERROR: Exceeded max elements limit" << std::endl;
                    throw std::runtime_error("The number of elements exceeds the specified limit");
                };

                cur_element_count++;
                label_lookup_[label] = cur_c;
                // std::cerr << "[addPointPlain_with_id] Added to label_lookup: label=" << label << " -> internal_id=" << cur_c << std::endl;
            }

            // Take update lock to prevent race conditions on an element with insertion/update at the same time.
            std::unique_lock <std::mutex> lock_el_update(link_list_update_locks_[(cur_c & (max_update_element_locks - 1))]);
            std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
            // std::cerr << "[addPointPlain_with_id] Acquired locks for internal_id=" << cur_c << std::endl;
            
            // int curlevel = getRandomLevel(mult_);
            int curlevel = 0;
            // std::cerr << "[addPointPlain_with_id] Assigned level: " << curlevel << std::endl;

            element_levels_[cur_c] = curlevel;

            std::unique_lock <std::mutex> templock(global);
            int maxlevelcopy = maxlevel_;
            if (curlevel <= maxlevelcopy)
                templock.unlock();
            tableint currObj = enterpoint_node_;
            tableint enterpoint_copy = enterpoint_node_;
            // std::cerr << "[addPointPlain_with_id] Current maxlevel: " << maxlevelcopy << ", enterpoint: " << enterpoint_copy << std::endl;
            // tableint currObj = rand() % (cur_element_count - 1);
            // tableint enterpoint_copy = currObj;

            // std::cerr << "[addPointPlain_with_id] Clearing memory for internal_id=" << cur_c << std::endl;
            memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

            // Initialisation of the dat and labela
            // std::cerr << "[addPointPlain_with_id] Copying data and label" << std::endl;
            memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
            memcpy(getDataByInternalId(cur_c), data_point, data_size_);
            // std::cerr << "[addPointPlain_with_id] Data copied successfully" << std::endl;
            // if(cur_c != label){
            //     std::cout << cur_c << " ," << label << std::endl;
            // }

            if ((signed)currObj != -1) { // not the first object
                // std::cerr << "[addPointPlain_with_id] Not the first object, starting graph traversal" << std::endl;

                if (curlevel < maxlevelcopy) {
                    // std::cerr << "[addPointPlain_with_id] Current level < maxlevel, traversing higher levels" << std::endl;
                    dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                    // std::cerr << "[addPointPlain_with_id] Initial distance to current object: " << curdist << std::endl;
                    
                    for (int level = maxlevelcopy; level > curlevel; level--) {
                        // std::cerr << "[addPointPlain_with_id] Traversing level " << level << std::endl;

                        bool changed = true;
                        while (changed) {
                            changed = false;
                            unsigned int *data;
                            std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                            data = get_linklist(currObj,level);
                            int size = getListCount(data);
                            // std::cerr << "[addPointPlain_with_id] Level " << level << " has " << size << " neighbors" << std::endl;

                            tableint *datal = (tableint *) (data + 1);
                            for (int i = 0; i < size; i++) {
                                tableint cand = datal[i];
                                if (cand < 0 || cand > max_elements_)
                                    throw std::runtime_error("cand error");
                                dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                                // std::cerr << "[addPointPlain_with_id] Candidate " << cand << " distance: " << d << " (current best: " << curdist << ")" << std::endl;
                                if (d < curdist) {
                                    curdist = d;
                                    currObj = cand;
                                    changed = true;
                                    // std::cerr << "[addPointPlain_with_id] Found better candidate: " << cand << " with distance: " << d << std::endl;
                                }
                            }
                        }
                    }
                }

                bool epDeleted = isMarkedDeleted(enterpoint_copy);
                // std::cerr << "[addPointPlain_with_id] Entry point deleted: " << epDeleted << std::endl;
                
                for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                    // std::cerr << "[addPointPlain_with_id] Processing level " << level << " for connections" << std::endl;
                    if (level > maxlevelcopy || level < 0)  // possible?
                        throw std::runtime_error("Level error");

                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                            currObj, data_point, level);
                    // std::cerr << "[addPointPlain_with_id] Found " << top_candidates.size() << " candidates at level " << level << std::endl;
                    
                    if (epDeleted) {
                        dist_t ep_dist = fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_);
                        // std::cerr << "[addPointPlain_with_id] Adding deleted entry point with distance: " << ep_dist << std::endl;
                        top_candidates.emplace(ep_dist, enterpoint_copy);
                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();
                    }
                    currObj = mutuallyConnectNewElement_timed(data_point, cur_c, top_candidates, level, false);
                    // std::cerr << "[addPointPlain_with_id] Level " << level << " connections established, new currObj: " << currObj << std::endl;
                }

            } else {
                // Do nothing for the first element
                // std::cerr << "[addPointPlain_with_id] First element in graph" << std::endl;
                enterpoint_node_ = 0;
                maxlevel_ = curlevel;
            }

            //Releasing lock for the maximum level
            if (curlevel > maxlevelcopy) {
                // std::cerr << "[addPointPlain_with_id] New element has higher level, updating entry point and maxlevel" << std::endl;
                enterpoint_node_ = cur_c;
                maxlevel_ = curlevel;
            }
            
            // std::cerr << "[addPointPlain_with_id] Successfully completed for internal_id=" << cur_c << std::endl;
            return cur_c;
        };




/** Prepare fixed IDs and deterministic levels before parallel graph mutation. */
int preparePointWithId(const void *data_point, labeltype label, int internal_id) {
    const tableint cur_c = static_cast<tableint>(internal_id);
    if (cur_c >= max_elements_)
        throw std::runtime_error("preparePointWithId: internal ID exceeds capacity");
    if (cur_element_count != cur_c)
        throw std::runtime_error("preparePointWithId: IDs must be prepared in order");

    const int curlevel = getRandomLevelSeeded(mult_);
    element_levels_[cur_c] = curlevel;
    memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_,
           0, size_data_per_element_);
    memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
    memcpy(getDataByInternalId(cur_c), data_point, data_size_);
    if (curlevel) {
        linkLists_[cur_c] = static_cast<char *>(malloc(size_links_per_element_ * curlevel + 1));
        if (linkLists_[cur_c] == nullptr)
            throw std::runtime_error("preparePointWithId: upper-level allocation failed");
        memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
    }
    label_lookup_[label] = cur_c;
    ++cur_element_count;
    return curlevel;
}

void initializePreparedEntryPoint(tableint internal_id = 0) {
    if (cur_element_count == 0 || internal_id >= cur_element_count)
        throw std::runtime_error("initializePreparedEntryPoint: point is not prepared");
    std::lock_guard<std::mutex> lock(global);
    enterpoint_node_ = internal_id;
    maxlevel_ = element_levels_[internal_id];
}

/** Connect one prepared point.  No path holds two node locks at once. */
tableint connectPreparedPoint(tableint cur_c) {
    if (cur_c == 0)
        return cur_c;
    if (cur_c >= cur_element_count)
        throw std::runtime_error("connectPreparedPoint: point is not prepared");

    const void *data_point = getDataByInternalId(cur_c);
    const int curlevel = element_levels_[cur_c];
    std::unique_lock<std::mutex> entry_lock(global);
    const int maxlevelcopy = maxlevel_;
    const tableint enterpoint_copy = enterpoint_node_;
    tableint currObj = enterpoint_copy;
    if (curlevel <= maxlevelcopy)
        entry_lock.unlock();

    if (curlevel < maxlevelcopy) {
        dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
        for (int level = maxlevelcopy; level > curlevel; --level) {
            bool changed = true;
            while (changed) {
                changed = false;
                std::unique_lock<std::mutex> node_lock(link_list_locks_[currObj]);
                unsigned int *links = get_linklist(currObj, level);
                const int size = getListCount(links);
                tableint *neighbors = reinterpret_cast<tableint *>(links + 1);
                for (int i = 0; i < size; ++i) {
                    const tableint cand = neighbors[i];
                    const dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }
    }

    for (int level = std::min(curlevel, maxlevelcopy); level >= 0; --level) {
        auto top_candidates = searchBaseLayer(currObj, data_point, level);
        std::priority_queue<std::pair<dist_t, tableint>,
                            std::vector<std::pair<dist_t, tableint>>, CompareByFirst> filtered;
        while (!top_candidates.empty()) {
            const auto candidate = top_candidates.top();
            top_candidates.pop();
            if (candidate.second != cur_c && element_levels_[candidate.second] >= level)
                filtered.push(candidate);
        }
        if (filtered.empty())
            continue;
        currObj = mutuallyConnectNewElement_timed(
            data_point, cur_c, filtered, level, false,
            /*caller_holds_cur_lock=*/false);
    }

    // A hierarchy-raising point retains the global entry lock until here.
    if (curlevel > maxlevelcopy) {
        enterpoint_node_ = cur_c;
        maxlevel_ = curlevel;
    }
    return cur_c;
}

tableint addPointPlain_with_id_timed(const void *data_point, labeltype label, int internal_id
                                    ) {
    auto total_start = std::chrono::high_resolution_clock::now();
    
    tableint cur_c = internal_id;
    
    // Step 1: Label lookup and element count update
    auto label_lookup_start = std::chrono::high_resolution_clock::now();
    HNSW_LOG_DEBUG("[TIMED_INSERT] Step 1: Label lookup and element count update...");
    {
        std::unique_lock <std::mutex> templock_curr(cur_element_count_guard_);
        
        if (cur_element_count >= max_elements_) {
            throw std::runtime_error("The number of elements exceeds the specified limit");
        };

        cur_element_count++;
        label_lookup_[label] = cur_c;
    }
    auto label_lookup_end = std::chrono::high_resolution_clock::now();
    auto label_lookup_duration = std::chrono::duration_cast<std::chrono::microseconds>(label_lookup_end - label_lookup_start);
    // g_label_lookup_time_us += label_lookup_duration.count();
    HNSW_LOG_DEBUG("[TIMED_INSERT]   Step 1 completed in ", label_lookup_duration.count(), " us");

    // Step 2: Acquire element locks
    auto lock_start = std::chrono::high_resolution_clock::now();
    HNSW_LOG_DEBUG("[TIMED_INSERT] Step 2: Acquiring element locks...");
    std::unique_lock <std::mutex> lock_el_update(link_list_update_locks_[(cur_c & (max_update_element_locks - 1))]);
    std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
    auto lock_end = std::chrono::high_resolution_clock::now();
    auto lock_duration = std::chrono::duration_cast<std::chrono::microseconds>(lock_end - lock_start);
    // g_lock_acquisition_time_us += lock_duration.count();
    HNSW_LOG_DEBUG("[TIMED_INSERT]   Step 2 completed in ", lock_duration.count(), " us");
    
    int curlevel = 0;
    element_levels_[cur_c] = curlevel;

    // Step 3: Global lock for max level
    auto global_lock_start = std::chrono::high_resolution_clock::now();
    HNSW_LOG_DEBUG("[TIMED_INSERT] Step 3: Acquiring global lock for max level...");
    std::unique_lock <std::mutex> templock(global);
    int maxlevelcopy = maxlevel_;
    if (curlevel <= maxlevelcopy)
        templock.unlock();
    tableint currObj = enterpoint_node_;
    tableint enterpoint_copy = enterpoint_node_;
    auto global_lock_end = std::chrono::high_resolution_clock::now();
    auto global_lock_duration = std::chrono::duration_cast<std::chrono::microseconds>(global_lock_end - global_lock_start);
    // g_lock_acquisition_time_us += global_lock_duration.count();
    HNSW_LOG_DEBUG("[TIMED_INSERT]   Step 3 completed in ", global_lock_duration.count(), " us");
    HNSW_LOG_DEBUG("[TIMED_INSERT]   Current maxlevel: ", maxlevelcopy, ", enterpoint: ", enterpoint_copy);

    // Step 4: Memory initialization
    auto mem_init_start = std::chrono::high_resolution_clock::now();
    HNSW_LOG_DEBUG("[TIMED_INSERT] Step 4: Memory initialization...");
    memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);
    memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
    memcpy(getDataByInternalId(cur_c), data_point, data_size_);
    auto mem_init_end = std::chrono::high_resolution_clock::now();
    auto mem_init_duration = std::chrono::duration_cast<std::chrono::microseconds>(mem_init_end - mem_init_start);
    // g_memory_init_time_us += mem_init_duration.count();
    HNSW_LOG_DEBUG("[TIMED_INSERT]   Step 4 completed in ", mem_init_duration.count(), " us");
    HNSW_LOG_DEBUG("[TIMED_INSERT]   Data size: ", data_size_, " bytes");

    if ((signed)currObj != -1) { // not the first object
        HNSW_LOG_DEBUG("[TIMED_INSERT] Not the first object, starting graph traversal...");
        
        // Step 5: Traverse higher levels (if any)
        auto traversal_start = std::chrono::high_resolution_clock::now();
        HNSW_LOG_DEBUG("[TIMED_INSERT] Step 5: Traversing higher levels...");
        if (curlevel < maxlevelcopy) {
            dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
            // g_distance_calc_count++;
            
            for (int level = maxlevelcopy; level > curlevel; level--) {
                bool changed = true;
                int level_iterations = 0;
                while (changed) {
                    changed = false;
                    level_iterations++;
                    unsigned int *data;
                    std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                    data = get_linklist(currObj,level);
                    int size = getListCount(data);

                    tableint *datal = (tableint *) (data + 1);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                        dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                        // g_distance_calc_count++;
                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
                HNSW_LOG_DEBUG("[TIMED_INSERT]     Level ", level, " completed in ", level_iterations, " iterations");
            } 
            }
            else {
                HNSW_LOG_DEBUG("[TIMED_INSERT]   Current level (", curlevel, ") >= maxlevel (", maxlevelcopy, "), skipping higher level traversal");
            }
            auto traversal_end = std::chrono::high_resolution_clock::now();
            auto traversal_duration = std::chrono::duration_cast<std::chrono::microseconds>(traversal_end - traversal_start);
            // g_higher_level_traversal_time_us += traversal_duration.count();
            HNSW_LOG_DEBUG("[TIMED_INSERT]   Step 5 completed in ", traversal_duration.count(), " us");

            // Step 6: Search base layer and establish connections
            auto search_base_start = std::chrono::high_resolution_clock::now();
            HNSW_LOG_DEBUG("[TIMED_INSERT] Step 6: Searching base layer and establishing connections...");
            bool epDeleted = isMarkedDeleted(enterpoint_copy);
            // HNSW_LOG_DEBUG("[TIMED_INSERT]   Entry point deleted: ", (epDeleted ? "true" : "false"));
            
            for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                if (level > maxlevelcopy || level < 0)
                    throw std::runtime_error("Level error");

                // HNSW_LOG_DEBUG("[TIMED_INSERT]     Processing level ", level, " for connections");
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, typename HierarchicalNSW<dist_t>::CompareByFirst> top_candidates = searchBaseLayer_timed(
                        currObj, data_point, level);
                // HNSW_LOG_DEBUG("[TIMED_INSERT]       Found ", top_candidates.size(), " candidates at level ", level);
                
                if (epDeleted) {
                    dist_t ep_dist = fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_);
                    // g_distance_calc_count++;
                    top_candidates.emplace(ep_dist, enterpoint_copy);
                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();
                    // HNSW_LOG_DEBUG("[TIMED_INSERT]       Added deleted entry point with distance: ", ep_dist);
                }
                
                auto mutual_connect_start = std::chrono::high_resolution_clock::now();
                currObj = mutuallyConnectNewElement_timed(data_point, cur_c, top_candidates, level, false);
                auto mutual_connect_end = std::chrono::high_resolution_clock::now();
                auto mutual_connect_duration = std::chrono::duration_cast<std::chrono::microseconds>(mutual_connect_end - mutual_connect_start);
                // g_mutual_connect_time_us += mutual_connect_duration.count();
                HNSW_LOG_DEBUG("[TIMED_INSERT]       Level ", level, " connections established in ", mutual_connect_duration.count(), " us, new currObj: ", currObj);
            }
            auto search_base_end = std::chrono::high_resolution_clock::now();
            auto search_base_duration = std::chrono::duration_cast<std::chrono::microseconds>(search_base_end - search_base_start);
            // g_search_base_time_us += search_base_duration.count();
            HNSW_LOG_DEBUG("[TIMED_INSERT]   Step 6 completed in ", search_base_duration.count(), " us");   

    }
    else {
        // First element
        HNSW_LOG_DEBUG("[TIMED_INSERT] First element in graph");
        enterpoint_node_ = 0;
        maxlevel_ = curlevel;
    }

    // Update entry point if needed
    if (curlevel > maxlevelcopy) {
        HNSW_LOG_DEBUG("[TIMED_INSERT] New element has higher level, updating entry point and maxlevel");
        enterpoint_node_ = cur_c;
        maxlevel_ = curlevel;
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start);
    // g_total_time_us += total_duration.count();
    
    // HNSW_LOG_DEBUG("[TIMED_INSERT] Insertion completed successfully for internal_id=", cur_c);
    // HNSW_LOG_DEBUG("[TIMED_INSERT] Total insertion time: ", total_duration.count(), " us");
    // HNSW_LOG_DEBUG("[TIMED_INSERT] Breakdown:");
    // HNSW_LOG_DEBUG("[TIMED_INSERT]   Label lookup: ", label_lookup_duration.count(), " us (", (100.0 * label_lookup_duration.count() / total_duration.count()), "%)");
    // HNSW_LOG_DEBUG("[TIMED_INSERT]   Lock acquisition: ", (lock_duration.count() + global_lock_duration.count()), " us (", (100.0 * (lock_duration.count() + global_lock_duration.count()) / total_duration.count()), "%)");
    // HNSW_LOG_DEBUG("[TIMED_INSERT]   Memory initialization: ", mem_init_duration.count(), " us (", (100.0 * mem_init_duration.count() / total_duration.count()), "%)");
    // if ((signed)currObj != -1) {
    //     HNSW_LOG_DEBUG("[TIMED_INSERT]   Higher level traversal: ", traversal_duration.count(), " us (", (100.0 * traversal_duration.count() / total_duration.count()), "%)");
    //     HNSW_LOG_DEBUG("[TIMED_INSERT]   Search base layer: ", search_base_duration.count(), " us (", (100.0 * search_base_duration.count() / total_duration.count()), "%)");
    // }
    // HNSW_LOG_DEBUG("[TIMED_INSERT] =========================================");
    
    return cur_c;
};

tableint addPoint_with_id_timed(const void *data_point, labeltype label, int internal_id
                                    ) {
    auto total_start = std::chrono::high_resolution_clock::now();
    
    tableint cur_c = internal_id;
    
    // Step 1: Label lookup and element count update
    auto label_lookup_start = std::chrono::high_resolution_clock::now();
    HNSW_LOG_DEBUG("[TIMED_INSERT] Step 1: Label lookup and element count update...");
    {
        std::unique_lock <std::mutex> templock_curr(cur_element_count_guard_);
        
        if (cur_element_count >= max_elements_) {
            throw std::runtime_error("The number of elements exceeds the specified limit");
        };

        cur_element_count++;
        label_lookup_[label] = cur_c;
    }
    auto label_lookup_end = std::chrono::high_resolution_clock::now();
    auto label_lookup_duration = std::chrono::duration_cast<std::chrono::microseconds>(label_lookup_end - label_lookup_start);
    HNSW_LOG_DEBUG("[TIMED_INSERT]   Step 1 completed in ", label_lookup_duration.count(), " us");

    // Step 2: Acquire element locks
    auto lock_start = std::chrono::high_resolution_clock::now();
    HNSW_LOG_DEBUG("[TIMED_INSERT] Step 2: Acquiring element locks...");
    std::unique_lock <std::mutex> lock_el_update(link_list_update_locks_[(cur_c & (max_update_element_locks - 1))]);
    std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
    auto lock_end = std::chrono::high_resolution_clock::now();
    auto lock_duration = std::chrono::duration_cast<std::chrono::microseconds>(lock_end - lock_start);
    HNSW_LOG_DEBUG("[TIMED_INSERT]   Step 2 completed in ", lock_duration.count(), " us");
    
    // Layered insertion with reproducible RNG tied to level_generator_ seed.
    int curlevel = getRandomLevelSeeded(mult_);
    element_levels_[cur_c] = curlevel;
    HNSW_LOG_DEBUG("[TIMED_INSERT]   Assigned level: ", curlevel);

    // Step 3: Global lock for max level
    auto global_lock_start = std::chrono::high_resolution_clock::now();
    HNSW_LOG_DEBUG("[TIMED_INSERT] Step 3: Acquiring global lock for max level...");
    std::unique_lock <std::mutex> templock(global);
    int maxlevelcopy = maxlevel_;
    if (curlevel <= maxlevelcopy)
        templock.unlock();
    tableint currObj = enterpoint_node_;
    tableint enterpoint_copy = enterpoint_node_;
    auto global_lock_end = std::chrono::high_resolution_clock::now();
    auto global_lock_duration = std::chrono::duration_cast<std::chrono::microseconds>(global_lock_end - global_lock_start);
    HNSW_LOG_DEBUG("[TIMED_INSERT]   Step 3 completed in ", global_lock_duration.count(), " us");
    HNSW_LOG_DEBUG("[TIMED_INSERT]   Current maxlevel: ", maxlevelcopy, ", enterpoint: ", enterpoint_copy);

    // Step 4: Memory initialization
    auto mem_init_start = std::chrono::high_resolution_clock::now();
    HNSW_LOG_DEBUG("[TIMED_INSERT] Step 4: Memory initialization...");
    memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);
    memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
    memcpy(getDataByInternalId(cur_c), data_point, data_size_);
    auto mem_init_end = std::chrono::high_resolution_clock::now();
    auto mem_init_duration = std::chrono::duration_cast<std::chrono::microseconds>(mem_init_end - mem_init_start);
    HNSW_LOG_DEBUG("[TIMED_INSERT]   Step 4 completed in ", mem_init_duration.count(), " us");
    HNSW_LOG_DEBUG("[TIMED_INSERT]   Data size: ", data_size_, " bytes");

    if (curlevel) {
        auto level_alloc_start = std::chrono::high_resolution_clock::now();
        linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
        if (linkLists_[cur_c] == nullptr)
            throw std::runtime_error("Not enough memory: addPoint_with_id_timed failed to allocate linklist");
        memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
        auto level_alloc_end = std::chrono::high_resolution_clock::now();
        auto level_alloc_duration = std::chrono::duration_cast<std::chrono::microseconds>(level_alloc_end - level_alloc_start);
        HNSW_LOG_DEBUG("[TIMED_INSERT]   Upper-level linklist allocation completed in ", level_alloc_duration.count(), " us");
    }

    if ((signed)currObj != -1) { // not the first object
        HNSW_LOG_DEBUG("[TIMED_INSERT] Not the first object, starting graph traversal...");
        
        // Step 5: Traverse higher levels (if any)
        auto traversal_start = std::chrono::high_resolution_clock::now();
        HNSW_LOG_DEBUG("[TIMED_INSERT] Step 5: Traversing higher levels...");
        if (curlevel < maxlevelcopy) {
            dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
            
            for (int level = maxlevelcopy; level > curlevel; level--) {
                bool changed = true;
                int level_iterations = 0;
                while (changed) {
                    changed = false;
                    level_iterations++;
                    unsigned int *data;
                    std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                    data = get_linklist(currObj,level);
                    int size = getListCount(data);

                    tableint *datal = (tableint *) (data + 1);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                        dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
                HNSW_LOG_DEBUG("[TIMED_INSERT]     Level ", level, " completed in ", level_iterations, " iterations");
            } 
        } else {
            HNSW_LOG_DEBUG("[TIMED_INSERT]   Current level (", curlevel, ") >= maxlevel (", maxlevelcopy, "), skipping higher level traversal");
        }
        auto traversal_end = std::chrono::high_resolution_clock::now();
        auto traversal_duration = std::chrono::duration_cast<std::chrono::microseconds>(traversal_end - traversal_start);
        HNSW_LOG_DEBUG("[TIMED_INSERT]   Step 5 completed in ", traversal_duration.count(), " us");

        // Step 6: Search base layer and establish connections
        auto search_base_start = std::chrono::high_resolution_clock::now();
        HNSW_LOG_DEBUG("[TIMED_INSERT] Step 6: Searching base layer and establishing connections...");
        bool epDeleted = isMarkedDeleted(enterpoint_copy);
        
        for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
            if (level > maxlevelcopy || level < 0)
                throw std::runtime_error("Level error");

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, typename HierarchicalNSW<dist_t>::CompareByFirst> top_candidates = searchBaseLayer_timed(
                    currObj, data_point, level);
            
            if (epDeleted) {
                dist_t ep_dist = fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_);
                top_candidates.emplace(ep_dist, enterpoint_copy);
                if (top_candidates.size() > ef_construction_)
                    top_candidates.pop();
            }

            // Filter out invalid candidates that do not exist on this layer.
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, typename HierarchicalNSW<dist_t>::CompareByFirst> filtered_top_candidates;
            while (!top_candidates.empty()) {
                const auto& c = top_candidates.top();
                if (element_levels_[c.second] >= level) {
                    filtered_top_candidates.push(c);
                }
                top_candidates.pop();
            }
            if (filtered_top_candidates.empty()) {
                continue;
            }
            
            auto mutual_connect_start = std::chrono::high_resolution_clock::now();
            currObj = mutuallyConnectNewElement_timed(data_point, cur_c, filtered_top_candidates, level, false);
            auto mutual_connect_end = std::chrono::high_resolution_clock::now();
            auto mutual_connect_duration = std::chrono::duration_cast<std::chrono::microseconds>(mutual_connect_end - mutual_connect_start);
            HNSW_LOG_DEBUG("[TIMED_INSERT]       Level ", level, " connections established in ", mutual_connect_duration.count(), " us, new currObj: ", currObj);
        }
        auto search_base_end = std::chrono::high_resolution_clock::now();
        auto search_base_duration = std::chrono::duration_cast<std::chrono::microseconds>(search_base_end - search_base_start);
        HNSW_LOG_DEBUG("[TIMED_INSERT]   Step 6 completed in ", search_base_duration.count(), " us");   

    }
    else {
        // First element
        HNSW_LOG_DEBUG("[TIMED_INSERT] First element in graph");
        enterpoint_node_ = cur_c;
        maxlevel_ = curlevel;
    }

    // Update entry point if needed
    if (curlevel > maxlevelcopy) {
        HNSW_LOG_DEBUG("[TIMED_INSERT] New element has higher level, updating entry point and maxlevel");
        enterpoint_node_ = cur_c;
        maxlevel_ = curlevel;
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start);
    (void)total_duration;
    
    return cur_c;
};




        //max heap
        std::priority_queue<std::pair<dist_t, labeltype >>
        searchKnn(void *query_data, size_t k, int adaptive=0) const {
            
            std::priority_queue<std::pair<dist_t, labeltype >> result;
            if (cur_element_count == 0) return result;

            tableint currObj = enterpoint_node_;
#ifdef COUNT_DIST_TIME
            StopW stopw = StopW();
#endif
#ifndef ED2IP
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);
#else
            dist_t curdist = hnswlib::L2Sqr_by_IP(query_data, getDataByInternalId(enterpoint_node_), getExternalLabel(enterpoint_node_), dist_func_param_);
#endif
#ifdef COUNT_DIST_TIME
            adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
            adsampling::tot_dist_calculation ++;
            adsampling::tot_full_dist++;
            for (int level = maxlevel_; level > 0; level--) {
                
                bool changed = true;
                while (changed) {
                    
                    changed = false;
                    unsigned int *data;

                    data = (unsigned int *) get_linklist(currObj, level);
                    int size = getListCount(data);
                    metric_hops++;
                    metric_distance_computations+=size;
                    adsampling::tot_hops++;
                    tableint *datal = (tableint *) (data + 1);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                         else {
#ifdef COUNT_DIST_TIME
                            StopW stopw = StopW();
#endif
#ifndef ED2IP
            dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);
#else
            dist_t d = hnswlib::L2Sqr_by_IP(query_data, getDataByInternalId(cand), getExternalLabel(cand), dist_func_param_);
#endif
#ifdef COUNT_DIST_TIME
                            adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
#ifdef COUNT_DIMENSION
                            if (adaptive == 10)
                                adsampling::tot_dimension+= finger::D;
#endif
                            adsampling::tot_full_dist++;
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }
            
            //max heap
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>> top_candidates;
            // std::cerr << currObj << ",";
            
            if (num_deleted_) {
                top_candidates=searchBaseLayerST<true,true>(currObj, query_data, std::max(ef_, k));
            }
            else{
                top_candidates=searchBaseLayerST<false,true>(currObj, query_data, std::max(ef_, k));
            }

            //cerr << "search baselayer succeed!" << endl;
            while (top_candidates.size() > k) {
                top_candidates.pop();
            }
            while (top_candidates.size() > 0) {
                std::pair<dist_t, tableint> rez = top_candidates.top();
                result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
                top_candidates.pop();
            }
            return result;
        };



        std::priority_queue<std::pair<dist_t, labeltype >>
        searchKnnNDC(void *query_data, size_t k, int adaptive=0, Metric* metric = nullptr) const {
            
            std::priority_queue<std::pair<dist_t, labeltype >> result;
            if (cur_element_count == 0) return result;

            tableint currObj = enterpoint_node_;
#ifdef COUNT_DIST_TIME
            StopW stopw = StopW();
#endif
#ifndef ED2IP
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);
#else
            dist_t curdist = hnswlib::L2Sqr_by_IP(query_data, getDataByInternalId(enterpoint_node_), getExternalLabel(enterpoint_node_), dist_func_param_);
#endif
#ifdef COUNT_DIST_TIME
            adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
            adsampling::tot_dist_calculation ++;
            adsampling::tot_full_dist++;
            if (metric != nullptr)
                metric->hierarchical_ndc++;
            for (int level = maxlevel_; level > 0; level--) {
                
                bool changed = true;
                while (changed) {
                    
                    changed = false;
                    unsigned int *data;

                    data = (unsigned int *) get_linklist(currObj, level);
                    int size = getListCount(data);
                    metric_hops++;
                    metric_distance_computations+=size;
                    adsampling::tot_hops++;
                    tableint *datal = (tableint *) (data + 1);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                         else {
#ifdef COUNT_DIST_TIME
                            StopW stopw = StopW();
#endif
#ifndef ED2IP
            dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);
#else
            dist_t d = hnswlib::L2Sqr_by_IP(query_data, getDataByInternalId(cand), getExternalLabel(cand), dist_func_param_);
#endif
#ifdef COUNT_DIST_TIME
                            adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
#ifdef COUNT_DIMENSION
                            if (adaptive == 10)
                                adsampling::tot_dimension+= finger::D;
#endif
                            adsampling::tot_full_dist++;
                            if (metric != nullptr)
                                metric->hierarchical_ndc++;
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }
            
            //max heap
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>> top_candidates;
            // std::cerr << currObj << ",";
            
            if (num_deleted_) {
                top_candidates=searchBaseLayerST<true,true>(currObj, query_data, std::max(ef_, k), metric);
            }
            else{
                top_candidates=searchBaseLayerST<false,true>(currObj, query_data, std::max(ef_, k), metric);
            }

            //cerr << "search baselayer succeed!" << endl;
            while (top_candidates.size() > k) {
                top_candidates.pop();
            }
            while (top_candidates.size() > 0) {
                std::pair<dist_t, tableint> rez = top_candidates.top();
                result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
                top_candidates.pop();
            }
            return result;
        };

        // Same as searchKnnNDC, but takes ef as an explicit per-call argument
        // instead of reading the shared, unsynchronized ef_ member. Required
        // for any caller that shares one HierarchicalNSW instance across
        // #pragma omp parallel threads with per-thread/per-probe ef values
        // (e.g. the mode-13 binary-search collector) -- setEf() on a shared
        // index object races across concurrent threads.
        std::priority_queue<std::pair<dist_t, labeltype >>
        searchKnnNDCExplicitEf(void *query_data, size_t k, size_t ef, int adaptive=0, Metric* metric = nullptr) const {

            std::priority_queue<std::pair<dist_t, labeltype >> result;
            if (cur_element_count == 0) return result;

            tableint currObj = enterpoint_node_;
#ifdef COUNT_DIST_TIME
            StopW stopw = StopW();
#endif
#ifndef ED2IP
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);
#else
            dist_t curdist = hnswlib::L2Sqr_by_IP(query_data, getDataByInternalId(enterpoint_node_), getExternalLabel(enterpoint_node_), dist_func_param_);
#endif
#ifdef COUNT_DIST_TIME
            adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
            adsampling::tot_dist_calculation ++;
            adsampling::tot_full_dist++;
            if (metric != nullptr)
                metric->hierarchical_ndc++;
            for (int level = maxlevel_; level > 0; level--) {

                bool changed = true;
                while (changed) {

                    changed = false;
                    unsigned int *data;

                    data = (unsigned int *) get_linklist(currObj, level);
                    int size = getListCount(data);
                    metric_hops++;
                    metric_distance_computations+=size;
                    adsampling::tot_hops++;
                    tableint *datal = (tableint *) (data + 1);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                         else {
#ifdef COUNT_DIST_TIME
                            StopW stopw = StopW();
#endif
#ifndef ED2IP
            dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);
#else
            dist_t d = hnswlib::L2Sqr_by_IP(query_data, getDataByInternalId(cand), getExternalLabel(cand), dist_func_param_);
#endif
#ifdef COUNT_DIST_TIME
                            adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
#ifdef COUNT_DIMENSION
                            if (adaptive == 10)
                                adsampling::tot_dimension+= finger::D;
#endif
                            adsampling::tot_full_dist++;
                            if (metric != nullptr)
                                metric->hierarchical_ndc++;
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            //max heap
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>> top_candidates;

            if (num_deleted_) {
                top_candidates=searchBaseLayerST<true,true>(currObj, query_data, std::max(ef, k), metric);
            }
            else{
                top_candidates=searchBaseLayerST<false,true>(currObj, query_data, std::max(ef, k), metric);
            }

            while (top_candidates.size() > k) {
                top_candidates.pop();
            }
            while (top_candidates.size() > 0) {
                std::pair<dist_t, tableint> rez = top_candidates.top();
                result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
                top_candidates.pop();
            }
            return result;
        };

        // Same as searchKnnNDC, but additionally reports the query-specific
        // base-layer entry point (currObj after the upper-layer descent) via
        // entry_point_out, so callers can seed an external base-layer-only
        // search (e.g. a Python simulation over base_adj_*.bin) from the
        // same entry point the real multi-layer index would have used.
        std::priority_queue<std::pair<dist_t, labeltype >>
        searchKnnNDCWithEntry(void *query_data, size_t k, tableint* entry_point_out, int adaptive=0, Metric* metric = nullptr) const {

            std::priority_queue<std::pair<dist_t, labeltype >> result;
            if (cur_element_count == 0) return result;

            tableint currObj = enterpoint_node_;
#ifdef COUNT_DIST_TIME
            StopW stopw = StopW();
#endif
#ifndef ED2IP
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);
#else
            dist_t curdist = hnswlib::L2Sqr_by_IP(query_data, getDataByInternalId(enterpoint_node_), getExternalLabel(enterpoint_node_), dist_func_param_);
#endif
#ifdef COUNT_DIST_TIME
            adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
            adsampling::tot_dist_calculation ++;
            adsampling::tot_full_dist++;
            if (metric != nullptr)
                metric->hierarchical_ndc++;
            for (int level = maxlevel_; level > 0; level--) {

                bool changed = true;
                while (changed) {

                    changed = false;
                    unsigned int *data;

                    data = (unsigned int *) get_linklist(currObj, level);
                    int size = getListCount(data);
                    metric_hops++;
                    metric_distance_computations+=size;
                    adsampling::tot_hops++;
                    tableint *datal = (tableint *) (data + 1);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                         else {
#ifdef COUNT_DIST_TIME
                            StopW stopw = StopW();
#endif
#ifndef ED2IP
            dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);
#else
            dist_t d = hnswlib::L2Sqr_by_IP(query_data, getDataByInternalId(cand), getExternalLabel(cand), dist_func_param_);
#endif
#ifdef COUNT_DIST_TIME
                            adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
#ifdef COUNT_DIMENSION
                            if (adaptive == 10)
                                adsampling::tot_dimension+= finger::D;
#endif
                            adsampling::tot_full_dist++;
                            if (metric != nullptr)
                                metric->hierarchical_ndc++;
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            if (entry_point_out != nullptr)
                *entry_point_out = currObj;

            //max heap
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>> top_candidates;

            if (num_deleted_) {
                top_candidates=searchBaseLayerST<true,true>(currObj, query_data, std::max(ef_, k), metric);
            }
            else{
                top_candidates=searchBaseLayerST<false,true>(currObj, query_data, std::max(ef_, k), metric);
            }

            while (top_candidates.size() > k) {
                top_candidates.pop();
            }
            while (top_candidates.size() > 0) {
                std::pair<dist_t, tableint> rez = top_candidates.top();
                result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
                top_candidates.pop();
            }
            return result;
        };

        BridgeRescueSearchResult searchKnnBridgeRescue(
            void *query_data,
            size_t k,
            const BridgeRescueParams& params = BridgeRescueParams(),
            int adaptive = 0,
            Metric* metric = nullptr) const {
            (void)adaptive;

            BridgeRescueSearchResult result;
            if (cur_element_count == 0) return result;

            tableint currObj = enterpoint_node_;
#ifndef ED2IP
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);
#else
            dist_t curdist = hnswlib::L2Sqr_by_IP(query_data, getDataByInternalId(enterpoint_node_), getExternalLabel(enterpoint_node_), dist_func_param_);
#endif
            adsampling::tot_dist_calculation++;
            adsampling::tot_full_dist++;
            if (metric != nullptr) metric->hierarchical_ndc++;

            for (int level = maxlevel_; level > 0; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data = (unsigned int *) get_linklist(currObj, level);
                    int size = getListCount(data);
                    metric_hops++;
                    metric_distance_computations += size;
                    adsampling::tot_hops++;
                    tableint *datal = (tableint *) (data + 1);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_) {
                            throw std::runtime_error("cand error");
                        }
#ifndef ED2IP
                        dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);
#else
                        dist_t d = hnswlib::L2Sqr_by_IP(query_data, getDataByInternalId(cand), getExternalLabel(cand), dist_func_param_);
#endif
                        adsampling::tot_full_dist++;
                        if (metric != nullptr) metric->hierarchical_ndc++;
                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }

            BridgeRescueBaseLayerResult base_result;
            if (num_deleted_) {
                base_result = searchBaseLayerBridgeRescue<true, true>(
                    currObj, query_data, std::max(ef_, k), params, metric);
            } else {
                base_result = searchBaseLayerBridgeRescue<false, true>(
                    currObj, query_data, std::max(ef_, k), params, metric);
            }

            auto top_candidates = base_result.top_candidates;
            while (top_candidates.size() > k) {
                top_candidates.pop();
            }
            while (!top_candidates.empty()) {
                std::pair<dist_t, tableint> rez = top_candidates.top();
                result.neighbors.emplace(rez.first, getExternalLabel(rez.second));
                top_candidates.pop();
            }
            result.stats = base_result.stats;
            return result;
        }

        BridgeRescueSearchResult searchKnnRFBridgeRescue(
            void *query_data,
            size_t k,
            const BridgeRescueParams& params = BridgeRescueParams(),
            int adaptive = 0,
            Metric* metric = nullptr) const {
            (void)adaptive;

            BridgeRescueSearchResult result;
            if (cur_element_count == 0) return result;

            tableint currObj = enterpoint_node_;
#ifndef ED2IP
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);
#else
            dist_t curdist = hnswlib::L2Sqr_by_IP(query_data, getDataByInternalId(enterpoint_node_), getExternalLabel(enterpoint_node_), dist_func_param_);
#endif
            adsampling::tot_dist_calculation++;
            adsampling::tot_full_dist++;
            if (metric != nullptr) metric->hierarchical_ndc++;

            for (int level = maxlevel_; level > 0; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data = (unsigned int *) get_linklist(currObj, level);
                    int size = getListCount(data);
                    metric_hops++;
                    metric_distance_computations += size;
                    adsampling::tot_hops++;
                    tableint *datal = (tableint *) (data + 1);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_) {
                            throw std::runtime_error("cand error");
                        }
#ifndef ED2IP
                        dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);
#else
                        dist_t d = hnswlib::L2Sqr_by_IP(query_data, getDataByInternalId(cand), getExternalLabel(cand), dist_func_param_);
#endif
                        adsampling::tot_full_dist++;
                        if (metric != nullptr) metric->hierarchical_ndc++;
                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }

            BridgeRescueBaseLayerResult base_result;
            if (num_deleted_) {
                base_result = searchBaseLayerRFBridgeRescue<true, true>(
                    currObj, query_data, std::max(ef_, k), params, metric);
            } else {
                base_result = searchBaseLayerRFBridgeRescue<false, true>(
                    currObj, query_data, std::max(ef_, k), params, metric);
            }

            auto top_candidates = base_result.top_candidates;
            while (top_candidates.size() > k) {
                top_candidates.pop();
            }
            while (!top_candidates.empty()) {
                std::pair<dist_t, tableint> rez = top_candidates.top();
                result.neighbors.emplace(rez.first, getExternalLabel(rez.second));
                top_candidates.pop();
            }
            result.stats = base_result.stats;
            return result;
        }


        // Add a struct for data collection statistics
        struct DataCollectionStats {
            std::priority_queue<std::pair<dist_t, labeltype>> neighbors;
            int skipped_nodes;
            float entry_dist;
        };

        


        // Refactor searchKnnDataCollection to return DataCollectionStats
        DataCollectionStats searchKnnDataCollection(void *query_data, size_t k, int adaptive=0, Metric *metric = nullptr, size_t m = 5, size_t ef=10) const {
            DataCollectionStats stats;
            stats.skipped_nodes = 0;
            stats.entry_dist = 0.0f;
            std::priority_queue<std::pair<dist_t, labeltype>> &result = stats.neighbors;
            if (cur_element_count == 0) {
                HNSW_LOG_DEBUG("Graph is empty. Returning empty result.");
                return stats;
            }
            tableint currObj = enterpoint_node_;
#ifndef ED2IP
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);
#else
            dist_t curdist = hnswlib::L2Sqr_by_IP(query_data, getDataByInternalId(enterpoint_node_), getExternalLabel(enterpoint_node_), dist_func_param_);
#endif
            stats.entry_dist = curdist;
            // HNSW_LOG_INFO("Starting search from entry point: ", currObj, ", distance: ", curdist);
            adsampling::tot_dist_calculation ++;
            adsampling::tot_full_dist++;

            /* Traverse the graph from the top level to the base layer */ 
            for (int level = maxlevel_; level > 0; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data;
                    data = (unsigned int *) get_linklist(currObj, level);
                    int size = getListCount(data);
                    metric_hops++;
                    metric_distance_computations+=size;
                    adsampling::tot_hops++;
                    tableint *datal = (tableint *) (data + 1);
                    HNSW_LOG_DEBUG("Exploring level ", level, ", current node: ", currObj, ", neighbors: ", size);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                        else {
#ifndef ED2IP
                            dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);
#else
                            dist_t d = hnswlib::L2Sqr_by_IP(query_data, getDataByInternalId(cand), getExternalLabel(cand), dist_func_param_);
#endif
                            adsampling::tot_full_dist++;
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                                HNSW_LOG_DEBUG("Found closer node: ", cand, ", distance: ", d);
                            }
                        }
                    }
                }
            }
            // Use the updated searchBaseLayerDataCollection function to collect neighbors and skipped node count
            std::vector<std::pair<dist_t, int>> neighbors = searchBaseLayerDataCollection<false, true>(currObj, query_data, m, ef, metric, &stats.skipped_nodes);
            // HNSW_LOG_INFO("Collected ", neighbors.size(), " neighbors from base layer.");
            for (const auto &neighbor : neighbors) {
                dist_t distance = neighbor.first;
                int hop_count = neighbor.second;
                // Note: neighbor.second is hop_count, not internal_id. You may want to return internal_id in the tuple if needed.
                // Here, we just use hop_count as a placeholder for labeltype if needed.
                result.emplace(distance, hop_count);
            }
            // HNSW_LOG_INFO("Search completed. Returning top ", k, " neighbors.");
            return stats;
        }

        // Static variables for timing statistics
        // static int query_count = 0;
        // static auto start_time = std::chrono::high_resolution_clock::now();
        // static auto last_checkpoint_time = std::chrono::high_resolution_clock::now();
        
        // // Function to reset timing statistics
        // static void resetTimingStatistics() {
        //     query_count = 0;
        //     start_time = std::chrono::high_resolution_clock::now();
        //     last_checkpoint_time = start_time;
        //     HNSW_LOG_INFO("[TIMING] Timing statistics reset");
        // }
        
        EffectiveBranchingResult searchKnnDataCollectionWithBranching(void *query_data, size_t k, int adaptive=0, Metric *metric = nullptr, size_t m = 5, size_t ef=10, float lid_estimate=0.0f, float lid_r_k=0.0f, float prior_branching_factor=0.0f) const {
            // int query_count = 0;
            // auto start_time = std::chrono::high_resolution_clock::now();
            // auto last_checkpoint_time = start_time;

            EffectiveBranchingResult stats;
            // stats.skipped_nodes = 0;
            // stats.entry_dist = 0.0f;
            // std::priority_queue<std::pair<dist_t, labeltype>> &neighbors_queue = stats.neighbors;
            if (cur_element_count == 0) {
                HNSW_LOG_DEBUG("Graph is empty. Returning empty result.");
                return stats;
            }
            
            // Increment query count and check timing
            // query_count++;
            // auto current_time = std::chrono::high_resolution_clock::now();
            
            // Log timing every 500 queries
            // if (query_count % 500 == 0) {
            //     auto checkpoint_duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_checkpoint_time);
            //     auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
                
            //     HNSW_LOG_INFO("[TIMING] Processed ", query_count, " queries");
            //     HNSW_LOG_INFO("[TIMING] Last 500 queries took: ", checkpoint_duration.count(), " ms");
            //     HNSW_LOG_INFO("[TIMING] Total time so far: ", total_duration.count(), " ms");
            //     HNSW_LOG_INFO("[TIMING] Average time per query: ", total_duration.count() / query_count, " ms");
                
            //     last_checkpoint_time = current_time;
            // }
            tableint currObj = enterpoint_node_;
#ifndef ED2IP
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);
#else
            dist_t curdist = hnswlib::L2Sqr_by_IP(query_data, getDataByInternalId(enterpoint_node_), getExternalLabel(enterpoint_node_), dist_func_param_);
#endif
            // stats.entry_dist = curdist;
            // HNSW_LOG_INFO("Starting search from entry point: ", currObj, ", distance: ", curdist);
            adsampling::tot_dist_calculation ++;
            adsampling::tot_full_dist++;

            /* Traverse the graph from the top level to the base layer */ 
            for (int level = maxlevel_; level > 0; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data;
                    data = (unsigned int *) get_linklist(currObj, level);
                    int size = getListCount(data);
                    metric_hops++;
                    metric_distance_computations+=size;
                    adsampling::tot_hops++;
                    tableint *datal = (tableint *) (data + 1);
                    HNSW_LOG_DEBUG("Exploring level ", level, ", current node: ", currObj, ", neighbors: ", size);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                        else {
#ifndef ED2IP
                            dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);
#else
                            dist_t d = hnswlib::L2Sqr_by_IP(query_data, getDataByInternalId(cand), getExternalLabel(cand), dist_func_param_);
#endif
                            adsampling::tot_full_dist++;
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                                HNSW_LOG_DEBUG("Found closer node: ", cand, ", distance: ", d);
                            }
                        }
                    }
                }
            }
            // Use the updated searchBaseLayerDataCollectionWithBranching function to collect neighbors and branching factor analysis
            // EffectiveBranchingResult branching_result = searchBaseLayerDataCollectionWithBranching<false, true>(
                // currObj, query_data, m, ef, nullptr, nullptr, 16.0f, 0.3f, k = 10, 0.9f, 1.0f,  0.1f, 1.0f, lid_estimate, lid_r_k, 0.5f, true, prior_branching_factor);
            EffectiveBranchingResult branching_result = searchBaseLayerRouting<false, true>(currObj, query_data, m, ef, 10);
            // HNSW_LOG_INFO("Collected ", branching_result.neighbors.size(), " neighbors from base layer.");
            
            // Convert EffectiveBranchingResult neighbors to the format expected by DataCollectionStats
            // for (const auto &neighbor : branching_result.neighbors) {
            //     dist_t distance = neighbor.first;
            //     int hop_count = neighbor.second;
                
            //     // Convert hop_count to labeltype (using the hop_count as a simple identifier)
            //     // In practice, you might want to store the actual node ID or a different identifier
            //     labeltype neighbor_label = static_cast<labeltype>(hop_count);
                
            //     neighbors_queue.emplace(distance, neighbor_label);
            // }
            
            // Log branching factor analysis results
            // HNSW_LOG_INFO("Branching factor analysis results:");
            // HNSW_LOG_INFO("  - Effective branching factor: ", branching_result.effective_branching_factor);
            // HNSW_LOG_INFO("  - Novelty rate: ", branching_result.novelty_rate);
            // HNSW_LOG_INFO("  - Local clustering coefficient: ", branching_result.local_clustering_coeff);
            // HNSW_LOG_INFO("  - Replacement rate: ", branching_result.replacement_rate);
            // HNSW_LOG_INFO("  - Basin size estimate: ", branching_result.basin_size_estimate);
            // HNSW_LOG_INFO("  - Estimated path hops: ", branching_result.estimated_path_hops);
            
            if (branching_result.lid_estimate > 0) {
                // HNSW_LOG_INFO("  - LID estimate: ", branching_result.lid_estimate);
                // HNSW_LOG_INFO("  - Current best distance: ", branching_result.current_best_distance);
                // HNSW_LOG_INFO("  - Basin size (LID method): ", branching_result.basin_size_lid_method);
            }
            
            // HNSW_LOG_INFO("Search completed. Returning top ", k, " neighbors.");
            
            // Log final timing statistics when query count is a multiple of 500 or at the end
            
            // auto final_time = std::chrono::high_resolution_clock::now();
            // auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(final_time - start_time);
            
            // HNSW_LOG_INFO("[FINAL_TIMING] Total queries processed: ", query_count);
            // HNSW_LOG_INFO("[FINAL_TIMING] Total execution time: ", total_duration.count(), " ms");
            // HNSW_LOG_INFO("[FINAL_TIMING] Average time per query: ", total_duration.count() / query_count, " ms");
        
            
            return branching_result;
        }

        SearchWithStatsResult searchKnnDataCollectionWithStats(void *query_data, size_t k, int adaptive=0, Metric *metric = nullptr, size_t m = 50, size_t ef=10) const {
            SearchWithStatsResult stats;
            // stats.skipped_nodes = 0;
            // stats.entry_dist = 0.0f;
            // std::priority_queue<std::pair<dist_t, labeltype>> &result = stats.neighbors;
            if (cur_element_count == 0) {
                HNSW_LOG_INFO("Graph is empty. Returning empty result.");
                return stats;
        }
            tableint currObj = enterpoint_node_;
#ifndef ED2IP
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);
#else
            dist_t curdist = hnswlib::L2Sqr_by_IP(query_data, getDataByInternalId(enterpoint_node_), getExternalLabel(enterpoint_node_), dist_func_param_);
#endif
            // stats.entry_dist = curdist;
            // HNSW_LOG_INFO("Starting search from entry point: ", currObj, ", distance: ", curdist);
            adsampling::tot_dist_calculation ++;
            adsampling::tot_full_dist++;
            for (int level = maxlevel_; level > 0; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data;
                    data = (unsigned int *) get_linklist(currObj, level);
                    int size = getListCount(data);
                    metric_hops++;
                    metric_distance_computations+=size;
                    adsampling::tot_hops++;
                    tableint *datal = (tableint *) (data + 1);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                        else {
                            #ifndef ED2IP

                            dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);
#else
                            dist_t d = hnswlib::L2Sqr_by_IP(query_data, getDataByInternalId(cand), getExternalLabel(cand), dist_func_param_);
#endif
                            adsampling::tot_full_dist++;
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            //max heap
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>> top_candidates;
            // std::cerr << currObj << ",";
            stats = searchBaseLayerWithStepStats<false, true>(currObj, query_data, k, m, 5, 4, metric);
            // result = stats.top_candidates;
            // stats.skipped_nodes = stats.skipped_nodes;
            // stats.entry_dist = stats.entry_dist;
            return stats;
        }

            // randomly select entry point
//             std::uniform_int_distribution<int> distribution(0, cur_element_count - 1);
//             unsigned entry_point = distribution(get_random_engine());
//             // cout << entry_point << ":";

//             tableint currObj = entry_point;
// #ifdef COUNT_DIST_TIME
//             StopW stopw = StopW();
// #endif
// #ifndef ED2IP
//             dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(entry_point), dist_func_param_);
// #else
//             dist_t curdist = hnswlib::L2Sqr_by_IP(query_data, getDataByInternalId(entry_point), getExternalLabel(entry_point), dist_func_param_);
// #endif
// #ifdef COUNT_DIST_TIME
//             adsampling::distance_time += stopw.getElapsedTimeMicro();
// #endif
//             adsampling::tot_dist_calculation ++;
//             adsampling::tot_full_dist++;
//             //max heap
//             std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>> top_candidates;
//             // std::cerr << currObj << ",";
            
//             if (num_deleted_) {
//                 top_candidates=searchBaseLayerST<true,true>(currObj, query_data, std::max(ef_, k));
//             }
//             else{
//                  {
//                     if(use_ef >= k) top_candidates=searchBaseLayerST<false,true>(currObj, query_data, std::max(use_ef, k), metric);
//                     else top_candidates=searchBaseLayernStep<false,true>(currObj, query_data, use_ef, k, metric);

//                 }
                
//             }

//             //cerr << "search baselayer succeed!" << endl;
//             while (top_candidates.size() > k) {
//                 top_candidates.pop();
//             }
//             while (top_candidates.size() > 0) {
//                 std::pair<dist_t, tableint> rez = top_candidates.top();
//                 result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
//                 top_candidates.pop();
//             }
//             return result;
//         };





        /**
 * @brief Perform probe-based neighborhood sampling in HNSW.
 *
 * @tparam data_t The data type of the query vectors (e.g., float, uint8_t).
 * @tparam dist_t The data type used for distance calculations (e.g., float, int).
 *
 * @param query Pointer to the query vector.
 * @param vecdim The dimensionality of the query vector.
 * @param ef_probe Beam size for lightweight probing.
 * @param m Maximum number of distinct neighbors to collect.
 * @return A vector of pairs containing distance (`r_i`) and hop count (`h_i`) for each neighbor.
 */
template<typename data_t>
std::vector<std::pair<dist_t, int>> probeNeighborhood(
    data_t* query,
    size_t vecdim,
    size_t ef_probe,
    size_t m
) {
    // Set ef_probe for lightweight probing
    setEf(ef_probe);


    // Initialize visited list and result container
    VisitedList* vl = visited_list_pool_->getFreeVisitedList();
    vl_type* visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::vector<std::pair<dist_t, int>> neighbors; // Stores (distance, hop count)
    std::unordered_set<tableint> distinct_neighbors; // Track distinct neighbors

    // Start from the entry point
    tableint currObj = enterpoint_node_;
    dist_t curdist = fstdistfunc_(query, getDataByInternalId(currObj), dist_func_param_);

    // Priority queue for candidate nodes
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, typename HierarchicalNSW<dist_t>::CompareByFirst> candidate_set;
    candidate_set.emplace(-curdist, currObj); // candidate nodes use max-heap, so we negate the distance to ensure the closest nodes are at the top

    visited_array[currObj] = visited_array_tag;

    int hop_count = 0; // Tracks the number of hops

    // Perform lightweight probing
    while (!candidate_set.empty() && neighbors.size() < m) {
        std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
        candidate_set.pop();

        tableint current_node_id = current_node_pair.second;

        // If the node is already visited, skip it
        if (distinct_neighbors.find(current_node_id) != distinct_neighbors.end()) {
            continue;
        }

        // Record the neighbor and its distance/hop count
        neighbors.emplace_back(-current_node_pair.first, hop_count);
        distinct_neighbors.insert(current_node_id);

        // Explore neighbors of the current node
        int* data = (int*)get_linklist0(current_node_id);
        size_t size = getListCount((linklistsizeint*)data);

        for (size_t j = 1; j <= size; j++) {
            tableint candidate_id = *(data + j);

            if (visited_array[candidate_id] != visited_array_tag) {
                visited_array[candidate_id] = visited_array_tag;

                // Compute distance to the candidate
                dist_t dist = fstdistfunc_(query, getDataByInternalId(candidate_id), dist_func_param_);

                // Add candidate to the priority queue
                candidate_set.emplace(-dist, candidate_id);
            }
        }

        hop_count++; // Increment hop count for each traversal
    }

    // Release visited list
    visited_list_pool_->releaseVisitedList(vl);

    return neighbors;
}

        
        /**
         * @brief Calculate entropy of a distance distribution.
         */
        float calculateEntropy(const std::vector<float>& distances) const {
            if (distances.empty()) return 0.0f;
            
            // Normalize distances to probabilities (using inverse distance as similarity)
            float sum = 0.0f;
            std::vector<float> similarities;
            for (float d : distances) {
                float sim = 1.0f / (1.0f + d);  // Convert distance to similarity
                similarities.push_back(sim);
                sum += sim;
            }
            
            if (sum < 1e-8f) return 0.0f;
            
            // Normalize to probabilities
            float entropy = 0.0f;
            for (float sim : similarities) {
                float p = sim / sum;
                if (p > 1e-8f) {
                    entropy -= p * log(p);
                }
            }
            
            return entropy;
        }

        void checkIntegrity(){
            int connections_checked=0;
            std::vector <int > inbound_connections_num(cur_element_count,0);
            for(int i = 0;i < cur_element_count; i++){
                for(int l = 0;l <= element_levels_[i]; l++){
                    linklistsizeint *ll_cur = get_linklist_at_level(i,l);
                    int size = getListCount(ll_cur);
                    tableint *data = (tableint *) (ll_cur + 1);
                    std::unordered_set<tableint> s;
                    for (int j=0; j<size; j++){
                        assert(data[j] > 0);
                        assert(data[j] < cur_element_count);
                        assert (data[j] != i);
                        inbound_connections_num[data[j]]++;
                        s.insert(data[j]);
                        connections_checked++;

                    }
                    assert(s.size() == size);
                }
            }
            if(cur_element_count > 1){
                int min1=inbound_connections_num[0], max1=inbound_connections_num[0];
                for(int i=0; i < cur_element_count; i++){
                    assert(inbound_connections_num[i] > 0);
                    min1=std::min(inbound_connections_num[i],min1);
                    max1=std::max(inbound_connections_num[i],max1);
                }
                std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
            }
            std::cout << "integrity ok, checked " << connections_checked << " connections\n";

        }

        void getDegrees(vector<int>& out_degree, vector<int>&in_degree){
            for(int i = 0;i < cur_element_count; i++){
                int *data = (int *) get_linklist0(i);
                size_t size = getListCount((linklistsizeint*)data);
                out_degree[i] = size;
                for(int j = 1; j <= size; ++j)
                    in_degree[data[j]]++;
            }
        }

        void printGraph(){
            for(int i = 0;i < cur_element_count; i++){
                int *data = (int *) get_linklist0(i);
                size_t size = getListCount((linklistsizeint*)data);
                cout << i << ": ";
                for(int j = 1; j <= size; ++j)
                    cout << data[j] << ",";
                cout << endl;       
            }
        }

    };

}
