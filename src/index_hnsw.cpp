#include <iostream>
#include <fstream>
#include <queue>
#include <getopt.h>
#include <unordered_set>
#include <chrono>
#include <iomanip>
#include <random>
#include <algorithm>
#include <atomic>
#include <exception>
#include <mutex>
#include <cstdio>

#include <filesystem>
#include "matrix.h"
#include "utils.h"
#include "hnswlib/hnswlib.h"
#include "hnswlib/logging.h"
#include "adsampling.h"
#include <string>

using namespace std;
using namespace hnswlib;

// Function to create HNSW index with 1% of the dataset and store it
void create_partial_hnsw_index(const char* data_path, const char* index_path, size_t efConstruction, size_t M, char data_type) {
    int read_N = 500000000;
    // Load the full matrix to get the total number of rows
    size_t total_N = 0;
    size_t D = 0;
    // Determine total_N and D for the data type
    if (data_type == 0) {
        Matrix<float> X_full(const_cast<char*>(data_path), read_N);
        total_N = X_full.n;
        D = X_full.d;
    } else if (data_type == 1) {
        Matrix<uint8_t> X_full(const_cast<char*>(data_path), read_N);
        total_N = X_full.n;
        D = X_full.d;
    } else if (data_type == 2) {
        Matrix<int8_t> X_full(const_cast<char*>(data_path), read_N);
        total_N = X_full.n;
        D = X_full.d;
    }
    // Use only 1% of the data
    size_t partial_N = std::max<size_t>(1, total_N / 100);
    std::cerr << "Partial index: using " << partial_N << " / " << total_N << " points." << std::endl;
    // Build the partial index
    std::string partial_index_path = std::string(index_path);
    if (data_type == 0) {
        Matrix<float> X((char*)data_path, (int)partial_N);
        L2Space space(D);
        HierarchicalNSW<float>* appr_alg = new HierarchicalNSW<float>(&space, partial_N, M, efConstruction);
        int curr = 0;
        // SEQUENTIAL INSERTION - Remove OpenMP to avoid deadlocks
        for (int i = 0; i < (int)partial_N; i++) {
            appr_alg->addPointPlain_with_id(X.data + i * D, i, i);
            
            if (++curr % 10000 == 0) {
                std::cerr << "Partial Processing - " << curr << " / " << partial_N << std::endl;
            }
        }
        appr_alg->saveIndex(partial_index_path);
        delete appr_alg;
    } else if (data_type == 1) {
        Matrix<uint8_t> X((char*)data_path, (int)partial_N);
        L2SpaceU8 space(D);
        HierarchicalNSW<int>* appr_alg = new HierarchicalNSW<int>(&space, partial_N, M, efConstruction);
        int curr = 0;
        // SEQUENTIAL INSERTION - Remove OpenMP to avoid deadlocks
        for (int i = 0; i < (int)partial_N; i++) {
            appr_alg->addPointPlain_with_id(X.data + i * D, i, i);
            
            if (++curr % 10000 == 0) {
                std::cerr << "Partial Processing - " << curr << " / " << partial_N << std::endl;
            }
        }
        appr_alg->saveIndex(partial_index_path);
        delete appr_alg;
    } else if (data_type == 2) {
        Matrix<int8_t> X((char*)data_path, (int)partial_N);
        L2SpaceI8 space(D);
        HierarchicalNSW<int>* appr_alg = new HierarchicalNSW<int>(&space, partial_N, M, efConstruction);
        int curr = 0;
        // SEQUENTIAL INSERTION - Remove OpenMP to avoid deadlocks
        for (int i = 0; i < (int)partial_N; i++) {
            appr_alg->addPointPlain_with_id(X.data + i * D, i, i);
            
            if (++curr % 10000 == 0) {
                std::cerr << "Partial Processing - " << curr << " / " << partial_N << std::endl;
            }
        }
        appr_alg->saveIndex(partial_index_path);
        delete appr_alg;
    }
}

// Function to create HNSW index with random subset of the dataset and store it
void create_random_partial_hnsw_index(char* data_path, const char* index_path, size_t efConstruction, size_t M, char data_type, double subset_ratio = 0.01, unsigned int seed = 42) {
    int read_N = 500000000;
    // Load the full matrix to get the total number of rows
    size_t total_N = 0;
    size_t D = 0;
    
    // Add timing variables
    auto total_start = std::chrono::high_resolution_clock::now();
    auto data_load_start = std::chrono::high_resolution_clock::now();
    
    // Load data based on type - only load once
    if (data_type == 0) {
        Matrix<float> *X_full = new Matrix<float>(data_path, read_N);
        total_N = X_full->n;
        D = X_full->d;
        
        // Calculate subset size
        size_t subset_N = std::max<size_t>(1, (size_t)(total_N * subset_ratio));
        std::cerr << "Random partial index: using " << subset_N << " / " << total_N << " points (" 
                  << std::fixed << std::setprecision(1) << (100.0 * subset_ratio) << "%)" << std::endl;
        
        // Generate random indices for subset selection
        std::vector<size_t> all_indices(total_N);
        std::iota(all_indices.begin(), all_indices.end(), 0);
        
        // Use a fixed seed for reproducibility
        std::mt19937 gen(seed);
        std::shuffle(all_indices.begin(), all_indices.end(), gen);
        
        // Take the first subset_N indices
        std::vector<size_t> selected_indices(all_indices.begin(), all_indices.begin() + subset_N);
        std::sort(selected_indices.begin(), selected_indices.end()); // Sort for efficient access
        
        std::cerr << "Selected random indices from 0 to " << total_N - 1 << " using seed " << seed << std::endl;
        
        // Build the partial index
        std::string partial_index_path = std::string(index_path);
        int report = 1000;
        auto add_points_start = std::chrono::high_resolution_clock::now();
        auto last_timing = add_points_start;
        
        L2Space space(D);
        HierarchicalNSW<float>* appr_alg = new HierarchicalNSW<float>(&space, subset_N, M, efConstruction);
        
        int curr = 0;
        // Add points using random subset
        for (size_t subset_idx = 0; subset_idx < subset_N; subset_idx++) {
            size_t original_idx = selected_indices[subset_idx];
            appr_alg->addPointPlain_with_id_timed(X_full->data + original_idx * D, subset_idx, subset_idx);
            
            if (++curr % report == 0) {
                auto current_time = std::chrono::high_resolution_clock::now();
                    auto batch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_timing);
                    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - add_points_start);
                    double rate = (double)report / (batch_duration.count() / 1000.0);
                    std::cerr << "Processing - " << curr << " / " << subset_N 
                              << " (" << std::fixed << std::setprecision(1) << (100.0 * curr / subset_N) << "%)"
                              << " | Rate: " << std::fixed << std::setprecision(0) << rate << " points/sec"
                              << " | Total time: " << total_duration.count() << " ms" << std::endl;
                    last_timing = current_time;
            }
        }
        appr_alg->saveIndex(partial_index_path);
        delete appr_alg;
        delete X_full;
        
    } else if (data_type == 1) {
        Matrix<uint8_t> *X_full = new Matrix<uint8_t>(const_cast<char*>(data_path), read_N);
        total_N = X_full->n;
        D = X_full->d;
        
        // Calculate subset size
        size_t subset_N = std::max<size_t>(1, (size_t)(total_N * subset_ratio));
        std::cerr << "Random partial index: using " << subset_N << " / " << total_N << " points (" 
                  << std::fixed << std::setprecision(1) << (100.0 * subset_ratio) << "%)" << std::endl;
        
        // Generate random indices for subset selection
        std::vector<size_t> all_indices(total_N);
        std::iota(all_indices.begin(), all_indices.end(), 0);
        
        // Use a fixed seed for reproducibility
        std::mt19937 gen(seed);
        std::shuffle(all_indices.begin(), all_indices.end(), gen);
        
        // Take the first subset_N indices
        std::vector<size_t> selected_indices(all_indices.begin(), all_indices.begin() + subset_N);
        std::sort(selected_indices.begin(), selected_indices.end()); // Sort for efficient access
        
        std::cerr << "Selected random indices from 0 to " << total_N - 1 << " using seed " << seed << std::endl;
        
        // Build the partial index
        std::string partial_index_path = std::string(index_path);
        int report = 10000;
        auto add_points_start = std::chrono::high_resolution_clock::now();
        auto last_timing = add_points_start;
        
        L2SpaceU8 space(D);
        HierarchicalNSW<int>* appr_alg = new HierarchicalNSW<int>(&space, subset_N, M, efConstruction);
        
        int curr = 0;
        // Add points using random subset
        for (size_t subset_idx = 0; subset_idx < subset_N; subset_idx++) {
            size_t original_idx = selected_indices[subset_idx];
            appr_alg->addPointPlain_with_id_timed(X_full->data + original_idx * D, subset_idx, subset_idx);
            
            if (++curr % report == 0) {
                auto current_time = std::chrono::high_resolution_clock::now();
                    auto batch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_timing);
                    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - add_points_start);
                    double rate = (double)report / (batch_duration.count() / 1000.0);
                    std::cerr << "Processing - " << curr << " / " << subset_N 
                              << " (" << std::fixed << std::setprecision(1) << (100.0 * curr / subset_N) << "%)"
                              << " | Rate: " << std::fixed << std::setprecision(0) << rate << " points/sec"
                              << " | Total time: " << total_duration.count() << " ms" << std::endl;
                    last_timing = current_time;
            }
        }
        appr_alg->saveIndex(partial_index_path);
        delete appr_alg;
        delete X_full;
        
    } else if (data_type == 2) {
        Matrix<int8_t> *X_full = new Matrix<int8_t>(const_cast<char*>(data_path), read_N);
        total_N = X_full->n;
        D = X_full->d;
        
        // Calculate subset size
        size_t subset_N = std::max<size_t>(1, (size_t)(total_N * subset_ratio));
        std::cerr << "Random partial index: using " << subset_N << " / " << total_N << " points (" 
                  << std::fixed << std::setprecision(1) << (100.0 * subset_ratio) << "%)" << std::endl;
        
        // Generate random indices for subset selection
        std::vector<size_t> all_indices(total_N);
        std::iota(all_indices.begin(), all_indices.end(), 0);
        
        // Use a fixed seed for reproducibility
        std::mt19937 gen(seed);
        std::shuffle(all_indices.begin(), all_indices.end(), gen);
        
        // Take the first subset_N indices
        std::vector<size_t> selected_indices(all_indices.begin(), all_indices.begin() + subset_N);
        std::sort(selected_indices.begin(), selected_indices.end()); // Sort for efficient access
        
        std::cerr << "Selected random indices from 0 to " << total_N - 1 << " using seed " << seed << std::endl;
        
        // Build the partial index
        std::string partial_index_path = std::string(index_path);
        int report = 100;
        auto add_points_start = std::chrono::high_resolution_clock::now();
        auto last_timing = add_points_start;
        
        L2SpaceI8 space(D);
        HierarchicalNSW<int>* appr_alg = new HierarchicalNSW<int>(&space, subset_N, M, efConstruction);
        
        int curr = 0;
        // Add points using random subset
        for (size_t subset_idx = 0; subset_idx < subset_N; subset_idx++) {
            size_t original_idx = selected_indices[subset_idx];
            appr_alg->addPointPlain_with_id_timed(X_full->data + original_idx * D, subset_idx, subset_idx);
            
            if (++curr % report == 0) {
                auto current_time = std::chrono::high_resolution_clock::now();
                    auto batch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_timing);
                    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - add_points_start);
                    double rate = (double)report / (batch_duration.count() / 1000.0);
                    std::cerr << "Processing - " << curr << " / " << subset_N 
                              << " (" << std::fixed << std::setprecision(1) << (100.0 * curr / subset_N) << "%)"
                              << " | Rate: " << std::fixed << std::setprecision(0) << rate << " points/sec"
                              << " | Total time: " << total_duration.count() << " ms" << std::endl;
                    last_timing = current_time;
            }
        }
        appr_alg->saveIndex(partial_index_path);
        delete appr_alg;
        delete X_full;
    }
}

void print_usage(char* self) {
    std::cerr << "Usage: " << self << " [options]" << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  -d, --debug              Enable debug mode (full logging)" << std::endl;
    std::cerr << "  -l, --log-file FILE      Write logs to FILE (required)" << std::endl;
    std::cerr << "  -v, --log-level LEVEL    Set log level (error|warn|info|debug|trace)" << std::endl;
    std::cerr << "  -p, --path_data FILE     Input data file path (required)" << std::endl;
    std::cerr << "  -i, --path_index FILE    Output index file path (required)" << std::endl;
    std::cerr << "  -m, --M INT              HNSW parameter M (default: 16)" << std::endl;
    std::cerr << "  -e, --ef_construction INT HNSW parameter ef_construction (default: 100)" << std::endl;
    std::cerr << "  -t, --data_type INT      Data type (0:float, 1:uint8, 2:int8, default:0)" << std::endl;
    std::cerr << "  --build_mode MODE        Build mode (plain|hierarchy|parallel, default: plain)" << std::endl;
    std::cerr << "  --random_seed INT        Random seed for reproducible hierarchy (default: 100)" << std::endl;
    std::cerr << "  -h, --help               Show this help message" << std::endl;
}

void create_hnsw_index(char* data_path, char* index_path, size_t efConstruction, size_t M, char data_type, const std::string& build_mode, size_t random_seed,
                       bool enable_pruning_instrumentation = false,
                       const char* pruning_counts_path = nullptr,
                       int construction_threads = 1,
                       int max_points = -1,
                       const std::string& checkpoint_path = "",
                       int checkpoint_every = 0,
                       bool resume_checkpoint = false) {
    const struct option longopts[] ={
        // General Parameter
        {"help",                        no_argument,       0, 'h'}, 

        // Index Parameter
        {"efConstruction",              required_argument, 0, 'e'}, 
        {"M",                           required_argument, 0, 'm'}, 

        {"data_type",                   required_argument, 0, 't'},

        // Indexing Path 
        {"data_path",                   required_argument, 0, 'd'},
        {"index_path",                  required_argument, 0, 'i'},
    };

    int ind;
    int iarg = 0;
    opterr = 1;    //getopt error message (off: 0)

    
    int read_N = max_points > 0 ? max_points : 500000000;
    size_t report = 10000;
    
    // Add timing variables
    auto total_start = std::chrono::high_resolution_clock::now();
    auto data_load_start = std::chrono::high_resolution_clock::now();
    
    if(data_type == 0){
            HNSW_LOG_INFO("=== Starting HNSW Index Creation with Profiling ===");
            HNSW_LOG_INFO("Parameters: efConstruction=", efConstruction, ", M=", M, ", data_type=", (int)data_type);
            
            const std::string input_path(data_path);
            const bool is_global_header_bin = input_path.size() >= 3 &&
                input_path.compare(input_path.size() - 3, 3, "bin") == 0;
            size_t D = 0, N = 0;
            Matrix<float> *X = nullptr;
            if (resume_checkpoint && !checkpoint_path.empty()) {
                if (!is_global_header_bin)
                    throw std::runtime_error("checkpoint resume currently requires fbin input");
                std::ifstream header(data_path, std::ios::binary);
                uint32_t file_n = 0, file_d = 0;
                header.read(reinterpret_cast<char *>(&file_n), sizeof(file_n));
                header.read(reinterpret_cast<char *>(&file_d), sizeof(file_d));
                if (!header)
                    throw std::runtime_error("failed to read fbin header for checkpoint resume");
                D = file_d;
                N = max_points > 0 ? std::min<size_t>(file_n, max_points) : file_n;
            } else {
                HNSW_LOG_INFO("Step 1: Loading data from ", data_path, "...");
                X = is_global_header_bin && max_points > 0
                    ? new Matrix<float>(data_path, true, read_N)
                    : new Matrix<float>(data_path, read_N);
                D = X->d;
                N = X->n;
            }

            // Step 2: Initialize HNSW
            HNSW_LOG_INFO("Step 2: Initializing HNSW index...");
            L2Space space(D);
            HNSW_LOG_INFO("L2 space (FLOAT)");
            HierarchicalNSW<float>* appr_alg = nullptr;
            int resume_from = 1;
            if (resume_checkpoint) {
                if (checkpoint_path.empty())
                    throw std::runtime_error("--resume requires --checkpoint_path");
                std::ifstream state(checkpoint_path + ".state");
                size_t state_n = 0, state_m = 0, state_efc = 0, state_seed = 0;
                state >> resume_from >> state_n >> state_m >> state_efc >> state_seed;
                if (!state || state_n != N || state_m != M || state_efc != efConstruction || state_seed != random_seed)
                    throw std::runtime_error("checkpoint state is missing or does not match N/M/efConstruction/seed");
                HNSW_LOG_INFO("Loading checkpoint ", checkpoint_path, " at connected prefix ", resume_from);
                appr_alg = new HierarchicalNSW<float>(&space, checkpoint_path, false, N);
                if (appr_alg->cur_element_count != N)
                    throw std::runtime_error("checkpoint does not contain all prepared points");
            } else {
                appr_alg = new HierarchicalNSW<float>(&space, N, M, efConstruction, random_seed);
            }

            // Pruning instrumentation (Batch 2a-3): must be enabled BEFORE
            // any addPoint, since getNeighborsByHeuristic2 is called from
            // mutuallyConnectNewElement during every insertion.
            if (enable_pruning_instrumentation) {
                HNSW_LOG_INFO("Pruning instrumentation: enabled");
                appr_alg->enablePruningInstrumentation(N);
            }

            HNSW_LOG_INFO("will write index to ", index_path);
            
            // Step 3: Add points (this is likely the bottleneck)
            HNSW_LOG_INFO("Step 3: Adding points to index...");
            auto add_points_start = std::chrono::high_resolution_clock::now();
            int curr = 0;
            
            // Add timing for individual batches
            const int timing_batch_size = 1000;
            auto last_timing = add_points_start;
            
            if (build_mode == "parallel") {
                if (!resume_checkpoint) {
                    HNSW_LOG_INFO("Parallel construction: preparing fixed IDs and deterministic levels");
                    for (int i = 0; i < N; ++i) {
                        appr_alg->preparePointWithId(X->data + i * D, i, i);
                        if ((i + 1) % 1000000 == 0)
                            HNSW_LOG_INFO("Prepared - ", i + 1, " / ", N);
                    }
                    appr_alg->initializePreparedEntryPoint(0);
                    delete X;
                    X = nullptr;
                }

                HNSW_LOG_INFO("Parallel construction: connecting with ", construction_threads, " threads");
                std::atomic<int> completed{resume_from};
                std::atomic<bool> failed{false};
                std::exception_ptr first_error;
                std::mutex error_mutex;
                auto parallel_start = std::chrono::high_resolution_clock::now();

                auto save_checkpoint = [&](int connected_prefix) {
                    if (checkpoint_path.empty())
                        return;
                    const std::string index_tmp = checkpoint_path + ".tmp";
                    const std::string state_tmp = checkpoint_path + ".state.tmp";
                    HNSW_LOG_INFO("Checkpoint: saving connected prefix ", connected_prefix,
                                  " to ", checkpoint_path);
                    appr_alg->saveIndex(index_tmp);
                    if (std::rename(index_tmp.c_str(), checkpoint_path.c_str()) != 0)
                        throw std::runtime_error("failed to atomically publish checkpoint index");
                    {
                        std::ofstream state(state_tmp, std::ios::trunc);
                        state << connected_prefix << ' ' << N << ' ' << M << ' '
                              << efConstruction << ' ' << random_seed << '\n';
                        state.flush();
                        if (!state)
                            throw std::runtime_error("failed to write checkpoint state");
                    }
                    if (std::rename(state_tmp.c_str(), (checkpoint_path + ".state").c_str()) != 0)
                        throw std::runtime_error("failed to atomically publish checkpoint state");
                    HNSW_LOG_INFO("Checkpoint: complete at ", connected_prefix, " / ", N);
                };

                const int chunk = checkpoint_every > 0 ? checkpoint_every : static_cast<int>(N);
                for (int batch_begin = resume_from; batch_begin < static_cast<int>(N); batch_begin += chunk) {
                    const int batch_end = std::min<int>(N, batch_begin + chunk);
                    #pragma omp parallel for schedule(dynamic, 64) num_threads(construction_threads)
                    for (int i = batch_begin; i < batch_end; ++i) {
                        if (failed.load(std::memory_order_relaxed))
                            continue;
                        try {
                            appr_alg->connectPreparedPoint(static_cast<hnswlib::tableint>(i));
                            int done = completed.fetch_add(1, std::memory_order_relaxed) + 1;
                            if (done % 10000 == 0) {
                                #pragma omp critical(hnsw_progress)
                                {
                                    auto now = std::chrono::high_resolution_clock::now();
                                    double seconds = std::chrono::duration<double>(now - parallel_start).count();
                                    double rate = (done - resume_from) / std::max(seconds, 1e-9);
                                    double eta = (N - done) / std::max(rate, 1e-9);
                                    HNSW_LOG_INFO("Processing - ", done, " / ", N,
                                                  " (", std::fixed, std::setprecision(1), 100.0 * done / N,
                                                  "%) | Aggregate rate: ", std::setprecision(0), rate,
                                                  " points/sec | ETA: ", std::setprecision(1), eta / 3600.0, "h");
                                }
                            }
                        } catch (...) {
                            failed.store(true, std::memory_order_relaxed);
                            std::lock_guard<std::mutex> guard(error_mutex);
                            if (!first_error)
                                first_error = std::current_exception();
                        }
                    }
                    if (first_error)
                        std::rethrow_exception(first_error);
                    completed.store(batch_end);
                    if (batch_end < static_cast<int>(N))
                        save_checkpoint(batch_end);
                }
                curr = completed.load();
            } else for(int i=0;i<N;i++){
                if (build_mode == "hierarchy") {
                    appr_alg->addPoint_with_id_timed(X->data + i * D, i, i);
                } else {
                    appr_alg->addPointPlain_with_id_timed(X->data + i * D, i, i);
                }
                
                    if(++curr % report == 0){
                    auto current_time = std::chrono::high_resolution_clock::now();
                    auto batch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_timing);
                    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - add_points_start);
                    double rate = (double)report / (batch_duration.count() / 1000.0);
                    double eta_seconds = (N - curr) / rate;
                    int eta_minutes = (int)(eta_seconds / 60);
                    int eta_seconds_remainder = (int)(eta_seconds) % 60;
                    
                    HNSW_LOG_INFO("Processing - ", curr, " / ", N,
                              " (", std::fixed, std::setprecision(1), (100.0 * curr / N), "%)",
                              " | Rate: ", std::fixed, std::setprecision(0), rate, " points/sec",
                              " | ETA: ", eta_minutes, "m ", eta_seconds_remainder, "s",
                              " | Total time: ", total_duration.count(), " ms");
                    last_timing = current_time;
                }
            }
            auto add_points_end = std::chrono::high_resolution_clock::now();
            auto add_points_duration = std::chrono::duration_cast<std::chrono::milliseconds>(add_points_end - add_points_start);
            HNSW_LOG_INFO("Point addition completed in: ", add_points_duration.count(), " ms");
            HNSW_LOG_INFO("Average rate: ", std::fixed, std::setprecision(0), (N / (add_points_duration.count() / 1000.0)), " points/sec");

            // Step 4: Save index
            HNSW_LOG_INFO("Step 4: Saving index to ", index_path, "...");
            appr_alg->saveIndex(index_path);

            // Step 4b: Dump pruning counts (Batch 2a-3)
            if (enable_pruning_instrumentation && pruning_counts_path != nullptr && pruning_counts_path[0] != '\0') {
                HNSW_LOG_INFO("Step 4b: Dumping pruning counts to ", pruning_counts_path);
                appr_alg->dumpPruningCounts(pruning_counts_path);
            }

            // Step 5: Calculate degrees (optional)
            HNSW_LOG_INFO("Step 5: Calculating graph degrees...");

            vector<int> out_degree(N, 0);
            vector<int> in_degree(N, 0);
            appr_alg->getDegrees(out_degree, in_degree);

            // Calculate and display statistics
            double avg_out_degree = 0.0;
            double avg_in_degree = 0.0;
            for(int i = 0; i < N; i++) {
                avg_out_degree += out_degree[i];
                avg_in_degree += in_degree[i];
            }
            avg_out_degree /= N;
            avg_in_degree /= N;
            HNSW_LOG_INFO("Graph statistics - Avg out-degree: ", std::fixed, std::setprecision(2), avg_out_degree,
                      ", Avg in-degree: ", avg_in_degree);
            
         

    }else if(data_type == 1){
            Matrix<uint8_t> *X = new Matrix<uint8_t>(data_path, read_N);
            size_t D = X->d;
            size_t N = X->n;

            L2SpaceU8 space(D);
            HNSW_LOG_INFO("L2 space (UINT 8)");
            HierarchicalNSW<int>* appr_alg = new HierarchicalNSW<int> (&space, N, M, efConstruction, random_seed);

            int curr = 0;
            // SEQUENTIAL INSERTION - Remove OpenMP to avoid deadlocks
            for(int i=0;i<N;i++){
                if (build_mode == "hierarchy") {
                    appr_alg->addPoint_with_id_timed(X->data + i * D, i, i);
                } else {
                    appr_alg->addPointPlain_with_id_timed(X->data + i * D, i, i);
                }
                
                    if(++curr % report == 0){
                        HNSW_LOG_INFO("Processing - ", curr, " / ", N);
                    }
            }

            appr_alg->saveIndex(index_path);
    }else if(data_type == 2){
            Matrix<int8_t> *X = new Matrix<int8_t>(data_path, read_N);
            size_t D = X->d;
            size_t N = X->n;

            L2SpaceI8 space(D);
            HNSW_LOG_INFO("L2 space (INT 8)");
            HierarchicalNSW<int>* appr_alg = new HierarchicalNSW<int> (&space, N, M, efConstruction, random_seed);

            int curr = 0;
            // SEQUENTIAL INSERTION - Remove OpenMP to avoid deadlocks
            for(int i=0;i<N;i++){
                if (build_mode == "hierarchy") {
                    appr_alg->addPoint_with_id_timed(X->data + i * D, i, i);
                } else {
                    appr_alg->addPointPlain_with_id_timed(X->data + i * D, i, i);
                }
                
                    if(++curr % report == 0){
                        HNSW_LOG_INFO("Processing - ", curr, " / ", N);
                    }
            }

            appr_alg->saveIndex(index_path);
    }


}
int main(int argc, char * argv[]) {
    LogMode log_mode = LogMode::DEBUG;
    LogLevel log_level = LogLevel::INFO;
    std::string log_file_path;
    char* path_data = nullptr;
    char* path_index = nullptr;
    int M = 16, ef_construction = 100;
    int data_type = 0;
    std::string build_mode = "plain";
    size_t random_seed = 100;
    int construction_threads = 1;
    int max_points = -1;
    std::string checkpoint_path;
    int checkpoint_every = 0;
    bool resume_checkpoint = false;

    // Batch 2a-3: optional pruning instrumentation
    bool enable_pruning_instrumentation = false;
    std::string pruning_counts_path;

    const int OPT_ENABLE_PRUNING_INSTRUMENTATION = 2001;
    const int OPT_PRUNING_COUNTS_PATH = 2002;
    const int OPT_CHECKPOINT_PATH = 2003;
    const int OPT_CHECKPOINT_EVERY = 2004;
    const int OPT_RESUME = 2005;

    const struct option longopts[] = {
        {"debug", no_argument, 0, 'd'},
        {"log-file", required_argument, 0, 'l'},
        {"log-level", required_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
        {"path_data", required_argument, 0, 'p'},
        {"path_index", required_argument, 0, 'i'},
        {"M", required_argument, 0, 'm'},
        {"ef_construction", required_argument, 0, 'e'},
        {"data_type", required_argument, 0, 't'},
        {"build_mode", required_argument, 0, 'b'},
        {"random_seed", required_argument, 0, 'r'},
        {"threads", required_argument, 0, 'j'},
        {"max_points", required_argument, 0, 'n'},
        {"checkpoint_path", required_argument, 0, OPT_CHECKPOINT_PATH},
        {"checkpoint_every", required_argument, 0, OPT_CHECKPOINT_EVERY},
        {"resume", no_argument, 0, OPT_RESUME},
        {"enable_pruning_instrumentation", no_argument, 0, OPT_ENABLE_PRUNING_INSTRUMENTATION},
        {"pruning_counts_path", required_argument, 0, OPT_PRUNING_COUNTS_PATH},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "dl:v:hp:i:m:e:t:b:r:j:n:", longopts, nullptr)) != -1) {
        switch (opt) {
            case 'd':
                log_mode = LogMode::DEBUG;
                break;
            case 'l':
                log_file_path = optarg;
                break;
            case 'v':
                if (strcmp(optarg, "error") == 0) log_level = LogLevel::ERROR;
                else if (strcmp(optarg, "warn") == 0) log_level = LogLevel::WARN;
                else if (strcmp(optarg, "info") == 0) log_level = LogLevel::INFO;
                else if (strcmp(optarg, "debug") == 0) log_level = LogLevel::DEBUG;
                else if (strcmp(optarg, "trace") == 0) log_level = LogLevel::TRACE;
                else {
                    std::cerr << "Unknown log level: " << optarg << std::endl;
                    return 1;
                }
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            case 'p':
                path_data = optarg;
                break;
            case 'i':
                path_index = optarg;
                break;
            case 'm':
                M = atoi(optarg);
                break;
            case 'e':
                ef_construction = atoi(optarg);
                break;
            case 't':
                data_type = atoi(optarg);
                break;
            case 'b':
                build_mode = optarg;
                if (build_mode != "plain" && build_mode != "hierarchy" && build_mode != "parallel") {
                    std::cerr << "Unknown build_mode: " << build_mode << " (expected plain|hierarchy|parallel)" << std::endl;
                    return 1;
                }
                break;
            case 'r':
                random_seed = static_cast<size_t>(strtoull(optarg, nullptr, 10));
                break;
            case 'j':
                construction_threads = std::max(1, atoi(optarg));
                break;
            case 'n':
                max_points = atoi(optarg);
                break;
            case OPT_CHECKPOINT_PATH:
                checkpoint_path = optarg;
                break;
            case OPT_CHECKPOINT_EVERY:
                checkpoint_every = std::max(0, atoi(optarg));
                break;
            case OPT_RESUME:
                resume_checkpoint = true;
                break;
            case OPT_ENABLE_PRUNING_INSTRUMENTATION:
                enable_pruning_instrumentation = true;
                break;
            case OPT_PRUNING_COUNTS_PATH:
                pruning_counts_path = optarg;
                break;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }

    if (!path_data || !path_index || log_file_path.empty() || M <= 0 || ef_construction <= 0) {
        std::cerr << "Required: --path_data FILE --path_index FILE --log-file FILE; M and ef_construction must be positive.\n";
        return 2;
    }
    if (!std::filesystem::is_regular_file(path_data)) {
        std::cerr << "Input file does not exist: " << path_data << '\n'; return 2;
    }
    for (const auto& path : {std::string(path_index), log_file_path}) {
        auto parent = std::filesystem::path(path).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
    }
    if (std::filesystem::absolute(path_data).lexically_normal() == std::filesystem::absolute(path_index).lexically_normal() ||
        std::filesystem::absolute(path_data).lexically_normal() == std::filesystem::absolute(log_file_path).lexically_normal() ||
        std::filesystem::absolute(path_index).lexically_normal() == std::filesystem::absolute(log_file_path).lexically_normal()) {
        std::cerr << "Data, index and log must use distinct paths.\n"; return 2;
    }
    std::ofstream log_check(log_file_path, std::ios::app);
    if (!log_check) { std::cerr << "Cannot write log file\n"; return 2; }
    log_check.close();
    initLogging(log_mode, log_level, log_file_path);

    HNSW_LOG_INFO("Building index with M=", M, ", ef_construction=", ef_construction,
                  ", build_mode=", build_mode, ", random_seed=", random_seed,
                  ", threads=", construction_threads, ", max_points=", max_points,
                  ", checkpoint_every=", checkpoint_every,
                  ", resume=", resume_checkpoint);

    // create_random_partial_hnsw_index(path_data, path_index, ef_construction, M, data_type, 0.01, 2025);
    
    create_hnsw_index(path_data, path_index, ef_construction, M, data_type, build_mode, random_seed,
                      enable_pruning_instrumentation,
                      pruning_counts_path.empty() ? nullptr : pruning_counts_path.c_str(),
                      construction_threads, max_points, checkpoint_path,
                      checkpoint_every, resume_checkpoint);

    HNSW_LOG_INFO("Index building complete.");
    return 0;
}
