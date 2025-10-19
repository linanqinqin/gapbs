// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef GAPBS_PTHREADS_H_
#define GAPBS_PTHREADS_H_

#include <pthread.h>
#include <atomic>
#include <vector>
#include <functional>
#include <algorithm>
#include <thread>
#include <memory>
#include <utility>
#include <cstdlib>
#include <cstring>

// Atomic operations are now handled by platform_atomics.h

/*
GAP Benchmark Suite
File:   GAPBS Pthreads Wrapper
Author: Custom Implementation

Direct pthreads implementation to replace OpenMP pragmas
- Provides equivalent functionality to OpenMP parallel constructs
- Designed for cooperative threading model integration
- Maintains same performance characteristics as OpenMP
*/

class GAPBSPthreads {
private:
    int num_threads_;
    std::vector<pthread_t> threads_;
    std::vector<pthread_mutex_t> mutexes_;
    std::vector<pthread_cond_t> conditions_;
    
    // Thread pool implementation
    struct ThreadPool {
        std::vector<pthread_t> threads;
        std::vector<pthread_mutex_t> work_mutexes;
        std::vector<pthread_cond_t> work_conditions;
        std::vector<bool> thread_busy;
        std::vector<void*> work_data;
        std::vector<std::function<void*(void*)>> work_functions;
        pthread_mutex_t pool_mutex;
        pthread_cond_t pool_condition;
        bool shutdown;
        int active_threads;
        
        // Barrier and critical section support for worker threads
        pthread_mutex_t barrier_mutex;
        pthread_cond_t barrier_condition;
        int barrier_count;
        int barrier_generation;
        bool barrier_active;
        
        pthread_mutex_t critical_mutex;
        
        ThreadPool(int num_threads) : shutdown(false), active_threads(0), 
                                      barrier_count(0), barrier_generation(0), barrier_active(false) {
            threads.resize(num_threads);
            work_mutexes.resize(num_threads);
            work_conditions.resize(num_threads);
            thread_busy.resize(num_threads, false);
            work_data.resize(num_threads, nullptr);
            work_functions.resize(num_threads);
            
            pthread_mutex_init(&pool_mutex, nullptr);
            pthread_cond_init(&pool_condition, nullptr);
            pthread_mutex_init(&barrier_mutex, nullptr);
            pthread_cond_init(&barrier_condition, nullptr);
            pthread_mutex_init(&critical_mutex, nullptr);
            
            for (int i = 0; i < num_threads; i++) {
                pthread_mutex_init(&work_mutexes[i], nullptr);
                pthread_cond_init(&work_conditions[i], nullptr);
            }
        }
        
        ~ThreadPool() {
            shutdown = true;
            pthread_cond_broadcast(&pool_condition);
            
            for (size_t i = 0; i < threads.size(); i++) {
                pthread_join(threads[i], nullptr);
                pthread_mutex_destroy(&work_mutexes[i]);
                pthread_cond_destroy(&work_conditions[i]);
            }
            
            pthread_mutex_destroy(&pool_mutex);
            pthread_cond_destroy(&pool_condition);
            pthread_mutex_destroy(&barrier_mutex);
            pthread_cond_destroy(&barrier_condition);
            pthread_mutex_destroy(&critical_mutex);
        }
        
        void start_threads() {
            for (size_t i = 0; i < threads.size(); i++) {
                pthread_create(&threads[i], nullptr, worker_thread, this);
            }
        }
        
        static void* worker_thread(void* arg) {
            ThreadPool* pool = static_cast<ThreadPool*>(arg);
            int thread_id = -1;
            
            // Find our thread ID
            pthread_mutex_lock(&pool->pool_mutex);
            for (size_t i = 0; i < pool->threads.size(); i++) {
                if (pthread_equal(pthread_self(), pool->threads[i])) {
                    thread_id = static_cast<int>(i);
                    break;
                }
            }
            pthread_mutex_unlock(&pool->pool_mutex);
            
            while (true) {
                pthread_mutex_lock(&pool->work_mutexes[thread_id]);
                
                while (!pool->thread_busy[thread_id] && !pool->shutdown) {
                    pthread_cond_wait(&pool->work_conditions[thread_id], &pool->work_mutexes[thread_id]);
                }
                
                if (pool->shutdown) {
                    pthread_mutex_unlock(&pool->work_mutexes[thread_id]);
                    break;
                }
                
                // Execute work
                if (pool->work_functions[thread_id]) {
                    pool->work_functions[thread_id](pool->work_data[thread_id]);
                }
                
                pool->thread_busy[thread_id] = false;
                pool->work_functions[thread_id] = nullptr;
                pool->work_data[thread_id] = nullptr;
                
                pthread_mutex_unlock(&pool->work_mutexes[thread_id]);
                
                // Notify pool that thread is done
                pthread_mutex_lock(&pool->pool_mutex);
                pool->active_threads--;
                if (pool->active_threads == 0) {
                    pthread_cond_signal(&pool->pool_condition);
                }
                pthread_mutex_unlock(&pool->pool_mutex);
            }
            
            return nullptr;
        }
        
        void execute_work(int thread_id, std::function<void*(void*)> func, void* data) {
            pthread_mutex_lock(&work_mutexes[thread_id]);
            work_functions[thread_id] = func;
            work_data[thread_id] = data;
            thread_busy[thread_id] = true;
            pthread_cond_signal(&work_conditions[thread_id]);
            pthread_mutex_unlock(&work_mutexes[thread_id]);
        }
        
        void wait_for_completion() {
            pthread_mutex_lock(&pool_mutex);
            while (active_threads > 0) {
                pthread_cond_wait(&pool_condition, &pool_mutex);
            }
            pthread_mutex_unlock(&pool_mutex);
        }
        
        // Barrier synchronization for worker threads
        void worker_barrier() {
            pthread_mutex_lock(&barrier_mutex);
            barrier_count++;
            
            if (barrier_count == static_cast<int>(threads.size())) {
                barrier_count = 0;
                barrier_generation++;
                pthread_cond_broadcast(&barrier_condition);
            } else {
                int current_generation = barrier_generation;
                while (current_generation == barrier_generation) {
                    pthread_cond_wait(&barrier_condition, &barrier_mutex);
                }
            }
            pthread_mutex_unlock(&barrier_mutex);
        }
        
        // Critical section for worker threads
        void worker_critical_section(std::function<void()> func) {
            pthread_mutex_lock(&critical_mutex);
            func();
            pthread_mutex_unlock(&critical_mutex);
        }
    };
    
    std::unique_ptr<ThreadPool> thread_pool_;
    
    // Barrier implementation
    struct Barrier {
        pthread_mutex_t mutex;
        pthread_cond_t condition;
        int count;
        int total_threads;
        int generation;
        
        Barrier(int total) : count(0), total_threads(total), generation(0) {
            pthread_mutex_init(&mutex, nullptr);
            pthread_cond_init(&condition, nullptr);
        }
        
        ~Barrier() {
            pthread_mutex_destroy(&mutex);
            pthread_cond_destroy(&condition);
        }
        
        void wait() {
            pthread_mutex_lock(&mutex);
            count++;
            
            if (count == total_threads) {
                count = 0;
                generation++;
                pthread_cond_broadcast(&condition);
            } else {
                int current_generation = generation;
                while (current_generation == generation) {
                    pthread_cond_wait(&condition, &mutex);
                }
            }
            pthread_mutex_unlock(&mutex);
        }
    };
    
    std::unique_ptr<Barrier> barrier_;
    
    // C++11 compatible make_unique implementation
    template<typename T, typename... Args>
    std::unique_ptr<T> make_unique(Args&&... args) {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }
    
public:
    explicit GAPBSPthreads(int num_threads = 0) 
        : num_threads_(num_threads ? num_threads : get_default_thread_count()) {
        threads_.resize(num_threads_);
        mutexes_.resize(num_threads_);
        conditions_.resize(num_threads_);
        
        for (int i = 0; i < num_threads_; i++) {
            pthread_mutex_init(&mutexes_[i], nullptr);
            pthread_cond_init(&conditions_[i], nullptr);
        }
        
        // Initialize thread pool
        thread_pool_ = make_unique<ThreadPool>(num_threads_);
        thread_pool_->start_threads();
    }
    
    // Set number of threads at runtime
    void set_num_threads(int num_threads) {
        if (num_threads > 0 && num_threads != num_threads_) {
            // Clean up existing resources
            for (int i = 0; i < num_threads_; i++) {
                pthread_mutex_destroy(&mutexes_[i]);
                pthread_cond_destroy(&conditions_[i]);
            }
            
            // Clean up thread pool
            thread_pool_.reset();
            
            num_threads_ = num_threads;
            threads_.resize(num_threads_);
            mutexes_.resize(num_threads_);
            conditions_.resize(num_threads_);
            
            for (int i = 0; i < num_threads_; i++) {
                pthread_mutex_init(&mutexes_[i], nullptr);
                pthread_cond_init(&conditions_[i], nullptr);
            }
            
            // Reinitialize thread pool
            thread_pool_ = make_unique<ThreadPool>(num_threads_);
            thread_pool_->start_threads();
        }
    }
    
private:
    static int get_default_thread_count() {
        // Check environment variable first
        const char* env_threads = getenv("OMP_NUM_THREADS");
        if (env_threads) {
            int threads = std::atoi(env_threads);
            if (threads > 0) return threads;
        }
        
        // Check GAPBS-specific environment variable
        const char* gapbs_threads = getenv("GAPBS_NUM_THREADS");
        if (gapbs_threads) {
            int threads = std::atoi(gapbs_threads);
            if (threads > 0) return threads;
        }
        
        // Default to hardware concurrency
        return std::thread::hardware_concurrency();
    }
    
    int get_num_threads() const { return num_threads_; }
    
public:
    // Barrier synchronization - works with thread pool
    void barrier() {
        if (!barrier_) {
            barrier_ = make_unique<Barrier>(num_threads_);
        }
        barrier_->wait();
    }
    
    // Critical section - works with thread pool
    void critical_section(std::function<void()> func) {
        static pthread_mutex_t critical_mutex = PTHREAD_MUTEX_INITIALIZER;
        pthread_mutex_lock(&critical_mutex);
        func();
        pthread_mutex_unlock(&critical_mutex);
    }
    
    // Thread pool barrier and critical section for use within worker threads
    void worker_barrier() {
        if (thread_pool_) {
            thread_pool_->worker_barrier();
        }
    }
    
    void worker_critical_section(std::function<void()> func) {
        if (thread_pool_) {
            thread_pool_->worker_critical_section(func);
        }
    }
    
    // Parallel for with reduction using thread pool
    template<typename Func>
    int64_t parallel_for_reduction(size_t count, Func func, int64_t initial_value = 0) {
        std::atomic<int64_t> result(initial_value);
        std::vector<int64_t> local_results(num_threads_, 0);
        
        struct ThreadData {
            Func func;  // Store by value, not pointer
            size_t count;
            int thread_id;
            int num_threads;
            std::vector<int64_t>* local_results;
        };
        
        auto worker = [](void* arg) -> void* {
            ThreadData* data = static_cast<ThreadData*>(arg);
            int64_t local_result = 0;
            
            // Dynamic scheduling simulation
            for (size_t i = data->thread_id; i < data->count; i += data->num_threads) {
                local_result += data->func(i);
            }
            
            (*data->local_results)[data->thread_id] = local_result;
            return nullptr;
        };
        
        // Use thread pool with proper data management
        std::vector<ThreadData> thread_data(num_threads_);
        for (int i = 0; i < num_threads_; i++) {
            thread_data[i] = {func, count, i, num_threads_, &local_results};
            thread_pool_->execute_work(i, worker, &thread_data[i]);
        }
        
        thread_pool_->wait_for_completion();
        
        int64_t total = initial_value;
        for (int64_t local : local_results) {
            total += local;
        }
        
        return total;
    }
    
    // Parallel for without reduction using thread pool
    template<typename Func>
    void parallel_for(size_t count, Func func) {
        struct ThreadData {
            Func func;  // Store by value, not pointer
            size_t count;
            int thread_id;
            int num_threads;
        };
        
        auto worker = [](void* arg) -> void* {
            ThreadData* data = static_cast<ThreadData*>(arg);
            
            for (size_t i = data->thread_id; i < data->count; i += data->num_threads) {
                data->func(i);
            }
            return nullptr;
        };
        
        // Use thread pool with proper data management
        std::vector<ThreadData> thread_data(num_threads_);
        for (int i = 0; i < num_threads_; i++) {
            thread_data[i] = {func, count, i, num_threads_};
            thread_pool_->execute_work(i, worker, &thread_data[i]);
        }
        
        thread_pool_->wait_for_completion();
    }
    
    // Helper function for static block distribution using thread pool
    template<typename Func>
    void parallel_for_range(size_t count, Func func) {
        struct ThreadData {
            Func func;  // Store by value, not pointer
            size_t count;
            int thread_id;
            int num_threads;
        };
        
        auto worker = [](void* arg) -> void* {
            ThreadData* data = static_cast<ThreadData*>(arg);
            
            // Static block distribution - each thread gets a contiguous range
            size_t start = (data->thread_id * data->count) / data->num_threads;
            size_t end = ((data->thread_id + 1) * data->count) / data->num_threads;
            
            for (size_t i = start; i < end; i++) {
                data->func(i);
            }
            return nullptr;
        };
        
        // Use thread pool with proper data management
        std::vector<ThreadData> thread_data(num_threads_);
        for (int i = 0; i < num_threads_; i++) {
            thread_data[i] = {func, count, i, num_threads_};
            thread_pool_->execute_work(i, worker, &thread_data[i]);
        }
        
        thread_pool_->wait_for_completion();
    }
    
    // Helper function for static block distribution with reduction using thread pool
    template<typename Func>
    int64_t parallel_for_range_reduction(size_t count, Func func, int64_t initial_value = 0) {
        std::vector<int64_t> local_results(num_threads_, 0);
        
        struct ThreadData {
            Func func;  // Store by value, not pointer
            size_t count;
            int thread_id;
            int num_threads;
            std::vector<int64_t>* local_results;
        };
        
        auto worker = [](void* arg) -> void* {
            ThreadData* data = static_cast<ThreadData*>(arg);
            int64_t local_result = 0;
            
            // Static block distribution - each thread gets a contiguous range
            size_t start = (data->thread_id * data->count) / data->num_threads;
            size_t end = ((data->thread_id + 1) * data->count) / data->num_threads;
            
            for (size_t i = start; i < end; i++) {
                local_result += data->func(i);
            }
            
            (*data->local_results)[data->thread_id] = local_result;
            return nullptr;
        };
        
        // Use thread pool with proper data management
        std::vector<ThreadData> thread_data(num_threads_);
        for (int i = 0; i < num_threads_; i++) {
            thread_data[i] = {func, count, i, num_threads_, &local_results};
            thread_pool_->execute_work(i, worker, &thread_data[i]);
        }
        
        thread_pool_->wait_for_completion();
        
        int64_t total = initial_value;
        for (int64_t local : local_results) {
            total += local;
        }
        
        return total;
    }
    
    // Parallel region with thread-local storage using thread pool
    template<typename Func>
    int64_t parallel_region(Func func) {
        std::atomic<int64_t> result(0);
        
        struct ThreadData {
            Func func;  // Store by value, not pointer
            int thread_id;
            int num_threads;
            std::atomic<int64_t>* result;
        };
        
        auto worker = [](void* arg) -> void* {
            ThreadData* data = static_cast<ThreadData*>(arg);
            int64_t local_result = data->func(data->thread_id, data->num_threads);
            data->result->fetch_add(local_result);
            return nullptr;
        };
        
        // Use thread pool with proper data management
        std::vector<ThreadData> thread_data(num_threads_);
        for (int i = 0; i < num_threads_; i++) {
            thread_data[i] = {func, i, num_threads_, &result};
            thread_pool_->execute_work(i, worker, &thread_data[i]);
        }
        
        thread_pool_->wait_for_completion();
        
        return result.load();
    }
    
public:
    ~GAPBSPthreads() {
        for (int i = 0; i < num_threads_; i++) {
            pthread_mutex_destroy(&mutexes_[i]);
            pthread_cond_destroy(&conditions_[i]);
        }
    }
};

// Global instance for easy access
extern GAPBSPthreads gapbs_pthreads;

// Initialize with specific number of threads
void gapbs_set_num_threads(int num_threads);

// Macros to replace OpenMP pragmas
#define GAPBS_PARALLEL_FOR_REDUCTION(count, func, result) \
    result = gapbs_pthreads.parallel_for_reduction(count, func, 0)

#define GAPBS_PARALLEL_FOR(count, func) \
    gapbs_pthreads.parallel_for(count, func)

#define GAPBS_PARALLEL_REGION(func, result) \
    result = gapbs_pthreads.parallel_region(func)

// New cleaner macros that hide work distribution logic
#define GAPBS_PARALLEL_FOR_RANGE(count, func) \
    gapbs_pthreads.parallel_for_range(count, func)

#define GAPBS_PARALLEL_FOR_RANGE_REDUCTION(count, func, result) \
    result = gapbs_pthreads.parallel_for_range_reduction(count, func, 0)

#define GAPBS_BARRIER() \
    gapbs_pthreads.barrier()

#define GAPBS_CRITICAL(func) \
    gapbs_pthreads.critical_section(func)

// Worker thread versions for use within parallel work functions
#define GAPBS_WORKER_BARRIER() \
    gapbs_pthreads.worker_barrier()

#define GAPBS_WORKER_CRITICAL(func) \
    gapbs_pthreads.worker_critical_section(func)

#endif  // GAPBS_PTHREADS_H_
