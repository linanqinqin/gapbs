// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef PTHREADPP_H_
#define PTHREADPP_H_

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
#include <chrono>

// Atomic operations are now handled by platform_atomics.h

/*
PthreadPP - pthread Parallel Programming Framework
File:   PthreadPP Header
Author: Custom Implementation

Direct pthreads implementation to replace OpenMP pragmas
- Provides equivalent functionality to OpenMP parallel constructs
- Designed for cooperative threading model integration
- Maintains same performance characteristics as OpenMP
- General-purpose parallel programming framework
*/

class PthreadPP {
private:
    int num_threads_;
    std::vector<pthread_t> threads_;
    std::vector<pthread_mutex_t> mutexes_;
    std::vector<pthread_cond_t> conditions_;
    
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
    explicit PthreadPP(int num_threads = 0) 
        : num_threads_(num_threads ? num_threads : get_default_thread_count()) {
        threads_.resize(num_threads_);
        mutexes_.resize(num_threads_);
        conditions_.resize(num_threads_);
        
        for (int i = 0; i < num_threads_; i++) {
            pthread_mutex_init(&mutexes_[i], nullptr);
            pthread_cond_init(&conditions_[i], nullptr);
        }
    }
    
    // Set number of threads at runtime
    void set_num_threads(int num_threads) {
        if (num_threads > 0 && num_threads != num_threads_) {
            // Clean up existing resources
            for (int i = 0; i < num_threads_; i++) {
                pthread_mutex_destroy(&mutexes_[i]);
                pthread_cond_destroy(&conditions_[i]);
            }
            
            num_threads_ = num_threads;
            threads_.resize(num_threads_);
            mutexes_.resize(num_threads_);
            conditions_.resize(num_threads_);
            
            for (int i = 0; i < num_threads_; i++) {
                pthread_mutex_init(&mutexes_[i], nullptr);
                pthread_cond_init(&conditions_[i], nullptr);
            }
        }
    }
    
private:
    static int get_default_thread_count() {
        // Check P3 environment variable
        const char* p3_threads = getenv("P3_NUM_THREADS");
        if (p3_threads) {
            int threads = std::atoi(p3_threads);
            if (threads > 0) return threads;
        }
        
        // Default to hardware concurrency
        return std::thread::hardware_concurrency();
    }
    
    int get_num_threads() const { return num_threads_; }
    
public:
    // Barrier synchronization
    void barrier() {
        if (!barrier_) {
            barrier_ = make_unique<Barrier>(num_threads_);
        }
        barrier_->wait();
    }
    
    // Critical section
    void critical_section(std::function<void()> func) {
        static pthread_mutex_t critical_mutex = PTHREAD_MUTEX_INITIALIZER;
        pthread_mutex_lock(&critical_mutex);
        func();
        pthread_mutex_unlock(&critical_mutex);
    }
    
    // Atomic operations support
    template<typename T>
    void atomic_add(T& variable, T value) {
        static pthread_mutex_t atomic_mutex = PTHREAD_MUTEX_INITIALIZER;
        pthread_mutex_lock(&atomic_mutex);
        variable += value;
        pthread_mutex_unlock(&atomic_mutex);
    }
    
    template<typename T>
    void atomic_max(T& variable, T value) {
        static pthread_mutex_t atomic_mutex = PTHREAD_MUTEX_INITIALIZER;
        pthread_mutex_lock(&atomic_mutex);
        if (value > variable) {
            variable = value;
        }
        pthread_mutex_unlock(&atomic_mutex);
    }
    
    // Single directive support - only one thread executes the function
    void single(std::function<void()> func) {
        static pthread_mutex_t single_mutex = PTHREAD_MUTEX_INITIALIZER;
        static std::atomic<int> thread_count{0};
        static std::atomic<bool> executed{false};
        
        // Reset for each parallel region
        int count = thread_count.fetch_add(1);
        if (count == 0) {
            executed.store(false);
        }
        
        bool expected = false;
        if (executed.compare_exchange_strong(expected, true)) {
            pthread_mutex_lock(&single_mutex);
            func();
            pthread_mutex_unlock(&single_mutex);
        }
        
        // Wait for all threads to reach this point
        while (thread_count.load() < num_threads_) {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
        
        if (count == num_threads_ - 1) {
            thread_count.store(0);
        }
    }
    // Parallel for with reduction
    template<typename Func>
    int64_t parallel_for_reduction(size_t count, Func func, int64_t initial_value = 0) {
        std::atomic<int64_t> result(initial_value);
        std::vector<int64_t> local_results(num_threads_, 0);
        
        struct ThreadData {
            Func* func;
            size_t count;
            int thread_id;
            int num_threads;
            std::vector<int64_t>* local_results;
            
            ThreadData() = default;
            ThreadData(Func* f, size_t c, int tid, int nt, std::vector<int64_t>* lr)
                : func(f), count(c), thread_id(tid), num_threads(nt), local_results(lr) {}
        };
        
        auto worker = [](void* arg) -> void* {
            ThreadData* data = static_cast<ThreadData*>(arg);
            int64_t local_result = 0;
            
            // Dynamic scheduling simulation
            for (size_t i = data->thread_id; i < data->count; i += data->num_threads) {
                local_result += (*data->func)(i);
            }
            
            (*data->local_results)[data->thread_id] = local_result;
            return nullptr;
        };
        
        std::vector<ThreadData> thread_data(num_threads_);
        for (int i = 0; i < num_threads_; i++) {
            thread_data[i] = ThreadData(&func, count, i, num_threads_, &local_results);
            pthread_create(&threads_[i], nullptr, worker, &thread_data[i]);
        }
        
        for (int i = 0; i < num_threads_; i++) {
            pthread_join(threads_[i], nullptr);
        }
        
        int64_t total = initial_value;
        for (int64_t local : local_results) {
            total += local;
        }
        
        return total;
    }
    
    // Parallel for with max reduction
    template<typename Func, typename T>
    T parallel_for_max_reduction(size_t count, Func func, T initial_value) {
        std::vector<T> local_results(num_threads_, initial_value);
        
        struct ThreadData {
            Func* func;
            size_t count;
            int thread_id;
            int num_threads;
            std::vector<T>* local_results;
            
            ThreadData() = default;
            ThreadData(Func* f, size_t c, int tid, int nt, std::vector<T>* lr)
                : func(f), count(c), thread_id(tid), num_threads(nt), local_results(lr) {}
        };
        
        auto worker = [](void* arg) -> void* {
            ThreadData* data = static_cast<ThreadData*>(arg);
            T local_max = data->local_results->at(0); // Use initial value
            
            for (size_t i = data->thread_id; i < data->count; i += data->num_threads) {
                T result = (*data->func)(i);
                if (result > local_max) {
                    local_max = result;
                }
            }
            
            (*data->local_results)[data->thread_id] = local_max;
            return nullptr;
        };
        
        std::vector<ThreadData> thread_data(num_threads_);
        for (int i = 0; i < num_threads_; i++) {
            thread_data[i] = ThreadData(&func, count, i, num_threads_, &local_results);
            pthread_create(&threads_[i], nullptr, worker, &thread_data[i]);
        }
        
        for (int i = 0; i < num_threads_; i++) {
            pthread_join(threads_[i], nullptr);
        }
        
        T max_result = initial_value;
        for (T local : local_results) {
            if (local > max_result) {
                max_result = local;
            }
        }
        
        return max_result;
    }
    
    // Parallel for without reduction
    template<typename Func>
    void parallel_for(size_t count, Func func) {
        struct ThreadData {
            Func* func;
            size_t count;
            int thread_id;
            int num_threads;
            
            ThreadData() = default;
            ThreadData(Func* f, size_t c, int tid, int nt)
                : func(f), count(c), thread_id(tid), num_threads(nt) {}
        };
        
        auto worker = [](void* arg) -> void* {
            ThreadData* data = static_cast<ThreadData*>(arg);
            
            for (size_t i = data->thread_id; i < data->count; i += data->num_threads) {
                (*data->func)(i);
            }
            return nullptr;
        };
        
        std::vector<ThreadData> thread_data(num_threads_);
        for (int i = 0; i < num_threads_; i++) {
            thread_data[i] = ThreadData(&func, count, i, num_threads_);
            pthread_create(&threads_[i], nullptr, worker, &thread_data[i]);
        }
        
        for (int i = 0; i < num_threads_; i++) {
            pthread_join(threads_[i], nullptr);
        }
    }
    
    // Parallel region with thread-local storage
    template<typename Func>
    int64_t parallel_region(Func func) {
        std::atomic<int64_t> result(0);
        
        struct ThreadData {
            Func* func;
            int thread_id;
            int num_threads;
            std::atomic<int64_t>* result;
            
            ThreadData() = default;
            ThreadData(Func* f, int tid, int nt, std::atomic<int64_t>* r)
                : func(f), thread_id(tid), num_threads(nt), result(r) {}
        };
        
        auto worker = [](void* arg) -> void* {
            ThreadData* data = static_cast<ThreadData*>(arg);
            int64_t local_result = (*data->func)(data->thread_id, data->num_threads);
            data->result->fetch_add(local_result);
            return nullptr;
        };
        
        std::vector<ThreadData> thread_data(num_threads_);
        for (int i = 0; i < num_threads_; i++) {
            thread_data[i] = ThreadData(&func, i, num_threads_, &result);
            pthread_create(&threads_[i], nullptr, worker, &thread_data[i]);
        }
        
        for (int i = 0; i < num_threads_; i++) {
            pthread_join(threads_[i], nullptr);
        }
        
        return result.load();
    }
    
    // Parallel region with private variables support
    template<typename Func>
    void parallel_region_private(Func func) {
        struct ThreadData {
            Func* func;
            int thread_id;
            int num_threads;
            
            ThreadData() = default;
            ThreadData(Func* f, int tid, int nt)
                : func(f), thread_id(tid), num_threads(nt) {}
        };
        
        auto worker = [](void* arg) -> void* {
            ThreadData* data = static_cast<ThreadData*>(arg);
            (*data->func)(data->thread_id, data->num_threads);
            return nullptr;
        };
        
        std::vector<ThreadData> thread_data(num_threads_);
        for (int i = 0; i < num_threads_; i++) {
            thread_data[i] = ThreadData(&func, i, num_threads_);
            pthread_create(&threads_[i], nullptr, worker, &thread_data[i]);
        }
        
        for (int i = 0; i < num_threads_; i++) {
            pthread_join(threads_[i], nullptr);
        }
    }
    
    // Parallel for with dynamic scheduling
    template<typename Func>
    void parallel_for_dynamic(size_t count, Func func, size_t chunk_size = 64) {
        struct ThreadData {
            Func* func;
            size_t count;
            size_t chunk_size;
            int thread_id;
            int num_threads;
            std::atomic<size_t>* next_chunk;
            
            ThreadData() = default;
            ThreadData(Func* f, size_t c, size_t cs, int tid, int nt, std::atomic<size_t>* nc)
                : func(f), count(c), chunk_size(cs), thread_id(tid), num_threads(nt), next_chunk(nc) {}
        };
        
        auto worker = [](void* arg) -> void* {
            ThreadData* data = static_cast<ThreadData*>(arg);
            
            while (true) {
                size_t chunk_start = data->next_chunk->fetch_add(data->chunk_size);
                if (chunk_start >= data->count) break;
                
                size_t chunk_end = std::min(chunk_start + data->chunk_size, data->count);
                for (size_t i = chunk_start; i < chunk_end; i++) {
                    (*data->func)(i);
                }
            }
            return nullptr;
        };
        
        std::atomic<size_t> next_chunk(0);
        std::vector<ThreadData> thread_data(num_threads_);
        for (int i = 0; i < num_threads_; i++) {
            thread_data[i] = ThreadData(&func, count, chunk_size, i, num_threads_, &next_chunk);
            pthread_create(&threads_[i], nullptr, worker, &thread_data[i]);
        }
        
        for (int i = 0; i < num_threads_; i++) {
            pthread_join(threads_[i], nullptr);
        }
    }
    
public:
    ~PthreadPP() {
        for (int i = 0; i < num_threads_; i++) {
            pthread_mutex_destroy(&mutexes_[i]);
            pthread_cond_destroy(&conditions_[i]);
        }
    }
};

// Singleton instance for global access
inline PthreadPP& get_p3_instance() {
    static PthreadPP instance;
    return instance;
}

// Initialize with specific number of threads
inline void p3_set_num_threads(int num_threads) {
    get_p3_instance().set_num_threads(num_threads);
}

// P3 Macros to replace OpenMP pragmas
#define P3_PARALLEL_FOR_REDUCTION(count, func, result) \
    result = get_p3_instance().parallel_for_reduction(count, func, 0)

#define P3_PARALLEL_FOR_MAX_REDUCTION(count, func, result, initial_value) \
    result = get_p3_instance().parallel_for_max_reduction(count, func, initial_value)

#define P3_PARALLEL_FOR(count, func) \
    get_p3_instance().parallel_for(count, func)

#define P3_PARALLEL_FOR_DYNAMIC(count, func, chunk_size) \
    get_p3_instance().parallel_for_dynamic(count, func, chunk_size)

#define P3_PARALLEL_REGION(func, result) \
    result = get_p3_instance().parallel_region(func)

#define P3_PARALLEL_REGION_PRIVATE(func) \
    get_p3_instance().parallel_region_private(func)

#define P3_BARRIER() \
    get_p3_instance().barrier()

#define P3_CRITICAL(func) \
    get_p3_instance().critical_section(func)

#define P3_ATOMIC_ADD(variable, value) \
    get_p3_instance().atomic_add(variable, value)

#define P3_ATOMIC_MAX(variable, value) \
    get_p3_instance().atomic_max(variable, value)

#define P3_SINGLE(func) \
    get_p3_instance().single(func)

#endif  // PTHREADPP_H_
