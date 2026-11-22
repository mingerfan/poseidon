// One bounded, persistent producer. Reuses the OpenMP team across matrices
// instead of spawning a new std::async thread (and OpenMP team) for each tile.
// One-slot lookahead matches the previous pipeline's memory/overlap boundary.
#pragma once
#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <stdexcept>
namespace cerium_source {
template<class Work> class BoundedPreparation {
    std::mutex mutex_;std::condition_variable changed_;std::optional<Work> slot_;
    std::exception_ptr failure_;bool stop_=false,finished_=false;std::thread worker_;
public:
    BoundedPreparation(std::size_t count,std::function<Work(std::size_t)> prepare)
        :worker_([this,count,prepare=std::move(prepare)] {
            try {
                for(std::size_t i=0;i<count;++i) {
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        changed_.wait(lock,[this]{return stop_||!slot_;});
                        if(stop_)return;
                    }
                    auto work=prepare(i);
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        if(stop_)return;
                        slot_.emplace(std::move(work));
                    }
                    changed_.notify_all();
                }
                {std::lock_guard<std::mutex> lock(mutex_);finished_=true;}
                changed_.notify_all();
            } catch(...) {
                {std::lock_guard<std::mutex> lock(mutex_);failure_=std::current_exception();}
                changed_.notify_all();
            }
        }) {}
    BoundedPreparation(const BoundedPreparation &)=delete;
    BoundedPreparation &operator=(const BoundedPreparation &)=delete;
    ~BoundedPreparation() {
        {std::lock_guard<std::mutex> lock(mutex_);stop_=true;}
        changed_.notify_all();worker_.join();
    }
    Work get() {
        std::unique_lock<std::mutex> lock(mutex_);
        changed_.wait(lock,[this]{return slot_||failure_||finished_;});
        if(failure_)std::rethrow_exception(failure_);
        if(!slot_)throw std::out_of_range("preparation pipeline exhausted");
        auto out=std::move(*slot_);slot_.reset();lock.unlock();changed_.notify_all();return out;
    }
};
} // namespace cerium_source
