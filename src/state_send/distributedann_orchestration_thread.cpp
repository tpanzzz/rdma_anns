#include "ssd_partition_index.h"

template <typename T, typename TagT>
uint64_t
SSDPartitionIndex<T,
                  TagT>::OrchestrationThread::get_random_scoring_server_id() {
  static thread_local std::random_device dev;
  static thread_local std::mt19937 rng(dev());
  static thread_local std::uniform_int_distribution<std::mt19937::result_type>
  dist(0, this->parent->other_peer_ids.size() - 1);
  return this->parent->other_peer_ids[dist(rng)];
}

template <typename T, typename TagT>
void SSDPartitionIndex<T, TagT>::OrchestrationThread::main_loop_batch() {
  std::array<SearchState<T, TagT> *, max_queries_balance> allocated_states;
  
  while (this->running) {
    uint64_t num_states_to_dequeue =
        this->parent->num_queries_balance - this->number_concurrent_queries;
    if (num_states_to_dequeue > 0) {
      size_t num_dequeued = this->parent->global_state_queue.try_dequeue_bulk(
          this->search_thread_consumer_token, allocated_states.begin(),
									      num_states_to_dequeue);
      add_states_to_batch(allocated_states.data(), num_dequeued);
    }
    search_result_t *computation_result;
    this->parent->preallocated_result_queue.dequeue_exact(1,
                                                          &computation_result);
    bool dequeued = computation_result_queue.try_dequeue(computation_result);
    if (!dequeued) {
      this->parent->preallocated_result_queue.free(computation_result);
      continue;
    }

    
  }

}



template class SSDPartitionIndex<float>::OrchestrationThread;
template class SSDPartitionIndex<uint8_t>::OrchestrationThread;
template class SSDPartitionIndex<int8_t>::OrchestrationThread;
