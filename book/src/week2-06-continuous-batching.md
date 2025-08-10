# Week 2 Day 6: Continuous Batching

Continuous batching is a key optimization for LLM serving systems that dramatically improves throughput by serving multiple requests simultaneously. Unlike traditional static batching, continuous batching allows requests to join and leave the batch dynamically, maximizing GPU utilization.

## The Problem with Static Batching

**Static batching** processes all requests in a batch together:
```
Batch: [Request1, Request2, Request3, Request4]
- All requests start together
- All requests finish together  
- Batch cannot start new requests until all current requests complete
- GPU underutilized when requests have different lengths
```

**Issues**:
- **Head-of-line blocking**: Fast requests wait for slow requests
- **Poor utilization**: GPU idle when fewer than batch_size requests available
- **Memory waste**: Padding sequences to maximum length in batch

## Continuous Batching Solution

**Continuous batching** allows dynamic request management:
```
Time 0: [Req1(new), Req2(new), Req3(new)]
Time 1: [Req1, Req2, Req3, Req4(new)]         # Add new request
Time 2: [Req1, Req3, Req4, Req5(new)]         # Req2 finished, add Req5  
Time 3: [Req3, Req4, Req5, Req6(new)]         # Req1 finished, add Req6
```

**Benefits**:
- **No head-of-line blocking**: Requests complete independently
- **Higher utilization**: Always fill available batch slots
- **Lower latency**: New requests start immediately when slots available

**Readings**

- [vLLM Continuous Batching](https://blog.vllm.ai/2023/06/20/vllm.html)
- [Orca: Continuous Batching Paper](https://www.usenix.org/conference/osdi22/presentation/yu)
- [PagedAttention and Dynamic Batching](https://arxiv.org/abs/2309.06180)

## Task 1: Understanding Request Lifecycle

Each request goes through multiple phases:

```python
class Request:
    def __init__(self, prompt: str, max_tokens: int):
        self.state = "waiting"     # waiting -> prefilling -> decoding -> finished
        self.tokens = tokenize(prompt)
        self.kv_cache = None
        self.generated_tokens = []
        self.offset = 0           # Current position in sequence
```

**Request states**:
1. **Waiting**: In queue, not yet started
2. **Prefilling**: Processing initial prompt tokens  
3. **Decoding**: Generating new tokens one by one
4. **Finished**: Completed (EOS token or max length)

## Task 2: Batching KV Cache Management

The key challenge is managing KV caches for variable-length sequences:

```python
class BatchingKvCache:
    def __init__(self, max_active_requests: int, max_seq_len: int):
        self.max_active_requests = max_active_requests
        self.max_seq_len = max_seq_len
        self.kv_caches = [None] * max_active_requests
        
    def add_request(self, prefilled_cache: TinyKvCache, slot_id: int):
        """Add a new request to the batch"""
        self.kv_caches[slot_id] = prefilled_cache
        
    def remove_request(self, slot_id: int):
        """Remove finished request from batch"""
        self.kv_caches[slot_id] = None
        
    def update_and_fetch(self, keys, values, mask_length=None, mask=None):
        """Update all active caches and return batched tensors"""
        # Handle variable sequence lengths
        # Pad/mask appropriately for batch processing
```

**Key challenges**:
- **Variable sequence lengths**: Requests have different current lengths
- **Memory layout**: Efficiently pack variable-length sequences
- **Attention masking**: Proper masks for each request's valid positions

## Task 3: Implement Batch Generation

Examine the batch generation implementation in `src/tiny_llm_ref/batch.py`:

```python
def batch_generate(
    model, tokenizer, prompts: list[str], 
    max_seq_len=512, batch_size=5, prefill_step=128
):
    # State management
    decode_requests = [None] * batch_size  # Active decode requests
    is_idle = [True] * batch_size          # Slot availability
    pending_prefill_request = None         # Request currently prefilling
    
    # Batched KV cache for all layers
    kv_cache = [
        BatchingKvCache(max_active_requests=batch_size, max_seq_len=max_seq_len)
        for _ in range(model.num_hidden_layers)
    ]
    
    while has_pending_work():
        # Phase 1: Prefill new requests
        if can_start_prefill():
            handle_prefill_phase()
            
        # Phase 2: Decode existing requests  
        if has_active_requests():
            handle_decode_phase()
            
        # Phase 3: Cleanup finished requests
        cleanup_finished_requests()
```

## Task 4: Prefill vs Decode Phases

**Prefill phase** (one request at a time):
```python
def handle_prefill_phase():
    if pending_prefill_request and not pending_prefill_request.is_prefill_done:
        # Process part of the prefill (chunked)
        tokens_to_process = min(prefill_step, remaining_tokens)
        logits = model(
            inputs=tokens_to_process,
            offset=current_offset,
            cache=request.kv_cache
        )
        pending_prefill_request.offset += tokens_to_process
        
        if prefill_complete():
            # Move to decode phase
            add_to_decode_batch(pending_prefill_request)
            pending_prefill_request = None
```

**Decode phase** (all requests together):
```python
def handle_decode_phase():
    # Collect next tokens from all active requests
    next_tokens = []
    offsets = []
    for slot_id, request in enumerate(decode_requests):
        if request is not None:
            next_tokens.append(request.next_token)
            offsets.append(request.offset)
        else:
            next_tokens.append(0)  # Padding for inactive slots
            offsets.append(0)
    
    # Batch decode all requests together
    logits = model(
        inputs=mx.array(next_tokens).reshape(-1, 1),
        offsets=offsets,
        cache=kv_cache  # Batched cache
    )
    
    # Update each request with its new token
    for slot_id, request in enumerate(decode_requests):
        if request is not None:
            new_token = sample(logits[slot_id])
            request.add_token(new_token)
```

## Task 5: Memory Management and Slot Allocation

Efficient slot management is crucial:

```python
class SlotManager:
    def __init__(self, max_slots: int):
        self.available_slots = set(range(max_slots))
        self.allocated_slots = {}  # request_id -> slot_id
        
    def allocate_slot(self, request_id: str) -> int | None:
        if not self.available_slots:
            return None  # No available slots
        
        slot_id = self.available_slots.pop()
        self.allocated_slots[request_id] = slot_id
        return slot_id
        
    def free_slot(self, request_id: str):
        if request_id in self.allocated_slots:
            slot_id = self.allocated_slots.pop(request_id)
            self.available_slots.add(slot_id)
            # Clear the KV cache for this slot
            self.clear_kv_cache(slot_id)
```

## Task 6: Attention Masking for Variable Lengths

Handle different sequence lengths in the same batch:

```python
def create_batch_mask(requests: list[Request], max_seq_len: int):
    batch_size = len(requests)
    masks = mx.full((batch_size, 1, 1, max_seq_len), -mx.inf)
    
    for i, request in enumerate(requests):
        if request is not None:
            seq_len = request.offset
            # Allow attention to all positions up to current length
            masks[i, 0, 0, :seq_len] = 0.0
    
    return masks
```

## Task 7: Performance Metrics and Monitoring

Track important metrics for continuous batching:

```python
class BatchingMetrics:
    def __init__(self):
        self.total_requests = 0
        self.active_requests = 0
        self.queue_size = 0
        self.throughput_tokens_per_sec = 0
        self.average_latency = 0
        self.gpu_utilization = 0
        
    def update(self, batch_state):
        self.active_requests = sum(1 for req in batch_state if req is not None)
        self.gpu_utilization = self.active_requests / self.max_batch_size
        # ... update other metrics
```

**Key metrics**:
- **Throughput**: Total tokens generated per second across all requests
- **Latency**: Time from request start to completion
- **Queue time**: Time spent waiting for available slot
- **GPU utilization**: Fraction of batch slots actively processing

## Task 8: Scheduling Strategies

Different strategies for request scheduling:

**FIFO (First In, First Out)**:
```python
def schedule_next_request():
    return request_queue.pop(0)  # Simple FIFO
```

**Priority-based**:
```python
def schedule_next_request():
    # Priority by estimated completion time
    return min(request_queue, key=lambda r: r.estimated_tokens_remaining)
```

**Load balancing**:
```python
def schedule_next_request():
    # Balance sequence lengths in current batch
    current_avg_length = sum(req.offset for req in active_requests) / len(active_requests)
    return best_request_for_balance(request_queue, current_avg_length)
```

## Task 9: Testing and Benchmarking

```bash
# Test continuous batching implementation
pdm run test --week 2 --day 6 -- -k continuous_batching

# Benchmark throughput with different batch sizes
pdm run python -c "
from tiny_llm_ref.batch import batch_generate
from mlx_lm import load

model, tokenizer = load('Qwen/Qwen2-0.5B-Instruct-MLX')
prompts = ['Tell me about'] * 10

# Compare different batch sizes
for batch_size in [1, 2, 4, 8]:
    start_time = time.time()
    results = batch_generate(model, tokenizer, prompts, batch_size=batch_size)
    total_time = time.time() - start_time
    print(f'Batch size {batch_size}: {total_time:.2f}s')
"
```

Expected results:
- **Higher throughput**: 5-15x improvement over sequential processing
- **Lower latency**: Requests start immediately when slots available
- **Better utilization**: GPU stays busy even with variable request lengths

## Integration with Real Serving

Production LLM serving systems extend this pattern:

1. **Request queuing**: HTTP endpoints add requests to queues
2. **Load balancing**: Multiple model instances handle requests
3. **Memory management**: More sophisticated KV cache strategies (paged attention)
4. **Scheduling**: Advanced algorithms for fairness and efficiency

At the end of this day, you should understand how continuous batching enables efficient multi-request serving with dynamic request management.

```bash
pdm run test --week 2 --day 6
```

{{#include copyright.md}}
