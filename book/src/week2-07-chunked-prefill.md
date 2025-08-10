# Week 2 Day 7: Chunked Prefill

Chunked prefill is an optimization technique that breaks large prefill sequences into smaller chunks, enabling better interleaving with decode requests and improved system responsiveness. This is especially important for long-context applications where prefill can take seconds.

## The Prefill vs Decode Problem

**Prefill characteristics**:
- Processes many tokens at once (100s to 1000s)
- High compute utilization (parallel processing)
- High memory bandwidth usage
- Can take seconds for long contexts

**Decode characteristics**:
- Processes one token at a time
- Lower compute utilization (more memory-bound)
- Fast per-step (milliseconds)
- Interactive - users expect low latency

**The conflict**: A long prefill blocks all decode requests, causing poor user experience.

## Traditional vs Chunked Prefill

**Traditional prefill**:
```
Prefill: [████████████████████████████████] (2 seconds, blocks everything)
Decode:                                      [█][█][█][█] (fast, but delayed)
```

**Chunked prefill**:
```
Chunk 1: [████████] 
Decode:           [█][█]
Chunk 2:               [████████]
Decode:                         [█][█]  
Chunk 3:                             [████████]
Decode:                                       [█][█]
```

**Benefits**:
- **Lower decode latency**: Decode requests interleaved with prefill chunks
- **Better fairness**: No single request monopolizes the system
- **Graceful degradation**: System remains responsive under load

**Readings**

- [vLLM Chunked Prefill Implementation](https://github.com/vllm-project/vllm/pull/4484)
- [Orca Paper on Request Scheduling](https://www.usenix.org/conference/osdi22/presentation/yu)
- [Splitting Long Sequences in Transformers](https://arxiv.org/abs/2004.05150)

## Task 1: Understanding Chunking Strategy

The key insight is that prefill can be split across multiple iterations:

```python
# Original prefill: process all 1000 tokens at once
tokens = [1, 2, 3, ..., 1000]
logits = model(tokens, offset=0, cache=kv_cache)  # Blocks for 2 seconds

# Chunked prefill: process in chunks of 128 tokens
chunk_size = 128
for i in range(0, len(tokens), chunk_size):
    chunk = tokens[i:i + chunk_size]
    logits = model(chunk, offset=i, cache=kv_cache)  # 0.2 seconds each
    # Between chunks: process decode requests
    handle_decode_requests()
```

## Task 2: Implement Chunked Prefill in Request Class

Modify the `Request` class to support chunked prefill:

```python
class Request:
    def __init__(self, prompt: str, prefill_max_step: int = 128):
        self.prefill_tokens = tokenize(prompt)
        self.prefill_max_step = prefill_max_step  # Chunk size
        self.is_prefill_done = False
        self.offset = 0  # Current position in prefill
        
    def try_prefill(self):
        """Process one chunk of prefill tokens"""
        if self.is_prefill_done:
            raise ValueError("Prefill already complete")
            
        # Determine chunk size
        remaining_tokens = len(self.prefill_tokens) - self.offset
        tokens_to_process = min(self.prefill_max_step, remaining_tokens)
        
        # Process this chunk
        chunk = self.prefill_tokens[self.offset:self.offset + tokens_to_process]
        logits = self.model(chunk, offset=self.offset, cache=self.kv_cache)
        
        # Update state
        self.offset += tokens_to_process
        
        # Check if prefill is complete
        if self.offset >= len(self.prefill_tokens):
            self.is_prefill_done = True
            # Extract the last token for decode phase
            self.next_token = sample(logits[:, -1, :])
            
        return tokens_to_process
```

## Task 3: Scheduler Integration

The batch scheduler must now handle partial prefill requests:

```python
def batch_generate_with_chunked_prefill(
    model, tokenizer, prompts, 
    batch_size=5, prefill_step=128, max_seq_len=512
):
    decode_requests = [None] * batch_size
    is_idle = [True] * batch_size
    pending_prefill_request = None
    request_queue = [Request(p, prefill_step) for p in prompts]
    
    while has_work_remaining():
        made_progress = False
        
        # Phase 1: Continue or start prefill
        if handle_prefill_phase():
            made_progress = True
            
        # Phase 2: Process decode requests
        if handle_decode_phase():
            made_progress = True
            
        # Phase 3: Move completed prefills to decode
        if move_prefill_to_decode():
            made_progress = True
            
        if not made_progress:
            break
            
    return collect_results()

def handle_prefill_phase():
    """Process one chunk of the current prefill request"""
    global pending_prefill_request
    
    # Start new prefill if none active
    if pending_prefill_request is None and request_queue:
        pending_prefill_request = request_queue.pop(0)
        
    # Process one chunk
    if pending_prefill_request and not pending_prefill_request.is_prefill_done:
        pending_prefill_request.try_prefill()
        return True
        
    return False
```

## Task 4: Memory and Compute Trade-offs

Chunked prefill involves several trade-offs:

**Chunk size selection**:
```python
def select_optimal_chunk_size(sequence_length: int, available_memory: int, 
                            active_decode_requests: int):
    # Smaller chunks: Better interleaving, more overhead
    # Larger chunks: Better efficiency, worse interleaving
    
    if active_decode_requests > batch_size * 0.8:
        return 64   # Prioritize decode responsiveness
    elif sequence_length > 2048:
        return 256  # Efficiency for long sequences
    else:
        return 128  # Balanced default
```

**Memory considerations**:
- **KV cache growth**: Each chunk adds to the cache
- **Peak memory**: Chunk size affects peak memory usage
- **Memory fragmentation**: Partial caches use memory less efficiently

## Task 5: Adaptive Chunking

Advanced implementations use adaptive chunk sizes:

```python
class AdaptiveChunker:
    def __init__(self):
        self.decode_latency_target = 0.05  # 50ms target
        self.recent_chunk_times = []
        
    def select_chunk_size(self, base_chunk_size: int, decode_pressure: float):
        # Measure recent prefill chunk timing
        avg_chunk_time = sum(self.recent_chunk_times) / len(self.recent_chunk_times)
        
        # Adjust based on decode pressure
        if decode_pressure > 0.8:  # Many waiting decode requests
            # Use smaller chunks for better interleaving
            return max(32, base_chunk_size // 2)
        elif avg_chunk_time < self.decode_latency_target:
            # We have headroom, can use larger chunks
            return min(512, base_chunk_size * 2)
        else:
            return base_chunk_size
```

## Task 6: Attention Pattern Considerations

Chunked prefill must maintain correct attention patterns:

```python
def chunked_attention_mask(chunk_start: int, chunk_size: int, total_length: int):
    """Create attention mask for a prefill chunk"""
    L = chunk_size
    S = chunk_start + chunk_size  # Total context seen so far
    
    # Each token in chunk can attend to:
    # 1. All previous tokens (from earlier chunks)
    # 2. Tokens up to its position in current chunk
    mask = mx.full((L, S), 0.0)
    
    for i in range(L):
        current_pos = chunk_start + i
        # Causal mask: can only attend to positions <= current_pos
        mask[i, current_pos + 1:] = -mx.inf
        
    return mask
```

**Key insight**: Each chunk sees the full context built up from previous chunks, maintaining the causal attention property.

## Task 7: Performance Monitoring

Track metrics specific to chunked prefill:

```python
class ChunkedPrefillMetrics:
    def __init__(self):
        self.prefill_chunks_processed = 0
        self.avg_chunk_time = 0.0
        self.decode_requests_served = 0
        self.prefill_to_decode_ratio = 0.0
        
    def record_prefill_chunk(self, chunk_time: float, chunk_size: int):
        self.prefill_chunks_processed += 1
        self.avg_chunk_time = (
            self.avg_chunk_time * 0.9 + chunk_time * 0.1
        )  # Exponential moving average
        
    def record_decode_step(self, decode_time: float):
        self.decode_requests_served += 1
        
    def get_interleaving_ratio(self):
        """Ratio of prefill chunks to decode steps"""
        if self.decode_requests_served == 0:
            return float('inf')
        return self.prefill_chunks_processed / self.decode_requests_served
```

## Task 8: Testing Chunked Prefill

```bash
# Test chunked prefill implementation
pdm run test --week 2 --day 7 -- -k chunked_prefill

# Benchmark latency improvement
pdm run python -c "
import time
from tiny_llm_ref.batch import batch_generate

# Test with long prompts
long_prompts = ['Write a detailed essay about ' + 'artificial intelligence ' * 100] * 3
short_prompts = ['Hello'] * 10

# Mixed workload: long prefills + short interactive requests
all_prompts = long_prompts + short_prompts

# Compare chunked vs non-chunked
for prefill_step in [32, 128, 512, 999999]:  # 999999 = no chunking
    start = time.time()
    results = batch_generate(
        model, tokenizer, all_prompts, 
        prefill_step=prefill_step, batch_size=5
    )
    total_time = time.time() - start
    print(f'Chunk size {prefill_step}: {total_time:.2f}s')
"
```

Expected results:
- **Smaller chunks**: Lower latency for decode requests, slightly higher total time
- **Larger chunks**: Higher efficiency, but worse interactivity
- **Optimal chunk size**: Balance between efficiency and responsiveness

## Task 9: Production Considerations

Real serving systems extend chunked prefill with:

**Priority queues**:
```python
class PriorityScheduler:
    def __init__(self):
        self.high_priority_queue = []  # Interactive requests
        self.low_priority_queue = []   # Batch processing
        
    def schedule_next(self):
        # Always prioritize interactive requests
        if self.high_priority_queue:
            return self.high_priority_queue.pop(0)
        return self.low_priority_queue.pop(0)
```

**Dynamic resource allocation**:
- Reserve some batch slots for decode-only
- Scale chunk sizes based on system load
- Preempt long prefills during peak load

**Memory management**:
- Stream prefill chunks to avoid memory spikes
- Use compression for inactive KV caches
- Implement cache eviction policies

At the end of this day, you should understand how chunked prefill enables better system responsiveness by interleaving long prefill operations with fast decode requests.

```bash
pdm run test --week 2 --day 7
```

{{#include copyright.md}}
