# Agent 2 Probability Generation Optimization Plan

## Overview
This is a stepwise plan to optimize the performance of Agent 2 probability generation in the environment inspection tool. The current implementation processes visibility-based and gap-based probabilities, which can be computationally expensive for large maps with many nodes.

**TARGET FILE**: `inspect_environment.py` and its underlying `environment_inspection_simulation.py`

**CLEANUP NOTE**: Remove the optimization reference comment from `inspect_environment.py` when all optimization phases are complete.

## Current Performance Analysis
- **Primary bottleneck**: Agent 2 probability calculations in `environment_inspection_simulation.py` (lines 2649+)
- **Key components**: 
  - Visibility-based probability calculation (800px range)
  - Gap-based sweep processing (45-degree arcs with 20 angles)
  - Node-to-rod distance calculations for each map graph node
- **Current timing**: Measured using `pygame.time.get_ticks()` in the code

## Optimization Plan

### Phase 1: Performance Profiling and Baseline Measurement ⏳
**Status**: PENDING
**Goal**: Establish current performance metrics and identify specific bottlenecks

#### Tasks:
1. **Add detailed timing measurements**
   - Instrument individual sections of agent 2 probability calculation
   - Measure visibility calculation time vs gap processing time
   - Track node count impact on performance
   
2. **Create performance logging system**
   - Log timing data to file for analysis
   - Track frame rate impact during agent 2 probability mode
   - Measure memory usage patterns

3. **Identify hotspots**
   - Profile which specific calculations take the most time
   - Analyze if gap processing or visibility checks dominate
   - Determine if node iteration or distance calculations are the issue

**Deliverables**:
- Baseline performance metrics
- Detailed timing breakdown
- Identification of primary bottlenecks

---

### Phase 2: Data Structure Optimization ⏳
**Status**: PENDING (depends on Phase 1)
**Goal**: Optimize data structures and reduce redundant calculations

#### Tasks:
1. **Implement spatial indexing**
   - Add spatial hash or quadtree for faster node lookup
   - Pre-compute node neighborhoods within 800px range
   - Cache distance calculations between agent and nodes

2. **Optimize gap calculation caching**
   - Cache gap line calculations between frames
   - Store sweep geometry when agent hasn't moved significantly
   - Pre-compute rod positions for common angles

3. **Reduce node iteration overhead**
   - Filter nodes early by distance before detailed calculations
   - Use bounding box checks before precise distance calculations
   - Implement early termination for out-of-range nodes

**Deliverables**:
- Spatial indexing system for map nodes
- Cached distance calculation system
- Optimized node filtering pipeline

---

### Phase 3: Algorithm Optimization ⏳
**Status**: PENDING (depends on Phase 2)
**Goal**: Improve the core algorithms for probability calculation

#### Tasks:
1. **Vectorize distance calculations**
   - Use NumPy for batch distance calculations
   - Replace individual math.dist() calls with vectorized operations
   - Implement SIMD-friendly algorithms where possible

2. **Optimize gap sweep processing**
   - Reduce from 20 angles to adaptive angle count based on density
   - Use analytical solutions instead of iterative rod checking
   - Implement incremental updates instead of full recalculation

3. **Improve visibility lookup efficiency**
   - Cache visibility map results for nearby positions
   - Use interpolation for intermediate agent positions
   - Implement hierarchical visibility checks

**Deliverables**:
- Vectorized calculation implementations
- Reduced-complexity gap processing
- Enhanced visibility lookup system

---

### Phase 4: Multi-threading and Parallelization ⏳
**Status**: PENDING (depends on Phase 3)
**Goal**: Leverage multiple CPU cores for probability calculations

#### Tasks:
1. **Implement parallel node processing**
   - Split node list across worker threads
   - Use multiprocessing.Pool for CPU-intensive calculations
   - Implement lock-free data structures for results

2. **Asynchronous probability updates**
   - Move probability calculation to background thread
   - Use double-buffering for smooth visualization
   - Implement progressive calculation (partial updates per frame)

3. **GPU acceleration exploration**
   - Investigate OpenCL/CUDA for distance calculations
   - Explore compute shaders for gap processing
   - Benchmark GPU vs CPU performance trade-offs

**Deliverables**:
- Multi-threaded probability calculation system
- Asynchronous update mechanism
- GPU acceleration feasibility study

---

### Phase 5: Caching and Prediction ⏳
**Status**: PENDING (depends on Phase 4)
**Goal**: Implement intelligent caching to avoid redundant calculations

#### Tasks:
1. **Agent movement prediction**
   - Predict agent movement to pre-calculate probabilities
   - Cache results for likely future positions
   - Implement motion-based cache invalidation

2. **Hierarchical probability caching**
   - Cache coarse-grained probability maps
   - Refine only changed regions
   - Use temporal coherence for smooth updates

3. **Adaptive quality control**
   - Reduce calculation precision when agent moves quickly
   - Increase precision when agent is stationary
   - Implement distance-based level-of-detail

**Deliverables**:
- Predictive caching system
- Hierarchical probability maps
- Adaptive quality control mechanism

---

### Phase 6: User Experience Optimization ⏳
**Status**: PENDING (depends on Phase 5)
**Goal**: Ensure optimizations don't degrade visual quality or responsiveness

#### Tasks:
1. **Progressive rendering**
   - Display partial results during calculation
   - Show loading indicators for heavy computations
   - Maintain responsive UI during background processing

2. **Quality vs performance settings**
   - Add user controls for performance/quality trade-offs
   - Implement preset performance profiles
   - Allow runtime adjustment of calculation parameters

3. **Benchmarking and validation**
   - Verify visual quality is maintained
   - Test on various hardware configurations
   - Validate performance improvements across different map sizes

**Deliverables**:
- Progressive rendering system
- User performance controls
- Comprehensive performance validation

---

## Success Metrics
- **Target**: 50%+ reduction in agent 2 probability calculation time
- **Frame rate**: Maintain 60 FPS during agent 2 probability mode
- **Memory usage**: No significant increase in memory consumption
- **Visual quality**: No degradation in probability visualization accuracy

## Implementation Notes
- Each phase will be implemented and tested before proceeding to the next
- Performance measurements will be taken after each phase
- Rollback plan available if optimizations cause regressions
- Code will maintain compatibility with existing features

## Files to Modify
- `multitrack/simulation/environment_inspection_simulation.py` (main optimization target)
- `multitrack/utils/map_graph.py` (for spatial indexing)
- `multitrack/utils/config.py` (for performance settings)
- `probability_visibility_overlay.py` (if relevant optimizations apply)

## Backup and Testing Strategy
- Create performance test suite before modifications
- Maintain original implementation as fallback
- Test on multiple map configurations and hardware setups
- Validate against current visual output for consistency

---

**Next Step**: Begin Phase 1 with detailed performance profiling and measurement.
