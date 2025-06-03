# Agent 2 Probability Generation Optimization Plan

## Overview
This is a stepwise plan to optimize the performance of Agent 2 probability generation in the environment inspection tool. The current implementation processes visibility-based and gap-based probabilities, which can be computationally expensive for large maps with many nodes.

**TARGET FILE**: `inspect_environment.py` and its underlying `environment_inspection_simulation.py`

**CLEANUP NOTE**: Remove the optimization reference comment from `inspect_environment.py` when all optimization phases are complete.

**IMPORTANT FOR LLMs**: This file serves as both a roadmap AND a living memory system. When making progress on any optimization task:
1. **ALWAYS UPDATE** the "Progress Memory & Status Tracker" section to reflect new achievements
2. **MOVE COMPLETED ITEMS** from ❌ "Remaining Tasks" to ✅ "Completed Tasks" 
3. **UPDATE STATUS** of phases from "PENDING" to "IN PROGRESS" to "COMPLETED"
4. **ADD NEW INSIGHTS** to the "Known Performance Hotspots" section if discovered
5. **UPDATE BASELINE METRICS** when new measurements are available
This ensures future LLM sessions have accurate context without re-analyzing the entire codebase.

---

## Progress Memory & Status Tracker
*This section serves as a memory for all updates and achievements. When returning to this project via `inspect_environment.py`, this tracker shows what has been accomplished and what remains.*

### ✅ Completed Work:
1. **Basic Agent 2 Probability System** - Fully implemented visibility-based and gap-based probability calculations
2. **Performance Timing Infrastructure** - Basic timing measurements using `pygame.time.get_ticks()` in lines 2649-2860 of `environment_inspection_simulation.py`
3. **Agent 2 Visualization System** - Complete pink-green probability overlay with 800px range visualization
4. **Gap-based Sweep Processing** - 20-angle instantaneous sweep system with rod-based probability assignment
5. **Integration Layer** - Visibility probabilities override gap probabilities when both exist

### 🔄 Currently Working On:
- **Phase 1**: Performance Profiling and Baseline Measurement
  - Status: **Partially Started** - Basic timing exists, need detailed breakdown and logging

### ❌ Not Yet Started:
- **Phase 2**: Data Structure Optimization (spatial indexing, caching)
- **Phase 3**: Algorithm Optimization (vectorization, analytical solutions)  
- **Phase 4**: Multi-threading and Parallelization
- **Phase 5**: Caching and Prediction
- **Phase 6**: User Experience Optimization

### 🎯 Next Immediate Steps:
1. Complete Phase 1 by adding detailed timing breakdown for:
   - Visibility calculation time vs gap processing time
   - Node iteration and distance calculation timing
   - Performance logging system to CSV/files
2. Identify specific bottlenecks before proceeding to Phase 2

### 📝 LLM Context for Future Sessions:
**Primary Goal**: Optimize Agent 2 probability calculations that cause frame rate drops
**Current Problem**: Gap processing (20 angles × gaps × all nodes) is computationally expensive
**Key Insight**: Need granular profiling before optimizing - don't guess where bottlenecks are
**Technical Context**: This is a real-time pygame application with 60 FPS target
**User Workflow**: Run app → Press J → Move agent with WASD → Observe performance drops

**🤖 IMPORTANT FOR ALL FUTURE LLM SESSIONS**: 
- **ALWAYS UPDATE THIS FILE** after completing any work on this project
- **UPDATE THE PROGRESS TRACKER** sections (✅ Completed, 🔄 Currently Working On, ❌ Not Yet Started)
- **ADD NEW INSIGHTS** to the "Key Insight" section when you discover performance bottlenecks
- **UPDATE STATUS** indicators (✅ ❌ 🔄 ⏳) throughout the document to reflect current state
- **RECORD PERFORMANCE METRICS** when you gather new timing data or benchmarks
- **ADD TROUBLESHOOTING NOTES** if you encounter new issues or solutions
- **THIS FILE IS YOUR PERSISTENT MEMORY** - treat it as a living document that accumulates knowledge across sessions

### 📊 Current Performance Baseline:
- Total Agent 2 computation time is measured and displayed in FPS panel
- No granular breakdown of individual operation costs yet
- Target: 50%+ reduction in computation time while maintaining 60 FPS

### 🔧 How to Test/Reproduce Issues:
1. **Run the environment**: `python inspect_environment.py`
2. **Enable Agent 2 probability mode**: Press `J` key in the application
3. **Observe performance**: Check FPS panel for "Agent 2 computation: X ms"
4. **Test with different map sizes**: Use `V` to generate visibility data, then `J` to see impact
5. **Monitor frame drops**: Performance issues most visible when agent moves (WASD keys)

### 💡 Known Performance Hotspots:
- **Gap processing**: 20 angles × multiple gaps × all map nodes = expensive
- **Distance calculations**: `math.dist()` called for every node-to-rod distance
- **Node iteration**: No spatial filtering - processes ALL map nodes regardless of distance
- **Visibility lookups**: Linear search through visibility map for each calculation

## Current Performance Analysis
- **Primary bottleneck**: Agent 2 probability calculations in `environment_inspection_simulation.py` (lines 2649+)
- **Key components**: 
  - Visibility-based probability calculation (800px range)
  - Gap-based sweep processing (45-degree arcs with 20 angles)
  - Node-to-rod distance calculations for each map graph node
- **Current timing**: Measured using `pygame.time.get_ticks()` in the code

## Optimization Plan

### Phase 1: Performance Profiling and Baseline Measurement 🔄
**Status**: PARTIALLY STARTED (Basic timing infrastructure exists)
**Goal**: Establish current performance metrics and identify specific bottlenecks

#### ✅ Completed Tasks:
- Basic performance timing using `pygame.time.get_ticks()` (lines 2649-2860 in `environment_inspection_simulation.py`)
- Agent 2 computation time display in FPS panel
- Operation counters partially implemented (total_gaps_processed, total_angles_processed, etc.)

#### 🔄 Currently Working On:
1. **Add detailed timing measurements**
   - Instrument individual sections of agent 2 probability calculation
   - Measure visibility calculation time vs gap processing time
   - Track node count impact on performance

#### ❌ Remaining Tasks:
1. **Create performance logging system**
   - Log timing data to file for analysis
   - Track frame rate impact during agent 2 probability mode
   - Measure memory usage patterns

2. **Identify hotspots**
   - Profile which specific calculations take the most time
   - Analyze if gap processing or visibility checks dominate
   - Determine if node iteration or distance calculations are the issue

**Implementation Hints for Next Session**:
- Add timing markers around visibility lookup, gap processing, and node iteration loops
- Create a CSV logger to track: timestamp, total_time, visibility_time, gap_time, node_count, fps
- Use `time.perf_counter()` for high-precision timing measurements
- Consider using Python's `cProfile` module for detailed function-level profiling

**Concrete Code Changes Needed**:
```python
# Add these imports to environment_inspection_simulation.py
import time
import csv
from datetime import datetime

# Add timing variables before the main computation section (around line 2649)
visibility_start = time.perf_counter()
# ... existing visibility code ...
visibility_time = time.perf_counter() - visibility_start

gap_start = time.perf_counter()
# ... existing gap processing code ...
gap_time = time.perf_counter() - gap_start

# Log to CSV file
performance_data = {
    'timestamp': datetime.now().isoformat(),
    'total_time_ms': total_time,
    'visibility_time_ms': visibility_time * 1000,
    'gap_time_ms': gap_time * 1000,
    'node_count': len(map_graph.nodes),
    'fps': current_fps
}
```

**Deliverables**:
- Baseline performance metrics
- Detailed timing breakdown
- Identification of primary bottlenecks

---

### Phase 2: Data Structure Optimization ⏳
**Status**: PENDING (depends on Phase 1 completion)
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
**Status**: PENDING (depends on Phase 2 completion)
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
**Status**: PENDING (depends on Phase 3 completion)
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
**Status**: PENDING (depends on Phase 4 completion)
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
**Status**: PENDING (depends on Phase 5 completion)
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

**Next Step**: Complete Phase 1 by implementing detailed timing breakdown and performance logging system before proceeding to Phase 2.

## Quick Reference When Returning to This Project:
1. **From `inspect_environment.py`**: Look for the optimization comment referencing this plan
2. **Current Focus**: Phase 1 - Add granular timing measurements and logging
3. **Key Files**: `environment_inspection_simulation.py` (lines 2649+) contains the optimization target
4. **Progress**: Basic infrastructure exists, need detailed profiling to proceed

## Development Environment Setup:
```bash
# Navigate to project directory
cd /home/smandal/Documents/PivotedTracking

# Run the environment inspection tool
python inspect_environment.py

# Key controls for testing Agent 2 performance:
# J - Toggle Agent 2 probability overlay (main performance target)
# K - Toggle Agent 2 visibility gaps  
# V - Generate visibility data (prerequisite)
# WASD - Move Agent 2 (triggers recalculation)
```

## Code Navigation Guide:
- **Main optimization target**: `multitrack/simulation/environment_inspection_simulation.py:2649-2860`
- **Performance timing**: Look for `PERFORMANCE TIMING` comments
- **Agent 2 controls**: Search for `show_agent2_probability_overlay` variable
- **Gap processing**: Search for `agent2_gap_probabilities` dictionary
- **Visibility system**: `multitrack/utils/vision.py` and visibility_map usage

## Dependencies & Requirements:
- Python 3.x with pygame, numpy, math libraries
- Map graph cache files (auto-generated on first run)
- Visibility cache files (generated with `V` key or `--load-visibility` flag)

## Expected Behavior & Troubleshooting:
- **Normal operation**: Agent 2 computation should be <10ms for smooth 60 FPS
- **Performance issue**: Computation times >16ms cause visible frame drops
- **Troubleshooting**: If app crashes on `J` key, ensure visibility data exists (press `V` first)
- **Map size impact**: Larger maps = more nodes = longer computation times

## Sample Performance Data Format:
```csv
timestamp,total_time_ms,visibility_time_ms,gap_time_ms,node_count,fps,gaps_processed,angles_processed
2025-06-03T10:30:15,45.2,12.1,31.8,1247,42,8,160
2025-06-03T10:30:16,48.7,11.9,35.2,1247,38,9,180
```

## Architecture Overview for LLMs:
This is a **real-time pygame simulation** with:
- **Main loop**: 60 FPS target in `environment_inspection_simulation.py`
- **Agent system**: Two agents (Agent 1: arrow keys, Agent 2: WASD)
- **Probability calculation**: Runs every frame when enabled (expensive!)
- **Visualization**: Color-coded nodes based on probability values
- **Performance bottleneck**: Agent 2 probability mode processes all map nodes every frame

---

## 🚨 CRITICAL INSTRUCTIONS FOR ALL FUTURE LLMS 🚨

**THIS OPTIMIZATION PLAN IS A LIVING MEMORY SYSTEM**

When you work on this Agent 2 optimization project, you MUST:

### 1. 📝 Update This File After Any Work
- **NEVER** complete optimization work without updating this file
- Move completed items from ❌ to ✅ 
- Update phase statuses (⏳ → 🔄 → ✅)
- Add new performance metrics, insights, or troubleshooting notes

### 2. 🔍 Check Progress First
- **ALWAYS** read the "Progress Memory & Status Tracker" section first
- Look at the "Currently Working On 🔄" section to continue where left off
- Don't duplicate work that's already marked as completed ✅

### 3. 📊 Record Performance Data
- **UPDATE baseline metrics** when you gather new timing data
- **ADD new bottlenecks** to the "Known Performance Hotspots" section
- **RECORD any frame rate improvements** achieved

### 4. 🔧 Update Implementation Guidance
- **ENHANCE** the "Concrete Code Changes Needed" sections with your discoveries
- **ADD troubleshooting notes** if you encounter issues
- **UPDATE** file paths or line numbers if code structure changes

### 5. 🎯 Maintain Context for Future Sessions
- **ASSUME** the next LLM session will start from scratch with no memory
- **WRITE** as if explaining to a colleague taking over the project
- **INCLUDE** enough context for immediate continuation without re-analysis

**Example Update Pattern:**
```markdown
### ✅ Completed Work: (ADD YOUR ACHIEVEMENT HERE)
- ✅ **Granular Timing Breakdown** - Added visibility vs gap processing timing (Phase 1.1 complete)

### 🔄 Currently Working On: (UPDATE STATUS)  
- **Phase 1.2**: Performance logging system implementation

### 📊 Current Performance Baseline: (UPDATE WITH NEW DATA)
- Visibility processing: 8.2ms average  
- Gap processing: 31.4ms average (PRIMARY BOTTLENECK CONFIRMED)
```

**Remember: This file is your persistent memory across sessions. Treat it like the most important documentation in the project.**
