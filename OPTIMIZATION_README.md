# Agent 2 Optimization - Documentation Index

This directory contains multiple files related to the Agent 2 probability calculation optimization project. Here's what each file contains and when to use it:

## 📋 ACTIVE PROJECT FILES (Use These First)

### `agent2_optimization_plan.md` ⭐ **START HERE**
- **Purpose**: Current active plan, concise and focused (<100 lines)
- **Contains**: Current phase status, next actions, key performance data
- **Use When**: Starting work on optimization, checking current status
- **LLM Instructions**: Always update this file when making progress

### `inspect_environment.py` 🚀 **ENTRY POINT**
- **Purpose**: Main application entry point with optimization notes
- **Contains**: How to run tests, where to find bottlenecks
- **Use When**: Running the application, understanding the workflow

### `agent2_performance_log.csv` 📊 **LIVE DATA**
- **Purpose**: Real-time performance measurements from the application
- **Contains**: Timing data, node counts, FPS measurements
- **Use When**: Analyzing performance, validating optimizations

## 📚 REFERENCE DOCUMENTATION

### `PHASE1_COMPLETION_SUMMARY.md` ✅ **PHASE 1 RESULTS**
- **Purpose**: Detailed findings from Phase 1 profiling work
- **Contains**: Complete performance analysis, implementation details
- **Use When**: Understanding what was already accomplished

### `agent2_optimization_archive.md` 📖 **FULL HISTORY**
- **Purpose**: Complete historical context and detailed planning
- **Contains**: Original 6-phase plan, full methodology, historical updates
- **Use When**: Need deep background context or reference details

## 🎯 TARGET SOURCE CODE

### `multitrack/simulation/environment_inspection_simulation.py`
- **Critical Lines**: 2649-2860 (Agent 2 computation)
- **Current Issue**: 4.4-4.6s computation per frame
- **Goal**: Reduce to <16ms for 60 FPS

## 🔄 WORKFLOW FOR FUTURE LLMs

1. **Check Status**: Read `agent2_optimization_plan.md` for current phase
2. **Run Tests**: Use `inspect_environment.py` → Press 'J' → Move with WASD
3. **Analyze Data**: Check `agent2_performance_log.csv` for measurements
4. **Get Context**: Reference `PHASE1_COMPLETION_SUMMARY.md` for background
5. **Update Progress**: Always update `agent2_optimization_plan.md` when making changes

## 🎯 CURRENT STATUS (June 3, 2025)

- **Phase 1**: ✅ COMPLETED (Performance profiling)
- **Phase 2**: 🔄 READY (Spatial optimization - 95% performance improvement expected)
- **Critical Finding**: Need to filter 11,762 nodes down to ~200-500 within 800px range

**Next Action**: Implement spatial filtering in gap processing loop around line 2670-2790
