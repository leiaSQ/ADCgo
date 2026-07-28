package main

import (
	"fmt"
	"os"
	"runtime/debug"
	"strconv"
	"strings"
)

// applyCgroupMemLimit sets Go's soft memory limit from the cgroup's memory cap when the process
// is running under one (a SLURM --mem allocation always is), leaving headroom for memory the Go
// heap does not account for (the CUDA driver, cgo/pinned host buffers). The Go runtime returns
// freed spans to the OS lazily; without a limit, the churn of large transients — the growing-Dim
// projection copy and the basis download chunks in the SIP checkpoint path — stacks RSS until the
// cgroup OOM-kills the run (production SIP, 733 GB against --mem=700gb). A soft limit makes the GC
// scavenge aggressively as RSS approaches it, holding the process under the cap.
//
// It is a no-op if GOMEMLIMIT is set (the runtime already honours that) or if no finite cgroup
// limit is found. headroomFrac of the cgroup limit is left for non-heap host memory.
func applyCgroupMemLimit() {
	if os.Getenv("GOMEMLIMIT") != "" {
		return // user/runtime already set an explicit limit
	}
	limit, ok := cgroupMemLimitBytes()
	if !ok {
		return
	}
	const headroomFrac = 0.85 // heap ≤ 85% of the cgroup cap; the rest for driver/cgo/pinned
	soft := int64(float64(limit) * headroomFrac)
	if soft <= 0 {
		return
	}
	debug.SetMemoryLimit(soft)
	fmt.Fprintf(os.Stderr, "adcgo: GOMEMLIMIT auto-set to %.0f GiB (85%% of the %.0f GiB cgroup cap) "+
		"to keep RSS under the SLURM --mem limit\n", float64(soft)/(1<<30), float64(limit)/(1<<30))
}

// cgroupMemLimitBytes reads the process's cgroup memory cap, trying cgroup v2 then v1. It returns
// (0,false) when no cgroup applies or the cap is unlimited ("max", or the v1 sentinel).
func cgroupMemLimitBytes() (uint64, bool) {
	v2, _ := os.ReadFile("/sys/fs/cgroup/memory.max")
	v1, _ := os.ReadFile("/sys/fs/cgroup/memory/memory.limit_in_bytes")
	return parseCgroupLimit(string(v2), string(v1))
}

// parseCgroupLimit turns the raw contents of the v2 (memory.max) and v1
// (memory.limit_in_bytes) files into a finite byte cap, preferring v2. "max" (v2) and the v1
// "unlimited" sentinel (a near-int64-max value) both yield (0,false). Empty strings mean the
// file was absent. Split out so the sentinel/blank handling is unit-testable without a cgroup.
func parseCgroupLimit(v2, v1 string) (uint64, bool) {
	if s := strings.TrimSpace(v2); s != "" && s != "max" {
		if v, err := strconv.ParseUint(s, 10, 64); err == nil && v > 0 {
			return v, true
		}
	}
	// v1 "unlimited" is a huge sentinel (~page-aligned int64 max), so reject implausible values.
	if s := strings.TrimSpace(v1); s != "" {
		if v, err := strconv.ParseUint(s, 10, 64); err == nil && v > 0 && v < (1<<62) {
			return v, true
		}
	}
	return 0, false
}
