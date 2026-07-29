package main

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime/debug"
	"slices"
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

// cgroupMemLimitBytes reports the memory cap that actually applies to THIS process.
//
// It must resolve the process's own cgroup rather than read the hierarchy root: on a SLURM node
// the root is always "unlimited" and the real cap lives on a nested job cgroup (`--mem`), while
// intermediate ancestors may carry their own caps. Reading only the root is why the earlier
// version of this guard silently no-op'd through the production SIP OOM — verified on a Helix login
// node, where the root and the leaf both read the v1 "unlimited" sentinel while the *ancestor*
// user slice carried a real 20 GiB cap.
//
// The effective limit is the MINIMUM finite cap over the process's cgroup and all its ancestors,
// which is what the walk below computes.
func cgroupMemLimitBytes() (uint64, bool) {
	self, err := os.ReadFile("/proc/self/cgroup")
	if err != nil {
		return 0, false
	}
	return cgroupLimitFrom("/sys/fs/cgroup", string(self))
}

// cgroupLimitFrom resolves the effective cap under filesystem root `root` given the contents of
// /proc/self/cgroup. Split out (with root injected) so it is testable against a fixture tree.
//
// /proc/self/cgroup lines are "hierarchy:controllers:path". The unified (v2) entry has an empty
// controller field ("0::/some/path"); v1 entries name their controllers ("9:memory:/some/path").
func cgroupLimitFrom(root, selfCgroup string) (uint64, bool) {
	var v2Path, v1Path string
	for _, line := range strings.Split(strings.TrimSpace(selfCgroup), "\n") {
		f := strings.SplitN(strings.TrimSpace(line), ":", 3)
		if len(f) != 3 {
			continue
		}
		switch {
		case f[1] == "":
			v2Path = f[2]
		case slices.Contains(strings.Split(f[1], ","), "memory"):
			v1Path = f[2]
		}
	}

	best, found := uint64(0), false
	consider := func(raw string, v2 bool) {
		var v uint64
		var ok bool
		if v2 {
			v, ok = parseCgroupLimit(raw, "")
		} else {
			v, ok = parseCgroupLimit("", raw)
		}
		if ok && (!found || v < best) {
			best, found = v, true
		}
	}
	// Walk each hierarchy from the process's own cgroup up to the root, taking the minimum
	// finite cap. A cap may sit on any ancestor (SLURM puts the job's --mem partway up).
	walk := func(dir, path, file string, v2 bool) {
		for p := path; ; p = filepath.Dir(p) {
			b, err := os.ReadFile(filepath.Join(dir, filepath.FromSlash(p), file))
			if err == nil {
				consider(string(b), v2)
			}
			if p == "/" || p == "." || p == "" {
				return
			}
		}
	}
	if v2Path != "" {
		walk(root, v2Path, "memory.max", true)
	}
	if v1Path != "" {
		walk(filepath.Join(root, "memory"), v1Path, "memory.limit_in_bytes", false)
	}
	return best, found
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
