package main

import (
	"os"
	"path/filepath"
	"testing"
)

// TestParseCgroupLimit pins the sentinel/blank handling that decides whether the SIP RSS guard
// (applyCgroupMemLimit) engages. A wrong "unlimited" case would either leave the run unbounded
// (the 733 GB OOM) or clamp Go's heap to a bogus tiny/huge value.
func TestParseCgroupLimit(t *testing.T) {
	const v1Unlimited = "9223372036854771712" // ~int64 max, page-aligned: cgroup v1 "no limit"
	cases := []struct {
		name   string
		v2, v1 string
		want   uint64
		wantOK bool
	}{
		{"v2 finite preferred", "751619276800\n", "123\n", 751619276800, true},
		{"v2 max falls through to v1", "max\n", "268435456\n", 268435456, true},
		{"v2 max, v1 unlimited", "max", v1Unlimited, 0, false},
		{"both absent", "", "", 0, false},
		{"v2 absent, v1 finite", "", "700000000000\n", 700000000000, true},
		{"v2 unlimited sentinel is not 'max' but is rejected as too large", "", "9223372036854775807", 0, false},
		{"garbage", "notanumber", "alsobad", 0, false},
		{"v2 zero ignored", "0", "500\n", 500, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := parseCgroupLimit(tc.v2, tc.v1)
			if got != tc.want || ok != tc.wantOK {
				t.Errorf("parseCgroupLimit(%q,%q) = (%d,%v), want (%d,%v)",
					tc.v2, tc.v1, got, ok, tc.want, tc.wantOK)
			}
		})
	}
}

// TestCgroupLimitFromWalksAncestors is the regression test for the bug that let the production SIP
// job OOM at 733 GB: the guard read only the cgroup hierarchy ROOT, which is always "unlimited",
// so it never engaged. The real cap can sit on the process's own cgroup OR on any ancestor —
// SLURM puts a job's --mem partway up, and on a Helix login node the root and the leaf both read
// the v1 unlimited sentinel while the ancestor user slice carried a real 20 GiB cap.
//
// The effective limit is the minimum finite cap along that chain, which is what this pins.
func TestCgroupLimitFromWalksAncestors(t *testing.T) {
	const unlimited = "9223372036854771712"

	// write lays out a fake cgroup tree: files[relative path] = contents.
	write := func(t *testing.T, files map[string]string) string {
		t.Helper()
		root := t.TempDir()
		for rel, body := range files {
			full := filepath.Join(root, filepath.FromSlash(rel))
			if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(full, []byte(body), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		return root
	}

	t.Run("v1 cap on an ancestor, leaf and root unlimited", func(t *testing.T) {
		root := write(t, map[string]string{
			"memory/memory.limit_in_bytes":                                       unlimited,
			"memory/user.slice/memory.limit_in_bytes":                            "246960619520",
			"memory/user.slice/user-1.slice/memory.limit_in_bytes":               "21474836480", // 20 GiB
			"memory/user.slice/user-1.slice/session.scope/memory.limit_in_bytes": unlimited,
		})
		self := "9:memory:/user.slice/user-1.slice/session.scope\n1:name=systemd:/user.slice\n"
		got, ok := cgroupLimitFrom(root, self)
		if !ok || got != 21474836480 {
			t.Errorf("got (%d,%v), want (21474836480,true) — the minimum finite ancestor cap", got, ok)
		}
	})

	t.Run("v1 all unlimited yields no limit", func(t *testing.T) {
		root := write(t, map[string]string{
			"memory/memory.limit_in_bytes":            unlimited,
			"memory/user.slice/memory.limit_in_bytes": unlimited,
		})
		if got, ok := cgroupLimitFrom(root, "9:memory:/user.slice\n"); ok {
			t.Errorf("got (%d,true), want no limit", got)
		}
	})

	t.Run("v2 nested job cgroup, root is max", func(t *testing.T) {
		root := write(t, map[string]string{
			"memory.max":                     "max",
			"slurm/memory.max":               "max",
			"slurm/job_42/memory.max":        "751619276800", // 700 GiB, the SLURM --mem shape
			"slurm/job_42/step_0/memory.max": "max",
		})
		self := "0::/slurm/job_42/step_0\n"
		got, ok := cgroupLimitFrom(root, self)
		if !ok || got != 751619276800 {
			t.Errorf("got (%d,%v), want (751619276800,true)", got, ok)
		}
	})

	t.Run("multiple controllers on one line", func(t *testing.T) {
		root := write(t, map[string]string{
			"memory/grp/memory.limit_in_bytes": "1048576",
		})
		// The memory controller can share a line with others, e.g. "cpu,memory".
		if got, ok := cgroupLimitFrom(root, "5:cpu,memory:/grp\n"); !ok || got != 1048576 {
			t.Errorf("got (%d,%v), want (1048576,true)", got, ok)
		}
	})

	t.Run("no cgroup entry yields no limit", func(t *testing.T) {
		if got, ok := cgroupLimitFrom(write(t, nil), "3:cpuset:/\n"); ok {
			t.Errorf("got (%d,true), want no limit", got)
		}
	})
}
