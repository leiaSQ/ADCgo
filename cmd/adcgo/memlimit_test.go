package main

import "testing"

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
