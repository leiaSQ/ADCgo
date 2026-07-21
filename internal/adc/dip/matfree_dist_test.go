package dip

import "testing"

// TestSatRowBands pins the per-device 3h1p row-band derivation. This arithmetic has no safety net
// downstream: unlike the dense blocks — which go through distVec.Slice and hit its
// crossing-boundary panic — the per-device applier hands raw offsets straight to the kernel
// launcher, so an off-by-one here silently applies the operator to the wrong rows or skips rows
// entirely. The cases below are exactly the ones the plan calls out as must-cover.
func TestSatRowBands(t *testing.T) {
	for _, tc := range []struct {
		name    string
		bounds  []int
		main, n int
		wantLo  []int
		wantHi  []int
	}{{
		// Every boundary past the main block: plain satellite-relative shift.
		name:   "all partitions inside the satellite region",
		bounds: []int{0, 40, 70, 100}, main: 10, n: 100,
		wantLo: []int{0, 30, 60}, wantHi: []int{30, 60, 90},
	}, {
		// Partition 0 straddles the main/satellite boundary — the normal case, since main is
		// tiny relative to n. Its band must start at 0 (satellite-relative), not at -main.
		name:   "first partition straddles the boundary",
		bounds: []int{0, 40, 100}, main: 25, n: 100,
		wantLo: []int{0, 15}, wantHi: []int{15, 75},
	}, {
		// Partition 0 lies WHOLLY inside the main block: empty band, must be skipped.
		name:   "partition owns no satellite rows",
		bounds: []int{0, 20, 60, 100}, main: 20, n: 100,
		wantLo: []int{0, 0, 40}, wantHi: []int{0, 40, 80},
	}, {
		// Boundary exactly on main: no rows lost at the seam between partitions 0 and 1.
		name:   "boundary exactly at main",
		bounds: []int{0, 30, 100}, main: 30, n: 100,
		wantLo: []int{0, 0}, wantHi: []int{0, 70},
	}, {
		name:   "single partition covers everything",
		bounds: []int{0, 100}, main: 7, n: 100,
		wantLo: []int{0}, wantHi: []int{93},
	}} {
		t.Run(tc.name, func(t *testing.T) {
			lo, hi := satRowBands(tc.bounds, tc.main, tc.n)
			for d := range tc.wantLo {
				if lo[d] != tc.wantLo[d] || hi[d] != tc.wantHi[d] {
					t.Errorf("part %d: got [%d,%d), want [%d,%d)", d, lo[d], hi[d], tc.wantLo[d], tc.wantHi[d])
				}
			}

			// Whatever the partitioning, the bands must exactly tile the satellite region
			// [0, n-main) with no gap and no overlap — that is the property the apply depends
			// on for every output row to be computed exactly once.
			covered := 0
			prevHi := 0
			for d := range lo {
				if hi[d] < lo[d] {
					t.Fatalf("part %d: inverted band [%d,%d)", d, lo[d], hi[d])
				}
				if lo[d] < prevHi {
					t.Errorf("part %d: band starts at %d, overlapping the previous band end %d", d, lo[d], prevHi)
				}
				if lo[d] > prevHi {
					t.Errorf("part %d: band starts at %d, leaving rows [%d,%d) uncomputed", d, lo[d], prevHi, lo[d])
				}
				covered += hi[d] - lo[d]
				prevHi = hi[d]
			}
			if want := tc.n - tc.main; covered != want {
				t.Errorf("bands cover %d satellite rows, want %d", covered, want)
			}
			if prevHi != tc.n-tc.main {
				t.Errorf("last band ends at %d, want %d", prevHi, tc.n-tc.main)
			}
		})
	}
}
