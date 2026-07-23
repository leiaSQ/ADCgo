package backend

import "testing"

// TestGonumDownload2DGathersStridedBlock pins StridedDownloader's contract on the reference
// backend: a compact rows×cols column-major copy of the sub-block whose columns sit ld apart.
//
// The Ritz back-transform (lanczos.go) reads the leading `main` rows of every basis column
// through this, so an off-by-one in the stride arithmetic would silently transpose or shift the
// Ritz vectors rather than fail loudly. ld > rows deliberately — equal values would hide exactly
// the indexing bug this guards.
func TestGonumDownload2DGathersStridedBlock(t *testing.T) {
	const (
		ld   = 5 // panel leading dimension (full column height)
		rows = 3 // sub-block height: the leading rows of each column
		cols = 4
	)
	be := Gonum{}

	// Column-major panel: element (r, c) = 100*c + r, so a wrong stride is unmistakable.
	host := make([]float64, ld*cols)
	for c := range cols {
		for r := range ld {
			host[c*ld+r] = float64(100*c + r)
		}
	}
	v := be.Upload(host)

	got := be.Download2D(v, rows, cols, ld)
	if len(got) != rows*cols {
		t.Fatalf("length: got %d, want %d", len(got), rows*cols)
	}
	for c := range cols {
		for r := range rows {
			want := float64(100*c + r)
			if g := got[c*rows+r]; g != want {
				t.Errorf("(row %d, col %d): got %g, want %g", r, c, g, want)
			}
		}
	}

	// The rows beyond the sub-block (r >= rows) must not appear anywhere in the result — the
	// failure mode where a contiguous copy silently pulls in the tail of each column.
	for _, v := range got {
		if r := int(v) % 100; r >= rows {
			t.Errorf("value %g comes from row %d, outside the requested %d rows", v, r, rows)
		}
	}
}
