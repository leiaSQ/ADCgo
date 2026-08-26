package backend

import (
	"reflect"
	"testing"
)

// TestDistBackendDownloadInto guards a trap that cost a production run 40 h of compute.
//
// distBackend EMBEDS Gonum, to inherit the host SymEig and anything else it does not override.
// That embedding also makes distBackend satisfy BufferedDownloader whether or not it implements
// DownloadInto itself — and the inherited Gonum.DownloadInto does host(v), i.e. v.(hostVec), which
// panics on a distVec. So a caller doing the ordinary
//
//	bd, ok := be.(BufferedDownloader)
//
// gets ok == true and a method that cannot work, instead of falling back to the allocating
// Download path it intended. That is precisely how the production DIP probe (job 14158038) died at
// its first checkpoint with "interface conversion: backend.Vector is backend.distVec, not
// backend.hostVec", losing a 40 h block.
//
// The test therefore asserts both halves: DownloadInto must agree with Download element-for-element
// on a distributed panel, and it must do so WITHOUT panicking — which only holds while distBackend
// overrides the embedded method. Deleting the override reintroduces the outage.
func TestDistBackendDownloadInto(t *testing.T) {
	const n, main = 12, 2 // n > 2·main² = 8
	// Deliberately uneven bands, so a per-device offset error cannot cancel out.
	subs := []Backend{Gonum{}, Gonum{}, Gonum{}}
	bounds := []int{0, 3, 7, n}
	be, err := NewDistributed(subs, n, main, bounds)
	if err != nil {
		t.Fatalf("NewDistributed: %v", err)
	}

	bd, ok := be.(BufferedDownloader)
	if !ok {
		t.Fatal("distBackend does not satisfy BufferedDownloader")
	}

	for _, cols := range []int{1, 2, 5} {
		host := make([]float64, n*cols)
		for i := range host {
			host[i] = float64(i) * 0.5
		}
		v := be.Upload(host)

		want := be.Download(v)
		dst := make([]float64, n*cols)
		bd.DownloadInto(dst, v)
		if !reflect.DeepEqual(dst, want) {
			t.Errorf("cols=%d: DownloadInto disagrees with Download\n got %v\nwant %v", cols, dst, want)
		}
		if !reflect.DeepEqual(dst, host) {
			t.Errorf("cols=%d: round trip lost data", cols)
		}

		// A column sub-range must work too: that is what the checkpoint writer actually streams.
		if cols >= 2 {
			panel := BlockView{V: v, Rows: n, Cols: cols, Ld: n}
			sub := panel.ColRange(1, cols).V
			subWant := be.Download(sub)
			subDst := make([]float64, len(subWant))
			bd.DownloadInto(subDst, sub)
			if !reflect.DeepEqual(subDst, subWant) {
				t.Errorf("cols=%d: DownloadInto on a ColRange disagrees with Download", cols)
			}
		}
		be.Free(v)
	}

	// Reusing the staging buffers across calls must not leak state between them.
	a := be.Upload(make([]float64, n*2))
	big := make([]float64, n*2)
	for i := range big {
		big[i] = 7
	}
	bcv := be.Upload(big)
	dst := make([]float64, n*2)
	bd.DownloadInto(dst, bcv)
	bd.DownloadInto(dst, a)
	for i, v := range dst {
		if v != 0 {
			t.Fatalf("stale staging data at %d: got %g, want 0", i, v)
		}
	}
}

// TestDistBackendDownloadIntoRejectsSmallDst pins the same contract gpuBackend states: a dst that
// cannot hold the panel is a programming error, reported as such rather than silently truncating.
func TestDistBackendDownloadIntoRejectsSmallDst(t *testing.T) {
	const n, main = 12, 2
	be, err := NewDistributed([]Backend{Gonum{}, Gonum{}}, n, main, []int{0, 6, n})
	if err != nil {
		t.Fatalf("NewDistributed: %v", err)
	}
	v := be.Upload(make([]float64, n*2))
	defer be.Free(v)

	defer func() {
		if recover() == nil {
			t.Error("DownloadInto accepted an undersized dst")
		}
	}()
	be.(BufferedDownloader).DownloadInto(make([]float64, n), v)
}
