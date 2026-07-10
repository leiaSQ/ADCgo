package dip

import (
	"math"
	"math/rand"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

// TestApplyBlockMatchesApplyFull pins the level-3 mat-vec to the level-2 one it
// replaces: for every (spin, irrep) sector, M applied to a block of random vectors
// must equal M applied to each vector individually.
//
// This is the correctness gate for the whole apply-side optimization. It also
// exercises BlockView.RowRange and the row-major/column-major flag inversion in
// GemmMat, which is where a silent transpose would hide.
func TestApplyBlockMatchesApplyFull(t *testing.T) {
	d, err := fcidump.ReadFile("../../../testdata/h2o.fcidump")
	if err != nil {
		t.Fatalf("read fcidump: %v", err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	ints := integrals.New(d, nocc, d.OrbSym)
	rng := rand.New(rand.NewSource(3))

	const blk = 5
	// Every backend compiled into this build: under -tags cuda/hip this also covers
	// GemmMat's row-major -> column-major flag inversion on the device.
	for _, name := range backend.Available() {
		be, err := backend.New(name)
		if err != nil {
			t.Fatalf("backend.New(%q): %v", name, err)
		}
		t.Run(name, func(t *testing.T) { applyBlockVsFull(t, be, d, nocc, eps, ints, rng, blk) })
	}
}

func applyBlockVsFull(t *testing.T, be backend.Backend, d *fcidump.Data, nocc int,
	eps []float64, ints *integrals.Store, rng *rand.Rand, blk int) {
	tested := 0
	for _, spin := range []Spin{Singlet, Triplet} {
		for sym := range 4 {
			sp := NewSpace(nocc, d.NORB, d.OrbSym, sym, spin)
			n := sp.Size()
			if n == 0 {
				continue
			}
			mx := New(sp, ints, eps, be)

			// Random n×blk input panel, column-major.
			inData := make([]float64, n*blk)
			for i := range inData {
				inData[i] = rng.NormFloat64()
			}
			in := backend.BlockView{V: be.Upload(inData), Rows: n, Cols: blk, Ld: n}
			out := backend.BlockView{V: be.Alloc(n * blk), Rows: n, Cols: blk, Ld: n}
			mx.ApplyBlock(out, in)
			got := be.Download(out.V)

			// Column-by-column reference through ApplyFull.
			var maxDiff, scale float64
			for j := range blk {
				col := be.Upload(inData[j*n : (j+1)*n])
				ref := be.Alloc(n)
				mx.ApplyFull(ref, col)
				want := be.Download(ref)
				for i := range n {
					scale = math.Max(scale, math.Abs(want[i]))
					maxDiff = math.Max(maxDiff, math.Abs(want[i]-got[j*n+i]))
				}
			}
			rel := maxDiff / math.Max(scale, 1e-300)
			if rel > 1e-12 {
				t.Errorf("spin=%v sym=%d n=%d: ApplyBlock vs ApplyFull relative diff %.3e", spin, sym, n, rel)
			}
			tested++
		}
	}
	if tested == 0 {
		t.Fatal("no sectors exercised")
	}
	t.Logf("%d sectors: ApplyBlock == ApplyFull", tested)
}
