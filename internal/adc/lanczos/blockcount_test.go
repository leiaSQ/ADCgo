package lanczos

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/dip"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

// The block count is the one solver parameter that has to mean the same thing here as
// in theADCcode, because it is what a user sets to reproduce a reference run. There,
// `iter N` diagonalizes a Krylov space of N blocks of `main_block_size()` columns:
// Iterate() is called N+1 times (the first only registers the start block), reaching
// dim = (N+1)·block, and Diagonalize() sets dimd = dim − block = N·block, which is the
// "Size of Lanczos space" it prints (../ADC/libLanczos/lanczos.h:226-238, :257;
// ../ADC/analysis/adc_analyzer.cpp:229-247). The trailing block only supplies the
// coupling for the residuals — exactly the discarded orthogonalization in Solve.
//
// So MaxBlocks == iter, and the subspace is MaxBlocks·main. These tests pin that; they
// fail against the older (MaxBlocks+1)·main convention.

func TestSubspaceDimCountsBlocks(t *testing.T) {
	cases := []struct {
		n, main, blocks, want int
	}{
		{139, 5, 100, 139}, // reference sym1 singlet: 100·5 = 500 overshoots n, we cap
		{151, 1, 100, 100}, // reference sym1 triplet: "Size of Lanczos space: 100"
		{123, 2, 100, 123}, // reference sym3 singlet: 100·2 = 200 overshoots n
		{10000, 7, 30, 210},
		{10000, 7, 1, 7}, // one block == just the start block
		{500, 5, 0, 500}, // 0 → unbounded, capped at n
		{0, 5, 30, 0},
		{500, 0, 30, 0},
	}
	for _, c := range cases {
		if got := SubspaceDim(c.n, c.main, Options{MaxBlocks: c.blocks}); got != c.want {
			t.Errorf("SubspaceDim(n=%d, main=%d, blocks=%d) = %d, want %d",
				c.n, c.main, c.blocks, got, c.want)
		}
	}
}

// TestSolveBuildsMaxBlocksBlocks checks that Solve actually stops at MaxBlocks blocks,
// not MaxBlocks+1: the Ritz count is the subspace dimension it built.
func TestSolveBuildsMaxBlocksBlocks(t *testing.T) {
	be := backend.Gonum{}
	mx := buildH2O(t, dip.Singlet)
	n, main := mx.Size(), mx.MainBlockSize()

	for _, blocks := range []int{1, 2, 5} {
		// Spelled out rather than taken from SubspaceDim: this must fail if both drift
		// together.
		want := blocks * main
		if want >= n {
			t.Fatalf("blocks=%d saturates the %d-dim sector; test needs a truncated run", blocks, n)
		}
		res := Solve(mx, be, Options{MaxBlocks: blocks})
		if got := len(res.Values); got != want {
			t.Errorf("blocks=%d: Solve returned %d Ritz values, want %d (= %d blocks × %d main)",
				blocks, got, want, blocks, main)
		}
		if got := SubspaceDim(n, main, Options{MaxBlocks: blocks}); got != want {
			t.Errorf("blocks=%d: SubspaceDim = %d, want %d — Solve and SubspaceDim disagree",
				blocks, got, want)
		}
	}
}

// refSector is one (irrep, spin) block of theADCcode's h2o/DZP DIP run, read off
// testdata/reference/adcdip{irrep}.out. `main` is its "block size" line, `size` its
// "Number of ISR configurations" line, and lancSpace its "Size of Lanczos space" —
// all four files ran `iter 100`.
type refSector struct {
	irrep     int
	spin      dip.Spin
	main      int
	size      int
	lancSpace int
}

var refSectors = []refSector{
	{0, dip.Singlet, 5, 139, 500},
	{0, dip.Triplet, 1, 151, 100},
	{1, dip.Singlet, 1, 117, 100},
	{1, dip.Triplet, 1, 151, 100},
	{2, dip.Singlet, 2, 123, 200},
	{2, dip.Triplet, 2, 152, 200},
	{3, dip.Singlet, 2, 131, 200},
	{3, dip.Triplet, 2, 152, 200},
}

// TestReferenceBlockAndSpaceSizes ties the two quantities theADCcode prints per sector —
// the Lanczos block size and the Lanczos space — to ADCgo's MainBlockSize() and
// SubspaceDim(). The matched FCIDUMP carries theADCcode's own ORBSYM, so ADCgo sector
// index N is reference file adcdip{N+1}.out.
func TestReferenceBlockAndSpaceSizes(t *testing.T) {
	const refIter = 100 // the `iter` value used in every adcdip*.out run

	path := filepath.Join("..", "..", "..", "testdata", "reference", "h2o_dzp.matched.fcidump")
	d, err := fcidump.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			t.Skip("matched FCIDUMP not present (regenerate with ../ADC/fcidump_export)")
		}
		t.Fatalf("read matched fcidump: %v", err)
	}
	nocc := mp.NOcc(d)

	for _, r := range refSectors {
		sp := dip.NewSpace(nocc, d.NORB, d.OrbSym, r.irrep, r.spin)
		if got := sp.MainBlockSize(); got != r.main {
			t.Errorf("adcdip%d spin %d: MainBlockSize = %d, reference 'block size' = %d",
				r.irrep+1, spinCode(r.spin), got, r.main)
		}
		if got := sp.Size(); got != r.size {
			t.Errorf("adcdip%d spin %d: Size = %d, reference 'Number of ISR configurations' = %d",
				r.irrep+1, spinCode(r.spin), got, r.size)
		}
		// MaxBlocks == iter: the unclamped subspace is exactly what the reference prints.
		if got := refIter * r.main; got != r.lancSpace {
			t.Errorf("adcdip%d spin %d: %d blocks × %d main = %d, reference 'Size of Lanczos space' = %d",
				r.irrep+1, spinCode(r.spin), refIter, r.main, got, r.lancSpace)
		}
		// ADCgo additionally clamps at the sector dimension rather than generating the
		// ghost roots the reference then drops (adcdip1: 361 of 500 spurious).
		want := min(r.size, r.lancSpace)
		if got := SubspaceDim(sp.Size(), sp.MainBlockSize(), Options{MaxBlocks: refIter}); got != want {
			t.Errorf("adcdip%d spin %d: SubspaceDim = %d, want %d",
				r.irrep+1, spinCode(r.spin), got, want)
		}
	}
}

func spinCode(s dip.Spin) int {
	if s == dip.Triplet {
		return 3
	}
	return 1
}

// TestSolveReportsProgressPerBlock pins the contract the production logs depend on: Solve
// calls Options.Progress exactly once per block, with ascending 0-based block indices, so
// counting "progress" lines in a job's stderr counts the blocks it actually built. Job
// 14717237 had to have its 200 blocks reconstructed from Slurm's recorded checkpoint write
// volume because Solve reported nothing at all; this is what replaces that arithmetic.
func TestSolveReportsProgressPerBlock(t *testing.T) {
	be := backend.Gonum{}
	mx := buildH2O(t, dip.Singlet)
	n, main := mx.Size(), mx.MainBlockSize()

	type call struct{ iter, dim, size int }
	for _, blocks := range []int{1, 2, 5} {
		if blocks*main >= n {
			t.Fatalf("blocks=%d saturates the %d-dim sector; test needs a truncated run", blocks, n)
		}
		var got []call
		res := Solve(mx, be, Options{
			MaxBlocks: blocks,
			Progress: func(iter, dim, size int, _ Timing) {
				got = append(got, call{iter, dim, size})
			},
		})
		if len(got) != blocks {
			t.Fatalf("blocks=%d: Progress fired %d times, want %d (once per block)",
				blocks, len(got), blocks)
		}
		for i, c := range got {
			if c.iter != i {
				t.Errorf("blocks=%d: call %d reported block=%d, want %d", blocks, i, c.iter, i)
			}
			if c.size != main {
				t.Errorf("blocks=%d: call %d reported size=%d, want the undeflated block %d",
					blocks, i, c.size, main)
			}
		}
		// The last line's dim is the subspace the run ends with, which is what makes a
		// truncated log readable without waiting for the result.
		if last := got[len(got)-1].dim; last != len(res.Values) {
			t.Errorf("blocks=%d: final reported dim=%d, but Solve returned %d Ritz values",
				blocks, last, len(res.Values))
		}
		// The trailing iteration only builds the discarded R_{j+1}, so it repeats the
		// previous dim rather than growing the basis.
		if blocks > 1 {
			if a, b := got[len(got)-2].dim, got[len(got)-1].dim; a != b {
				t.Errorf("blocks=%d: truncating block moved dim %d -> %d, want it unchanged",
					blocks, a, b)
			}
		}
	}
}
