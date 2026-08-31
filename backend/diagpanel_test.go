package backend

import (
	"math"
	"math/rand"
	"testing"
)

// TestAddDiagPanelMatchesTheColumnLoop pins the free function to the definition it is
// a batched form of: c(i,j) += d(i)*a(i,j), column by column. The oracle is written out
// here rather than taken from AxpyDiag, because an oracle sharing code with the routine
// it checks would pass a sign error in both.
//
// Padded panels (Ld > Rows) are covered because the leading dimension is exactly what a
// whole-panel implementation has to get right and a column loop cannot get wrong.
func TestAddDiagPanelMatchesTheColumnLoop(t *testing.T) {
	be := Gonum{}
	rng := rand.New(rand.NewSource(11))

	for _, c := range []struct{ rows, cols, ldA, ldC int }{
		{4, 3, 4, 4},     // both compact
		{4, 3, 6, 4},     // padded source
		{4, 3, 4, 7},     // padded destination
		{5, 1, 5, 5},     // single column
		{1, 6, 1, 1},     // single row
		{68, 14, 68, 68}, // hh2's rd mode: sub x nspf
	} {
		d := make([]float64, c.rows)
		for i := range d {
			d[i] = rng.NormFloat64()
		}
		a := make([]float64, c.ldA*c.cols)
		got := make([]float64, c.ldC*c.cols)
		for i := range a {
			a[i] = rng.NormFloat64()
		}
		for i := range got {
			got[i] = rng.NormFloat64()
		}
		want := append([]float64(nil), got...)
		for j := 0; j < c.cols; j++ {
			for i := 0; i < c.rows; i++ {
				want[j*c.ldC+i] += d[i] * a[j*c.ldA+i]
			}
		}

		av := BlockView{V: be.Upload(a), Rows: c.rows, Cols: c.cols, Ld: c.ldA}
		cv := BlockView{V: be.Upload(got), Rows: c.rows, Cols: c.cols, Ld: c.ldC}
		AddDiagPanel(be, be.Upload(d), av, cv)

		out := be.Download(cv.V)
		for i := range want {
			if math.Abs(out[i]-want[i]) > 1e-14 {
				t.Fatalf("rows=%d cols=%d ldA=%d ldC=%d: element %d is %g, want %g",
					c.rows, c.cols, c.ldA, c.ldC, i, out[i], want[i])
			}
		}
	}
}

// TestAddDiagPanelRejectsMismatchedShapes: the contract is checked in the free function,
// so every implementation inherits it and none has to repeat it.
func TestAddDiagPanelRejectsMismatchedShapes(t *testing.T) {
	be := Gonum{}
	mk := func(rows, cols int) BlockView {
		return BlockView{V: be.Alloc(rows * cols), Rows: rows, Cols: cols, Ld: rows}
	}
	for _, c := range []struct {
		name string
		d    int
		a, b BlockView
	}{
		{"different shapes", 3, mk(3, 2), mk(3, 4)},
		{"diagonal too short", 2, mk(3, 2), mk(3, 2)},
	} {
		t.Run(c.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("expected a panic")
				}
			}()
			AddDiagPanel(be, be.Alloc(c.d), c.a, c.b)
		})
	}
}
