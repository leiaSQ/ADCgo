package backend

import "testing"

// TestDistBackendAddPanel guards the matrix-free -mgpu scatter-add: AddPanel must add a full
// host panel into only the first cols columns of a wider allocated output panel (the solver
// sizes panels to the max block width, so an apply commonly uses fewer columns than allocated).
// The regression it locks: adding across the whole partition storage instead of the first
// cols·rowsOn(d) contiguous columns, which mismatched lengths and corrupted the strided layout.
func TestDistBackendAddPanel(t *testing.T) {
	const n, main = 12, 2 // n > 2·main² = 8, so the shape invariant holds
	subs := []Backend{Gonum{}, Gonum{}}
	bounds := []int{0, 6, n}
	be, err := NewDistributed(subs, n, main, bounds)
	if err != nil {
		t.Fatalf("NewDistributed: %v", err)
	}

	const cols = 4    // allocated panel width
	const addCols = 2 // apply width (fewer than allocated)
	out := be.Alloc(n * cols)
	be.Zero(out)

	full := make([]float64, n*addCols)
	for i := range full {
		full[i] = float64(i + 1)
	}
	be.(PanelScatterAdd).AddPanel(out, full)

	got := be.Download(out) // n × cols, column-major
	for c := range cols {
		for r := range n {
			want := 0.0
			if c < addCols {
				want = full[c*n+r]
			}
			if g := got[c*n+r]; g != want {
				t.Errorf("col %d row %d: got %g, want %g (columns >= addCols must stay untouched)", c, r, g, want)
			}
		}
	}
}
