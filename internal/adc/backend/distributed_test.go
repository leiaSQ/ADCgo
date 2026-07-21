package backend

import (
	"strings"
	"testing"
)

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

// TestDistBackendPartitionedDevices covers the PartitionedDevices capability that backs the
// per-device satellite apply: the partition metadata a caller needs (count, bounds, sub-backend,
// device kernels, peer capability) and the per-device panel storage. Uneven bands are used
// deliberately — equal splits would hide an off-by-one in the row-band arithmetic. Gonum
// sub-backends give full coverage of everything except the kernel launch itself.
func TestDistBackendPartitionedDevices(t *testing.T) {
	const n, main = 12, 2
	subs := []Backend{Gonum{}, Gonum{}, Gonum{}}
	bounds := []int{0, 3, 7, n} // uneven: 3, 4, 5 rows
	be, err := NewDistributed(subs, n, main, bounds)
	if err != nil {
		t.Fatalf("NewDistributed: %v", err)
	}
	pd, ok := be.(PartitionedDevices)
	if !ok {
		t.Fatal("distributed backend does not implement PartitionedDevices")
	}

	if got := pd.NumParts(); got != len(subs) {
		t.Errorf("NumParts = %d, want %d", got, len(subs))
	}

	// Bounds must be a defensive copy — a caller mutating it must not repartition the backend.
	got := pd.Bounds()
	for i, w := range bounds {
		if got[i] != w {
			t.Fatalf("Bounds()[%d] = %d, want %d", i, got[i], w)
		}
	}
	got[1] = 999
	if again := pd.Bounds(); again[1] != bounds[1] {
		t.Errorf("Bounds() is not a copy: caller mutation leaked into the backend (%d)", again[1])
	}

	for d := range subs {
		if pd.PartBackend(d) != subs[d] {
			t.Errorf("PartBackend(%d) returned the wrong sub-backend", d)
		}
		if _, ok := pd.PartKernels(d); ok {
			t.Errorf("PartKernels(%d) reported device kernels for a Gonum sub-backend", d)
		}
	}
	if pd.AllPeered() {
		t.Error("AllPeered = true over Gonum sub-backends; must be false so the host fallback stays")
	}

	// PartVector must hand back exactly device d's row band, column-major with ld = rowsOn(d).
	const cols = 2
	host := make([]float64, n*cols)
	for i := range host {
		host[i] = float64(i + 1)
	}
	v := be.Upload(host)
	for d := range subs {
		lo, hi := bounds[d], bounds[d+1]
		rd := hi - lo
		local := subs[d].Download(pd.PartVector(v, d))
		if len(local) != rd*cols {
			t.Fatalf("PartVector(%d) length = %d, want %d", d, len(local), rd*cols)
		}
		for c := range cols {
			for r := range rd {
				want := host[c*n+lo+r]
				if g := local[c*rd+r]; g != want {
					t.Errorf("part %d col %d row %d: got %g, want %g", d, c, r, g, want)
				}
			}
		}
	}

	// Shapes the per-device apply must never be handed: a replicated small buffer, and a
	// located row band. Accepting either silently would apply the operator to the wrong rows.
	// The message is asserted too: without it a panic raised somewhere else (e.g. inside Slice)
	// would pass the test while PartVector's guard was never reached.
	mustPanic := func(name, want string, fn func()) {
		t.Helper()
		defer func() {
			r := recover()
			if r == nil {
				t.Errorf("PartVector accepted %s; want panic", name)
				return
			}
			if msg, _ := r.(string); !strings.Contains(msg, want) {
				t.Errorf("PartVector(%s) panicked with %q; want a message containing %q", name, r, want)
			}
		}()
		fn()
	}
	mustPanic("a replicated buffer", "replicated", func() { pd.PartVector(be.Alloc(main*main), 0) })
	mustPanic("a located row band", "located row band", func() {
		pd.PartVector(v.Slice(bounds[1], cols*n-(n-1)), 1)
	})
}
