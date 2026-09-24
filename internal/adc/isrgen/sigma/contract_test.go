package sigma

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
)

func randTens(e *engine, rng *rand.Rand, labels []uint8, dims []int) tens {
	h := make([]float64, size(dims))
	for i := range h {
		h[i] = rng.NormFloat64()
	}
	return tens{v: e.be.Upload(h), labels: labels, dims: dims}
}

func naive(t *testing.T, a, b tens, out []uint8, be backend.Backend) []float64 {
	t.Helper()
	ext := map[uint8]int{}
	for _, x := range []tens{a, b} {
		for m, l := range x.labels {
			ext[l] = x.dims[m]
		}
	}
	var all []uint8
	for l := range ext {
		all = append(all, l)
	}
	slices.Sort(all)
	odims := make([]int, len(out))
	for m, l := range out {
		odims[m] = ext[l]
	}
	res := make([]float64, size(odims))
	ad, bd := be.Download(a.v), be.Download(b.v)
	val := map[uint8]int{}
	at := func(x tens, d []float64) float64 {
		off, s := 0, 1
		for m, l := range x.labels {
			off += val[l] * s
			s *= x.dims[m]
		}
		return d[off]
	}
	var rec func(i int)
	rec = func(i int) {
		if i == len(all) {
			off, s := 0, 1
			for m, l := range out {
				off += val[l] * s
				s *= odims[m]
			}
			res[off] += at(a, ad) * at(b, bd)
			return
		}
		for x := range ext[all[i]] {
			val[all[i]] = x
			rec(i + 1)
		}
	}
	rec(0)
	return res
}

func TestContractMatchesNaive(t *testing.T) {
	e := &engine{be: backend.Gonum{}, k: hostKernels{h: backend.Gonum{}, be: backend.Gonum{}}}
	rng := rand.New(rand.NewPCG(1, 2))
	D := map[uint8]int{0: 3, 1: 4, 2: 2, 3: 5, 4: 3, 5: 2, colLabel: 3}
	mk := func(ls ...uint8) ([]uint8, []int) {
		d := make([]int, len(ls))
		for m, l := range ls {
			d[m] = D[l]
		}
		return ls, d
	}
	cases := []struct {
		name string
		a, b []uint8
		out  []uint8
	}{
		{"gemm", []uint8{0, 1}, []uint8{1, 2}, []uint8{0, 2}},
		{"gemm out permuted", []uint8{0, 1}, []uint8{1, 2}, []uint8{2, 0}},
		{"transposed A", []uint8{1, 0}, []uint8{1, 2}, []uint8{0, 2}},
		{"transposed B", []uint8{0, 1}, []uint8{2, 1}, []uint8{0, 2}},
		{"two contracted", []uint8{0, 3, 1}, []uint8{1, 2, 3}, []uint8{2, 0}},
		{"batch", []uint8{0, 1, 4}, []uint8{1, 4, 2}, []uint8{4, 2, 0}},
		{"batch column", []uint8{3, 1}, []uint8{1, 2, colLabel}, []uint8{3, 2, colLabel}},
		{"hadamard", []uint8{0, 1}, []uint8{0, 1, colLabel}, []uint8{0, 1, colLabel}},
		{"outer", []uint8{0}, []uint8{2, colLabel}, []uint8{2, 0, colLabel}},
		{"diagonal in A", []uint8{0, 0, 1}, []uint8{1, 2}, []uint8{0, 2}},
		{"trace in A", []uint8{3, 1, 3}, []uint8{1, 2}, []uint8{2}},
		{"summed in B only", []uint8{0, 1}, []uint8{1, 5, 2}, []uint8{0, 2}},
		{"scalar", []uint8{0, 1}, []uint8{0, 1}, nil},
		{"no free A", []uint8{1}, []uint8{1, 2, colLabel}, []uint8{2, colLabel}},
	}
	for _, c := range cases {
		la, da := mk(c.a...)
		lb, db := mk(c.b...)
		a, b := randTens(e, rng, la, da), randTens(e, rng, lb, db)
		got, err := e.contract(a, b, c.out)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		want := naive(t, a, b, c.out, e.be)
		if len(got.labels) != len(c.out) {
			t.Fatalf("%s: result labels %v for out %v", c.name, got.labels, c.out)
		}
		// the result may come in any mode order: read it back in out's order
		inOrder, _, err := e.permuted(got, c.out)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		g := e.be.Download(inOrder.v)
		var worst float64
		for i := range want {
			worst = math.Max(worst, math.Abs(g[i]-want[i]))
		}
		if worst > 1e-12 {
			t.Errorf("%s: max deviation %.2e", c.name, worst)
		}
	}
}

func TestContractRejectsRepeatedOutput(t *testing.T) {
	e := &engine{be: backend.Gonum{}, k: hostKernels{h: backend.Gonum{}, be: backend.Gonum{}}}
	rng := rand.New(rand.NewPCG(3, 4))
	a := randTens(e, rng, []uint8{0, 1}, []int{2, 2})
	b := randTens(e, rng, []uint8{1, 2}, []int{2, 2})
	if _, err := e.contract(a, b, []uint8{0, 0}); err == nil {
		t.Fatal("repeated output labels accepted")
	}
}
