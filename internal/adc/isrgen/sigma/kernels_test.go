package sigma

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
)

func TestDeviceKernelsMatchNaive(t *testing.T) {
	dev, err := backend.New("cuda")
	if err != nil {
		t.Skipf("no cuda backend/device: %v", err)
	}
	tk, ok := dev.(backend.TensorKernels)
	if !ok {
		t.Fatalf("cuda backend %T does not implement backend.TensorKernels", dev)
	}
	e := &engine{be: dev, k: deviceKernels{tk}}
	rng := rand.New(rand.NewPCG(9, 10))
	D := map[uint8]int{0: 3, 1: 4, 2: 2, 3: 5, 4: 3, 5: 2, colLabel: 1}
	for _, c := range [][3][]uint8{
		{{0, 1}, {1, 2}, {0, 2}},
		{{1, 0}, {1, 2}, {0, 2}},
		{{0, 3, 1}, {1, 2, 3}, {2, 0}},
		{{0, 1, 4}, {1, 4, 2}, {4, 2, 0}},
		{{3, 1}, {1, 2, colLabel}, {3, 2, colLabel}},
		{{0, 1}, {0, 1, colLabel}, {0, 1, colLabel}},
		{{0, 0, 1}, {1, 2}, {0, 2}},
		{{3, 1, 3}, {1, 2}, {2}},
		{{0, 1}, {0, 1}, nil},
	} {
		mk := func(ls []uint8) []int {
			d := make([]int, len(ls))
			for m, l := range ls {
				d[m] = D[l]
			}
			return d
		}
		a, b := randTens(e, rng, c[0], mk(c[0])), randTens(e, rng, c[1], mk(c[1]))
		got, err := e.contract(a, b, c[2])
		if err != nil {
			t.Fatalf("%v·%v→%v: %v", c[0], c[1], c[2], err)
		}
		inOrder, _, err := e.permuted(got, c[2])
		if err != nil {
			t.Fatal(err)
		}
		want := naive(t, a, b, c[2], dev)
		g := dev.Download(inOrder.v)
		for i := range want {
			if math.Abs(g[i]-want[i]) > 1e-12 {
				t.Fatalf("%v·%v→%v: element %d = %g, want %g", c[0], c[1], c[2], i, g[i], want[i])
			}
		}
	}
}
