package sigma

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/khci"
)

func canonicalPatterns(nPart, nHole int) []string {
	var out []string
	for pa := 0; pa <= nPart; pa++ {
		for ha := 0; ha <= nHole; ha++ {
			s := ""
			for i := range nPart {
				if i < pa {
					s += "a"
				} else {
					s += "b"
				}
			}
			for i := range nHole {
				if i < ha {
					s += "a"
				} else {
					s += "b"
				}
			}
			out = append(out, s)
		}
	}
	return out
}

func TestPackMaps(t *testing.T) {
	for _, tc := range []struct{ K, twoMs int }{{2, 0}, {2, 2}, {1, 1}, {3, 1}} {
		sp, err := khci.NewSpace(khci.Options{K: tc.K, NOcc: 3, NVir: 3, TwoMs: tc.twoMs, MaxClass: tc.K + 2})
		if err != nil {
			t.Fatal(err)
		}
		n := sp.Size()
		seen := make([]int, n)
		rng := rand.New(rand.NewPCG(5, 6))
		y := make([]float64, n)
		for i := range y {
			y[i] = rng.NormFloat64()
		}
		back := make([]float64, n)
		be := backend.Gonum{}
		k := hostKernels{h: be, be: be}
		for c := 0; c <= 2; c++ {
			for _, pat := range canonicalPatterns(c, tc.K+c) {
				if patternMs(pat, c) != tc.twoMs {
					continue
				}
				canon, err := buildPackMap(sp, c, pat, 3, 3, false)
				if err != nil {
					t.Fatal(err)
				}
				full, err := buildPackMap(sp, c, pat, 3, 3, true)
				if err != nil {
					t.Fatal(err)
				}
				for _, r := range canon.row {
					seen[r]++
				}
				dims := (&engine{nocc: 3, nvir: 3}).modeDims(c, len(pat))
				tv := be.Alloc(size(dims))
				k.unpack(tv, size(dims), be.Upload(y), n, 1, full)
				// antisymmetry: every full entry holds sign·y[row]
				td := be.Download(tv)
				for i, e := range full.elem {
					if got, want := td[e], float64(full.sign[i])*y[full.row[i]]; got != want {
						t.Fatalf("K=%d %s: element %d holds %g, want %g", tc.K, pat, e, got, want)
					}
				}
				bv := be.Upload(make([]float64, n))
				k.pack(bv, n, 1, tv, size(dims), canon)
				for i, v := range be.Download(bv) {
					back[i] += v
				}
			}
		}
		for r, s := range seen {
			if s != 1 {
				t.Fatalf("K=%d 2Ms=%d: row %d in %d canonical blocks", tc.K, tc.twoMs, r, s)
			}
		}
		for i := range y {
			if math.Abs(back[i]-y[i]) > 0 {
				t.Fatalf("K=%d: pack(unpack(y)) row %d = %g, want %g", tc.K, i, back[i], y[i])
			}
		}
	}
}
