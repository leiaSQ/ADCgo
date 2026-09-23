package dip

import (
	"fmt"
	"math"
	"math/bits"
	"path/filepath"
	"sort"
	"testing"

	"gonum.org/v1/gonum/mat"

	"github.com/leiaSQ/ADCgo/backend"
	dippkg "github.com/leiaSQ/ADCgo/internal/adc/dip"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/khci"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

// crosscheck_test.go (hand-written; survives regeneration): the adcgen-generated
// DIP-ADC evaluators against the hand-ported package dip (Tarantelli's DIP-ADC(2)).

func containsAll(got, want []float64) float64 {
	var worst float64
	for _, w := range want {
		k := sort.SearchFloat64s(got, w)
		d := math.Inf(1)
		for _, c := range []int{k - 1, k} {
			if c >= 0 && c < len(got) {
				d = min(d, math.Abs(got[c]-w))
			}
		}
		worst = max(worst, d)
	}
	return worst
}

// TestSpectraMatchDipPackage: the singlet and triplet spectra of package dip (h2o,
// symmetry off) are contained in the Ms = 0 spin-orbital spectrum of the generated
// evaluators under the scheme that is Tarantelli's DIP-ADC(2). That scheme is adc2x
// (the 3h1p/3h1p block through first order); strict:2 (3h1p/3h1p zeroth order) is
// logged for contrast and must NOT match, or the scheme identification is vacuous.
func TestSpectraMatchDipPackage(t *testing.T) {
	d, err := fcidump.ReadFile(filepath.Join("..", "..", "..", "..", "testdata", "h2o.fcidump"))
	if err != nil {
		t.Fatal(err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	ints := integrals.New(d, nocc, nil)
	be := backend.Gonum{}
	var want []float64
	for _, spin := range []dippkg.Spin{dippkg.Singlet, dippkg.Triplet} {
		sp := dippkg.NewSpace(nocc, d.NORB, nil, 0, spin)
		w, _ := be.SymEig(dippkg.New(sp, ints, eps, be).BuildMatrix())
		want = append(want, w...)
	}
	sp, err := khci.NewSpace(khci.Options{K: 2, NOcc: nocc, NVir: d.NORB - nocc, TwoMs: 0, MaxClass: 3})
	if err != nil {
		t.Fatal(err)
	}
	gaps := map[string]float64{}
	for _, scheme := range []string{"adc2x", "strict:2"} {
		el, err := New(ints, eps, nocc, scheme)
		if err != nil {
			t.Fatal(err)
		}
		got, _ := be.SymEig(el.BuildDense(sp))
		sort.Float64s(got)
		gaps[scheme] = containsAll(got, want)
		t.Logf("%s: %d spin-orbital roots vs %d dip roots, max gap %.2e", scheme, len(got), len(want), gaps[scheme])
	}
	if gaps["adc2x"] > 1e-9 {
		t.Errorf("dip package eigenvalue missing from the adc2x spectrum, max gap %g", gaps["adc2x"])
	}
	if gaps["strict:2"] < 1e-6 {
		t.Errorf("strict:2 also reproduces the dip spectrum (gap %g): the scheme check is vacuous", gaps["strict:2"])
	}
}

// spatialKey names a configuration's spatial occupation: the hole orbitals (a doubly
// emptied orbital twice) and the particle orbitals, both ascending. Every spin function
// of one spatial configuration shares it.
func spatialKey(holes, parts []int) string {
	h := append([]int(nil), holes...)
	p := append([]int(nil), parts...)
	sort.Ints(h)
	sort.Ints(p)
	return fmt.Sprint(h, p)
}

// fermiSign is (-1)^(number of occupied spin orbitals below p) on a canonical
// (ascending) determinant.
func fermiSign(occ uint64, p int) float64 {
	if bits.OnesCount64(occ&(uint64(1)<<uint(p)-1))&1 == 1 {
		return -1
	}
	return 1
}

// shiftSpin applies sum_p c+_(p,to) c_(p,from) (S- for from = alpha, S+ for from =
// beta) to canonical determinants, spatial orbitals p < norb.
func shiftSpin(in map[uint64]float64, norb, from int) map[uint64]float64 {
	out := map[uint64]float64{}
	for occ, c := range in {
		for p := range norb {
			f, t := 2*p+from, 2*p+1-from
			if occ>>uint(f)&1 == 0 || occ>>uint(t)&1 == 1 {
				continue
			}
			s := fermiSign(occ, f)
			mid := occ &^ (1 << uint(f))
			s *= fermiSign(mid, t)
			out[mid|1<<uint(t)] += s * c
		}
	}
	return out
}

// spinBases returns, for every spatial configuration of an Ms = 0 khci space, its rows
// and an orthonormal basis (columns over those rows) of the S = 0 and S = 1 eigenspaces
// of S^2 = S+ S- (Ms = 0), built in the khci row phase (|row> = sign |det>).
func spinBases(t *testing.T, sp *khci.Space, norb int) (rows map[string][]int, basis map[string][2]*mat.Dense) {
	t.Helper()
	if 2*norb > 64 {
		t.Fatalf("%d spin orbitals do not fit a determinant mask", 2*norb)
	}
	rows = map[string][]int{}
	var order []string
	for r := range sp.Size() {
		k := spatialKey(sp.Holes(r, nil), sp.Particles(r, nil))
		if _, ok := rows[k]; !ok {
			order = append(order, k)
		}
		rows[k] = append(rows[k], r)
	}
	basis = map[string][2]*mat.Dense{}
	for _, k := range order {
		rs := rows[k]
		n := len(rs)
		mask := make([]uint64, n)
		sign := make([]float64, n)
		at := map[uint64]int{}
		for i, r := range rs {
			occ, parts, s := sp.Det(r)
			for _, p := range parts {
				occ |= 1 << uint(p)
			}
			mask[i], sign[i], at[occ] = occ, s, i
		}
		s2 := mat.NewSymDense(n, nil)
		for j := range n {
			img := shiftSpin(shiftSpin(map[uint64]float64{mask[j]: 1}, norb, 0), norb, 1)
			for occ, c := range img {
				i, ok := at[occ]
				if !ok {
					t.Fatalf("S^2 leaves the spatial configuration %s", k)
				}
				if i >= j {
					s2.SetSym(i, j, sign[i]*sign[j]*c)
				}
			}
		}
		var eig mat.EigenSym
		if !eig.Factorize(s2, true) {
			t.Fatalf("S^2 eigendecomposition failed for %s", k)
		}
		var vec mat.Dense
		eig.VectorsTo(&vec)
		var out [2]*mat.Dense
		for S, want := range []float64{0, 2} {
			var cols []int
			for c, w := range eig.Values(nil) {
				if math.Abs(w-want) < 1e-9 {
					cols = append(cols, c)
				}
			}
			if len(cols) == 0 {
				continue
			}
			b := mat.NewDense(n, len(cols), nil)
			for jj, c := range cols {
				for i := range n {
					b.Set(i, jj, vec.At(i, c))
				}
			}
			out[S] = b
		}
		basis[k] = out
	}
	return rows, basis
}

// dipKey is the spatial configuration of a row of package dip (config.go): 2h rows use
// Occ[0], Occ[1]; 3h1p rows all three holes and the particle.
func dipKey(sp *dippkg.Space, r int) string {
	c := sp.Configs[r]
	if r < sp.MainBlockSize() {
		return spatialKey([]int{c.Occ[0], c.Occ[1]}, nil)
	}
	return spatialKey(c.Occ[:], []int{c.Vir})
}

// singular returns the singular values of m, descending.
func singular(m mat.Matrix) []float64 {
	var svd mat.SVD
	if !svd.Factorize(m, mat.SVDNone) {
		panic("svd failed")
	}
	return svd.Values(nil)
}

// TestSpinAdaptedBlocksMatchDipPackage: second source for package dip's spin-adapted
// elements, block by block. Tarantelli's spin functions of one spatial configuration
// (two singlets and three triplets for an |ijkr> group) are fixed only up to a rotation
// within that configuration, so what can be compared without transcribing them are the
// invariants of that gauge: for every pair of spatial configurations (G, G'), the
// singular values of dip's block M[G,G'] must equal those of the generated adc2x block
// projected onto the S = 0 or S = 1 eigenspace of S^2 in G and G'; and every spatial
// configuration must carry as many dip rows as S^2 eigenvectors. Unlike spectral
// containment (TestSpectraMatchDipPackage) this pins each coupling block separately.
func TestSpinAdaptedBlocksMatchDipPackage(t *testing.T) {
	d, err := fcidump.ReadFile(filepath.Join("..", "..", "..", "..", "testdata", "h2o.fcidump"))
	if err != nil {
		t.Fatal(err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	ints := integrals.New(d, nocc, nil)
	be := backend.Gonum{}
	ksp, err := khci.NewSpace(khci.Options{K: 2, NOcc: nocc, NVir: d.NORB - nocc, TwoMs: 0, MaxClass: 3})
	if err != nil {
		t.Fatal(err)
	}
	el, err := New(ints, eps, nocc, "adc2x")
	if err != nil {
		t.Fatal(err)
	}
	M := el.BuildDense(ksp)
	Mg := mat.NewDense(M.Rows, M.Cols, M.Data)
	krows, kbasis := spinBases(t, ksp, d.NORB)
	names := [2]string{"2h", "3h1p"}
	for S, spin := range []dippkg.Spin{dippkg.Singlet, dippkg.Triplet} {
		sp := dippkg.NewSpace(nocc, d.NORB, nil, 0, spin)
		R := dippkg.New(sp, ints, eps, be).BuildMatrix()
		Rd := mat.NewDense(R.Rows, R.Cols, R.Data)
		drows := map[string][]int{}
		var keys []string
		for r := range sp.Size() {
			k := dipKey(sp, r)
			if _, ok := drows[k]; !ok {
				keys = append(keys, k)
			}
			drows[k] = append(drows[k], r)
		}
		// completeness: every spatial configuration with spin-S functions is in dip, with
		// as many rows as functions
		for k, b := range kbasis {
			nk := 0
			if b[S] != nil {
				_, nk = b[S].Dims()
			}
			if len(drows[k]) != nk {
				t.Errorf("spin %d, configuration %s: dip has %d rows, S^2 gives %d functions",
					spin, k, len(drows[k]), nk)
			}
		}
		// projected generated rows: P[k] = B_k^T M[rows_k, :], over all khci columns
		proj := map[string]*mat.Dense{}
		for _, k := range keys {
			b := kbasis[k][S]
			rs := krows[k]
			sub := mat.NewDense(len(rs), M.Cols, nil)
			for i, r := range rs {
				sub.SetRow(i, Mg.RawRowView(r))
			}
			var p mat.Dense
			p.Mul(b.T(), sub)
			proj[k] = &p
		}
		var worst [2][2]float64
		cls := func(k string) int {
			if drows[k][0] < sp.MainBlockSize() {
				return 0
			}
			return 1
		}
		for a, ka := range keys {
			for _, kb := range keys[a:] {
				ra, rb := drows[ka], drows[kb]
				want := mat.NewDense(len(ra), len(rb), nil)
				for i, r := range ra {
					for j, c := range rb {
						want.Set(i, j, Rd.At(r, c))
					}
				}
				bb := kbasis[kb][S]
				cols := krows[kb]
				pa := proj[ka]
				blk := mat.NewDense(len(ra), len(cols), nil)
				for i := range len(ra) {
					for j, c := range cols {
						blk.Set(i, j, pa.At(i, c))
					}
				}
				var got mat.Dense
				got.Mul(blk, bb)
				sw, sg := singular(want), singular(&got)
				ca, cb := cls(ka), cls(kb)
				for i := range sw {
					worst[ca][cb] = max(worst[ca][cb], math.Abs(sw[i]-sg[i]))
				}
			}
		}
		for ca := range 2 {
			for cb := ca; cb < 2; cb++ {
				t.Logf("spin %d %s/%s: max singular-value difference %.2e", spin, names[ca], names[cb], worst[ca][cb])
				if worst[ca][cb] > 1e-11 {
					t.Errorf("spin %d %s/%s: dip block differs from the adcgen adc2x derivation by %.2e",
						spin, names[ca], names[cb], worst[ca][cb])
				}
			}
		}
	}
}
