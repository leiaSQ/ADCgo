package khci_test

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/isrgen/dip"
	"github.com/leiaSQ/ADCgo/internal/adc/isrgen/qip"
	"github.com/leiaSQ/ADCgo/internal/adc/isrgen/tip"
	"github.com/leiaSQ/ADCgo/internal/adc/khci"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

// ci_test.go — the gates of the CI operator: against pyscf FCI in
// class-complete sectors, orbital-rotation invariance, the generator's numpy
// Slater-Condon values, the ISR(<=1) evaluators, and its own enumeration.

var testdata = filepath.Join("..", "..", "..", "testdata")

func readDump(t *testing.T, rel string) *fcidump.Data {
	t.Helper()
	d, err := fcidump.ReadFile(filepath.Join(testdata, rel))
	if err != nil {
		t.Fatal(err)
	}
	return d
}

func newCI(t *testing.T, d *fcidump.Data, o khci.Options) *khci.CI {
	t.Helper()
	sp, err := khci.NewSpace(o)
	if err != nil {
		t.Fatal(err)
	}
	ci, err := khci.NewCI(sp, d, backend.Gonum{})
	if err != nil {
		t.Fatal(err)
	}
	return ci
}

func spectrum(t *testing.T, ci *khci.CI) ([]float64, backend.Mat) {
	t.Helper()
	w, v := backend.Gonum{}.SymEig(ci.BuildMatrix())
	return w, v
}

// fixtures is testdata/khci/fixtures_fci.json (pyscf FCI of every root per sector).
type fixtures struct {
	Systems map[string]struct {
		NOrb      int     `json:"norb"`
		NOcc      int     `json:"nocc"`
		EScf      float64 `json:"e_scf"`
		Canonical string  `json:"fcidump_canonical"`
		Localized string  `json:"fcidump_localized"`
		Sectors   map[string]struct {
			TwoMs    int       `json:"twoMs"`
			Dim      int       `json:"dim"`
			Complete bool      `json:"class_complete_through_k_plus_2"`
			E        []float64 `json:"E_minus_Escf"`
		} `json:"sectors"`
	} `json:"systems"`
}

func loadFixtures(t *testing.T) fixtures {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(testdata, "khci", "fixtures_fci.json"))
	if err != nil {
		t.Fatal(err)
	}
	var f fixtures
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

// TestClassCompleteEqualsFCI: where kh | (k+1)h1p | (k+2)h2p spans the whole
// N-k electron sector (He2 k = 2, 3 and He3 k = 4 in 6-31G), the CI spectrum IS the FCI
// spectrum, in both the canonical and the localized orbitals. E_HF + E_core equals
// pyscf's SCF energy, which fixes the zero.
func TestClassCompleteEqualsFCI(t *testing.T) {
	f := loadFixtures(t)
	checked := 0
	for name, sys := range f.Systems {
		for kstr, sec := range sys.Sectors {
			var k int
			fmt.Sscan(kstr, &k)
			if !sec.Complete || k > 4 {
				continue
			}
			for _, dump := range []string{sys.Canonical, sys.Localized} {
				d := readDump(t, dump)
				ci := newCI(t, d, khci.Options{K: k, NOcc: sys.NOcc, NVir: sys.NOrb - sys.NOcc, TwoMs: sec.TwoMs,
					MaxClass: min(k+2, 2*sys.NOcc)})
				if ci.Size() != sec.Dim {
					t.Fatalf("%s k=%d: CI space %d, FCI sector %d", name, k, ci.Size(), sec.Dim)
				}
				if e := ci.EHF() + d.Ecore - sys.EScf; math.Abs(e) > 1e-9 {
					t.Errorf("%s %s: E_HF + E_core misses the SCF energy by %.2e", name, dump, e)
				}
				w, _ := spectrum(t, ci)
				want := append([]float64(nil), sec.E...)
				sort.Float64s(want)
				var worst float64
				for i := range w {
					worst = max(worst, math.Abs(w[i]-want[i]))
				}
				t.Logf("%s k=%d %s: %d roots, max |CI - FCI| %.2e", name, k, filepath.Base(dump), len(w), worst)
				if worst > 1e-9 {
					t.Errorf("%s k=%d %s: class-complete CI differs from FCI by %.2e", name, k, dump, worst)
				}
				checked++
			}
		}
	}
	if checked < 6 {
		t.Fatalf("only %d class-complete sectors checked", checked)
	}
}

// TestLocalizedEqualsCanonical: the classes are invariant under occupied-occupied and
// virtual-virtual rotations, so the truncated CI spectrum (He3 k = 1, 2, 3, where the
// classes do NOT span the sector) must be the same in Pipek-Mezey / compact-free
// orbitals as in canonical ones. This is what lets CI run in localized orbitals.
func TestLocalizedEqualsCanonical(t *testing.T) {
	f := loadFixtures(t)
	sys := f.Systems["he3"]
	for _, k := range []int{1, 2, 3} {
		o := khci.Options{K: k, NOcc: sys.NOcc, NVir: sys.NOrb - sys.NOcc, TwoMs: k % 2}
		wc, _ := spectrum(t, newCI(t, readDump(t, sys.Canonical), o))
		wl, _ := spectrum(t, newCI(t, readDump(t, sys.Localized), o))
		var worst float64
		for i := range wc {
			worst = max(worst, math.Abs(wc[i]-wl[i]))
		}
		t.Logf("he3 k=%d: %d roots, max |canonical - localized| %.2e", k, len(wc), worst)
		if worst > 1e-9 {
			t.Errorf("he3 k=%d: truncated CI spectrum changes under localization by %.2e", k, worst)
		}
	}
}

// adcgenRef is the part of isrgen/*/testdata/adcgen_ref.json this test reads.
type adcgenRef struct {
	K         int    `json:"k"`
	NOcc      int    `json:"nocc"`
	NOrb      int    `json:"norb"`
	Fcidump   string `json:"fcidump"`
	CIEntries []struct {
		Block string  `json:"block"`
		Idx   []int   `json:"idx"`
		Value float64 `json:"value"`
	} `json:"ci_entries"`
}

func loadAdcgenRef(t *testing.T, variant string) (adcgenRef, *fcidump.Data) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "isrgen", variant, "testdata", "adcgen_ref.json"))
	if err != nil {
		t.Fatal(err)
	}
	var ref adcgenRef
	if err := json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	d, err := fcidump.Read(strings.NewReader(ref.Fcidump))
	if err != nil {
		t.Fatal(err)
	}
	return ref, d
}

// blockClasses gives the (bra, ket) classes of a block name B<bra><ket> as particle counts.
func blockClasses(b string) (int, int) { return int(b[1] - '0'), int(b[2] - '0') }

// TestMatchesGeneratorSlaterCondon: every ci_entries value of the four
// generated references (numpy Slater-Condon in the khci phase convention, evaluated by
// scripts/codegen/generate_adc.py independently of this package) is reproduced by CI.Element,
// for k = 1..4 and every block including B02.
func TestMatchesGeneratorSlaterCondon(t *testing.T) {
	for _, v := range []string{"ip", "dip", "tip", "qip"} {
		ref, d := loadAdcgenRef(t, v)
		ci := newCI(t, d, khci.Options{K: ref.K, NOcc: ref.NOcc, NVir: ref.NOrb - ref.NOcc, AllMs: true})
		sp := ci.Space()
		row := func(parts, holes []int) int {
			var m uint64
			for _, h := range holes {
				m |= 1 << uint(h)
			}
			r, ok := sp.Index(m, parts)
			if !ok {
				t.Fatalf("%s: probe holes %v particles %v not in the space", v, holes, parts)
			}
			return r
		}
		worst := map[string]float64{}
		for _, e := range ref.CIEntries {
			bp, kp := blockClasses(e.Block)
			idx := e.Idx
			nb := bp + ref.K + bp
			bra := row(idx[:bp], idx[bp:nb])
			ket := row(idx[nb:nb+kp], idx[nb+kp:])
			worst[e.Block] = max(worst[e.Block], math.Abs(ci.Element(bra, ket)-e.Value))
		}
		if len(worst) == 0 {
			t.Fatalf("%s: no ci_entries", v)
		}
		for b, w := range worst {
			t.Logf("%s %s: max |CI - numpy Slater-Condon| %.2e", v, b, w)
			if w > 1e-12 {
				t.Errorf("%s %s: CI element differs from the generator's Slater-Condon value by %.2e", v, b, w)
			}
		}
	}
}

// isrCI is the generated first-order ISR evaluator of one variant, on the same system.
type isrCI interface {
	Element(sp *khci.Space, r, c int) float64
}

// TestFirstOrderISREqualsCI: the generated ISR truncated at first order (the
// "ci" scheme of isrgen/dip, tip, qip) equals khci's CI element by element on every
// generated block except B02, where the ISR starts at second order: there the ISR must
// vanish while CI does not.
func TestFirstOrderISREqualsCI(t *testing.T) {
	for _, v := range []string{"dip", "tip", "qip"} {
		ref, d := loadAdcgenRef(t, v)
		nocc := mp.NOcc(d)
		eps := mp.OrbitalEnergies(d, nocc)
		ints := integrals.New(d, nocc, nil)
		var el isrCI
		var err error
		var gen [6]int
		switch v {
		case "dip":
			el, err = dip.New(ints, eps, nocc, "ci")
			gen = dip.Schemes["ci"]
		case "tip":
			el, err = tip.New(ints, eps, nocc, "ci")
			gen = tip.Schemes["ci"]
		case "qip":
			el, err = qip.New(ints, eps, nocc, "ci")
			gen = qip.Schemes["ci"]
		}
		if err != nil {
			t.Fatal(err)
		}
		// the classes whose blocks are generated
		maxClass := ref.K
		if gen[1] >= 0 && gen[2] >= 0 {
			maxClass = ref.K + 1
		}
		if gen[3] >= 0 && gen[4] >= 0 && gen[5] >= 0 {
			maxClass = ref.K + 2
		}
		ci := newCI(t, d, khci.Options{K: ref.K, NOcc: nocc, NVir: d.NORB - nocc, TwoMs: ref.K % 2, MaxClass: maxClass})
		sp := ci.Space()
		var worst, b02ci, b02isr float64
		for r := range sp.Size() {
			for c := r; c < sp.Size(); c++ {
				g, e := el.Element(sp, r, c), ci.Element(r, c)
				if sp.Class(r) == ref.K && sp.Class(c) == ref.K+2 {
					b02ci = max(b02ci, math.Abs(e))
					b02isr = max(b02isr, math.Abs(g))
					continue
				}
				worst = max(worst, math.Abs(g-e))
			}
		}
		t.Logf("%s (classes %d..%d, %d rows): max |ISR(<=1) - CI| %.2e; B02 CI %.3f, ISR %.1e",
			v, ref.K, maxClass, sp.Size(), worst, b02ci, b02isr)
		if worst > 1e-12 {
			t.Errorf("%s: first-order ISR differs from CI by %.2e", v, worst)
		}
		if maxClass == ref.K+2 && (b02isr != 0 || b02ci < 1e-3) {
			t.Errorf("%s: B02 should be CI-only at first order (CI %.2e, ISR %.2e)", v, b02ci, b02isr)
		}
	}
}

// TestApplyMatchesElements: the applier's enumeration (BuildMatrix, ApplyFull,
// ApplyBlock, the CSR path) and the independent Element evaluation agree, the matrix is
// symmetric, and the apply is bit-for-bit independent of GOMAXPROCS. Covers k = 1..4 on
// the random reference system and k = 2 with C2v symmetry on water.
func TestApplyMatchesElements(t *testing.T) {
	type tc struct {
		name string
		d    *fcidump.Data
		o    khci.Options
	}
	var cases []tc
	for _, v := range []string{"ip", "dip", "tip", "qip"} {
		ref, d := loadAdcgenRef(t, v)
		cases = append(cases, tc{v, d, khci.Options{K: ref.K, NOcc: ref.NOcc, NVir: ref.NOrb - ref.NOcc, TwoMs: ref.K % 2}})
	}
	h2o := readDump(t, "h2o.fcidump")
	nocc := mp.NOcc(h2o)
	sym := make([]int, h2o.NORB)
	for i, s := range h2o.OrbSym {
		sym[i] = s - 1
	}
	cases = append(cases, tc{"h2o k=2 C2v", h2o, khci.Options{K: 2, NOcc: nocc, NVir: h2o.NORB - nocc,
		OrbSym: sym, TargetIrrep: 0, MaxClass: 3}})
	for _, c := range cases {
		ci := newCI(t, c.d, c.o)
		n := ci.Size()
		M := ci.BuildMatrix()
		var worst, asym float64
		for r := range n {
			for col := range n {
				worst = max(worst, math.Abs(M.At(r, col)-ci.Element(r, col)))
				asym = max(asym, math.Abs(M.At(r, col)-M.At(col, r)))
			}
		}
		rng := rand.New(rand.NewPCG(1, uint64(n)))
		x := make([]float64, 3*n)
		for i := range x {
			x[i] = rng.NormFloat64()
		}
		want := make([]float64, 3*n)
		for j := range 3 {
			for r := range n {
				var s float64
				for col := range n {
					s += M.At(r, col) * x[j*n+col]
				}
				want[j*n+r] = s
			}
		}
		y1, y8 := applyInto(t, ci, x, n, 1), applyInto(t, ci, x, n, 8)
		var dApply float64
		for i := range want {
			dApply = max(dApply, math.Abs(y8[i]-want[i]))
			if y1[i] != y8[i] {
				t.Fatalf("%s: apply at GOMAXPROCS 1 and 8 differs in element %d", c.name, i)
			}
		}
		if !ci.Materialize(1 << 30) {
			t.Fatalf("%s: CSR did not fit 1 GiB", c.name)
		}
		yc := applyInto(t, ci, x, n, 8)
		var dCSR float64
		for i := range want {
			dCSR = max(dCSR, math.Abs(yc[i]-y8[i]))
		}
		t.Logf("%s: %d rows, %d couplings; |enum - Element| %.1e, asym %.1e, |apply - dense| %.1e, |CSR - enum| %.1e",
			c.name, n, ci.NNZ(), worst, asym, dApply, dCSR)
		if worst > 1e-13 || asym > 1e-13 || dApply > 1e-11 || dCSR != 0 {
			t.Errorf("%s: enumeration/Element %.2e, asymmetry %.2e, apply %.2e, CSR %.2e",
				c.name, worst, asym, dApply, dCSR)
		}
	}
}

// applyInto runs ApplyBlock on three columns at the given GOMAXPROCS.
func applyInto(t *testing.T, ci *khci.CI, x []float64, n, procs int) []float64 {
	t.Helper()
	prev := runtime.GOMAXPROCS(procs)
	defer runtime.GOMAXPROCS(prev)
	be := backend.Gonum{}
	in := be.Upload(append([]float64(nil), x...))
	out := be.Alloc(3 * n)
	ci.ApplyBlock(backend.BlockView{V: out, Rows: n, Cols: 3, Ld: n}, backend.BlockView{V: in, Rows: n, Cols: 3, Ld: n})
	return be.Download(out)
}

// TestRestrictIsSubBlock: the restricted operator (a Fano Q or P block) is exactly the
// parent's sub-block, through the enumeration on the restricted space.
func TestRestrictIsSubBlock(t *testing.T) {
	ref, d := loadAdcgenRef(t, "tip")
	ci := newCI(t, d, khci.Options{K: ref.K, NOcc: ref.NOcc, NVir: ref.NOrb - ref.NOcc, TwoMs: 1})
	M := ci.BuildMatrix()
	var rows []int
	for r := range ci.Size() {
		if r%3 != 1 {
			rows = append(rows, r)
		}
	}
	sub, err := ci.Restrict(rows)
	if err != nil {
		t.Fatal(err)
	}
	S := sub.BuildMatrix()
	for i, r := range rows {
		for j, c := range rows {
			if S.At(i, j) != M.At(r, c) {
				t.Fatalf("restricted (%d,%d) = %g, parent (%d,%d) = %g", i, j, S.At(i, j), r, c, M.At(r, c))
			}
		}
	}
}

// TestSpinSquared: every eigenvector of a class-complete sector has
// <S^2> = S(S+1) with 2S of the parity of k (one Ms sector holds one component of each
// multiplet). The count of roots per S is logged.
func TestSpinSquared(t *testing.T) {
	f := loadFixtures(t)
	sys := f.Systems["he2"]
	for _, k := range []int{2, 3} {
		ci := newCI(t, readDump(t, sys.Canonical), khci.Options{K: k, NOcc: sys.NOcc, NVir: sys.NOrb - sys.NOcc,
			TwoMs: k % 2, MaxClass: min(k+2, 2*sys.NOcc)})
		w, v := spectrum(t, ci)
		n := ci.Size()
		count := map[float64]int{}
		for j := range n {
			x := make([]float64, n)
			for i := range n {
				x[i] = v.At(i, j)
			}
			s2 := ci.SpinSquared(x)
			S := (math.Sqrt(1+4*s2) - 1) / 2
			if math.Abs(2*S-math.Round(2*S)) > 1e-6 || int(math.Round(2*S))%2 != k%2 {
				t.Errorf("k=%d root %d (E %.6f): <S^2> = %.8f is not S(S+1) for an S of parity %d",
					k, j, w[j], s2, k%2)
			}
			count[math.Round(2*S)/2]++
		}
		t.Logf("he2 k=%d: roots by S %v", k, count)
	}
}
