package analyze

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/dip"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
	"github.com/leiaSQ/ADCgo/internal/adc/mo"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

// TestPopSumMatchesPS is the M1 population oracle: the atom-resolved two-hole
// populations (one-site + two-site) must sum to the state's pole strength / 100.
// This is the same invariant ADCanalysis's TestPopSumMatchesPS enforces on the
// parsed reference tables.
func TestPopSumMatchesPS(t *testing.T) {
	base := filepath.Join("..", "..", "..", "testdata")
	d, err := fcidump.ReadFile(filepath.Join(base, "h2o.fcidump"))
	if err != nil {
		t.Fatal(err)
	}
	md, err := mo.ReadFile(filepath.Join(base, "h2o.mo.json"))
	if err != nil {
		t.Fatal(err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	ints := integrals.New(d, nocc, nil)
	be := backend.Gonum{}

	for _, spin := range []dip.Spin{dip.Singlet, dip.Triplet} {
		sp := dip.NewSpace(nocc, d.NORB, nil, 0, spin)
		mx := dip.New(sp, ints, eps, be)
		res := lanczos.SolveDense(mx, be)
		pe := NewPopEngine(sp, md)
		sec := BuildSector(sp, res, Options{PSThresh: 1, CoeffThresh: 0.1}, pe)

		for _, s := range sec.States {
			if s.Pop == nil {
				t.Fatalf("spin %d state %d has no population", spin, s.Index)
			}
			if d := math.Abs(s.Pop.Sum() - s.PSPercent/100); d > 1e-6 {
				t.Errorf("spin %d state %d: PopSum %.6f vs ps/100 %.6f (Δ=%.1e)",
					spin, s.Index, s.Pop.Sum(), s.PSPercent/100, d)
			}
		}
	}
}

// TestGroundStateLocalized checks the physical picture: the H2O dication ground
// state is a two-hole state localized on oxygen (Auger@O), so its O one-site
// weight dominates.
func TestGroundStateLocalized(t *testing.T) {
	base := filepath.Join("..", "..", "..", "testdata")
	d, _ := fcidump.ReadFile(filepath.Join(base, "h2o.fcidump"))
	md, _ := mo.ReadFile(filepath.Join(base, "h2o.mo.json"))
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	be := backend.Gonum{}

	sp := dip.NewSpace(nocc, d.NORB, nil, 0, dip.Singlet)
	mx := dip.New(sp, integrals.New(d, nocc, nil), eps, be)
	res := lanczos.SolveDense(mx, be)
	pe := NewPopEngine(sp, md)
	sec := BuildSector(sp, res, Options{PSThresh: 20, CoeffThresh: 0.1}, pe)

	g := sec.States[0]
	if o := g.Pop.OneSite["O"]; o < 0.7 {
		t.Errorf("ground state O one-site weight %.3f, want > 0.7 (localized on O)", o)
	}
}

// serialUO rebuilds U and O with the original single-threaded loop nesting, as the
// bit-identity reference for the parallel fills in NewPopEngine.
func serialUO(sp *dip.Space, md *mo.Data) (backend.Mat, backend.Mat) {
	main := sp.MainBlockSize()
	nao := md.NAO
	naopair := nao * (nao + 1) / 2
	fact1, fact2 := 1.0, 1.0
	if sp.Spin == dip.Triplet {
		fact1, fact2 = -1.0, 0.0
	}
	C := md.C

	U := backend.NewMat(naopair, main)
	for c := range main {
		i, j := sp.Configs[c].Occ[0], sp.Configs[c].Occ[1]
		for p := range nao {
			for q := 0; q <= p; q++ {
				var u float64
				switch {
				case i != j && p != q:
					u = C.At(p, i)*C.At(q, j) + fact1*C.At(q, i)*C.At(p, j)
				case i != j && p == q:
					u = C.At(p, i) * C.At(p, j) * fact2
				case i == j && p != q:
					u = sqrt2 * C.At(p, i) * C.At(q, i) * fact2
				default:
					u = sqrt1_2 * C.At(p, i) * C.At(p, i) * fact2
				}
				U.Set(triIdx(p, q), c, u)
			}
		}
	}

	S := md.S
	O := backend.NewMat(naopair, naopair)
	for p := range nao {
		for q := 0; q <= p; q++ {
			kpq := triIdx(p, q)
			for r := range nao {
				for s := 0; s <= r; s++ {
					O.Set(kpq, triIdx(r, s), S.At(p, r)*S.At(q, s)+fact1*S.At(p, s)*S.At(q, r))
				}
			}
		}
	}
	return U, O
}

// TestPopEngineMatricesBitIdentical: the U and O fills run one AO-pair row per work
// item, every cell written exactly once from immutable inputs with the per-element
// arithmetic unchanged. That makes them bit-identical to the serial loops, not merely
// close, and this asserts it with == rather than a tolerance.
func TestPopEngineMatricesBitIdentical(t *testing.T) {
	base := filepath.Join("..", "..", "..", "testdata")
	d, err := fcidump.ReadFile(filepath.Join(base, "h2o.fcidump"))
	if err != nil {
		t.Fatal(err)
	}
	md, err := mo.ReadFile(filepath.Join(base, "h2o.mo.json"))
	if err != nil {
		t.Fatal(err)
	}
	nocc := mp.NOcc(d)

	for _, spin := range []dip.Spin{dip.Singlet, dip.Triplet} {
		sp := dip.NewSpace(nocc, d.NORB, nil, 0, spin)
		wantU, wantO := serialUO(sp, md)
		pe := NewPopEngine(sp, md)
		for i, want := range wantU.Data {
			if pe.U.Data[i] != want {
				t.Fatalf("spin %d: U.Data[%d] = %.17g, want %.17g", spin, i, pe.U.Data[i], want)
			}
		}
		for i, want := range wantO.Data {
			if pe.O.Data[i] != want {
				t.Fatalf("spin %d: O.Data[%d] = %.17g, want %.17g", spin, i, pe.O.Data[i], want)
			}
		}
	}
}
