package main

import (
	"math"
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/mat"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fano"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/mo"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
	"github.com/leiaSQ/ADCgo/internal/adc/sip"
	"github.com/leiaSQ/ADCgo/internal/adc/spectrum"
)

// waterBlocks builds water's SIP ADC(2)x space and, per particle orbital, the block's
// rows, adapted vectors and determinant expansions.
func waterBlocks(t *testing.T) (nocc int, blocks []struct {
	rows []int
	U    backend.Mat
	dets [][]sip.DetTerm
}) {
	t.Helper()
	d := testFCIDUMP(t)
	nocc = mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	ints := integrals.New(d, nocc, nil)
	sp := sip.NewSpace(nocc, d.NORB, nil, 0)
	groups := map[int][]int{}
	for r := sp.BeginSat; r < len(sp.Configs); r++ {
		groups[sp.Configs[r].Vir] = append(groups[sp.Configs[r].Vir], r)
	}
	for p := range d.NORB - nocc {
		rows := groups[p]
		if len(rows) == 0 {
			continue
		}
		mx := sip.New(sp.Restrict(rows), ints, eps, 2, backend.Gonum{})
		_, U := backend.Gonum{}.SymEig(mx.BuildMatrix())
		dets := make([][]sip.DetTerm, len(rows))
		for i, r := range rows {
			var err error
			if dets[i], err = sp.DetExpansion(r); err != nil {
				t.Fatal(err)
			}
		}
		blocks = append(blocks, struct {
			rows []int
			U    backend.Mat
			dets [][]sip.DetTerm
		}{rows, U, dets})
	}
	return nocc, blocks
}

func randomOrthogonal(n int, seed int64) *mat.Dense {
	rng := rand.New(rand.NewSource(seed))
	A := mat.NewDense(n, n, nil)
	for i := range n {
		for j := range n {
			A.Set(i, j, rng.NormFloat64())
		}
	}
	var qr mat.QR
	qr.Factorize(A)
	var Q mat.Dense
	qr.QTo(&Q)
	return &Q
}

// TestTwoHoleCharacterInvariants: one site owns everything -> every adapted state is fully
// one-site; and for any orthogonal L the block totals of the one-site and inner weights
// are the same over the adapted states as over the configurations they rotate (a trace
// is basis independent), which pins the rotation and the normalization.
func TestTwoHoleCharacterInvariants(t *testing.T) {
	nocc, blocks := waterBlocks(t)
	if len(blocks) == 0 {
		t.Fatal("no particle blocks")
	}
	inner := make([]bool, nocc)
	inner[1] = true
	L := randomOrthogonal(nocc, 7)
	split := make([]int, nocc)
	for k := range nocc {
		split[k] = k % 2
	}
	one := make([]int, nocc)
	for _, bl := range blocks {
		m := len(bl.rows)
		u := make([]float64, m)
		var sumA, sumC, innA, innC float64
		for j := range m {
			for i := range m {
				u[i] = bl.U.At(i, j)
			}
			iw, os := twoHoleCharacter(u, bl.dets, nocc, L, split, inner)
			if os < -1e-12 || os > 1+1e-12 || iw < -1e-12 || iw > 1+1e-12 {
				t.Fatalf("weights outside [0,1]: inner %g, one-site %g", iw, os)
			}
			sumA += os
			innA += iw
			if _, all := twoHoleCharacter(u, bl.dets, nocc, L, one, inner); math.Abs(all-1) > 1e-12 {
				t.Fatalf("one site owns every orbital but the one-site weight is %g", all)
			}
			clear(u)
			u[j] = 1 // the configuration itself
			iwC, osC := twoHoleCharacter(u, bl.dets, nocc, L, split, inner)
			sumC += osC
			innC += iwC
		}
		if math.Abs(sumA-sumC) > 1e-10 || math.Abs(innA-innC) > 1e-10 {
			t.Errorf("block of %d: one-site total %.12f (adapted) vs %.12f (configurations); "+
				"inner %.12f vs %.12f", m, sumA, sumC, innA, innC)
		}
	}
}

// TestSchemeBRotatedBlocksAreDiagonal: with the particle index rotated (compact | free),
// the adapted operator T^T M T must be exactly diagonal on every particle block, with the
// block eigenvalues on its diagonal, and its spectrum must be M's (T is orthogonal). This
// pins the assembly of the rotated blocks from the dense 2h1p block, which nothing else
// checks. The hydrogens are treated as non-compact centres (charge 0) so that the compact
// span — the oxygen's AOs projected onto the virtuals — is a genuine rotation of them.
func TestSchemeBRotatedBlocksAreDiagonal(t *testing.T) {
	d, err := fcidump.ReadFile("../../testdata/h2o_dzp.fcidump")
	if err != nil {
		t.Fatal(err)
	}
	nocc := mp.NOcc(d)
	eps := mp.OrbitalEnergies(d, nocc)
	ints := integrals.New(d, nocc, nil)
	ssp := sip.NewSpace(nocc, d.NORB, nil, 0)
	pmx := sip.New(ssp, ints, eps, 2, backend.Gonum{})
	md, err := mo.ReadCanonical("../../testdata/h2o_dzp.mo.json")
	if err != nil {
		t.Fatal(err)
	}
	md.AtomCharges[1], md.AtomCharges[2] = 0, 0
	sel, err := fano.ParseClassRule("q:0:1", "")
	if err != nil {
		t.Fatal(err)
	}
	base := fano.NewPartition(ssp, sel)
	cfg := fanoConfig{sip: sipConfig{order: 2}, bInner: "e<-1.0", bCut: 0.5, bOneSite: "x",
		bParticle: "free", bCompactThresh: 0.02,
		sites: []spectrum.Site{{Name: "W", Members: []string{"O", "H1", "H2"}}}}
	f := sipFamily{parent: ssp, ints: ints, eps: eps, cfg: cfg.sip}
	part, fam, st, err := buildSchemeBWith(md, cfg, ssp, f, pmx, base, eps, nocc, nil)
	if err != nil {
		t.Fatal(err)
	}
	if st.CompactVirtuals == 0 || st.CompactVirtuals == d.NORB-nocc {
		t.Fatalf("%d of %d virtuals compact: the rotation would be trivial", st.CompactVirtuals, d.NORB-nocc)
	}
	A := fam.full.BuildMatrix()
	var maxOff, maxDiag float64
	for _, bl := range fam.full.Transform().Blocks() {
		for i, ri := range bl.Rows {
			maxDiag = math.Max(maxDiag, math.Abs(A.At(ri, ri)-bl.Values[i]))
			for j, rj := range bl.Rows {
				if i != j {
					maxOff = math.Max(maxOff, math.Abs(A.At(ri, rj)))
				}
			}
		}
	}
	if maxOff > 1e-10 || maxDiag > 1e-10 {
		t.Errorf("rotated particle blocks: max off-diagonal %.3g, max |diag - eigenvalue| %.3g", maxOff, maxDiag)
	}
	ea, _ := backend.Gonum{}.SymEig(A)
	em, _ := backend.Gonum{}.SymEig(pmx.BuildMatrix())
	var de float64
	for i := range ea {
		de = math.Max(de, math.Abs(ea[i]-em[i]))
	}
	if de > 1e-10 {
		t.Errorf("T^T M T and M have different spectra (max diff %.3g): T is not orthogonal", de)
	}
	t.Logf("%d compact of %d virtuals; %d blocks; adapted Q %d P %d X %d (bound %d); %s",
		st.CompactVirtuals, d.NORB-nocc, st.Blocks, st.AdaptedQ, st.AdaptedP, st.AdaptedX, st.AdaptedXBound,
		part.CensusString())
}
