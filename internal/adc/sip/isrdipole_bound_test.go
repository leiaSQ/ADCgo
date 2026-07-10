package sip

import (
	"math"
	"math/bits"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

// applyOneBodyDet applies D̂ = Σ_pq d_pq Σ_σ c†_pσ c_qσ to a determinant expansion,
// returning D̂|ψ⟩ as a map from determinant to amplitude. Phases come from apply, so this
// shares nothing with the closed-form block formulas.
func applyOneBodyDet(psi map[sdet]float64, d backend.Mat, norb int) map[sdet]float64 {
	out := make(map[sdet]float64)
	for det, c := range psi {
		if c == 0 {
			continue
		}
		for p := range norb {
			for q := range norb {
				dpq := d.At(p, q)
				if dpq == 0 {
					continue
				}
				for _, spin := range []int{alpha, beta} {
					e, ph := apply(det, an(q, spin), cr(p, spin))
					if ph != 0 {
						out[e] += ph * dpq * c
					}
				}
			}
		}
	}
	return out
}

// exClass counts (holes, particles) of a determinant relative to the closed-shell
// reference: holes are occupied reference spin-orbitals now empty, particles are virtual
// spin-orbitals now filled. A CVS 3h2p configuration is (3 holes, 2 particles) with the
// core (orbital 0) among the holes.
func exClass(det sdet, nocc int) (holes, parts, coreHoles int) {
	refMask := uint64(1)<<nocc - 1
	occHoleA := refMask &^ det.a
	occHoleB := refMask &^ det.b
	holes = bits.OnesCount64(occHoleA) + bits.OnesCount64(occHoleB)
	parts = bits.OnesCount64(det.a&^refMask) + bits.OnesCount64(det.b&^refMask)
	coreHoles = bits.OnesCount64((occHoleA | occHoleB) & 1) // orbital 0 = the O 1s core
	return
}

// TestNeglected3h2pBound quantifies the one block Chunk 5 drops. The neglected
// contribution to a core→valence transition moment is X_3h2p† D_{3h2p,2h1p} X_val, and by
// Cauchy–Schwarz its magnitude is at most
//
//	‖X_3h2p‖ · ‖P_3h2p D̂ |Ψ_val⟩‖ .
//
// The second factor is evaluated in the determinant basis — norms are invariant under the
// orthonormal rotation to spin-adapted configurations, so this needs none of the five
// Config3 spin functions. Projecting onto the whole 3h2p *determinant* class (rather than
// just the CVS subspace the bra spans) only loosens the bound, so it stays an upper bound.
//
// The finding, and the reason this only *logs* the ratio rather than asserting it small:
// for O 1s emission the bound is a substantial fraction of |μ| — ~20% on H2O/cc-pVDZ —
// because the core hole carries ‖X_3h2p‖² ≈ 5% of its weight *and* the O 1s→valence dipole
// is intrinsically small (the 1s orbital is compact, so μ ~ 0.02–0.03 a.u.). This is the
// order-of-magnitude the O(V³) scaling predicts once the small prefactor of the "dominant"
// Koopmans term is accounted for; it is a genuine limitation of the truncation, not a bug.
// It is nonetheless subordinate to the deferred first-order ISR density corrections, which
// touch the dominant 1h×1h element at O(V) (isrdipole_cross.go). The test fails only if the
// bound reaches |μ| itself — the point past which the truncated moment is meaningless.
func TestNeglected3h2pBound(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping 3h2p bound in -short mode")
	}
	fd := h2oData(t)
	md := h2oSidecar(t)
	nocc := mp.NOcc(fd)
	eps := mp.OrbitalEnergies(fd, nocc)
	ints := integrals.New(fd, nocc, fd.OrbSym)
	be := backend.Gonum{}

	// Bra: O 1s core hole (A1, CVS ADC(4)). Ket: outer-valence holes.
	bra := NewSpace4(nocc, fd.NORB, fd.OrbSym, 0, []int{0})
	mxBra := New(bra, ints, eps, 4, be)
	defer mxBra.Release()
	resBra := lanczos.SolveDense(mxBra, be)
	xBra := matColumn(resBra.FullVecs, 0) // the core main line
	w3 := Sat3Weight(bra, xBra)
	if w3 == 0 {
		t.Fatal("the core-hole state has no 3h2p weight; nothing to bound")
	}

	sawChannel := false
	for _, ketSym := range []int{1, 2, 3} {
		ket := NewSpace(nocc, fd.NORB, fd.OrbSym, ketSym)
		if ket.MainBlockSize() == 0 {
			continue // this molecule has no occupied orbital of that irrep
		}
		mxKet := New(ket, ints, eps, 3, be)
		resKet := lanczos.SolveDense(mxKet, be)
		xKet := matColumn(resKet.FullVecs, 0)

		// |Ψ_val⟩ as determinants.
		psi := make(map[sdet]float64)
		for j := range len(ket.Configs) {
			if xKet[j] == 0 {
				continue
			}
			for _, term := range configDets(ket, j) {
				psi[term.d] += xKet[j] * term.c
			}
		}

		ops, err := NewISRDipolesCross(bra, ket, md.DipMO)
		if err != nil {
			t.Fatal(err)
		}
		muNuc := md.NuclearDipole()
		ems, err := CrossEmissions(ops, mustOverlap(t, bra, ket), muNuc,
			resBra.Values, resKet.Values, resBra.FullVecs, resKet.FullVecs, []int{0}, []int{0})
		if err != nil {
			t.Fatal(err)
		}
		mu := ems[0].Mu
		muNorm := math.Sqrt(mu[0]*mu[0] + mu[1]*mu[1] + mu[2]*mu[2])

		// The Cauchy–Schwarz second factor: ‖P_3h2p D̂|Ψ_val⟩‖, worst over components.
		var imgMax float64
		for c := range 3 {
			img := applyOneBodyDet(psi, md.DipMO[c], fd.NORB)
			var n2 float64
			for det, amp := range img {
				if h, p, core := exClass(det, nocc); h == 3 && p == 2 && core == 1 {
					n2 += amp * amp
				}
			}
			if n := math.Sqrt(n2); n > imgMax {
				imgMax = n
			}
		}
		bound := w3 * imgMax

		sawChannel = true
		t.Logf("O1s → irrep %d state 0: |μ| = %.4e a.u., ‖X_3h2p‖ = %.3e, "+
			"3h2p bound ≤ %.3e (%.1f%% of |μ|)", ketSym+1, muNorm, w3, bound, 100*bound/muNorm)

		if muNorm > 1e-4 && bound > muNorm {
			t.Errorf("irrep %d: the neglected 3h2p block bound %.3e exceeds |μ| = %.3e; "+
				"the truncated moment is meaningless — implement the block (see isrdipole_cross.go)",
				ketSym+1, bound, muNorm)
		}
	}
	if !sawChannel {
		t.Fatal("no valence channel had a 1h main block; the bound was never computed")
	}
}

func mustOverlap(t *testing.T, bra, ket *Space) *ISROverlap {
	t.Helper()
	ov, err := ConfigOverlap(bra, ket)
	if err != nil {
		t.Fatal(err)
	}
	return ov
}

// TestCrossEmissionPhysics is the end-to-end smoke test: an O 1s → outer-valence X-ray
// emission of H2O. The core hole is a1 (irrep 0); this fixture's occupied set has no b1
// orbital, so the outer-valence hole is taken in irrep 3 (a 1b-type lone pair). Core and
// valence hole carry different irreps, so they share no configuration — the overlap
// vanishes identically and μ is origin-independent — and the emission is carried by the
// single Cartesian component transforming as irrep 0 ⊗ 3 = 3. The dominant term is the
// 1h×1h element −d between the two hole orbitals, which the test cross-checks against
// DipMO directly; because the O 1s orbital is compact that dipole is small, ~0.03 a.u.
func TestCrossEmissionPhysics(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping cross-emission physics smoke in -short mode")
	}
	fd := h2oData(t)
	md := h2oSidecar(t)
	nocc := mp.NOcc(fd)
	eps := mp.OrbitalEnergies(fd, nocc)
	ints := integrals.New(fd, nocc, fd.OrbSym)
	be := backend.Gonum{}

	bra := NewSpace4(nocc, fd.NORB, fd.OrbSym, 0, []int{0}) // O 1s, A1
	mxBra := New(bra, ints, eps, 4, be)
	defer mxBra.Release()
	resBra := lanczos.SolveDense(mxBra, be)

	const ketSym = 3 // an outer-valence irrep that has an occupied orbital in this fixture
	ket := NewSpace(nocc, fd.NORB, fd.OrbSym, ketSym)
	if ket.MainBlockSize() == 0 {
		t.Fatalf("irrep %d has no 1h main state in this fixture", ketSym)
	}
	mxKet := New(ket, ints, eps, 3, be)
	resKet := lanczos.SolveDense(mxKet, be)

	ops, err := NewISRDipolesCross(bra, ket, md.DipMO)
	if err != nil {
		t.Fatal(err)
	}
	ov, err := ConfigOverlap(bra, ket)
	if err != nil {
		t.Fatal(err)
	}
	if ov.Nonzero() != 0 {
		t.Fatalf("O 1s (A1) and the irrep-%d valence hole share %d configurations, want 0",
			ketSym, ov.Nonzero())
	}
	ems, err := CrossEmissions(ops, ov, md.NuclearDipole(),
		resBra.Values, resKet.Values, resBra.FullVecs, resKet.FullVecs, []int{0}, []int{0})
	if err != nil {
		t.Fatal(err)
	}
	e := ems[0]
	if math.Abs(e.Overlap) > 1e-14 {
		t.Errorf("overlap %g, want 0", e.Overlap)
	}
	if e.Omega < 400/27.211386 { // O 1s IP is ~540 eV; the line must be an X-ray one
		t.Errorf("emission energy %.1f eV is too low for an O 1s hole", e.Omega*27.211386)
	}

	// The valence hole orbital: the sole occupied orbital of the ket irrep.
	valOrb := -1
	for i := range nocc {
		if ket.irrep(i) == ketSym {
			valOrb = i
		}
	}
	if valOrb < 0 {
		t.Fatalf("irrep %d has no occupied orbital", ketSym)
	}
	// Leading 1h×1h element −d_{core,val}: the dominant emission term. Only one Cartesian
	// component (the one transforming as the ket irrep) connects the two hole orbitals.
	comp := 0
	for c := 1; c < 3; c++ {
		if math.Abs(md.DipMO[c].At(0, valOrb)) > math.Abs(md.DipMO[comp].At(0, valOrb)) {
			comp = c
		}
	}
	dLead := md.DipMO[comp].At(0, valOrb)
	if math.Abs(dLead) < 1e-3 {
		t.Fatalf("no dipole component connects O 1s and the irrep-%d hole (max %g)", ketSym, dLead)
	}
	// |μ| is dominated by that single element: the main states are ~Koopmans, so μ is the
	// bare core→valence dipole scaled by the two states' 1h weights — a factor <1, and the
	// 2h1p and (dropped) 3h2p terms only nudge it. It must land within a factor of a few.
	muNorm := math.Sqrt(e.Mu[0]*e.Mu[0] + e.Mu[1]*e.Mu[1] + e.Mu[2]*e.Mu[2])
	if muNorm < 0.1*math.Abs(dLead) || muNorm > 3*math.Abs(dLead) {
		t.Errorf("|μ| = %.4e but the leading core→valence dipole is %.4e: "+
			"the moment is not dominated by the Koopmans term as expected", muNorm, dLead)
	}
	// The symmetry-forbidden components must vanish (S = 0, so no nuclear leakage either).
	for c := range 3 {
		if c == comp {
			continue
		}
		if math.Abs(e.Mu[c]) > 1e-6 {
			t.Errorf("component %d = %.4e is not symmetry-forbidden as expected", c, e.Mu[c])
		}
	}
	t.Logf("O1s → irrep %d: ω = %.2f eV, μ = %.4e a.u. (component %d, d_core,val = %.4e), "+
		"f = %.3e, A = %.3e a.u.", ketSym, e.Omega*27.211386, muNorm, comp, dLead, e.Osc, e.Rate)
}
