package sip

import (
	"math"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

// ---------------------------------------------------------------------------
// The first-order wavefunction, as determinants.
//
// dyson.go asserts that f⁽¹⁾_{c,a} = −c12_1(a, c)/Δε — a closed form obtained by
// recognizing that the coupling element's spin algebra is the one that contracts the
// MP1 doubles against a 2h1p spin function. Everything below re-derives the same
// quantity from second quantization alone: it builds |Ψ₀⁽¹⁾⟩ as an explicit list of
// doubly-excited determinants with Rayleigh-Schrödinger coefficients, and evaluates
// ⟨Φ_c| c_aβ |Ψ₀⁽¹⁾⟩ by walking operators through occupation strings. It shares no
// line of code with dyson.go, and in particular it knows nothing of c12_1.
//
// Reuses the determinant machinery of isrdipole_test.go (sdet, apply, an/cr, exclusive,
// configDets, phys).
// ---------------------------------------------------------------------------

// less orders spin-orbitals the way sdet.occupied() emits them: every α before every β,
// ascending in orbital index within each spin.
func less(x, y so) bool {
	if x.spin != y.spin {
		return x.spin < y.spin
	}
	return x.p < y.p
}

// eriSO is the antisymmetrized spin-orbital integral <AB||IJ>.
func eriSO(fd *fcidump.Data, A, B, I, J so) float64 {
	var v float64
	if A.spin == I.spin && B.spin == J.spin {
		v += phys(fd, A.p, B.p, I.p, J.p)
	}
	if A.spin == J.spin && B.spin == I.spin {
		v -= phys(fd, A.p, B.p, J.p, I.p)
	}
	return v
}

// mp1 builds |Ψ₀⁽¹⁾⟩ = Σ_{I<J, A<B} C_{IJ}^{AB} |Φ_{IJ}^{AB}⟩ with
// C = <AB||IJ>/(ε_I+ε_J−ε_A−ε_B) and |Φ_{IJ}^{AB}⟩ = c†_A c†_B c_J c_I |0⟩, as a map
// from canonical determinant to amplitude. The phase relating the operator string to
// the canonically ordered determinant comes from apply, not from a counting rule.
func mp1(fd *fcidump.Data, nocc int, eps []float64) map[sdet]float64 {
	norb := fd.NORB
	ref := sdet{a: uint64(1)<<nocc - 1, b: uint64(1)<<nocc - 1}

	var occ, vir []so
	for _, s := range []int{alpha, beta} {
		for p := range nocc {
			occ = append(occ, so{p, s})
		}
		for p := nocc; p < norb; p++ {
			vir = append(vir, so{p, s})
		}
	}
	psi := make(map[sdet]float64)
	for i := range occ {
		for j := i + 1; j < len(occ); j++ {
			I, J := occ[i], occ[j] // I < J in canonical order
			for x := range vir {
				for y := x + 1; y < len(vir); y++ {
					A, B := vir[x], vir[y] // A < B
					num := eriSO(fd, A, B, I, J)
					if num == 0 {
						continue
					}
					den := eps[I.p] + eps[J.p] - eps[A.p] - eps[B.p]
					d, ph := apply(ref, an(I.p, I.spin), an(J.p, J.spin),
						cr(B.p, B.spin), cr(A.p, A.spin))
					if ph == 0 {
						continue
					}
					psi[d] += ph * num / den
				}
			}
		}
	}
	// less() is what makes the I<J / A<B enumeration above canonical; assert it.
	for i := 1; i < len(occ); i++ {
		if !less(occ[i-1], occ[i]) {
			panic("mp1: occupied spin-orbitals are not in canonical order")
		}
	}
	return psi
}

// dysonVirtDet is ⟨Φ_c| c_aβ |Ψ₀⁽¹⁾⟩, negated to match dyson.go's convention that the
// zeroth-order occupied block is +δ.
func dysonVirtDet(sp *Space, psi map[sdet]float64, ci, a int) float64 {
	var acc float64
	for _, t := range configDets(sp, ci) {
		d, ph := apply(t.d, cr(a, beta))
		if ph == 0 {
			continue
		}
		acc += t.c * ph * psi[d]
	}
	return -acc
}

// ---------------------------------------------------------------------------

// TestDysonVirtualMatchesDeterminants is the correctness gate on f⁽¹⁾: every satellite
// configuration against every virtual orbital, closed form vs determinants. It pins the
// spin factors, the denominator, and — crucially — the sign of the virtual block
// relative to the occupied one, which no norm-like quantity can see.
func TestDysonVirtualMatchesDeterminants(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping determinant-level Dyson check in -short mode")
	}
	sp, mx, fd := dipoleSpace(t)
	nocc := mp.NOcc(fd)
	eps := mp.OrbitalEnergies(fd, nocc)
	psi := mp1(fd, nocc, eps)

	var maxErr, maxVal float64
	for ci := sp.BeginSat; ci < len(sp.Configs); ci++ {
		for av := range sp.Nvir {
			a := sp.Nocc + av
			got := mx.dysonVirtCoeff(a, sp.Configs[ci])
			want := dysonVirtDet(sp, psi, ci, a)
			if v := math.Abs(want); v > maxVal {
				maxVal = v
			}
			if e := math.Abs(got - want); e > maxErr {
				maxErr = e
				if e > 1e-10 {
					t.Fatalf("f(1)[config %v, virtual %d] = %g, determinants give %g",
						sp.Configs[ci], a, got, want)
				}
			}
		}
	}
	if maxVal < 1e-3 {
		t.Fatalf("largest determinant f(1) is %g: the test proved nothing", maxVal)
	}
	if maxErr > 1e-10 {
		t.Errorf("max |f(1) − determinant f(1)| = %g", maxErr)
	}
}

// TestDysonVirtualSumRule closes the loop on completeness. c_aβ|Ψ₀⁽¹⁾⟩ is a doublet
// with M_S = +1/2 living entirely in the 2h1p space, so the two spin functions per hole
// pair must span it exactly:
//
//	Σ_c f⁽¹⁾_{c,a} f⁽¹⁾_{c,b} = ⟨c_aβ Ψ₀⁽¹⁾ | c_bβ Ψ₀⁽¹⁾⟩
//
// which is the MP2 correction to the virtual one-particle density. A missing spin
// function, or a wrong normalization on one of them, fails this even where a lucky sign
// lets the per-element test through.
func TestDysonVirtualSumRule(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping Dyson sum rule in -short mode")
	}
	sp, mx, fd := dipoleSpace(t)
	nocc := mp.NOcc(fd)
	eps := mp.OrbitalEnergies(fd, nocc)
	psi := mp1(fd, nocc, eps)

	// annihilate builds c_pβ|Ψ₀⁽¹⁾⟩ as a determinant expansion.
	annihilate := func(p int) map[sdet]float64 {
		out := make(map[sdet]float64)
		for d, c := range psi {
			e, ph := apply(d, an(p, beta))
			if ph != 0 {
				out[e] += ph * c
			}
		}
		return out
	}
	dot := func(u, v map[sdet]float64) float64 {
		var acc float64
		for d, c := range u {
			acc += c * v[d]
		}
		return acc
	}

	var maxErr, maxVal float64
	for av := range sp.Nvir {
		a := sp.Nocc + av
		ca := annihilate(a)
		for bv := range sp.Nvir {
			b := sp.Nocc + bv
			var got float64
			for ci := sp.BeginSat; ci < len(sp.Configs); ci++ {
				got += mx.dysonVirtCoeff(a, sp.Configs[ci]) * mx.dysonVirtCoeff(b, sp.Configs[ci])
			}
			want := dot(ca, annihilate(b))
			if v := math.Abs(want); v > maxVal {
				maxVal = v
			}
			if e := math.Abs(got - want); e > maxErr {
				maxErr = e
			}
		}
	}
	if maxVal < 1e-4 {
		t.Fatalf("largest density element is %g: the test proved nothing", maxVal)
	}
	if maxErr > 1e-10 {
		t.Errorf("Σ_c f f departs from the MP2 virtual density by %g (largest element %g)",
			maxErr, maxVal)
	}
}

// TestDysonOccupiedBlockIsFY: the occupied projection of the Dyson orbital is exactly
// the spectroscopic amplitude analyze.BuildSIPSector already reports, and its norm the
// existing pole strength. The virtual components add to ‖d‖² but not to that.
func TestDysonOccupiedBlockIsFY(t *testing.T) {
	sp, mx, _ := dipoleSpace(t)
	res := lanczos.SolveDense(mx, backend.Gonum{})
	states := []int{0, 1, 2, 7}
	dy, err := mx.DysonOrbitals(res.FullVecs, states)
	if err != nil {
		t.Fatal(err)
	}
	f := mx.FMatrix()
	main := sp.BeginSat
	for si, k := range states {
		y := make([]float64, main)
		for c := range main {
			y[c] = res.FullVecs.At(c, k)
		}
		a := f.MulVec(y)
		d := matColumn(dy, si)

		var ps float64
		for c := range main {
			orb := sp.Configs[c].Occ[0]
			if e := math.Abs(d[orb] - a[c]); e > 1e-14 {
				t.Errorf("state %d orbital %d: Dyson %g, F·Y %g", k, orb, d[orb], a[c])
			}
			ps += a[c] * a[c]
		}
		if e := math.Abs(SpectroscopicFactor(sp, d) - ps); e > 1e-14 {
			t.Errorf("state %d: spectroscopic factor %g, ‖F·Y‖² %g",
				k, SpectroscopicFactor(sp, d), ps)
		}
		var virt float64
		for p := sp.Nocc; p < sp.Norb; p++ {
			virt += d[p] * d[p]
		}
		if k == 0 && virt < 1e-6 {
			t.Errorf("the main line's Dyson orbital has no virtual weight (%g): "+
				"the satellite rows are not being read", virt)
		}
	}
}

// TestDysonVirtualVanishesForPure1h: a bare 1h configuration has no satellite amplitude,
// so its Dyson orbital is pure Koopmans — F·Y on the occupied block, nothing above it.
func TestDysonVirtualVanishesForPure1h(t *testing.T) {
	sp, mx, _ := dipoleSpace(t)
	x := backend.NewMat(sp.Size(), 1)
	x.Set(0, 0, 1)
	dy, err := mx.DysonOrbitals(x, []int{0})
	if err != nil {
		t.Fatal(err)
	}
	for p := sp.Nocc; p < sp.Norb; p++ {
		if v := dy.At(p, 0); v != 0 {
			t.Errorf("virtual %d has amplitude %g with no satellite weight", p, v)
		}
	}
}

// TestDysonRejectsMainVecs: main-block-only eigenvectors would give a Dyson orbital with
// no continuum part at all — silently, and with the pole strength still right. Error out.
func TestDysonRejectsMainVecs(t *testing.T) {
	_, mx, _ := dipoleSpace(t)
	res := lanczos.SolveDense(mx, backend.Gonum{})
	if _, err := mx.DysonOrbitals(res.MainVecs, []int{0}); err == nil {
		t.Error("DysonOrbitals accepted main-block-only eigenvectors")
	}
	if _, err := mx.DysonOrbitals(res.FullVecs, []int{len(res.Values)}); err == nil {
		t.Error("DysonOrbitals accepted an out-of-range state index")
	}
}
