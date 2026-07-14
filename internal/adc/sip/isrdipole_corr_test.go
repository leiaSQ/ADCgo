package sip

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
	"github.com/leiaSQ/ADCgo/internal/adc/selfenergy"
)

// The order-by-order gate on the ISR correlation corrections (isrdipole_corr.go).
//
// There is no usable legacy oracle. theADCcode's own ISR-dipole analyzer would need a GUK
// dumpfile carrying dipole integrals, and it is wrong twice over besides — it indexes
// cap_elements by absolute orbital while being handed an irrep-local matrix (sound only in
// C1), and its 1h/2h1p block is negated relative to the secular matrix whose eigenvectors it
// contracts. So the gate here is hermetic, and it is stronger than a bit-for-bit comparison
// would have been: it checks that the operator is right *to the order it claims* and no
// further, which is the only thing a truncated ISR expansion actually promises.
//
// # The method
//
// Scale the fluctuation potential. With the Møller–Plesset partition H = F + V, define
//
//	H(λ) = F + λV
//
// and build everything from H(λ). Every quantity ADCgo computes is a polynomial in λ whose
// coefficients are the orders of perturbation theory: the ADC secular matrix reproduces the
// exact ISR matrix through a known order per block, and so does the ISR property matrix. The
// *residual* against an exact calculation therefore has to vanish at a known rate:
//
//	block      residual, corrected   uncorrected
//	1h/1h      O(λ³)  — 8× a halving  O(λ²) — 4×
//	1h/2h1p    O(λ²)  — 4×            O(λ¹) — 2×
//	2h1p/2h1p  O(λ²)  — 4×            unchanged: it has no corrections
//
// Each corrected block gains exactly one order. Note the 2h1p/2h1p block is already O(λ²)
// without any correction at all, and the *uncorrected* 1h/1h block is O(λ²) rather than
// O(λ¹) — both because a one-particle operator cannot change the particle–hole rank by two,
// which kills the would-be first-order terms. That is why the legacy carries no d22
// correlation terms, and why every d11 correction it does carry ((12a)–(12c), and ρ) is
// second order. TestISRCorrOrders pins the rates; TestISRCorrOrdersFailWithoutCorrections
// shows the uncorrected operator losing exactly one order on the two blocks the corrections
// touch — the check that the gate has teeth.
//
// Getting a term wrong — a sign, a factor, a missing contraction — breaks the rate, not just
// the value, and no amount of internal self-consistency can hide that.
//
// # The exact calculation
//
// In a model small enough for full CI: |Ψ₀(λ)⟩ by dense diagonalization over the N-electron
// determinant space, the correlated states ĉ_J|Ψ₀(λ)⟩ (the same operator strings the config
// basis is built from — configOps), then Schirmer's excitation-class orthogonalization to
// turn them into intermediate states, and finally D_IJ = ⟨Ψ̃_I|D̂|Ψ̃_J⟩ by Slater–Condon.
//
// The oracle itself is gated: TestISROracleReproducesSecularMatrix runs the identical
// construction with Ĥ−E₀ in place of D̂ and recovers the ADC *secular* matrix at its own
// orders. That matrix is already bit-exact against theADCcode, so if the oracle can rebuild
// it, the oracle's determinant algebra, its ground state, and its orthogonalization are all
// sound — and a disagreement on D is a bug in D.

// ---------------------------------------------------------------------------
// A model with a diagonal Fock matrix, by construction.
// ---------------------------------------------------------------------------

// isrModel is a synthetic closed-shell system: chosen orbital energies plus two-electron
// integrals with the full eight-fold permutational symmetry, built from a random Cholesky
// factor so that (pq|rs) = Σ_P L^P_pq L^P_rs is a legitimate integral tensor rather than
// random noise that no Hamiltonian could produce.
//
// The one-electron integrals are then *derived*: h_pq = δ_pq ε_p − (J−K)_pq makes the Fock
// matrix exactly diag(ε) — canonical orbitals, Brillouin's theorem, and an HF determinant
// that really is the reference the ADC formulas assume. Nothing here is fitted or fudged;
// the model is a consistent, if unphysical, closed-shell problem.
type isrModel struct {
	norb, nocc int
	eps        []float64
	eri        []float64 // chemist (pq|rs), norb⁴, eight-fold symmetric
}

func newISRModel(norb, nocc int, seed uint64) *isrModel {
	state := seed
	next := func() float64 {
		state = state*6364136223846793005 + 1442695040888963407
		return float64(int64(state>>11))/float64(int64(1)<<52) - 1
	}

	// A few Cholesky vectors, each a symmetric norb×norb matrix. The resulting integrals are
	// small compared with the orbital-energy gaps, so the perturbation series converges and
	// the λ-expansion means something.
	const nchol = 3
	const scale = 0.15
	l := make([]float64, nchol*norb*norb)
	for p := range nchol {
		for q := range norb {
			for r := 0; r <= q; r++ {
				v := scale * next()
				l[p*norb*norb+q*norb+r] = v
				l[p*norb*norb+r*norb+q] = v
			}
		}
	}
	eri := make([]float64, norb*norb*norb*norb)
	for p := range norb {
		for q := range norb {
			for r := range norb {
				for s := range norb {
					var acc float64
					for c := range nchol {
						acc += l[c*norb*norb+p*norb+q] * l[c*norb*norb+r*norb+s]
					}
					eri[((p*norb+q)*norb+r)*norb+s] = acc
				}
			}
		}
	}

	// Well-separated orbital energies: occupied below zero, virtual above, no degeneracies.
	eps := make([]float64, norb)
	for p := range norb {
		eps[p] = -1.2 + 0.9*float64(p) + 0.05*next()
	}
	return &isrModel{norb: norb, nocc: nocc, eps: eps, eri: eri}
}

func (m *isrModel) at(p, q, r, s int) float64 {
	n := m.norb
	return m.eri[((p*n+q)*n+r)*n+s]
}

// jk is the closed-shell mean field Σ_i [2(pq|ii) − (pi|iq)].
func (m *isrModel) jk(p, q int) float64 {
	var acc float64
	for i := range m.nocc {
		acc += 2*m.at(p, q, i, i) - m.at(p, i, i, q)
	}
	return acc
}

// fcidumpAt writes the model at fluctuation-potential strength λ and reads it back through
// the real parser. Scaling V by λ scales the two-electron integrals by λ and moves the
// one-electron ones to h(λ) = diag(ε) − λ(J−K), which leaves the Fock matrix — and hence the
// orbital energies, the reference determinant and Brillouin's theorem — untouched. That is
// the whole point: only the correlation is being turned down.
func (m *isrModel) fcidumpAt(t *testing.T, lambda float64) *fcidump.Data {
	t.Helper()
	var b strings.Builder
	fmt.Fprintf(&b, " &FCI NORB= %d,NELEC= %d,MS2= 0,\n", m.norb, 2*m.nocc)
	b.WriteString("  ORBSYM=")
	for range m.norb {
		b.WriteString("1,")
	}
	b.WriteString("\n  ISYM=1,\n &END\n")

	for p := range m.norb {
		for q := 0; q <= p; q++ {
			for r := 0; r <= p; r++ {
				for s := 0; s <= r; s++ {
					if p*(p+1)/2+q < r*(r+1)/2+s {
						continue
					}
					if v := lambda * m.at(p, q, r, s); v != 0 {
						fmt.Fprintf(&b, "%23.16e %3d %3d %3d %3d\n", v, p+1, q+1, r+1, s+1)
					}
				}
			}
		}
	}
	for p := range m.norb {
		for q := 0; q <= p; q++ {
			h := -lambda * m.jk(p, q)
			if p == q {
				h += m.eps[p]
			}
			fmt.Fprintf(&b, "%23.16e %3d %3d %3d %3d\n", h, p+1, q+1, 0, 0)
		}
	}
	fmt.Fprintf(&b, "%23.16e %3d %3d %3d %3d\n", 0.0, 0, 0, 0, 0)

	path := filepath.Join(t.TempDir(), "model.fcidump")
	if err := os.WriteFile(path, []byte(b.String()), 0o600); err != nil {
		t.Fatal(err)
	}
	d, err := fcidump.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	// The construction is only sound if the Fock matrix really did come out diagonal.
	eps := mp.OrbitalEnergies(d, m.nocc)
	for p := range m.norb {
		if math.Abs(eps[p]-m.eps[p]) > 1e-12 {
			t.Fatalf("λ=%g: orbital energy %d drifted: %.15f, want %.15f — the model's h(λ) "+
				"is not keeping the Fock matrix fixed", lambda, p, eps[p], m.eps[p])
		}
	}
	return d
}

// ---------------------------------------------------------------------------
// Full CI in the determinant basis.
// ---------------------------------------------------------------------------

// detList enumerates every determinant with na α and nb β electrons in norb orbitals.
func detList(norb, na, nb int) []sdet {
	var masks func(n, k int) []uint64
	masks = func(n, k int) []uint64 {
		if k == 0 {
			return []uint64{0}
		}
		var out []uint64
		for hi := k - 1; hi < n; hi++ {
			for _, rest := range masks(hi, k-1) {
				out = append(out, rest|uint64(1)<<hi)
			}
		}
		return out
	}
	var out []sdet
	for _, a := range masks(norb, na) {
		for _, b := range masks(norb, nb) {
			out = append(out, sdet{a: a, b: b})
		}
	}
	return out
}

// denseFrom builds a matrix over a determinant list from a two-determinant kernel.
func denseFrom(dets []sdet, me func(x, y sdet) float64) backend.Mat {
	n := len(dets)
	m := backend.NewMat(n, n)
	for i := range n {
		for j := range n {
			m.Set(i, j, me(dets[i], dets[j]))
		}
	}
	return m
}

// groundState is the lowest eigenpair of H over the N-electron determinant space.
func groundState(t *testing.T, dets []sdet, fd *fcidump.Data) ([]float64, float64) {
	t.Helper()
	h := denseFrom(dets, func(x, y sdet) float64 { return hamDet(x, y, fd) })
	vals, vecs := backend.Gonum{}.SymEig(h)
	psi := make([]float64, len(dets))
	for i := range dets {
		psi[i] = vecs.At(i, 0)
	}
	return psi, vals[0]
}

// ---------------------------------------------------------------------------
// The intermediate-state representation, built by hand.
// ---------------------------------------------------------------------------

// correlatedStates applies each configuration's operator string to the correlated ground
// state: |Ψ_J⟩ = ĉ_J|Ψ₀⟩, expressed over the (N−1)-electron determinant basis. These are the
// ISR's *precursor* states — not yet orthonormal, and that non-orthogonality is exactly the
// correlation the corrections describe.
func correlatedStates(sp *Space, psi0 []float64, dets0, dets1 []sdet) backend.Mat {
	index := make(map[sdet]int, len(dets1))
	for i, d := range dets1 {
		index[d] = i
	}
	y := backend.NewMat(len(dets1), sp.Size())
	for j := range sp.Size() {
		for _, term := range configOps(sp, j) {
			for c0, d0 := range dets0 {
				if psi0[c0] == 0 {
					continue
				}
				d, sign := apply(d0, term.ops...)
				if sign == 0 {
					continue
				}
				if row, ok := index[d]; ok {
					y.Set(row, j, y.At(row, j)+term.c*sign*psi0[c0])
				}
			}
		}
	}
	return y
}

// invSqrt returns S^{-1/2} for a symmetric positive-definite S.
func invSqrt(t *testing.T, s backend.Mat) backend.Mat {
	t.Helper()
	vals, vecs := backend.Gonum{}.SymEig(s)
	n := s.Rows
	out := backend.NewMat(n, n)
	for i := range n {
		for j := range n {
			var acc float64
			for k := range n {
				if vals[k] <= 1e-12 {
					t.Fatalf("overlap matrix is singular (eigenvalue %.3e): the precursor states "+
						"are linearly dependent", vals[k])
				}
				acc += vecs.At(i, k) * vecs.At(j, k) / math.Sqrt(vals[k])
			}
			out.Set(i, j, acc)
		}
	}
	return out
}

// gram is Aᵀ·B.
func gram(a, b backend.Mat) backend.Mat {
	out := backend.NewMat(a.Cols, b.Cols)
	for i := range a.Cols {
		for j := range b.Cols {
			var acc float64
			for k := range a.Rows {
				acc += a.At(k, i) * b.At(k, j)
			}
			out.Set(i, j, acc)
		}
	}
	return out
}

// mul is A·B.
func mul(a, b backend.Mat) backend.Mat {
	out := backend.NewMat(a.Rows, b.Cols)
	for i := range a.Rows {
		for j := range b.Cols {
			var acc float64
			for k := range a.Cols {
				acc += a.At(i, k) * b.At(k, j)
			}
			out.Set(i, j, acc)
		}
	}
	return out
}

// cols extracts columns [lo, hi) of m.
func cols(m backend.Mat, lo, hi int) backend.Mat {
	out := backend.NewMat(m.Rows, hi-lo)
	for i := range m.Rows {
		for j := lo; j < hi; j++ {
			out.Set(i, j-lo, m.At(i, j))
		}
	}
	return out
}

// eco performs Schirmer's excitation-class orthogonalization on the precursor states: the 1h
// class is symmetrically (Löwdin) orthonormalized among itself; the 2h1p class is then
// Gram–Schmidt projected orthogonal to it and symmetrically orthonormalized in turn. The
// class ordering is what makes the ISR well defined — a plain Löwdin over all of them at once
// would mix the 1h and 2h1p spaces and give a different, wrong operator.
func eco(t *testing.T, y backend.Mat, nMain int) backend.Mat {
	t.Helper()
	y1 := cols(y, 0, nMain)
	x1 := mul(y1, invSqrt(t, gram(y1, y1)))

	y2 := cols(y, nMain, y.Cols)
	proj := mul(x1, gram(x1, y2)) // the part of the 2h1p precursors already spanned by 1h
	for i := range y2.Rows {
		for j := range y2.Cols {
			y2.Set(i, j, y2.At(i, j)-proj.At(i, j))
		}
	}
	x2 := mul(y2, invSqrt(t, gram(y2, y2)))

	out := backend.NewMat(y.Rows, y.Cols)
	for i := range y.Rows {
		for j := range nMain {
			out.Set(i, j, x1.At(i, j))
		}
		for j := range x2.Cols {
			out.Set(i, nMain+j, x2.At(i, j))
		}
	}
	return out
}

// isrMatrix is the ISR representation ⟨Ψ̃_I|Ô|Ψ̃_J⟩ of the one-determinant kernel me.
func isrMatrix(x backend.Mat, dets1 []sdet, me func(a, b sdet) float64) backend.Mat {
	return gram(x, mul(denseFrom(dets1, me), x))
}

// ---------------------------------------------------------------------------
// The gate.
// ---------------------------------------------------------------------------

// isrOracle builds the exact ISR matrices of the model at fluctuation strength λ: the
// property matrix for the operator dmo, and the secular matrix Ĥ−E₀.
func isrOracle(t *testing.T, m *isrModel, sp *Space, lambda float64, dmo backend.Mat) (dOracle, mOracle backend.Mat) {
	t.Helper()
	fd := m.fcidumpAt(t, lambda)

	dets0 := detList(m.norb, m.nocc, m.nocc)
	dets1 := detList(m.norb, m.nocc, m.nocc-1) // one β electron removed: the SIP sector
	psi0, e0 := groundState(t, dets0, fd)

	x := eco(t, correlatedStates(sp, psi0, dets0, dets1), sp.MainBlockSize())

	dOracle = isrMatrix(x, dets1, func(a, b sdet) float64 { return oneBodyDet(a, b, dmo) })
	mOracle = isrMatrix(x, dets1, func(a, b sdet) float64 {
		v := hamDet(a, b, fd)
		if a == b {
			v -= e0
		}
		return v
	})
	return dOracle, mOracle
}

// blockResiduals returns the largest |oracle − adcgo| over each of the three blocks.
func blockResiduals(oracle, got backend.Mat, nMain int) (d11, d12, d22 float64) {
	upd := func(dst *float64, v float64) {
		if v > *dst {
			*dst = v
		}
	}
	for i := range oracle.Rows {
		for j := range oracle.Cols {
			r := math.Abs(oracle.At(i, j) - got.At(i, j))
			switch {
			case i < nMain && j < nMain:
				upd(&d11, r)
			case i >= nMain && j >= nMain:
				upd(&d22, r)
			default:
				upd(&d12, r)
			}
		}
	}
	return d11, d12, d22
}

// adcgoDipole builds ADCgo's ISR property matrix for the model at strength λ.
func adcgoDipole(t *testing.T, m *isrModel, sp *Space, lambda float64, dmo backend.Mat, corrected bool) backend.Mat {
	t.Helper()
	fd := m.fcidumpAt(t, lambda)
	ints := integrals.New(fd, m.nocc, nil)
	eps := mp.OrbitalEnergies(fd, m.nocc)

	if !corrected {
		op, err := NewISRDipole(sp, dmo)
		if err != nil {
			t.Fatal(err)
		}
		return op.BuildMatrix()
	}
	rho, err := selfenergy.Density(ints, eps, m.nocc, m.norb, 2)
	if err != nil {
		t.Fatal(err)
	}
	ops, err := NewISRDipolesWithCorr(sp, [3]backend.Mat{dmo, dmo, dmo},
		&ISROptions{Ints: ints, Eps: eps, Rho: rho.Func()})
	if err != nil {
		t.Fatal(err)
	}
	return ops[0].BuildMatrix()
}

// modelSpace is the shared fixture: 4 electrons in 5 orbitals, C1. The N-electron
// determinant space is 100 and the SIP one 50, both trivially diagonalizable, while the SIP
// configuration space (2 × 1h, 12 × 2h1p) still exercises every branch of every block.
func modelSpace(t *testing.T) (*isrModel, *Space, backend.Mat) {
	t.Helper()
	m := newISRModel(5, 2, 0xC0FFEE123)
	return m, NewSpace(m.nocc, m.norb, nil, 0), randomSymmetric(m.norb)
}

// TestISROracleReproducesSecularMatrix gates the oracle itself, before it is trusted to
// judge anything. The identical construction — full-CI ground state, precursor states,
// excitation-class orthogonalization — applied to Ĥ−E₀ must reproduce the ADC *secular*
// matrix at that matrix's own orders. ADC(3) is correct to O(λ³) on the 1h/1h block, O(λ²)
// on the coupling and O(λ¹) on 2h1p/2h1p, so the residuals must fall as λ⁴, λ³ and λ².
//
// The secular matrix is bit-exact against theADCcode. If the oracle can rebuild it, the
// oracle is sound.
func TestISROracleReproducesSecularMatrix(t *testing.T) {
	m, sp, _ := modelSpace(t)
	dmo := randomSymmetric(m.norb)

	type row struct{ d11, d12, d22 float64 }
	got := map[float64]row{}
	lambdas := []float64{0.08, 0.04, 0.02}
	for _, lam := range lambdas {
		_, oracle := isrOracle(t, m, sp, lam, dmo)

		fd := m.fcidumpAt(t, lam)
		ints := integrals.New(fd, m.nocc, nil)
		eps := mp.OrbitalEnergies(fd, m.nocc)
		mx := New(sp, ints, eps, 3, backend.Gonum{})
		defer mx.Release()

		// The 1h/1h block is only third-order complete once the *static self-energy* is
		// folded in — ADCgo, like theADCcode, keeps Σ in a separate module and subtracts it
		// from the main block rather than building it into the element formulas
		// (selfenergy/selfenergy.go). Σ⁽³⁾ is exactly the O(λ³) piece, and without it this
		// gate sees the 1h/1h residual fall as λ³ instead of λ⁴ — which is how it was found.
		sig, err := selfenergy.Static(ints, eps, m.nocc, m.norb, selfenergy.Three, selfenergy.Options{})
		if err != nil {
			t.Fatal(err)
		}
		mx.SetStaticSelfEnergy(sig.Func())

		a, b, c := blockResiduals(oracle, mx.BuildMatrix(), sp.MainBlockSize())
		got[lam] = row{a, b, c}
		t.Logf("λ=%-5g secular residuals: 1h/1h %.3e  1h/2h1p %.3e  2h1p/2h1p %.3e", lam, a, b, c)
	}
	// ADC(3): 1h/1h through 3rd order, coupling through 2nd, 2h1p/2h1p through 1st.
	checkOrder(t, "secular 1h/1h", lambdas, got, func(r row) float64 { return r.d11 }, 4)
	checkOrder(t, "secular 1h/2h1p", lambdas, got, func(r row) float64 { return r.d12 }, 3)
	checkOrder(t, "secular 2h1p/2h1p", lambdas, got, func(r row) float64 { return r.d22 }, 2)
}

// TestISRCorrOrders is the gate on the corrections themselves.
//
// The orders below are not guesses fitted to the output; they follow from the terms:
//
//   - 1h/1h. Every correction the legacy carries is *second* order — (12a), (12b) and (12c)
//     are each a product of two fluctuation integrals, and (13c)/(13a) read ρ, which starts at
//     ρ⁽²⁾. There is no first-order term to carry, and there could not be: the O(V) term of
//     ⟨c_iΨ₀|D̂|c_jΨ₀⟩ is ⟨1h|D̂|3h2p⟩, and a one-particle operator cannot change the
//     particle–hole rank by two, so it vanishes identically. Corrected: O(λ³).
//   - 1h/2h1p. (8a)–(8c) are first order — one integral over one denominator. O(λ²).
//   - 2h1p/2h1p. my_calc_d22_{diag,off}.c contain no integrals and no denominators at all: the
//     block is exact at zeroth order and gets *no* correction. That is only consistent if its
//     first-order term vanishes too, and it does — so the residual is already O(λ²), one order
//     better than the naive count, and isrdipole.go's satsat needs nothing. This test is what
//     turns that reading of the legacy into a checked fact.
func TestISRCorrOrders(t *testing.T) {
	m, sp, dmo := modelSpace(t)

	type row struct{ d11, d12, d22 float64 }
	got := map[float64]row{}
	lambdas := []float64{0.08, 0.04, 0.02}
	for _, lam := range lambdas {
		oracle, _ := isrOracle(t, m, sp, lam, dmo)
		a, b, c := blockResiduals(oracle, adcgoDipole(t, m, sp, lam, dmo, true), sp.MainBlockSize())
		got[lam] = row{a, b, c}
		t.Logf("λ=%-5g ISR-D residuals: 1h/1h %.3e  1h/2h1p %.3e  2h1p/2h1p %.3e", lam, a, b, c)
	}
	checkOrder(t, "D 1h/1h", lambdas, got, func(r row) float64 { return r.d11 }, 3)
	checkOrder(t, "D 1h/2h1p", lambdas, got, func(r row) float64 { return r.d12 }, 2)
	checkOrder(t, "D 2h1p/2h1p", lambdas, got, func(r row) float64 { return r.d22 }, 2)
}

// TestISRCorrOrdersFailWithoutCorrections gives the gate above its teeth. Strip the
// corrections and both blocks they touch must lose exactly one order — 1h/1h from O(λ³) to
// O(λ²), and the coupling from O(λ²) to O(λ¹). If the uncorrected operator converged just as
// fast, TestISRCorrOrders would be measuring nothing.
func TestISRCorrOrdersFailWithoutCorrections(t *testing.T) {
	m, sp, dmo := modelSpace(t)

	type row struct{ d11, d12, d22 float64 }
	got := map[float64]row{}
	lambdas := []float64{0.08, 0.04, 0.02}
	for _, lam := range lambdas {
		oracle, _ := isrOracle(t, m, sp, lam, dmo)
		a, b, c := blockResiduals(oracle, adcgoDipole(t, m, sp, lam, dmo, false), sp.MainBlockSize())
		got[lam] = row{a, b, c}
		t.Logf("λ=%-5g uncorrected: 1h/1h %.3e  1h/2h1p %.3e", lam, a, b)
	}
	checkOrder(t, "uncorrected D 1h/1h", lambdas, got, func(r row) float64 { return r.d11 }, 2)
	checkOrder(t, "uncorrected D 1h/2h1p", lambdas, got, func(r row) float64 { return r.d12 }, 1)
	// The block with no corrections must be untouched by their absence.
	checkOrder(t, "uncorrected D 2h1p/2h1p", lambdas, got, func(r row) float64 { return r.d22 }, 2)
}

// checkOrder asserts that a residual falls off as λ^want: halving λ must shrink it by 2^want.
func checkOrder[R any](t *testing.T, name string, lambdas []float64, got map[float64]R,
	pick func(R) float64, want int) {
	t.Helper()
	for i := 1; i < len(lambdas); i++ {
		hi, lo := pick(got[lambdas[i-1]]), pick(got[lambdas[i]])
		if lo < 1e-13 {
			continue // already at machine precision: nothing left to measure
		}
		ratio := hi / lo
		ideal := math.Pow(2, float64(want))
		// A generous band: the next order in λ contaminates the ratio, and the point is to
		// tell λ¹ from λ² from λ³, not to measure an exponent to three digits.
		if ratio < 0.65*ideal || ratio > 1.6*ideal {
			t.Errorf("%s: residual fell by %.2f× when λ was halved (%.3e → %.3e); O(λ^%d) demands ≈%.0f×",
				name, ratio, hi, lo, want, ideal)
		}
	}
}

// ---------------------------------------------------------------------------
// The symmetry-blocked operator.
// ---------------------------------------------------------------------------

// cfgKey identifies a configuration independently of which space holds it, and of the role
// order that space happens to store its two holes in.
type cfgKey struct{ lo, hi, vir, typ int }

func keyOf(sp *Space, idx int) (cfgKey, float64) {
	c := sp.Configs[idx]
	if idx < sp.BeginSat {
		return cfgKey{c.Occ[0], -1, -1, -1}, 1
	}
	cc, s := canonical(sp, c)
	return cfgKey{cc.Occ[0], cc.Occ[1], cc.Vir, cc.Typ}, s
}

// TestCorrSymmetryBlocksMatchFullOperator closes the chain from the λ-oracle to the operator
// a symmetric molecule actually uses.
//
// The oracle above runs in C1: it validates the *square* corrected operator over one
// unsymmetrized space, and nothing else. But isrdipole_corr.go's central claim is that its
// summation ranges need no irrep gating — that running every loop over the full occupied and
// virtual ranges and letting the integrals vanish where symmetry forbids them is not only
// simpler but is the only form that survives the rectangular, cross-sector case, where the
// legacy's irrep windows all shift by bra.Sym ⊗ ket.Sym.
//
// This test checks that claim directly. The symmetry-blocked spaces partition exactly the
// configurations of the unsymmetrized space, so every corrected element — within a sector and
// across two — must equal the corresponding element of the one big operator the oracle already
// certified. Get an irrep window wrong and the cross-sector blocks disagree here.
func TestCorrSymmetryBlocksMatchFullOperator(t *testing.T) {
	fd := h2oData(t)
	nocc := mp.NOcc(fd)
	eps := mp.OrbitalEnergies(fd, nocc)
	dmo := randomSymmetric(fd.NORB)

	build := func(bra, ket *Space, orbSym []int) backend.Mat {
		ints := integrals.New(fd, nocc, orbSym)
		rho, err := selfenergy.Density(ints, eps, nocc, fd.NORB, 2)
		if err != nil {
			t.Fatal(err)
		}
		ops, err := NewISRDipolesCrossWithCorr(bra, ket, [3]backend.Mat{dmo, dmo, dmo},
			&ISROptions{Ints: ints, Eps: eps, Rho: rho.Func()})
		if err != nil {
			t.Fatal(err)
		}
		return ops[0].BuildMatrix()
	}

	full := NewSpace(nocc, fd.NORB, nil, 0)
	dFull := build(full, full, nil)
	index := make(map[cfgKey]int, full.Size())
	signs := make(map[cfgKey]float64, full.Size())
	for i := range full.Size() {
		k, s := keyOf(full, i)
		index[k], signs[k] = i, s
	}

	nsym := 1
	for o := range fd.NORB {
		for nsym < fd.OrbSym[o] {
			nsym <<= 1
		}
	}
	var secs []*Space
	for g := range nsym {
		if sp := NewSpace(nocc, fd.NORB, fd.OrbSym, g); sp.MainBlockSize() > 0 {
			secs = append(secs, sp)
		}
	}
	if len(secs) < 2 {
		t.Fatal("need at least two irrep sectors to exercise the cross-sector operator")
	}

	var checked, crossChecked int
	for _, bra := range secs {
		for _, ket := range secs {
			d := build(bra, ket, fd.OrbSym)
			for i := range bra.Size() {
				ki, si := keyOf(bra, i)
				for j := range ket.Size() {
					kj, sj := keyOf(ket, j)
					// Both spaces may store a hole pair in either role order; the canonical
					// sign of each side converts to the shared convention.
					want := dFull.At(index[ki], index[kj]) * si * sj * signs[ki] * signs[kj]
					if got := d.At(i, j); math.Abs(got-want) > 1e-12 {
						t.Fatalf("irrep %d×%d element (%d,%d): blocked %.15g, unsymmetrized %.15g",
							bra.Sym, ket.Sym, i, j, got, want)
					}
					checked++
					if bra.Sym != ket.Sym {
						crossChecked++
					}
				}
			}
		}
	}
	if crossChecked == 0 {
		t.Fatal("no cross-sector elements were compared")
	}
	t.Logf("%d elements agree (%d of them cross-sector)", checked, crossChecked)
}

// TestCorrDipoleSymmetric: D represents a symmetric operator, so it must come out symmetric —
// the correction densities are built per (i,j) with i and j in asymmetric roles, and only the
// underlying physics makes the contraction come out the same both ways.
func TestCorrDipoleSymmetric(t *testing.T) {
	m, sp, dmo := modelSpace(t)
	d := adcgoDipole(t, m, sp, 1.0, dmo, true)
	for i := range d.Rows {
		for j := range i {
			if math.Abs(d.At(i, j)-d.At(j, i)) > 1e-12 {
				t.Fatalf("D is asymmetric at (%d,%d): %.15g vs %.15g", i, j, d.At(i, j), d.At(j, i))
			}
		}
	}
}
