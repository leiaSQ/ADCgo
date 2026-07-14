// Package selfenergy computes the static self-energy Σ(∞) that the ADC main (1h/1h) block
// needs but that the ADC matrix code itself does not build.
//
// theADCcode keeps this in a separate module (`&self-energy`, ../ADC/self_energy) and its
// propagators subtract the result from the main block: ndadc3_ip's build_main_block() ends with
// `main_block->daxpy(-1., *sigma_)`, and the ADC(4) core reads Σ as an external input to adc_().
// ADCgo mirrors that split — sip.Matrix.SetStaticSelfEnergy takes what this package returns —
// so the sign convention here is theADCcode's: the caller does `main -= Σ`.
//
// Omitting Σ is worth ~0.2–0.35 eV on every SIP main line (satellites are untouched: they never
// enter the 1h block), which is why the pyscf cross-check only ever showed a vague band on the
// mains. See docs/adc4_sip_spec.md Finding F1.
//
// Schemes, in increasing completeness, mirroring theADCcode's `&self-energy` orders:
//
//	Three     Σ⁽³⁾   — ρ⁽²⁾ contracted with the integrals (Schirmer 1998 eq. A25)
//	Four      Σ⁽⁴⁾   — ρ⁽²⁾+ρ⁽³⁾ contracted
//	FourPlus  Σ(4+)  — Four, then the ph linear equation (von Niessen 1984 App. B, B.5/B.6)
//	Infinite  Σ(∞)   — FourPlus's linear equation with an all-order resolvent density
//
// Working equations: J. Schirmer, A. B. Trofimov, G. Stelter, J. Chem. Phys. 109, 4734 (1998),
// Appendix A; W. von Niessen, J. Schirmer, L. S. Cederbaum, Comput. Phys. Rep. 1, 57 (1984),
// Appendices B/C.
package selfenergy

import (
	"fmt"

	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
)

// Scheme selects how far the self-energy is carried.
type Scheme int

const (
	Three    Scheme = iota + 3 // Σ⁽³⁾
	Four                       // Σ⁽⁴⁾
	FourPlus                   // Σ(4+)
	Infinite                   // Σ(∞)
)

func (s Scheme) String() string {
	switch s {
	case Three:
		return "three"
	case Four:
		return "four"
	case FourPlus:
		return "fplus"
	case Infinite:
		return "infinite"
	}
	return fmt.Sprintf("Scheme(%d)", int(s))
}

// ParseScheme maps theADCcode's `&self-energy` keyword onto a Scheme.
func ParseScheme(s string) (Scheme, error) {
	switch s {
	case "three":
		return Three, nil
	case "four":
		return Four, nil
	case "fplus", "fourplus":
		return FourPlus, nil
	case "infinite", "inf", "on":
		return Infinite, nil
	}
	return 0, fmt.Errorf("unknown self-energy scheme %q (want three|four|fplus|infinite)", s)
}

// Sigma is a symmetric norb×norb static self-energy over absolute 0-based orbital indices,
// in Hartree. Callers subtract it from the main block.
type Sigma struct {
	n int
	d []float64
}

// At returns Σ_ij.
func (s *Sigma) At(i, j int) float64 { return s.d[i*s.n+j] }

func (s *Sigma) set(i, j int, v float64) { s.d[i*s.n+j] = v }

// Func adapts Σ to the sip.Matrix.SetStaticSelfEnergy signature. Out-of-range indices give 0
// so a caller may pass it orbital pairs outside the correlated space without special-casing.
func (s *Sigma) Func() func(i, j int) float64 {
	return func(i, j int) float64 {
		if i < 0 || j < 0 || i >= s.n || j >= s.n {
			return 0
		}
		return s.d[i*s.n+j]
	}
}

func newSigma(n int) *Sigma { return &Sigma{n: n, d: make([]float64, n*n)} }

// engine carries the space bookkeeping every stage needs: the orbital partition by irrep and
// the 2h1p satellite list. It mirrors Original_self_energy's constructor
// (../ADC/self_energy/original/original_self_energy.cpp).
type engine struct {
	ints *integrals.Store
	eps  []float64
	nocc int
	norb int
	nsym int

	occs [][]int // occupied orbitals by irrep
	virs [][]int // virtual orbitals by irrep
	sats [][]conf
}

// conf is one 2h1p satellite |a k l; type>. type distinguishes the two doublet spin couplings
// when the holes differ; for k == l only type 0 exists.
type conf struct {
	a, k, l int
	typ     int
}

// v is the physicists' integral <ab|cd>, the same accessor sip/elements.go uses, and identical
// to theADCcode's V1212(a,b,c,d) = integral(a,c,b,d).
func (e *engine) v(a, b, c, d int) float64 { return e.ints.Eri(a, c, b, d) }

// so is the 0-based irrep of an absolute orbital; the direct product is XOR.
func (e *engine) so(o int) int { return e.ints.OrbIrrep(o) }

func newEngine(ints *integrals.Store, eps []float64, nocc, norb int) *engine {
	e := &engine{ints: ints, eps: eps, nocc: nocc, norb: norb, nsym: ints.NSym()}
	e.occs = make([][]int, e.nsym)
	e.virs = make([][]int, e.nsym)
	e.sats = make([][]conf, e.nsym)
	for i := range nocc {
		s := e.so(i)
		e.occs[s] = append(e.occs[s], i)
	}
	for i := nocc; i < norb; i++ {
		s := e.so(i)
		e.virs[s] = append(e.virs[s], i)
	}
	// 2h1p configurations, k >= l, grouped by irrep(a)⊗irrep(k)⊗irrep(l).
	for k := range nocc {
		for l := 0; l <= k; l++ {
			for a := nocc; a < norb; a++ {
				s := e.so(a) ^ e.so(k) ^ e.so(l)
				if k != l {
					e.sats[s] = append(e.sats[s], conf{a, k, l, 0}, conf{a, k, l, 1})
				} else {
					e.sats[s] = append(e.sats[s], conf{a, k, l, 0})
				}
			}
		}
	}
	return e
}

// Density returns the ground-state one-particle density matrix ρ over the correlated space —
// the same object Static builds internally before contracting it into Σ, exposed for the other
// consumer of ρ: the ISR representation of a one-particle operator (sip/isrdipole_corr.go),
// whose 1h/1h block carries a ρ-weighted term (theADCcode's ndadc3_prop (13c), which reads its ρ
// from this very module — ND_ADC3_CAP_matrix::build_rho() calls Self_energy::density()).
//
// This is the *correlation part only*: the zeroth-order δ_ij n_i is never added, matching the
// convention density2.go documents. Callers that need the full density add it themselves.
//
// order 2 gives ρ⁽²⁾ — what the legacy property module uses — and order 3 adds ρ⁽³⁾.
func Density(ints *integrals.Store, eps []float64, nocc, norb, order int) (*Sigma, error) {
	if err := checkSpace(eps, nocc, norb); err != nil {
		return nil, err
	}
	if order != 2 && order != 3 {
		return nil, fmt.Errorf("selfenergy: density order %d not available (want 2 or 3)", order)
	}
	e := newEngine(ints, eps, nocc, norb)
	rho := newSigma(norb)
	e.rho2(rho)
	if order == 3 {
		e.rho3(rho, e.dynamicSelfEnergy3())
	}
	return rho, nil
}

// checkSpace validates the orbital space both entry points share.
func checkSpace(eps []float64, nocc, norb int) error {
	if nocc <= 0 || nocc >= norb {
		return fmt.Errorf("selfenergy: bad space (nocc=%d, norb=%d)", nocc, norb)
	}
	if len(eps) < norb {
		return fmt.Errorf("selfenergy: need %d orbital energies, got %d", norb, len(eps))
	}
	return nil
}

// Static computes the static self-energy over the correlated space. eps are the canonical
// orbital energies (mp.OrbitalEnergies); ints must carry the same orbSym the ADC space uses.
func Static(ints *integrals.Store, eps []float64, nocc, norb int, scheme Scheme, opts Options) (*Sigma, error) {
	if err := checkSpace(eps, nocc, norb); err != nil {
		return nil, err
	}
	e := newEngine(ints, eps, nocc, norb)

	// Σ(∞) throws the perturbative density away entirely and rebuilds it from the all-order
	// resolvent, so don't pay for ρ⁽²⁾/ρ⁽³⁾ at all.
	if scheme == Infinite {
		sig := e.rhoToSigma(e.densityAllOrder(opts))
		if err := e.solvePH(sig); err != nil {
			return nil, err
		}
		return sig, nil
	}

	rho := newSigma(norb) // the correlation density; overwritten by the contraction below
	e.rho2(rho)

	if scheme >= Four {
		mak := e.dynamicSelfEnergy3()
		if scheme == Four {
			// Σ⁽⁴⁾ folds the *static* third-order ph self-energy into the ph density
			// alongside the dynamic M⁽³⁾ (static_selfenergy_4: m_ak += sigma3(vir,occ)).
			// Σ(4+) does not — there the same piece is resummed by the linear equation
			// below, and double-counting it would be wrong.
			sig3 := e.rhoToSigma(rho) // rho is still exactly ρ⁽²⁾ here
			for a := nocc; a < norb; a++ {
				for k := range nocc {
					mak[(a-nocc)*nocc+k] += sig3.At(a, k)
				}
			}
		}
		e.rho3(rho, mak)
	}

	sig := e.rhoToSigma(rho)
	if scheme >= FourPlus {
		if err := e.solvePH(sig); err != nil {
			return nil, err
		}
	}
	return sig, nil
}
