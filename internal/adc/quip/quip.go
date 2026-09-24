// Package quip is the ADC(2,2) secular matrix for quadruple ionization, ADC(2,2)-QUIP,
// over the 4h | 5h1p | 6h2p configurations of a khci.Space, and the rotated
// representation that carries it into localized orbitals (where net-charge partitions are
// defined).
//
// # Assembly
//
// ADC(2,2) (Kolorenc & Averbukh, J. Chem. Phys. 152, 214107 (2020), Table I) fixes a
// maximum perturbation order per block; for quadruple ionization the blocks come from
// two sources, joined in the khci row basis (whose phase is adcgen's, so they agree sign
// for sign):
//
//	block            order (m / x / f)   source
//	4h   / 4h        2                   generated, isrgen/qip B00
//	4h   / 5h1p      1 / 1 / 2           generated, isrgen/qip B01
//	5h1p / 5h1p      2                   generated, isrgen/qip B11
//	4h   / 6h2p      absent              zero: the ISR block starts at second order
//	5h1p / 6h2p      1                   khci.CI (first-order ISR = CI, khci tests)
//	6h2p / 6h2p      0 / 1 / 1           m: orbital-energy differences; x, f: khci.CI
//
// The 4h/6h2p block is where ISR and CI part: CI couples the two classes at first order
// (a double excitation), the ISR does not, so taking the CI block there
// would be neither ADC nor consistent.
//
// Everything is in CANONICAL Hartree-Fock orbitals, which the generated expressions
// assume; New refuses a dump whose Fock matrix is not diagonal.
//
// # Rotated representation
//
// A net-charge partition and its decaying state are defined in localized occupied and
// compact/free virtual orbitals. Under orbital rotations U (occupied) and W
// (virtual), a localized row c+_a1 c+_a2 c_i1 ... c_in |Phi_0> expands over canonical rows
// with coefficient det(W[P, A]) det(U[H, I]) (the minors of the spin-orbital rotations on
// the particle and hole sets, both ascending, which is the khci phase convention), so
// M_loc = R^T M_can R with R block diagonal by class. It is exact whenever the row set is
// closed under the rotation: every class complete in its Ms sector. A free-particle limit
// (khci.Options.LimitFree) is defined in the localized virtuals and is not closed under
// W, so Rotation refuses such a space.
package quip

import (
	"fmt"
	"math"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/isrgen/qip"
	"github.com/leiaSQ/ADCgo/internal/adc/isrgen/sigma"
	"github.com/leiaSQ/ADCgo/internal/adc/khci"
	"github.com/leiaSQ/ADCgo/internal/adc/mo"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
	"github.com/leiaSQ/ADCgo/internal/adc/parallel"
	"gonum.org/v1/gonum/mat"
)

// hybrid is the order of the blocks the generated package does not carry: 4h/6h2p,
// 5h1p/6h2p and 6h2p/6h2p (-1 absent, 0 zeroth order, 1 CI), per scheme.
var hybrid = map[string][3]int{
	"adc2x":  {-1, -1, -1}, // no 6h2p class
	"adc22m": {-1, 1, 0},
	"adc22x": {-1, 1, 1},
	"adc22f": {-1, 1, 1},
}

// Schemes lists the ADC schemes Op can assemble, with the classes each needs.
func Schemes() map[string]int {
	out := map[string]int{}
	for s := range hybrid {
		if s == "adc2x" {
			out[s] = qip.K + 1
		} else {
			out[s] = qip.K + 2
		}
	}
	return out
}

// Op is ADC(2,2)-QUIP (or ADC(2)x) on the rows of a K = 4 khci.Space, canonical orbitals.
type Op struct {
	sp     *khci.Space
	ci     *khci.CI
	el     *qip.Elem
	eps    []float64 // spatial orbital energies
	scheme string
	orders [3]int
	nso    int
}

// New builds the operator. The space must have K = 4 and reach the classes the scheme
// needs (5h1p for adc2x, 6h2p for adc22*), and the dump must be canonical.
func New(sp *khci.Space, d *fcidump.Data, scheme string) (*Op, error) {
	o := sp.Options()
	if o.K != qip.K {
		return nil, fmt.Errorf("quip: the space has K = %d, ADC(2,2)-QUIP needs %d", o.K, qip.K)
	}
	h, ok := hybrid[scheme]
	if !ok {
		return nil, fmt.Errorf("quip: unknown scheme %q (have adc2x, adc22m, adc22x, adc22f)", scheme)
	}
	if need := Schemes()[scheme]; o.MaxClass != need {
		return nil, fmt.Errorf("quip: scheme %s is defined on classes %d..%d, the space reaches %d",
			scheme, o.K, need, o.MaxClass)
	}
	nocc := mp.NOcc(d)
	if nocc != o.NOcc {
		return nil, fmt.Errorf("quip: dump has %d occupied orbitals, the space %d", nocc, o.NOcc)
	}
	if err := mp.RequireCanonical(d, nocc); err != nil {
		return nil, fmt.Errorf("quip: the generated ISR blocks assume canonical orbitals: %w", err)
	}
	eps := mp.OrbitalEnergies(d, nocc)
	el, err := qip.New(integrals.New(d, nocc, nil), eps, nocc, scheme)
	if err != nil {
		return nil, fmt.Errorf("quip: %w (regenerate isrgen/qip with this scheme)", err)
	}
	ci, err := khci.NewCI(sp, d, backend.Gonum{})
	if err != nil {
		return nil, err
	}
	return &Op{sp: sp, ci: ci, el: el, eps: eps, scheme: scheme, orders: h, nso: 2 * nocc}, nil
}

// sigmaOrders is each scheme's per-block maximum order (B00 B01 B11 B02 B12 B22) on the
// σ-build: the qip ISR orders of B00, B01 and B11, B02 absent, and the hybrid's B12/B22 —
// through first order plain CI (the σ program derives them so, generate_adc.py
// --ci-blocks), or 0 for the orbital-energy diagonal of the m variant.
var sigmaOrders = map[string][6]int{
	"adc2x":  {2, 1, 1, -1, -1, -1},
	"adc22m": {2, 1, 2, -1, 1, 0},
	"adc22x": {2, 1, 2, -1, 1, 1},
	"adc22f": {2, 2, 2, -1, 1, 1},
}

// NewSigma is the matrix-free twin of New: the same operator as tensor contractions on
// the generated qip σ program, on be (host, a TensorKernels device, or a row-partitioned
// multi-device backend). A scheme whose blocks the σ program was not derived to (the
// adc22 schemes need the order-2 5h1p/5h1p block) is an error, not a truncation.
func NewSigma(sp *khci.Space, d *fcidump.Data, scheme string, be backend.Backend) (*sigma.Operator, error) {
	o := sp.Options()
	if o.K != qip.K {
		return nil, fmt.Errorf("quip: the space has K = %d, ADC(2,2)-QUIP needs %d", o.K, qip.K)
	}
	orders, ok := sigmaOrders[scheme]
	if !ok {
		return nil, fmt.Errorf("quip: unknown scheme %q (have adc2x, adc22m, adc22x, adc22f)", scheme)
	}
	if need := Schemes()[scheme]; o.MaxClass != need {
		return nil, fmt.Errorf("quip: scheme %s is defined on classes %d..%d, the space reaches %d",
			scheme, o.K, need, o.MaxClass)
	}
	nocc := mp.NOcc(d)
	if nocc != o.NOcc {
		return nil, fmt.Errorf("quip: dump has %d occupied orbitals, the space %d", nocc, o.NOcc)
	}
	if err := mp.RequireCanonical(d, nocc); err != nil {
		return nil, fmt.Errorf("quip: the generated ISR blocks assume canonical orbitals: %w", err)
	}
	op, err := qip.NewSigmaOrders(sp, integrals.New(d, nocc, nil), mp.OrbitalEnergies(d, nocc), nocc, orders, be)
	if err != nil {
		return nil, fmt.Errorf("quip: %w (regenerate isrgen/qip --sigma for this scheme)", err)
	}
	return op, nil
}

// Space is the row space.
func (op *Op) Space() *khci.Space { return op.sp }

// Size is the number of rows.
func (op *Op) Size() int { return op.sp.Size() }

// Element is <r|M|c>.
func (op *Op) Element(r, c int) float64 {
	cr, cc := op.sp.Class(r)-qip.K, op.sp.Class(c)-qip.K
	if cr > cc {
		r, c, cr, cc = c, r, cc, cr
	}
	switch {
	case cc <= 1:
		return op.el.Element(op.sp, r, c) // B00, B01, B11
	case cr == 0: // B02
		if op.orders[0] < 0 {
			return 0
		}
		return op.ci.Element(r, c)
	case cr == 1: // B12
		if op.orders[1] < 0 {
			return 0
		}
		return op.ci.Element(r, c)
	}
	// B22
	switch op.orders[2] {
	case 1:
		return op.ci.Element(r, c)
	case 0:
		if r != c {
			return 0
		}
		return op.zerothDiagonal(r)
	}
	return 0
}

// zerothDiagonal is the zeroth-order (k+2)h2p diagonal: sum of particle minus sum of hole
// orbital energies.
func (op *Op) zerothDiagonal(r int) float64 {
	var v float64
	for _, i := range op.sp.Holes(r, nil) {
		v -= op.eps[i]
	}
	for _, a := range op.sp.Particles(r, nil) {
		v += op.eps[op.sp.Options().NOcc+a]
	}
	return v
}

// BuildMatrix assembles M densely, rows in parallel.
func (op *Op) BuildMatrix() backend.Mat {
	n := op.Size()
	m := backend.NewMat(n, n)
	parallel.HeavyRows(n, func(r int) {
		for c := r; c < n; c++ {
			m.Data[r*n+c] = op.Element(r, c)
		}
	})
	for r := range n {
		for c := range r {
			m.Data[r*n+c] = m.Data[c*n+r]
		}
	}
	return m
}

// ---------------------------------------------------------------------------
// Rotated representation
// ---------------------------------------------------------------------------

// Rotations returns the occupied and virtual orbital rotations U (nocc x nocc) and W
// (nvir x nvir) from canonical to localized MOs, U_pi = C_can[:,p]^T S C_loc[:,i], from
// the two sidecars of the same SCF. It fails when the localized orbitals leave the
// occupied (or virtual) space of the canonical ones by more than tol.
func Rotations(can, loc *mo.Data, nocc int, tol float64) (U, W *mat.Dense, err error) {
	if can.NAO != loc.NAO || can.NMO != loc.NMO {
		return nil, nil, fmt.Errorf("quip: sidecars differ in size (%dx%d vs %dx%d)",
			can.NAO, can.NMO, loc.NAO, loc.NMO)
	}
	n, nao := can.NMO, can.NAO
	full := mat.NewDense(n, n, nil)
	for p := range n {
		for i := range n {
			var s float64
			for mu := range nao {
				var t float64
				for nu := range nao {
					t += can.S.At(mu, nu) * loc.C.At(nu, i)
				}
				s += can.C.At(mu, p) * t
			}
			full.Set(p, i, s)
		}
	}
	// the occupied-virtual block must vanish: localization rotates within each space
	var leak float64
	for p := range n {
		for i := range n {
			if (p < nocc) != (i < nocc) {
				leak = math.Max(leak, math.Abs(full.At(p, i)))
			}
		}
	}
	if leak > tol {
		return nil, nil, fmt.Errorf("quip: the localized orbitals mix occupied and virtual spaces (%.2e)", leak)
	}
	U = mat.DenseCopyOf(full.Slice(0, nocc, 0, nocc))
	W = mat.DenseCopyOf(full.Slice(nocc, n, nocc, n))
	return U, W, nil
}

// minor is det(M[rows, cols]) for spin-orbital index lists over a spatial rotation M
// (spin orbital x = 2*spatial + spin, the rotation block diagonal in spin), with the
// index offset off subtracted from the spatial parts.
func minor(M *mat.Dense, rows, cols []int, off int) float64 {
	k := len(rows)
	if k == 0 {
		return 1
	}
	a := mat.NewDense(k, k, nil)
	for i, p := range rows {
		for j, q := range cols {
			if p&1 == q&1 {
				a.Set(i, j, M.At(p>>1-off, q>>1-off))
			}
		}
	}
	return mat.Det(a)
}

// Rotation is R with |loc row l> = sum_c R[c][l] |can row c>, over one space used for
// both orbital sets: R[c][l] = det(W[P_c, A_l]) det(U[H_c, I_l]) within a class, zero
// across classes. The space must be closed under the rotation (no free-particle limit;
// point-group symmetry must be off, since a localized orbital need not carry one irrep).
func Rotation(sp *khci.Space, U, W *mat.Dense) (backend.Mat, error) {
	o := sp.Options()
	if o.LimitFree {
		return backend.Mat{}, fmt.Errorf("quip: a free-particle limit is not closed under the virtual rotation")
	}
	if o.OrbSym != nil {
		return backend.Mat{}, fmt.Errorf("quip: the rotated representation needs symmetry off")
	}
	nso := 2 * o.NOcc
	n := sp.Size()
	holes := make([][]int, n)
	parts := make([][]int, n)
	for r := range n {
		m := sp.HoleMask(r)
		for q := range nso {
			if m>>uint(q)&1 == 1 {
				holes[r] = append(holes[r], q)
			}
		}
		for _, a := range sp.PartSO(r, nil) {
			parts[r] = append(parts[r], nso+a)
		}
	}
	R := backend.NewMat(n, n)
	parallel.HeavyRows(n, func(c int) {
		cls := sp.Class(c)
		for l := sp.ClassStart(cls); l < sp.ClassEnd(cls); l++ {
			h := minor(U, holes[c], holes[l], 0)
			if h == 0 {
				continue
			}
			p := minor(W, parts[c], parts[l], o.NOcc)
			R.Data[c*n+l] = h * p
		}
	})
	return R, nil
}

// Rotate returns R^T M R.
func Rotate(M, R backend.Mat) backend.Mat {
	n := M.Rows
	m := mat.NewDense(n, n, M.Data)
	r := mat.NewDense(n, n, R.Data)
	var t, out mat.Dense
	t.Mul(m, r)
	out.Mul(r.T(), &t)
	res := backend.NewMat(n, n)
	for i := range n {
		for j := range n {
			res.Data[i*n+j] = out.At(i, j)
		}
	}
	return res
}
