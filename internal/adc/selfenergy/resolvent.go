package selfenergy

// resolvent.go — the all-order density behind Σ(∞). Ported from
// ../ADC/self_energy/constanti/constanti/inversion.f (INVERT / INVPRD / INVQKL).
//
// For each orbital p, solve at the frozen HF frequency
//
//	(ε_p − K − C)·y(p) = U(·,p)
//
// on both satellite blocks, then contract the amplitudes into the correlation density. On the
// 2h1p block the system is solved for the *virtual* orbitals only, and on 2p1h for the
// *occupied* ones; the other set's raw coupling amplitudes U are what the occupied/virtual block
// of the density consumes.
//
// The reference does NOT solve the system: INVPRD is a Jacobi/Neumann fixed point, truncated
// per-orbital when Σ_I(Δx_I)² < Akrit. Its Σ(∞) therefore carries the iteration's truncation
// error (~1e-6 at the default Akrit=1e-9), so reproducing theADCcode bit-for-bit means
// reproducing the iteration, not solving the system. Options carries the two knobs; converging
// them tightly gives the true fixed point, which is the exact Σ(∞).

// Options tunes the resolvent iteration. The zero value means "converge properly"; pass
// theADCcode's own defaults (Akrit 1e-9, MaxIt 30) to reproduce its output exactly.
type Options struct {
	Akrit float64 // per-orbital convergence threshold on Σ(Δx)²; 0 → 1e-16
	MaxIt int     // iteration cap; 0 → 200
}

func (o Options) akrit() float64 {
	if o.Akrit <= 0 {
		return 1e-16
	}
	return o.Akrit
}

func (o Options) maxIt() int {
	if o.MaxIt <= 0 {
		return 200
	}
	return o.MaxIt
}

// TheADCcodeDefaults are constanti's own iteration settings (input_data.cpp:438-446). Using them
// reproduces theADCcode's Σ(∞) including its truncation.
var TheADCcodeDefaults = Options{Akrit: 1e-9, MaxIt: 30}

// solveResolvent runs INVERT/INVPRD for one satellite block and returns y for each solved-for
// orbital, indexed by its position within the irrep (the orbsOfSym ordering). Columns that are
// not solved for on this block are left nil.
func (e *engine) solveResolvent(sp *satSpace, m *satMatrix, u []float64, opts Options) [][]float64 {
	orbs := e.orbsOfSym(sp.sym)
	nc, n := len(orbs), sp.dim
	nocc := len(e.occs[sp.sym])

	// The block solves for the complementary set: virtuals on 2h1p, occupieds on 2p1h.
	lo, hi := nocc, nc
	if sp.blk == iab2p1h {
		lo, hi = 0, nocc
	}

	y := make([][]float64, nc)
	akrit, maxIt := opts.akrit(), opts.maxIt()

	for np := lo; np < hi; np++ {
		epsP := e.eps[orbs[np]]

		// rhs = U(·,p); x⁰ = rhs.
		rhs := make([]float64, n)
		for i := range n {
			rhs[i] = u[i*nc+np]
		}
		x := append([]float64(nil), rhs...)
		acc := make([]float64, n)

		// Jacobi/Neumann: x ← U + Σ_{J≠I} (K+C)_IJ · x_J /(ε_p − (K+C)_JJ).
		// The diagonal never enters the mat-vec; each stored triplet feeds both of its rows,
		// which is how the reference realises the symmetric off-diagonal from one triangle.
		for range maxIt {
			clear(acc)
			for _, t := range m.off {
				acc[t.i] += t.v * x[t.j] / (epsP - m.diag[t.j])
				acc[t.j] += t.v * x[t.i] / (epsP - m.diag[t.i])
			}
			var delta float64
			for i := range n {
				acc[i] += rhs[i]
				d := x[i] - acc[i]
				delta += d * d
			}
			copy(x, acc)
			if delta < akrit {
				break
			}
		}

		// y = x/(ε_p − (K+C)_II), applied once at the end (inversion.f:83-86).
		yi := make([]float64, n)
		for i := range n {
			yi[i] = x[i] / (epsP - m.diag[i])
		}
		y[np] = yi
	}
	return y
}

// blockData is one solved satellite block: its couplings and its resolvent amplitudes.
type blockData struct {
	sp *satSpace
	u  []float64
	y  [][]float64
}

// densityAllOrder builds the all-order correlation density ρ (INVQKL, inversion.f:181-265),
// accumulated over both blocks and every irrep. ρ is returned as a full symmetric matrix over
// absolute orbital indices — algebraically identical to the reference's triangular QKL with its
// ½ diagonal factor, and exactly what rhoToSigma already consumes.
func (e *engine) densityAllOrder(opts Options) *Sigma {
	rho := newSigma(e.norb)

	for sym := range e.nsym {
		orbs := e.orbsOfSym(sym)
		if len(orbs) == 0 {
			continue
		}
		nocc := len(e.occs[sym])
		nc := len(orbs)

		// The reference only ever visits irreps that HAVE occupied orbitals: constanti drives
		// everything from NRSYM (adc1.f), and sigma.f likewise restricts the (k,l) sum of the
		// V·ρ contraction to those irreps. An irrep with no occupied orbital therefore
		// contributes nothing to ρ — not even its virtual/virtual block, which is physically
		// non-zero (h2o's A2 2h1p space is not empty: |a(A1) 1b2 1b1> configurations exist).
		// That is a real omission in theADCcode, but reproducing Σ(∞) bit-exactly means
		// reproducing it. See TestSigmaInfinite.
		if nocc == 0 {
			continue
		}

		blocks := map[iab]*blockData{}
		for _, blk := range []iab{iab2h1p, iab2p1h} {
			sp := e.buildSatSpace(blk, sym)
			if sp.dim == 0 {
				continue
			}
			m := e.buildSatMatrix(sp)
			u := e.coupling(sp)
			blocks[blk] = &blockData{sp: sp, u: u, y: e.solveResolvent(sp, m, u, opts)}
		}

		hp, ph := blocks[iab2h1p], blocks[iab2p1h]

		// occ/occ — from the 2p1h block, with a minus sign ((1−IAB) = −1).
		if ph != nil {
			for ki := range nocc {
				for li := range nocc {
					yk, yl := ph.y[ki], ph.y[li]
					if yk == nil || yl == nil {
						continue
					}
					var s float64
					for i := range ph.sp.dim {
						s += yk[i] * yl[i]
					}
					rho.set(orbs[ki], orbs[li], rho.At(orbs[ki], orbs[li])-s)
				}
			}
		}

		// vir/vir — from the 2h1p block ((2−IAB) = +1).
		if hp != nil {
			for ai := nocc; ai < nc; ai++ {
				for bi := nocc; bi < nc; bi++ {
					ya, yb := hp.y[ai], hp.y[bi]
					if ya == nil || yb == nil {
						continue
					}
					var s float64
					for i := range hp.sp.dim {
						s += ya[i] * yb[i]
					}
					rho.set(orbs[ai], orbs[bi], rho.At(orbs[ai], orbs[bi])+s)
				}
			}
		}

		// occ/vir — both blocks contribute: on 2h1p the resolvent of the virtual meets the raw
		// coupling of the occupied, and on 2p1h the other way round.
		for ki := range nocc {
			for ai := nocc; ai < nc; ai++ {
				k, a := orbs[ki], orbs[ai]
				diff := e.eps[k] - e.eps[a]
				var s float64
				if hp != nil && hp.y[ai] != nil {
					for i := range hp.sp.dim {
						s += hp.y[ai][i] * hp.u[i*nc+ki]
					}
				}
				if ph != nil && ph.y[ki] != nil {
					for i := range ph.sp.dim {
						s += ph.y[ki][i] * ph.u[i*nc+ai]
					}
				}
				s /= diff
				rho.set(k, a, rho.At(k, a)+s)
				rho.set(a, k, rho.At(a, k)+s)
			}
		}
	}
	return rho
}
