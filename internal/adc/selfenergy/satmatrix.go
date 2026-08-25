package selfenergy

import (
	"math"
	"runtime"

	"github.com/leiaSQ/ADCgo/internal/adc/parallel"
)

// satmatrix.go — the (K+C) satellite matrix. Ported from
// ../ADC/self_energy/constanti/common/aufbau2.f WERT1 (the elements, lines 439-612) and
// ../ADC/self_energy/constanti/constanti/ab3.f (the assembly, signs and sparsity).
//
// C is first order only: constanti runs at IORDER=3, and aufbau2.f:612 returns before the
// fourth-order (SUM1..SUM4) block. So this is the 2ph-TDA matrix of Schirmer & Cederbaum,
// J. Phys. B 11, 1889 (1978), eqs. (32) and (10).
//
// WERT1 builds −(K+C) (so the 2h1p eigenvalues come out positive); ab3 negates it again, so what
// is stored — and what the resolvent uses — is the physical (K+C). The resolvent denominator is
// then ε_p − (K+C)_JJ directly.

// tolmat is ab3.f's sparsity threshold (adc1.f:242). Elements below it are dropped — including,
// deliberately, diagonal ones: ab3.f applies the test *before* splitting diagonal from
// off-diagonal, so a tiny diagonal entry is silently discarded and shrinks the space. Replicated
// here so the matrix (and hence the truncated Jacobi iteration) is the reference's, exactly.
const tolmat = 1.0e-6

var ds075 = math.Sqrt(3) / 2

// satMatrix is one (K+C) block: a dense diagonal plus sparse off-diagonals. The off-diagonal is
// stored once per symmetric pair, as the reference does; the resolvent applies each triplet to
// both of its rows.
type satMatrix struct {
	diag []float64
	off  []offTriplet
}

type offTriplet struct {
	i, j int // 0-based spin-resolved indices
	v    float64
}

// wert1 returns W[rowSpin][colSpin] = −(K+C) for one pair of spatial configurations.
// row = (j,k,l), col = (jj,kk,ll) — the order ab3.f passes them in.
func (e *engine) wert1(row, col satConf, blk iab) [2][2]float64 {
	ep := e.eps
	j, k, l := row.j, row.k, row.l
	jj, kk, ll := col.j, col.k, col.l

	var w [2][2]float64

	// K: only for the identical configuration, and only on the spin diagonal.
	if j == jj && k == kk && l == ll {
		w[0][0] = ep[j] - ep[k] - ep[l]
		w[1][1] = w[0][0]
	}

	// C, first order: ten bare integrals selected by which orbitals coincide. The branch
	// structure is aufbau2.f:528-593 transcribed literally — several cases reuse A3/A4 (or
	// A9/A10, A5/A6) for the partners the coincidence makes equal.
	var a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 float64
	if j == jj {
		a1 = e.v(k, l, kk, ll)
		a2 = e.v(k, l, ll, kk)
	}

	done := false
	switch {
	case k == kk:
		a3 = e.v(j, ll, jj, l)
		a4 = e.v(j, ll, l, jj)
		switch {
		case k == l && kk == ll: // label 10
			a5, a6, a7, a8, a9, a10 = a3, a4, a3, a4, a3, a4
			done = true
		case k == l: // label 11
			a9, a10 = a3, a4
			done = true
		case kk == ll: // label 12
			a7, a8 = a3, a4
			done = true
		}
		// otherwise fall through to label 103, with A3/A4 already set
	case l == kk: // label 102
		a9 = e.v(j, ll, jj, k)
		a10 = e.v(j, ll, k, jj)
		if kk == ll {
			a5, a6 = a9, a10
		}
		done = true
	}
	if !done {
		if l == ll { // label 103
			a5 = e.v(j, kk, jj, k)
			a6 = e.v(j, kk, k, jj)
			if k == l {
				a7, a8 = a5, a6
			}
		} else if k == ll { // label 104
			a7 = e.v(j, kk, jj, l)
			a8 = e.v(j, kk, l, jj)
		}
	}

	// label 150: the spin table (aufbau2.f:594-601).
	f := faktor(row.maxS) * faktor(col.maxS) * asig(blk)
	w[0][0] += f * ((a1 + a2) - (a3 + a5 + a7 + a9) + 0.5*(a4+a6+a8+a10))
	w[1][1] += f * ((a1 - a2) - (a3 + a5 - a7 - a9) + 1.5*(a4+a6-a8-a10))
	w[0][1] += ds075 * f * (a4 - a6 - a8 + a10)
	w[1][0] += ds075 * f * (a4 - a6 + a8 - a10)
	return w
}

// satColumn walks one column of ab3.f's traversal — every earlier row configuration gives a full
// spin block, and the diagonal configuration contributes its lower spin triangle — handing each
// surviving element to emitOff (sparse off-diagonal) or emitDiag (dense diagonal) in traversal
// order. The two buildSatMatrix passes share it so the counting walk and the filling walk cannot
// drift apart: a discrepancy would silently truncate or misplace a column's triplets.
func (e *engine) satColumn(sp *satSpace, ci int, emitOff func(i, j int, v float64), emitDiag func(v float64)) {
	col := sp.confs[ci]
	for ri := 0; ri <= ci; ri++ {
		row := sp.confs[ri]
		w := e.wert1(row, col, sp.blk)

		if ri < ci {
			for mss := range col.maxS {
				for ms := range row.maxS {
					v := w[ms][mss]
					if math.Abs(v) < tolmat {
						continue
					}
					emitOff(row.off+ms, col.off+mss, -v)
				}
			}
			continue
		}
		// The configuration's own block: lower spin triangle only.
		for mss := range col.maxS {
			for ms := 0; ms <= mss; ms++ {
				v := w[ms][mss]
				if math.Abs(v) < tolmat {
					continue
				}
				i, jx := row.off+ms, col.off+mss
				if i != jx {
					emitOff(i, jx, -v)
				} else {
					emitDiag(-v)
				}
			}
		}
	}
}

// buildSatMatrix assembles (K+C) for one satellite space.
//
// The traversal is O(dim²/2) calls to wert1 — ten ERI lookups and a spin table each, pure Go, no
// BLAS — and it runs in the Σ(∞) build before any solver starts, so every core but one sat idle
// through it. It is parallelized over COLUMNS, in two passes: count each column's contribution,
// prefix-sum to a fixed offset per column, then fill. Both passes go through satColumn.
//
// Two passes rather than per-worker buckets, because this routine's binding constraint is host
// memory, not time — the production SIP triplet list is what exhausts the node before the solver is
// reached. Buckets concatenated at the end would peak at twice the list; counting first allocates
// off and diag at their exact final length, which peaks BELOW the serial version — append's
// geometric growth transiently held up to ~2x a full list and left the slack behind. The price is
// that wert1 runs twice, which pays for itself for any worker count above two.
//
// Bit-identical to the serial walk, and byte-identical in layout: a column's slot is decided by
// the prefix sum over column index, never by which worker got there first, so ordering is a pure
// function of the input. No arithmetic is reordered — each element is independent, there is no
// reduction here.
func (e *engine) buildSatMatrix(sp *satSpace) *satMatrix {
	n := len(sp.confs)
	if n == 0 {
		return &satMatrix{diag: make([]float64, 0, sp.dim)}
	}
	// Below parallel.Rows' serial-fallback threshold both passes would run serially, which is
	// just the double wert1 cost with none of the benefit. Small spaces keep the single-pass
	// walk. Output is identical either way — this selects a schedule, never a result.
	if n < 2*runtime.GOMAXPROCS(0) {
		m := &satMatrix{diag: make([]float64, 0, sp.dim)}
		for ci := range n {
			e.satColumn(sp, ci,
				func(i, j int, v float64) { m.off = append(m.off, offTriplet{i: i, j: j, v: v}) },
				func(v float64) { m.diag = append(m.diag, v) })
		}
		return m
	}

	offN := make([]int, n)
	diagN := make([]int, n)
	parallel.Rows(n, func(ci int) {
		var no, nd int
		e.satColumn(sp, ci, func(int, int, float64) { no++ }, func(float64) { nd++ })
		offN[ci], diagN[ci] = no, nd
	})

	// Exclusive prefix sums: column ci owns off[offAt[ci]:offAt[ci+1]], disjoint by construction.
	offAt := make([]int, n+1)
	diagAt := make([]int, n+1)
	for ci := range n {
		offAt[ci+1] = offAt[ci] + offN[ci]
		diagAt[ci+1] = diagAt[ci] + diagN[ci]
	}

	m := &satMatrix{off: make([]offTriplet, offAt[n]), diag: make([]float64, diagAt[n])}
	parallel.Rows(n, func(ci int) {
		off, diag := m.off[offAt[ci]:offAt[ci+1]], m.diag[diagAt[ci]:diagAt[ci+1]]
		var io, id int
		e.satColumn(sp, ci,
			func(i, j int, v float64) {
				off[io] = offTriplet{i: i, j: j, v: v}
				io++
			},
			func(v float64) {
				diag[id] = v
				id++
			})
	})
	return m
}
