package khci

// ci.go — the CI-flavour Hamiltonian over a k-hole space: M = H - E_HF on the rows of a Space, with H the full
// electronic Hamiltonian of the FCIDUMP and E_HF = <Phi_0|H|Phi_0>.
//
// There is no Fortran counterpart: theADCcode has no k-hole CI. The matrix elements
// are the Slater-Condon rules of internal/adc/sip/slater.go (hamElem, diffDet),
// rewritten relative to the reference so that no element sums over all electrons:
//
//	diagonal   E_D - E_HF = sum_P F_aa - sum_H F_ii + sum_{i<j in H} <ij||ij>
//	                      + sum_{a<b in P} <ab||ab> - sum_{i in H, a in P} <ai||ai>
//	single     f -> t     F_ft - sum_{i in H} <fi||ti> + sum_{a in P} <fa||ta>
//	double     f1 f2 -> t1 t2    <f1 f2||t1 t2>
//
// with H and P the holes and particles of the bra determinant and F the GENERAL
// (not necessarily diagonal) Fock matrix of the reference. Nothing assumes canonical
// orbitals: CI runs in localized occupied and compact/free virtual orbitals as well, and
// the classes kh | (k+1)h1p | (k+2)h2p are invariant under
// occupied-occupied and virtual-virtual rotations, so the spectrum is too
// (TestLocalizedEqualsCanonical).
//
// Signs follow diffDet: with both determinants canonical (spin orbitals created in
// ascending order), aligning the differing spin orbitals costs the sum of their
// positions, and a row relates to its canonical determinant by Space.Det's sign.
//
// The applier is direct CI: every output row enumerates the rows it couples to (at
// most a double replacement away and inside the space) and looks each up with
// Space.Index. Rows are independent, so the apply parallelizes over rows with a fixed
// per-row summation order: the result does not depend on the worker count. Materialize
// stores the enumerated couplings as CSR when they fit a memory budget, for repeated
// applies on a restricted (Q) space.

import (
	"fmt"
	"math"
	"math/bits"
	"sync/atomic"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/parallel"
)

// CI is H - E_HF on the rows of a Space. It satisfies lanczos.Operator,
// lanczos.PreconOperator, lanczos.DenseOperator and fano.Applier. It is immutable
// after construction apart from Materialize, and safe for concurrent applies.
type CI struct {
	sp   *Space
	d    *fcidump.Data
	be   backend.Backend
	norb int
	full uint64 // every reference-occupied spin orbital

	fock []float64 // spatial general Fock matrix, norb*norb
	ehf  float64   // <Phi_0|H|Phi_0>, electronic
	sign []int8    // |row> = sign |canonical determinant>
	diag []float64

	// virtuals by spin and irrep, as ABSOLUTE spin orbitals: vir[spin][irrep]
	vir   [2][][]int
	nirr  int
	irrep []int // spatial orbital -> irrep (all 0 without symmetry)

	csr atomic.Pointer[csrMat]
}

// csrMat is the materialized off-diagonal part, row-major.
type csrMat struct {
	start []int64
	col   []int32
	val   []float64
}

// NewCI builds the operator. The FCIDUMP must describe the closed-shell reference the
// space was built for: NORB = NOcc+NVir and NELEC = 2*NOcc.
func NewCI(sp *Space, d *fcidump.Data, be backend.Backend) (*CI, error) {
	o := sp.opt
	if d.NORB != o.NOcc+o.NVir {
		return nil, fmt.Errorf("khci: FCIDUMP has %d orbitals, the space %d occupied + %d virtual",
			d.NORB, o.NOcc, o.NVir)
	}
	if d.NELEC != 2*o.NOcc {
		return nil, fmt.Errorf("khci: FCIDUMP has %d electrons, the closed-shell reference %d",
			d.NELEC, 2*o.NOcc)
	}
	n := d.NORB
	ci := &CI{sp: sp, d: d, be: be, norb: n}
	if sp.nso == 64 {
		ci.full = ^uint64(0)
	} else {
		ci.full = uint64(1)<<uint(sp.nso) - 1
	}
	ci.irrep = make([]int, n)
	if o.OrbSym != nil {
		copy(ci.irrep, o.OrbSym)
	}
	for _, g := range ci.irrep {
		ci.nirr = max(ci.nirr, g+1)
	}
	ci.nirr = max(ci.nirr, 1)
	for s := range 2 {
		ci.vir[s] = make([][]int, ci.nirr)
	}
	for a := range sp.nv2 {
		x := sp.nso + a
		ci.vir[x&1][ci.irrep[x>>1]] = append(ci.vir[x&1][ci.irrep[x>>1]], x)
	}

	// general Fock matrix F_pq = h_pq + sum_i [2(pq|ii) - (pi|iq)] and
	// E_HF = sum_i (h_ii + F_ii)
	ci.fock = make([]float64, n*n)
	parallel.HeavyRows(n, func(p int) {
		for q := range n {
			v := d.OneE(p, q)
			for i := range o.NOcc {
				v += 2*d.TwoE(p, q, i, i) - d.TwoE(p, i, i, q)
			}
			ci.fock[p*n+q] = v
		}
	})
	for i := range o.NOcc {
		ci.ehf += d.OneE(i, i) + ci.fock[i*n+i]
	}

	rows := sp.Size()
	ci.sign = make([]int8, rows)
	ci.diag = make([]float64, rows)
	parallel.HeavyRows(rows, func(r int) {
		_, _, s := sp.Det(r)
		ci.sign[r] = int8(s)
		ci.diag[r] = ci.diagonal(ci.rep(r))
	})
	return ci, nil
}

// Space returns the configuration space.
func (ci *CI) Space() *Space { return ci.sp }

// EHF is <Phi_0|H|Phi_0> without the core energy; M is measured from it.
func (ci *CI) EHF() float64 { return ci.ehf }

// Fock returns the spatial general Fock element F_pq.
func (ci *CI) Fock(p, q int) float64 { return ci.fock[p*ci.norb+q] }

// Size is the number of rows.
func (ci *CI) Size() int { return ci.sp.Size() }

// Backend is the backend the operator was built for; device vectors are staged through
// the host (ApplyBlock).
func (ci *CI) Backend() backend.Backend { return ci.be }

// Release drops the materialized CSR, if any. The operator stays usable: later applies
// enumerate the couplings again.
func (ci *CI) Release() { ci.csr.Store(nil) }

// MainBlockSize is the size of the main (k-hole) class.
func (ci *CI) MainBlockSize() int { return ci.sp.MainBlockSize() }

// Restrict is the operator on the sub-space of the given rows (ascending, distinct):
// the QMQ / PMP blocks of a Fano partition. It shares the integrals and Fock matrix.
func (ci *CI) Restrict(rows []int) (*CI, error) {
	sub, err := ci.sp.Restrict(rows)
	if err != nil {
		return nil, err
	}
	out := &CI{sp: sub, d: ci.d, be: ci.be, norb: ci.norb, full: ci.full,
		fock: ci.fock, ehf: ci.ehf, vir: ci.vir, nirr: ci.nirr, irrep: ci.irrep}
	out.sign = make([]int8, len(rows))
	out.diag = make([]float64, len(rows))
	for i, r := range rows {
		out.sign[i] = ci.sign[r]
		out.diag[i] = ci.diag[r]
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// Determinants relative to the reference
// ---------------------------------------------------------------------------

// det is a determinant as its holes (reference spin orbitals emptied) and particles
// (ABSOLUTE virtual spin orbitals, ascending, np of them).
type det struct {
	h  uint64
	p  [2]int
	np int
}

func (ci *CI) rep(r int) det {
	rw := ci.sp.rows[r]
	d := det{h: rw.holes}
	for _, a := range rw.p {
		if a >= 0 {
			d.p[d.np] = ci.sp.nso + int(a)
			d.np++
		}
	}
	return d
}

// pos is the number of occupied spin orbitals of d below x: x's position in the
// canonical determinant (for x occupied) or the count its creation passes.
func (ci *CI) pos(d det, x int) int {
	nso := ci.sp.nso
	if x < nso {
		return bits.OnesCount64(ci.full &^ d.h & (uint64(1)<<uint(x) - 1))
	}
	n := bits.OnesCount64(ci.full &^ d.h)
	for i := range d.np {
		if d.p[i] < x {
			n++
		}
	}
	return n
}

// remove annihilates x (occupied in d) and create creates x (empty in d); both keep
// the particle list ascending.
func (ci *CI) remove(d det, x int) det {
	if x < ci.sp.nso {
		d.h |= 1 << uint(x)
		return d
	}
	if d.p[0] == x {
		d.p[0] = d.p[1]
	}
	d.p[1] = -1
	d.np--
	if d.np == 0 {
		d.p[0] = -1
	}
	return d
}

func (ci *CI) create(d det, x int) det {
	if x < ci.sp.nso {
		d.h &^= 1 << uint(x)
		return d
	}
	switch {
	case d.np == 0:
		d.p[0] = x
	case x < d.p[0]:
		d.p[1], d.p[0] = d.p[0], x
	default:
		d.p[1] = x
	}
	d.np++
	return d
}

// index is the row of d, if it is in the space.
func (ci *CI) index(d det) (int, bool) {
	var buf [2]int
	for i := range d.np {
		buf[i] = d.p[i] - ci.sp.nso
	}
	return ci.sp.Index(d.h, buf[:d.np])
}

// ---------------------------------------------------------------------------
// Integrals in spin orbitals (absolute index: spatial x>>1, spin x&1)
// ---------------------------------------------------------------------------

// asym is <pq||rs> = <pq|rs> - <pq|sr>, physicist notation, <pq|rs> = (pr|qs).
func (ci *CI) asym(p, q, r, s int) float64 {
	var v float64
	if p&1 == r&1 && q&1 == s&1 {
		v = ci.d.TwoE(p>>1, r>>1, q>>1, s>>1)
	}
	if p&1 == s&1 && q&1 == r&1 {
		v -= ci.d.TwoE(p>>1, s>>1, q>>1, r>>1)
	}
	return v
}

func (ci *CI) f(p, q int) float64 {
	if p&1 != q&1 {
		return 0
	}
	return ci.fock[(p>>1)*ci.norb+(q>>1)]
}

func (ci *CI) diagonal(d det) float64 {
	var v float64
	for q := d.h; q != 0; q &= q - 1 {
		i := bits.TrailingZeros64(q)
		v -= ci.f(i, i)
		for q2 := q & (q - 1); q2 != 0; q2 &= q2 - 1 {
			j := bits.TrailingZeros64(q2)
			v += ci.asym(i, j, i, j)
		}
		for k := range d.np {
			v -= ci.asym(d.p[k], i, d.p[k], i)
		}
	}
	for k := range d.np {
		a := d.p[k]
		v += ci.f(a, a)
		for l := k + 1; l < d.np; l++ {
			v += ci.asym(a, d.p[l], a, d.p[l])
		}
	}
	return v
}

// single is <D|H|D'> for D' = D with f replaced by t, without the sign.
func (ci *CI) single(d det, f, t int) float64 {
	v := ci.f(f, t)
	for q := d.h; q != 0; q &= q - 1 {
		i := bits.TrailingZeros64(q)
		v -= ci.asym(f, i, t, i)
	}
	for k := range d.np {
		v += ci.asym(f, d.p[k], t, d.p[k])
	}
	return v
}

func parity(n int) float64 {
	if n&1 == 1 {
		return -1
	}
	return 1
}

// ---------------------------------------------------------------------------
// Elements
// ---------------------------------------------------------------------------

// Element is <r|H - E_HF|c>, evaluated from the two rows' determinants directly (not
// through the applier's enumeration, which the tests compare it with).
func (ci *CI) Element(r, c int) float64 {
	if r == c {
		return ci.diag[r]
	}
	a, b := ci.rep(r), ci.rep(c)
	// in a, not in b: reference orbitals that are holes of b only, and a's particles
	// missing from b; symmetrically for "to"
	var from, to [2]int
	nf, nt := 0, 0
	add := func(dst *[2]int, n *int, x int) bool {
		if *n == 2 {
			return false
		}
		dst[*n] = x
		*n++
		return true
	}
	for q := b.h &^ a.h; q != 0; q &= q - 1 {
		if !add(&from, &nf, bits.TrailingZeros64(q)) {
			return 0
		}
	}
	for q := a.h &^ b.h; q != 0; q &= q - 1 {
		if !add(&to, &nt, bits.TrailingZeros64(q)) {
			return 0
		}
	}
	inList := func(d det, x int) bool {
		for k := range d.np {
			if d.p[k] == x {
				return true
			}
		}
		return false
	}
	for k := range a.np {
		if !inList(b, a.p[k]) && !add(&from, &nf, a.p[k]) {
			return 0
		}
	}
	for k := range b.np {
		if !inList(a, b.p[k]) && !add(&to, &nt, b.p[k]) {
			return 0
		}
	}
	if nf != nt || nf == 0 {
		return 0
	}
	// occupied orbitals precede particles, so from/to are already ascending
	s := float64(ci.sign[r]) * float64(ci.sign[c])
	if nf == 1 {
		return s * parity(ci.pos(a, from[0])+ci.pos(b, to[0])) * ci.single(a, from[0], to[0])
	}
	// diffDet's sign: the sum of the differing orbitals' positions in their own
	// determinants (the second position counts the first differing orbital, in both)
	p := ci.pos(a, from[0]) + ci.pos(a, from[1]) + ci.pos(b, to[0]) + ci.pos(b, to[1])
	return s * parity(p) * ci.asym(from[0], from[1], to[0], to[1])
}

// neighbors calls emit(c, M_rc) for every row c != r with M_rc possibly nonzero, in a
// fixed order: single replacements, then double replacements, each in ascending
// orbital order. The diagonal is not emitted.
func (ci *CI) neighbors(r int, emit func(c int, v float64)) {
	sp := ci.sp
	d := ci.rep(r)
	sr := float64(ci.sign[r])
	mmax := sp.opt.MaxClass - sp.opt.K
	nso := sp.nso

	// the electrons of d, ascending
	var el [66]int
	ne := 0
	for q := ci.full &^ d.h; q != 0; q &= q - 1 {
		el[ne] = bits.TrailingZeros64(q)
		ne++
	}
	for k := range d.np {
		el[ne] = d.p[k]
		ne++
	}
	isPart := func(x int) bool { return x >= nso }
	hasPart := func(x int) bool {
		for k := range d.np {
			if d.p[k] == x {
				return true
			}
		}
		return false
	}
	irr := func(x int) int { return ci.irrep[x>>1] }

	emitTo := func(e det, v float64) {
		if v == 0 {
			return
		}
		c, ok := ci.index(e)
		if !ok {
			return
		}
		emit(c, sr*float64(ci.sign[c])*v)
	}

	// singles f -> t, same spin and irrep
	for i := range ne {
		f := el[i]
		fp := 0
		if isPart(f) {
			fp = 1
		}
		mid := ci.remove(d, f)
		pf := ci.pos(d, f)
		// t a hole of d
		for q := d.h; q != 0; q &= q - 1 {
			t := bits.TrailingZeros64(q)
			if t&1 != f&1 || irr(t) != irr(f) {
				continue
			}
			e := ci.create(mid, t)
			emitTo(e, parity(pf+ci.pos(e, t))*ci.single(d, f, t))
		}
		// t a virtual not in d: the particle count goes to np - fp + 1
		if d.np-fp+1 > mmax {
			continue
		}
		for _, t := range ci.vir[f&1][irr(f)] {
			if hasPart(t) {
				continue
			}
			e := ci.create(mid, t)
			emitTo(e, parity(pf+ci.pos(e, t))*ci.single(d, f, t))
		}
	}

	// doubles f1 f2 -> t1 t2 (f1 < f2, t1 < t2), spin and irrep conserved
	var holes [64]int
	nh := 0
	for q := d.h; q != 0; q &= q - 1 {
		holes[nh] = bits.TrailingZeros64(q)
		nh++
	}
	for i := range ne {
		f1 := el[i]
		for j := i + 1; j < ne; j++ {
			f2 := el[j]
			npf := 0
			if isPart(f1) {
				npf++
			}
			if isPart(f2) {
				npf++
			}
			budget := mmax - (d.np - npf) // virtual targets allowed
			spinSum := f1&1 + f2&1
			irrSum := irr(f1) ^ irr(f2)
			mid := ci.remove(ci.remove(d, f1), f2)
			pf := ci.pos(d, f1) + ci.pos(d, f2)
			try := func(t1, t2 int) {
				e := ci.create(ci.create(mid, t1), t2)
				v := ci.asym(f1, f2, t1, t2)
				emitTo(e, parity(pf+ci.pos(e, t1)+ci.pos(e, t2))*v)
			}
			// two holes refilled
			for a := range nh {
				t1 := holes[a]
				for b := a + 1; b < nh; b++ {
					t2 := holes[b]
					if t1&1+t2&1 != spinSum || irr(t1)^irr(t2) != irrSum {
						continue
					}
					try(t1, t2)
				}
			}
			if budget < 1 {
				continue
			}
			// one hole refilled, one virtual
			for a := range nh {
				t1 := holes[a]
				s2 := spinSum - t1&1
				if s2 < 0 || s2 > 1 {
					continue
				}
				for _, t2 := range ci.vir[s2][irr(t1)^irrSum] {
					if hasPart(t2) {
						continue
					}
					try(t1, t2)
				}
			}
			if budget < 2 {
				continue
			}
			// two virtuals, t1 < t2
			for s1 := range 2 {
				s2 := spinSum - s1
				if s2 < 0 || s2 > 1 {
					continue
				}
				for g1 := range ci.nirr {
					for _, t1 := range ci.vir[s1][g1] {
						if hasPart(t1) {
							continue
						}
						for _, t2 := range ci.vir[s2][g1^irrSum] {
							if t2 <= t1 || hasPart(t2) {
								continue
							}
							try(t1, t2)
						}
					}
				}
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Operator interfaces
// ---------------------------------------------------------------------------

// Materialize stores the off-diagonal couplings as CSR when they need at most
// maxBytes (12 bytes per nonzero plus 8 per row), and reports whether it did. Later
// applies read the CSR instead of enumerating. It is the same matrix either way.
func (ci *CI) Materialize(maxBytes int64) bool {
	n := ci.Size()
	counts := make([]int64, n+1)
	parallel.HeavyRows(n, func(r int) {
		var k int64
		ci.neighbors(r, func(int, float64) { k++ })
		counts[r+1] = k
	})
	for r := range n {
		counts[r+1] += counts[r]
	}
	nnz := counts[n]
	if 12*nnz+8*int64(n+1) > maxBytes {
		return false
	}
	m := &csrMat{start: counts, col: make([]int32, nnz), val: make([]float64, nnz)}
	parallel.HeavyRows(n, func(r int) {
		k := m.start[r]
		ci.neighbors(r, func(c int, v float64) {
			m.col[k] = int32(c)
			m.val[k] = v
			k++
		})
	})
	ci.csr.Store(m)
	return true
}

// NNZ is the number of stored off-diagonal couplings, 0 when not materialized.
func (ci *CI) NNZ() int64 {
	if m := ci.csr.Load(); m != nil {
		return m.start[len(m.start)-1]
	}
	return 0
}

// applyHost computes y[:,j] = M x[:,j] for cols columns of leading dimensions ldx/ldy.
func (ci *CI) applyHost(y, x []float64, cols, ldx, ldy int) {
	n := ci.Size()
	m := ci.csr.Load()
	parallel.HeavyRows(n, func(r int) {
		for j := range cols {
			y[j*ldy+r] = ci.diag[r] * x[j*ldx+r]
		}
		if m != nil {
			for k := m.start[r]; k < m.start[r+1]; k++ {
				c, v := int(m.col[k]), m.val[k]
				for j := range cols {
					y[j*ldy+r] += v * x[j*ldx+c]
				}
			}
			return
		}
		ci.neighbors(r, func(c int, v float64) {
			for j := range cols {
				y[j*ldy+r] += v * x[j*ldx+c]
			}
		})
	})
}

// hostSlices returns the in-place host data of a vector, or nil when the backend keeps
// it elsewhere. Device backends embed Gonum and so satisfy HostData through promotion;
// DeviceKernels is tested first for that reason (backend/README.md).
func (ci *CI) hostSlice(v backend.Vector) []float64 {
	if _, dev := ci.be.(backend.DeviceKernels); dev {
		return nil
	}
	if hd, ok := ci.be.(backend.HostData); ok {
		return hd.HostSlice(v)
	}
	return nil
}

// ApplyFull is out = M in.
func (ci *CI) ApplyFull(out, in backend.Vector) {
	n := ci.Size()
	ci.ApplyBlock(backend.BlockView{V: out, Rows: n, Cols: 1, Ld: n},
		backend.BlockView{V: in, Rows: n, Cols: 1, Ld: n})
}

// ApplyBlock is out = M in, column by column, enumerating each row's couplings once for
// the whole block.
func (ci *CI) ApplyBlock(out, in backend.BlockView) {
	if in.Cols == 0 {
		return
	}
	x, y := ci.hostSlice(in.V), ci.hostSlice(out.V)
	if x != nil && y != nil {
		ci.applyHost(y, x, in.Cols, in.Ld, out.Ld)
		return
	}
	// device-resident vectors: stage through the host
	n := ci.Size()
	xh := ci.be.Download(in.V)
	yh := make([]float64, (out.Cols-1)*out.Ld+n)
	ci.applyHost(yh, xh, in.Cols, in.Ld, out.Ld)
	if out.Ld == n || out.Cols == 1 {
		dv := ci.be.Upload(yh)
		ci.be.Copy(out.V.Slice(0, len(yh)), dv)
		ci.be.Free(dv)
		return
	}
	for j := range out.Cols {
		dv := ci.be.Upload(yh[j*out.Ld : j*out.Ld+n])
		ci.be.Copy(out.Col(j), dv)
		ci.be.Free(dv)
	}
}

// Diagonal is M's diagonal, resident, for the Davidson preconditioner.
func (ci *CI) Diagonal(be backend.Backend) backend.Vector {
	return be.Upload(append([]float64(nil), ci.diag...))
}

// DiagonalHost is M's diagonal on the host (a copy).
func (ci *CI) DiagonalHost() []float64 { return append([]float64(nil), ci.diag...) }

// BuildMatrix assembles M densely from the applier's enumeration (rows in parallel).
func (ci *CI) BuildMatrix() backend.Mat {
	n := ci.Size()
	m := backend.NewMat(n, n)
	parallel.HeavyRows(n, func(r int) {
		m.Data[r*n+r] = ci.diag[r]
		ci.neighbors(r, func(c int, v float64) { m.Data[r*n+c] += v })
	})
	return m
}

// ---------------------------------------------------------------------------
// Spin
// ---------------------------------------------------------------------------

// shiftSpin applies sum_p c+_(p,t) c_(p,from) to one determinant (from = beta: S+,
// from = alpha: S-), calling emit with each image and its canonical sign.
func (ci *CI) shiftSpin(d det, from int, emit func(e det, s float64)) {
	occ := func(x int) bool {
		if x < ci.sp.nso {
			return d.h>>uint(x)&1 == 0
		}
		for k := range d.np {
			if d.p[k] == x {
				return true
			}
		}
		return false
	}
	for p := range ci.norb {
		f, t := 2*p+from, 2*p+1-from
		if !occ(f) || occ(t) {
			continue
		}
		mid := ci.remove(d, f)
		e := ci.create(mid, t)
		emit(e, parity(ci.pos(d, f)+ci.pos(mid, t)))
	}
}

// SpinSquared is <x|S^2|x> / <x|x> over the rows of the space. S^2 connects a row only
// to rows of the same spatial occupation, which the space always contains in its Ms
// sector (unless a free-particle or symmetry filter split them, which it cannot: both
// depend on spatial orbitals only). S^2 = S- S+ + Sz^2 + Sz.
func (ci *CI) SpinSquared(x []float64) float64 {
	n := ci.Size()
	if len(x) != n {
		panic(fmt.Sprintf("khci: SpinSquared: vector length %d, space %d", len(x), n))
	}
	sz := float64(ci.sp.opt.TwoMs) / 2
	var num, den float64
	part := make([]float64, n)
	parallel.HeavyRows(n, func(c int) {
		if x[c] == 0 {
			return
		}
		d := ci.rep(c)
		sc := float64(ci.sign[c])
		// S- S+ |c>: images are canonical determinants; <r|image> = sign_r * coef
		acc := map[int]float64{}
		ci.shiftSpin(d, 1, func(e det, s1 float64) {
			ci.shiftSpin(e, 0, func(g det, s2 float64) {
				r, ok := ci.index(g)
				if !ok {
					panic("khci: S^2 left the space; spin-flip partners must share its rows")
				}
				acc[r] += s1 * s2 * sc * float64(ci.sign[r])
			})
		})
		var v float64
		for r, a := range acc {
			v += x[r] * a * x[c]
		}
		part[c] = v + (sz*sz+sz)*x[c]*x[c]
	})
	for c := range n {
		num += part[c]
		den += x[c] * x[c]
	}
	if den == 0 {
		return math.NaN()
	}
	return num / den
}
