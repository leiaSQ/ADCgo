package fano

import (
	"fmt"
	"math"
	"slices"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
)

// adapted.go — scheme B, the partition over ADAPTED intermediate states (ADC22.pdf
// §III B 2, after Averbukh & Cederbaum 2005).
//
// Scheme A sorts configurations by their holes, which presumes that a hole pair is
// either one-site or two-site. When the occupied orbitals are shared between subunits —
// the g/u pairs of a homonuclear dimer, the partly delocalized outer valence of a water
// dimer — a 2h1p configuration is neither, and no hole rule can put it in the right
// subspace. Scheme B rotates first: the 2h1p configurations sharing a particle orbital
// form a small block of the secular matrix, whose eigenvectors (the adapted states) mirror
// the correlated dication states with an extra electron in that orbital. Each adapted
// state is then classified on its own two-hole character, and Q and P are index subsets
// of the ROTATED basis.
//
// The rotation T is block diagonal and orthogonal: identity on every row outside the
// blocks, U_b on block b. The secular matrix in the adapted basis is T^T M T; its
// diagonal on block b is U_b's eigenvalues, and QMQ, PMP and the coupling are sub-blocks
// of it as in scheme A. Nothing of M is formed: every product is one parent mat-vec with
// T applied before and T^T after, which is the paper's "one full multiplication and two
// projections" per step.
//
// The adapted state of block b occupies the parent row positions of that block, in
// eigenvalue order, so the class bands (1h | 2h1p | 3h2p) keep their positions and
// everything that reads a row's class from its hole count still works.

// AdaptedBlock is one group of parent rows rotated together.
type AdaptedBlock struct {
	Rows   []int       // parent rows, ascending
	U      backend.Mat // len(Rows) x len(Rows); column j is adapted state j over Rows
	Values []float64   // the block's eigenvalues, ascending: diag of T^T M T on Rows
}

// Adapted is the basis change T. With an inner transform set (Compose), T = Inner * B
// where B is this Adapted's own block rotation: B acts first in ToConfig.
type Adapted struct {
	n      int
	blocks []AdaptedBlock
	outer  *Adapted // applied after the blocks in ToConfig (before them in ToAdapted)
}

// Compose returns the transform outer * inner: ToConfig applies inner's blocks, then
// outer's. Both must be over the same dimension. The composition is orthogonal because
// both factors are; any orthogonal change of the configuration basis leaves the Fano
// problem exact, so scheme B may rotate the particle index (outer) before diagonalizing
// the particle blocks (inner).
func Compose(outer, inner *Adapted) (*Adapted, error) {
	if outer.n != inner.n {
		return nil, fmt.Errorf("fano: composing transforms of sizes %d and %d", outer.n, inner.n)
	}
	if inner.outer != nil {
		return nil, fmt.Errorf("fano: the inner transform is already a composition")
	}
	return &Adapted{n: inner.n, blocks: inner.blocks, outer: outer}, nil
}

// NewAdapted validates the blocks: disjoint rows inside [0, n), square orthogonal U.
func NewAdapted(n int, blocks []AdaptedBlock) (*Adapted, error) {
	seen := make([]bool, n)
	for b, bl := range blocks {
		m := len(bl.Rows)
		if bl.U.Rows != m || bl.U.Cols != m || len(bl.Values) != m {
			return nil, fmt.Errorf("fano: adapted block %d: %d rows, U %dx%d, %d values",
				b, m, bl.U.Rows, bl.U.Cols, len(bl.Values))
		}
		if !slices.IsSorted(bl.Rows) {
			return nil, fmt.Errorf("fano: adapted block %d rows are not ascending", b)
		}
		for _, r := range bl.Rows {
			if r < 0 || r >= n || seen[r] {
				return nil, fmt.Errorf("fano: adapted block %d: row %d out of range or in two blocks", b, r)
			}
			seen[r] = true
		}
		for i := range m {
			for j := range m {
				var s float64
				for k := range m {
					s += bl.U.At(k, i) * bl.U.At(k, j)
				}
				want := 0.0
				if i == j {
					want = 1
				}
				if math.Abs(s-want) > 1e-10 {
					return nil, fmt.Errorf("fano: adapted block %d is not orthogonal (U^T U)[%d,%d] = %g",
						b, i, j, s)
				}
			}
		}
	}
	return &Adapted{n: n, blocks: blocks}, nil
}

// Size is the dimension of the parent space.
func (a *Adapted) Size() int { return a.n }

// Blocks returns the blocks.
func (a *Adapted) Blocks() []AdaptedBlock { return a.blocks }

// ToConfig writes y = T x (adapted coordinates to configuration coordinates).
func (a *Adapted) ToConfig(x, y []float64) {
	if a.outer != nil {
		tmp := make([]float64, len(x))
		a.blockToConfig(x, tmp)
		a.outer.ToConfig(tmp, y)
		return
	}
	a.blockToConfig(x, y)
}

func (a *Adapted) blockToConfig(x, y []float64) {
	copy(y, x)
	for _, bl := range a.blocks {
		for i, ri := range bl.Rows {
			var s float64
			for j, rj := range bl.Rows {
				s += bl.U.At(i, j) * x[rj]
			}
			y[ri] = s
		}
	}
}

// ToAdapted writes x = T^T y.
func (a *Adapted) ToAdapted(y, x []float64) {
	if a.outer != nil {
		tmp := make([]float64, len(y))
		a.outer.ToAdapted(y, tmp)
		a.blockToAdapted(tmp, x)
		return
	}
	a.blockToAdapted(y, x)
}

func (a *Adapted) blockToAdapted(y, x []float64) {
	copy(x, y)
	for _, bl := range a.blocks {
		for j, rj := range bl.Rows {
			var s float64
			for i, ri := range bl.Rows {
				s += bl.U.At(i, j) * y[ri]
			}
			x[rj] = s
		}
	}
}

// AdaptedOperator is T^T M T over the full parent dimension: a lanczos.Operator and a
// coupling Applier. The parent is applied on the backend; T and T^T on the host.
type AdaptedOperator struct {
	parent lanczos.Operator
	be     backend.Backend
	ad     *Adapted
	diag   []float64
}

// NewAdaptedOperator wraps the parent operator. parentDiag is the parent's diagonal;
// the adapted diagonal is it with each block's eigenvalues in place of the block rows.
func NewAdaptedOperator(parent lanczos.Operator, be backend.Backend, ad *Adapted,
	parentDiag []float64) (*AdaptedOperator, error) {
	if parent.Size() != ad.Size() {
		return nil, fmt.Errorf("fano: adapted basis of size %d for an operator of size %d",
			ad.Size(), parent.Size())
	}
	if len(parentDiag) != ad.Size() {
		return nil, fmt.Errorf("fano: parent diagonal has %d entries, want %d", len(parentDiag), ad.Size())
	}
	d := slices.Clone(parentDiag)
	for _, bl := range ad.blocks {
		for j, r := range bl.Rows {
			d[r] = bl.Values[j]
		}
	}
	return &AdaptedOperator{parent: parent, be: be, ad: ad, diag: d}, nil
}

func (o *AdaptedOperator) Size() int          { return o.ad.Size() }
func (o *AdaptedOperator) MainBlockSize() int { return o.parent.MainBlockSize() }

// Transform is the basis change T.
func (o *AdaptedOperator) Transform() *Adapted { return o.ad }

// DiagonalHost is the diagonal of T^T M T.
func (o *AdaptedOperator) DiagonalHost() []float64 { return o.diag }

// ApplyFull computes out = T^T M T in.
func (o *AdaptedOperator) ApplyFull(out, in backend.Vector) {
	n := o.Size()
	x := o.be.Download(in)
	y := make([]float64, n)
	o.ad.ToConfig(x[:n], y)
	yin := o.be.Upload(y)
	yout := o.be.Alloc(n)
	o.parent.ApplyFull(yout, yin)
	z := o.be.Download(yout)
	o.be.Free(yin)
	o.be.Free(yout)
	o.ad.ToAdapted(z, x[:n])
	tmp := o.be.Upload(x[:n])
	o.be.Copy(out, tmp)
	o.be.Free(tmp)
}

// ApplyBlock applies T^T M T to every column of in. The parent sees one block, so its
// assembled or matrix-free blocks are streamed once per block as usual.
func (o *AdaptedOperator) ApplyBlock(out, in backend.BlockView) {
	n, nc := o.Size(), in.Cols
	host := o.be.Download(in.V)
	buf := make([]float64, n*nc)
	for c := range nc {
		o.ad.ToConfig(host[c*in.Ld:c*in.Ld+n], buf[c*n:(c+1)*n])
	}
	pin := o.be.Upload(buf)
	pout := o.be.Alloc(n * nc)
	o.parent.ApplyBlock(backend.BlockView{V: pout, Rows: n, Cols: nc, Ld: n},
		backend.BlockView{V: pin, Rows: n, Cols: nc, Ld: n})
	res := o.be.Download(pout)
	o.be.Free(pin)
	o.be.Free(pout)
	outHost := o.be.Download(out.V)
	for c := range nc {
		o.ad.ToAdapted(res[c*n:(c+1)*n], outHost[c*out.Ld:c*out.Ld+n])
	}
	tmp := o.be.Upload(outHost)
	o.be.Copy(out.V, tmp)
	o.be.Free(tmp)
}

// SubOperator restricts a full-dimension operator to a subset of its rows (Q or P of a
// scheme B partition): embed, apply, gather. It satisfies lanczos.PreconOperator.
type SubOperator struct {
	full lanczos.Operator
	be   backend.Backend
	rows []int
	main int
	diag []float64
}

// NewSubOperator builds the restriction. fullDiag is the full operator's diagonal; main
// is the number of leading rows of the subset that belong to the main (1h) class.
func NewSubOperator(full lanczos.Operator, be backend.Backend, rows []int, main int,
	fullDiag []float64) *SubOperator {
	d := make([]float64, len(rows))
	for i, r := range rows {
		d[i] = fullDiag[r]
	}
	return &SubOperator{full: full, be: be, rows: rows, main: main, diag: d}
}

func (s *SubOperator) Size() int          { return len(s.rows) }
func (s *SubOperator) MainBlockSize() int { return s.main }

// Diagonal uploads the restricted diagonal (the Davidson preconditioner).
func (s *SubOperator) Diagonal(be backend.Backend) backend.Vector { return be.Upload(s.diag) }

// ApplyFull computes out = (S^T A S) in for the row subset S.
func (s *SubOperator) ApplyFull(out, in backend.Vector) {
	n := s.full.Size()
	x := s.be.Download(in)
	emb := make([]float64, n)
	for i, r := range s.rows {
		emb[r] = x[i]
	}
	ein := s.be.Upload(emb)
	eout := s.be.Alloc(n)
	s.full.ApplyFull(eout, ein)
	y := s.be.Download(eout)
	s.be.Free(ein)
	s.be.Free(eout)
	g := make([]float64, len(s.rows))
	for i, r := range s.rows {
		g[i] = y[r]
	}
	tmp := s.be.Upload(g)
	s.be.Copy(out, tmp)
	s.be.Free(tmp)
}

// ApplyBlock applies the restriction to every column of in.
func (s *SubOperator) ApplyBlock(out, in backend.BlockView) {
	n, nc := s.full.Size(), in.Cols
	host := s.be.Download(in.V)
	emb := make([]float64, n*nc)
	for c := range nc {
		for i, r := range s.rows {
			emb[c*n+r] = host[c*in.Ld+i]
		}
	}
	ein := s.be.Upload(emb)
	eout := s.be.Alloc(n * nc)
	s.full.ApplyBlock(backend.BlockView{V: eout, Rows: n, Cols: nc, Ld: n},
		backend.BlockView{V: ein, Rows: n, Cols: nc, Ld: n})
	res := s.be.Download(eout)
	s.be.Free(ein)
	s.be.Free(eout)
	outHost := s.be.Download(out.V)
	for c := range nc {
		for i, r := range s.rows {
			outHost[c*out.Ld+i] = res[c*n+r]
		}
	}
	tmp := s.be.Upload(outHost)
	s.be.Copy(out.V, tmp)
	s.be.Free(tmp)
}

// BuildMatrix forms the restricted matrix densely, one unit vector per column. For the
// dense validation path on small spaces only.
func (s *SubOperator) BuildMatrix() backend.Mat {
	m := len(s.rows)
	out := backend.NewMat(m, m)
	e := make([]float64, m)
	for j := range m {
		clear(e)
		e[j] = 1
		in := s.be.Upload(e)
		y := s.be.Alloc(m)
		s.ApplyFull(y, in)
		col := s.be.Download(y)
		s.be.Free(in)
		s.be.Free(y)
		for i := range m {
			out.Set(i, j, col[i])
		}
	}
	return out
}

// Class is a row's subspace under a partition stated per row.
type Class int8

const (
	ClassQ Class = iota // bound
	ClassP              // continuum
	ClassX              // excluded from both
)

// classSelector describes a per-row partition for the run log; it decides nothing.
type classSelector struct{ desc string }

func (c classSelector) Bound([]int) bool { panic("fano: a per-row partition has no hole rule") }
func (c classSelector) String() string   { return c.desc }

// NewPartitionFromClasses builds a Partition from an explicit class per row — scheme B,
// where the rows are adapted states and no hole predicate can decide them. holes reports
// each row's hole count for the census (the adapted states keep their parent rows'
// classes). main is the number of leading main-class rows.
func NewPartitionFromClasses(class []Class, main int, holes func(row int) int, desc string) *Partition {
	n := len(class)
	pt := &Partition{sel: classSelector{desc}, census: map[int][3]int{}}
	pt.qOf = make([]int, n)
	pt.pOf = make([]int, n)
	for r, c := range class {
		pt.qOf[r], pt.pOf[r] = -1, -1
		cen := pt.census[holes(r)]
		switch c {
		case ClassQ:
			pt.qOf[r] = len(pt.Q)
			pt.Q = append(pt.Q, r)
			if r < main {
				pt.QMain++
			} else {
				pt.QSat++
			}
			cen[0]++
		case ClassP:
			pt.pOf[r] = len(pt.P)
			pt.P = append(pt.P, r)
			cen[1]++
		default:
			pt.X++
			cen[2]++
		}
		pt.census[holes(r)] = cen
	}
	return pt
}

// Diagonal uploads the diagonal of T^T M T (the Davidson preconditioner).
func (o *AdaptedOperator) Diagonal(be backend.Backend) backend.Vector { return be.Upload(o.diag) }

// BuildMatrix forms T^T M T densely, one unit vector per column (small spaces only).
func (o *AdaptedOperator) BuildMatrix() backend.Mat {
	all := make([]int, o.Size())
	for i := range all {
		all[i] = i
	}
	return NewSubOperator(o, o.be, all, o.MainBlockSize(), o.diag).BuildMatrix()
}

// Backend is the backend the parent operator runs on.
func (o *AdaptedOperator) Backend() backend.Backend { return o.be }

// Release is a no-op: the parent operator belongs to whoever built it, and it serves
// the QMQ, PMP and coupling stages alike.
func (o *AdaptedOperator) Release() {}

// Backend is the backend the wrapped operator runs on.
func (s *SubOperator) Backend() backend.Backend { return s.be }

// Release is a no-op; see AdaptedOperator.Release.
func (s *SubOperator) Release() {}
