package fano

import (
	"fmt"
	"slices"
	"strings"

	"github.com/leiaSQ/ADCgo/internal/adc/parallel"
)

// partition.go — the Feshbach split of a configuration space into the bound subspace
// Q and the continuum subspace P.

// Partition assigns every row of a Space to Q or P. The two are disjoint and together
// cover every row, so they are the index-space realization of Feshbach projectors
// satisfying Q + P = 1 and QP = 0 exactly — no overlap correction anywhere downstream.
//
// Q and P hold parent row indices in ascending order, which is what Space.Restrict
// consumes and what keeps a restricted space's class bands contiguous.
type Partition struct {
	Q []int // parent rows of the bound subspace, ascending
	P []int // parent rows of the continuum subspace, ascending

	qOf []int // parent row -> position in Q, or -1
	pOf []int // parent row -> position in P, or -1

	// Per-class Q counts, for the run log and for the sanity check that matters most:
	// a Q that contains no main-class configuration has no discrete state to decay.
	QMain, QSat int

	sel Selector

	// census[c] counts (Q, P) configurations of hole count c, for the run log.
	census map[int][2]int
}

// NewPartition classifies every row of sp with sel.
//
// Parallel over row chunks with per-worker lists concatenated in worker order, so the
// result is ascending and identical whatever the worker count. The classification
// itself is cheap, but the row count is the matrix dimension — 10^6 and up for the
// sectors ADC(2,2) is built for — and a serial pass over it with a per-row Holes call
// is not free.
//
// A ConfigSelector (net-charge rules) is evaluated on holes and particles and needs a
// ParticleSpace; handing it a hole-only space is a programming error and panics. Every
// other Selector takes the hole-only path, unchanged.
func NewPartition(sp Space, sel Selector) *Partition {
	n := sp.Size()
	main := sp.MainBlockSize()
	cs, byConfig := sel.(ConfigSelector)
	var psp ParticleSpace
	if byConfig {
		var ok bool
		if psp, ok = sp.(ParticleSpace); !ok {
			panic(fmt.Sprintf("fano: selector %s needs particles, but the space (%T) "+
				"does not provide them", sel, sp))
		}
	}
	W := parallel.ChunkWorkers(n)
	qs := make([][]int32, W)
	ps := make([][]int32, W)
	cens := make([]map[int][2]int, W)
	parallel.Chunks(n, W, func(w, lo, hi int) {
		holes := make([]int, 0, 8)
		parts := make([]int, 0, 2)
		q := make([]int32, 0, (hi-lo)/8+1)
		p := make([]int32, 0, hi-lo)
		cen := map[int][2]int{}
		for r := lo; r < hi; r++ {
			holes = sp.Holes(r, holes[:0])
			var bound bool
			if byConfig {
				parts = psp.Particles(r, parts[:0])
				bound = cs.BoundConfig(holes, parts)
			} else {
				bound = sel.Bound(holes)
			}
			c := cen[len(holes)]
			if bound {
				q = append(q, int32(r))
				c[0]++
			} else {
				p = append(p, int32(r))
				c[1]++
			}
			cen[len(holes)] = c
		}
		qs[w], ps[w] = q, p
		cens[w] = cen
	})

	pt := &Partition{sel: sel, census: map[int][2]int{}}
	for _, cen := range cens {
		for c, v := range cen {
			t := pt.census[c]
			pt.census[c] = [2]int{t[0] + v[0], t[1] + v[1]}
		}
	}
	nq, np := 0, 0
	for w := range W {
		nq += len(qs[w])
		np += len(ps[w])
	}
	pt.Q = make([]int, 0, nq)
	pt.P = make([]int, 0, np)
	for w := range W {
		for _, r := range qs[w] {
			pt.Q = append(pt.Q, int(r))
		}
		for _, r := range ps[w] {
			pt.P = append(pt.P, int(r))
		}
	}

	pt.qOf = make([]int, n)
	pt.pOf = make([]int, n)
	for i := range pt.qOf {
		pt.qOf[i], pt.pOf[i] = -1, -1
	}
	for i, r := range pt.Q {
		pt.qOf[r] = i
		if r < main {
			pt.QMain++
		} else {
			pt.QSat++
		}
	}
	for i, r := range pt.P {
		pt.pOf[r] = i
	}
	return pt
}

// QSize and PSize are the dimensions of the two sub-blocks.
func (p *Partition) QSize() int { return len(p.Q) }
func (p *Partition) PSize() int { return len(p.P) }

// QIndex and PIndex map a parent row to its position in Q resp. P, or -1 when the row
// is in the other subspace.
func (p *Partition) QIndex(row int) int { return p.qOf[row] }
func (p *Partition) PIndex(row int) int { return p.pOf[row] }

// Validate checks the partition invariants. They are cheap to verify and expensive to
// debug downstream: a Q with no main-class configuration yields no discrete state, and
// an empty P yields no continuum and hence a zero width, both of which otherwise show
// up as a silently wrong rate rather than an error.
func (p *Partition) Validate() error {
	if len(p.Q) == 0 {
		return fmt.Errorf("fano: the bound subspace Q is empty (%s); nothing can decay", p.sel)
	}
	if len(p.P) == 0 {
		return fmt.Errorf("fano: the continuum subspace P is empty (%s); "+
			"every configuration was assigned to Q, so there are no decay channels", p.sel)
	}
	if p.QMain == 0 {
		return fmt.Errorf("fano: Q holds %d configurations but none of the main class (%s); "+
			"the discrete state is selected by its leading main configuration, so there is "+
			"none to select", len(p.Q), p.sel)
	}
	return nil
}

// EmbedQ scatters a Q-subspace vector into parent coordinates, zero outside Q. out is
// reused when it is long enough; pass nil to allocate.
//
// This is the whole of the coupling machinery: applying the PARENT matrix to the
// embedded vector gives (M Phi)_j = sum_{i in Q} M_ji Phi_i for every parent row j,
// because Phi is zero off Q. Reading the P rows of the result off with GatherP is then
// the exact Q-to-P coupling, with no projector and no new element evaluation.
func (p *Partition) EmbedQ(q []float64, out []float64) []float64 {
	if len(q) != len(p.Q) {
		panic(fmt.Sprintf("fano: EmbedQ got a vector of length %d, want QSize() = %d", len(q), len(p.Q)))
	}
	n := len(p.qOf)
	if cap(out) < n {
		out = make([]float64, n)
	} else {
		out = out[:n]
		clear(out)
	}
	for i, r := range p.Q {
		out[r] = q[i]
	}
	return out
}

// GatherP reads the P rows out of a parent-coordinate vector. out is reused when long
// enough; pass nil to allocate.
func (p *Partition) GatherP(full []float64, out []float64) []float64 {
	if len(full) != len(p.qOf) {
		panic(fmt.Sprintf("fano: GatherP got a vector of length %d, want Size() = %d",
			len(full), len(p.qOf)))
	}
	if cap(out) < len(p.P) {
		out = make([]float64, len(p.P))
	} else {
		out = out[:len(p.P)]
	}
	for i, r := range p.P {
		out[i] = full[r]
	}
	return out
}

// GatherQ is GatherP's counterpart, for reading a Q-subspace vector back out of parent
// coordinates.
func (p *Partition) GatherQ(full []float64, out []float64) []float64 {
	if len(full) != len(p.qOf) {
		panic(fmt.Sprintf("fano: GatherQ got a vector of length %d, want Size() = %d",
			len(full), len(p.qOf)))
	}
	if cap(out) < len(p.Q) {
		out = make([]float64, len(p.Q))
	} else {
		out = out[:len(p.Q)]
	}
	for i, r := range p.Q {
		out[i] = full[r]
	}
	return out
}

// Census returns the (Q, P) configuration counts per hole count.
func (p *Partition) Census() map[int][2]int {
	out := make(map[int][2]int, len(p.census))
	for k, v := range p.census {
		out[k] = v
	}
	return out
}

// CensusString formats Census for the run log, classes in ascending hole count.
func (p *Partition) CensusString() string {
	keys := make([]int, 0, len(p.census))
	for k := range p.census {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	parts := make([]string, len(keys))
	for i, k := range keys {
		parts[i] = fmt.Sprintf("%dh: Q=%d P=%d", k, p.census[k][0], p.census[k][1])
	}
	return strings.Join(parts, ", ")
}

// Selector returns the criterion this partition was built with.
func (p *Partition) Selector() Selector { return p.sel }

// String summarizes the partition for the run log.
func (p *Partition) String() string {
	return fmt.Sprintf("Q=%d (main %d, satellite %d) P=%d of %d configurations; %s",
		len(p.Q), p.QMain, p.QSat, len(p.P), len(p.qOf), p.sel)
}
