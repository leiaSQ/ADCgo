package khci

import (
	"fmt"
	"math"
	"math/bits"
	"slices"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/parallel"
)

// spin.go — S² over the rows of a space, and the projector onto one multiplicity.
//
// On canonical determinants (spin orbitals created in ascending order, α = 2p before
// β = 2p+1) a spin flip within one spatial orbital moves an electron between adjacent
// spin orbitals and so carries no fermionic sign. With S² = S₋S₊ + S_z(S_z + 1) and
// S₊ = Σ_p c†_{pα} c_{pβ} this gives
//
//	S²|D> = (S_z(S_z+1) + n_β-open)|D> + Σ_{p β-open, q α-open, p≠q} |D with p and q flipped>,
//
// where an open orbital is singly occupied. The rows carry the phase sign of Det, so
// <r|S²|c> = sign_r · sign_c · <D_r|S²|D_c>. S² keeps each spatial occupation, hence each
// class, and a space closed under it (every khci space of one Ms is; a restriction that
// splits a spatial occupation is not) has a pure-spin basis.

// spinCSR is S² as a symmetric sparse matrix over a space's rows.
type spinCSR struct {
	ptr  []int64
	col  []int32
	val  []float64
	open []int8 // singly occupied spatial orbitals of each row
}

// openShells lists a row's singly occupied spatial orbitals with their spin (0 α, 1 β),
// over the occupied block (from the hole mask) and the virtual block (from the
// particles, offset by NOcc).
func (s *Space) openShells(r int, dst [][2]int) [][2]int {
	rw := s.rows[r]
	for q := rw.holes; q != 0; q &= q - 1 {
		p := bits.TrailingZeros64(q)
		partner := p ^ 1
		if rw.holes&(1<<uint(partner)) != 0 {
			continue // both spins removed: an empty orbital
		}
		// the hole removed spin p&1; the electron left has the other spin
		dst = append(dst, [2]int{p >> 1, (p & 1) ^ 1})
	}
	var ps [2]int
	np := 0
	for _, a := range rw.p {
		if a >= 0 {
			ps[np] = int(a)
			np++
		}
	}
	for i := range np {
		a := ps[i]
		if np == 2 && ps[0]>>1 == ps[1]>>1 {
			break // a doubly filled virtual orbital
		}
		dst = append(dst, [2]int{s.opt.NOcc + a>>1, a & 1})
	}
	return dst
}

// flip returns the row with the spins of spatial orbitals p (β→α) and q (α→β) exchanged.
func (s *Space) flip(r, p, q int) (int, bool) {
	rw := s.rows[r]
	holes := rw.holes
	parts := make([]int, 0, 2)
	for _, a := range rw.p {
		if a >= 0 {
			parts = append(parts, int(a))
		}
	}
	move := func(orb, from, to int) {
		if orb < s.opt.NOcc {
			// an electron of spin `from` in an occupied orbital: its hole has spin `to`
			holes = holes&^(1<<uint(2*orb+to)) | 1<<uint(2*orb+from)
			return
		}
		v := orb - s.opt.NOcc
		for i, a := range parts {
			if a == 2*v+from {
				parts[i] = 2*v + to
			}
		}
	}
	move(p, 1, 0)
	move(q, 0, 1)
	return s.Index(holes, parts)
}

func (s *Space) spinSquare() (*spinCSR, error) {
	n := len(s.rows)
	rows := make([][]int32, n)
	vals := make([][]float64, n)
	open := make([]int8, n)
	errs := make([]error, n)
	parallel.Rows(n, func(r int) {
		sh := s.openShells(r, make([][2]int, 0, 8))
		open[r] = int8(len(sh))
		var na, nb int
		for _, o := range sh {
			if o[1] == 0 {
				na++
			} else {
				nb++
			}
		}
		sz := float64(na-nb) / 2
		_, _, sr := s.Det(r)
		cs := []int32{int32(r)}
		vs := []float64{sz*(sz+1) + float64(nb)}
		for _, p := range sh {
			if p[1] != 1 {
				continue
			}
			for _, q := range sh {
				if q[1] != 0 {
					continue
				}
				c, ok := s.flip(r, p[0], q[0])
				if !ok {
					errs[r] = fmt.Errorf("khci: S² leaves the space from row %d (a restriction split a spatial occupation?)", r)
					return
				}
				_, _, sc := s.Det(c)
				cs = append(cs, int32(c))
				vs = append(vs, sr*sc)
			}
		}
		rows[r], vals[r] = cs, vs
	})
	for _, e := range errs {
		if e != nil {
			return nil, e
		}
	}
	m := &spinCSR{ptr: make([]int64, n+1), open: open}
	for r := range n {
		m.ptr[r+1] = m.ptr[r] + int64(len(rows[r]))
	}
	m.col = make([]int32, m.ptr[n])
	m.val = make([]float64, m.ptr[n])
	for r := range n {
		copy(m.col[m.ptr[r]:], rows[r])
		copy(m.val[m.ptr[r]:], vals[r])
	}
	return m, nil
}

// apply computes dst = S²·src over one column (dst and src distinct).
func (m *spinCSR) apply(dst, src []float64) {
	parallel.Rows(len(m.ptr)-1, func(r int) {
		var v float64
		for k := m.ptr[r]; k < m.ptr[r+1]; k++ {
			v += m.val[k] * src[m.col[k]]
		}
		dst[r] = v
	})
}

// MainSpinSquare is S² over the main-class rows (a MainBlockSize() square matrix) in
// the rows' own phases.
func (s *Space) MainSpinSquare() (backend.Mat, error) {
	n := s.MainBlockSize()
	m := backend.NewMat(n, n)
	for r := range n {
		sh := s.openShells(r, nil)
		var na, nb int
		for _, o := range sh {
			if o[1] == 0 {
				na++
			} else {
				nb++
			}
		}
		sz := float64(na-nb) / 2
		m.Data[r*n+r] = sz*(sz+1) + float64(nb)
		_, _, sr := s.Det(r)
		for _, p := range sh {
			for _, q := range sh {
				if p[1] != 1 || q[1] != 0 {
					continue
				}
				c, ok := s.flip(r, p[0], q[0])
				if !ok || c >= n {
					return m, fmt.Errorf("khci: S² leaves the main class of the space (row %d)", r)
				}
				_, _, sc := s.Det(c)
				m.Data[c*n+r] += sr * sc
			}
		}
	}
	return m, nil
}

// MainSpinVectors returns the main-class vectors of total spin twoS/2 as a column-major
// Size() × b panel (zero below the main class), for lanczos.Options.StartVecs with
// Options.Block = b. b is 0 when the main class holds no such state.
func (s *Space) MainSpinVectors(twoS int) ([]float64, int, error) {
	if twoS < 0 {
		return nil, 0, fmt.Errorf("khci: 2S = %d", twoS)
	}
	m, err := s.MainSpinSquare()
	if err != nil {
		return nil, 0, err
	}
	n, size := m.Rows, s.Size()
	if n == 0 {
		return nil, 0, nil
	}
	vals, vecs := backend.Gonum{}.SymEig(m)
	want := float64(twoS) * (float64(twoS) + 2) / 4
	var cols []int
	for i, v := range vals {
		if math.Abs(v-want) < 1e-8 {
			cols = append(cols, i)
			continue
		}
		// every eigenvalue of S² is S(S+1) for a half-integer S; anything else is a bug
		x := math.Sqrt(1+4*v) - 1 // = 2S
		if math.Abs(x-math.Round(x)) > 1e-6 {
			return nil, 0, fmt.Errorf("khci: S² eigenvalue %g is no S(S+1)", v)
		}
	}
	out := make([]float64, size*len(cols))
	for j, c := range cols {
		for r := range n {
			out[j*size+r] = vecs.At(r, c)
		}
	}
	return out, len(cols), nil
}

// SpinProjector is the Löwdin projector onto total spin TwoS/2 over a space's rows,
// P = Π_{S'≠S} (S² − S'(S'+1)) / (S(S+1) − S'(S'+1)) over every other multiplicity the
// rows can hold. It is an orthogonal projector commuting with any spin-free operator, so
// P·M·P restricted to its range is the pure-spin block of M.
type SpinProjector struct {
	TwoS   int
	s2     *spinCSR
	others []float64 // S'(S'+1) of the multiplicities removed
	target float64
}

// SpinProjector builds the projector onto 2S = twoS. It fails on a space that is not
// closed under S² or whose Ms cannot reach twoS.
func (s *Space) SpinProjector(twoS int) (*SpinProjector, error) {
	if s.opt.AllMs {
		return nil, fmt.Errorf("khci: a spin projector needs a single-Ms space")
	}
	ms := s.opt.TwoMs
	if twoS < abs(ms) || (twoS-ms)%2 != 0 {
		return nil, fmt.Errorf("khci: 2S = %d is not reachable at 2Ms = %d", twoS, ms)
	}
	m, err := s.spinSquare()
	if err != nil {
		return nil, err
	}
	maxOpen := int(slices.Max(append([]int8{0}, m.open...)))
	p := &SpinProjector{TwoS: twoS, s2: m, target: ss(twoS)}
	for t := abs(ms); t <= maxOpen; t += 2 {
		if t != twoS {
			p.others = append(p.others, ss(t))
		}
	}
	return p, nil
}

func ss(twoS int) float64 { return float64(twoS) * float64(twoS+2) / 4 }

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// Apply projects the column-major panel x (rows = the space size, cols columns, leading
// dimension ld) in place.
func (p *SpinProjector) Apply(x []float64, cols, ld int) {
	n := len(p.s2.ptr) - 1
	tmp := make([]float64, n)
	for c := range cols {
		col := x[c*ld : c*ld+n]
		for _, o := range p.others {
			p.s2.apply(tmp, col)
			f := 1 / (p.target - o)
			for i := range col {
				col[i] = f * (tmp[i] - o*col[i])
			}
		}
	}
}
