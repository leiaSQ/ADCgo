package sigma

import (
	"fmt"
	"math/bits"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/khci"
)

// packMap links rows of a khci.Space to elements of one spin-block tensor: row[i] of the
// packed vector is sign[i] times element elem[i] of the tensor (column-major over the
// spatial modes: particles first, then holes, as adcgen orders an amplitude's indices).
type packMap struct {
	elem []int64
	row  []int32
	sign []int8
	dev  backend.TensorMap // the resident copy on a TensorKernels backend (kernels.resident)
}

// patternMs is 2·Ms of the states a spin pattern (particles then holes) describes: the
// particles add their spin, the holes remove theirs (khci's TwoMs convention).
func patternMs(spin string, nPart int) int {
	s := 0
	for m, c := range spin {
		v := 1
		if c == 'b' {
			v = -1
		}
		if m < nPart {
			s += v
		} else {
			s -= v
		}
	}
	return s
}

// orb is a spatial orbital and its spin-orbital index (the sort key of the row phase).
type orb struct {
	spatial int
	so      int
}

// rowOrbs returns a row's particles (spin orbitals relative to the virtual block) and
// holes, each ascending: the order of the row's precursor c†a1 c†a2 c_i1 ... c_in.
func rowOrbs(sp *khci.Space, r int, pbuf []int) (parts, holes []orb) {
	for _, a := range sp.PartSO(r, pbuf[:0]) {
		parts = append(parts, orb{a >> 1, a})
	}
	for q := sp.HoleMask(r); q != 0; q &= q - 1 {
		p := bits.TrailingZeros64(q)
		holes = append(holes, orb{p >> 1, p})
	}
	return parts, holes
}

// inversions is the permutation parity (±1) that sorts the spin orbitals into ascending
// order: the sign relating the tensor element, whose modes hold them in this order, to
// the row amplitude.
func inversions(seq []orb) int {
	n := 0
	for i := range seq {
		for j := i + 1; j < len(seq); j++ {
			if seq[i].so > seq[j].so {
				n++
			}
		}
	}
	if n%2 == 1 {
		return -1
	}
	return 1
}

// buildPackMap maps every row of class nPart (particles) whose spins fit the pattern.
// all selects every element the row reaches through the antisymmetry of same-spin modes
// (what unpacking into a full spin-block tensor needs); otherwise only the element with
// each same-spin group in ascending order (what reading σ back needs).
func buildPackMap(sp *khci.Space, nPart int, spin string, nocc, nvir int, all bool) (*packMap, error) {
	K := sp.K()
	nh := K + nPart
	if len(spin) != nPart+nh {
		return nil, fmt.Errorf("sigma: spin pattern %q for %d particles and %d holes", spin, nPart, nh)
	}
	dims := make([]int, len(spin))
	for m := range dims {
		dims[m] = nocc
		if m < nPart {
			dims[m] = nvir
		}
	}
	strides := make([]int64, len(dims))
	s := int64(1)
	for m, d := range dims {
		strides[m] = s
		s *= int64(d)
	}
	// the positions of each (particle/hole, spin) group
	var posPA, posPB, posHA, posHB []int
	for m, c := range spin {
		switch {
		case m < nPart && c == 'a':
			posPA = append(posPA, m)
		case m < nPart:
			posPB = append(posPB, m)
		case c == 'a':
			posHA = append(posHA, m)
		default:
			posHB = append(posHB, m)
		}
	}
	pm := &packMap{}
	pbuf := make([]int, 0, 4)
	seqP := make([]orb, nPart)
	seqH := make([]orb, nh)
	for r := range sp.Size() {
		if sp.Class(r) != nh {
			continue
		}
		parts, holes := rowOrbs(sp, r, pbuf)
		var pa, pb, ha, hb []orb
		for _, o := range parts {
			if o.so&1 == 0 {
				pa = append(pa, o)
			} else {
				pb = append(pb, o)
			}
		}
		for _, o := range holes {
			if o.so&1 == 0 {
				ha = append(ha, o)
			} else {
				hb = append(hb, o)
			}
		}
		if len(pa) != len(posPA) || len(pb) != len(posPB) || len(ha) != len(posHA) || len(hb) != len(posHB) {
			continue
		}
		groups := [4][]orb{pa, pb, ha, hb}
		pos := [4][]int{posPA, posPB, posHA, posHB}
		emit := func() {
			var e int64
			for g := range 4 {
				for i, m := range pos[g] {
					o := groups[g][i]
					e += int64(o.spatial) * strides[m]
					if m < nPart {
						seqP[m] = o
					} else {
						seqH[m-nPart] = o
					}
				}
			}
			pm.elem = append(pm.elem, e)
			pm.row = append(pm.row, int32(r))
			pm.sign = append(pm.sign, int8(inversions(seqP)*inversions(seqH)))
		}
		if !all {
			emit()
			continue
		}
		permuteGroups(groups[:], 0, emit)
	}
	return pm, nil
}

// permuteGroups calls emit once for every combination of orderings of the groups
// (Heap's algorithm per group, nested), leaving the groups in their original order.
func permuteGroups(groups [][]orb, g int, emit func()) {
	if g == len(groups) {
		emit()
		return
	}
	a := groups[g]
	n := len(a)
	if n <= 1 {
		permuteGroups(groups, g+1, emit)
		return
	}
	c := make([]int, n)
	permuteGroups(groups, g+1, emit)
	for i := 0; i < n; {
		if c[i] < i {
			if i%2 == 0 {
				a[0], a[i] = a[i], a[0]
			} else {
				a[c[i]], a[i] = a[i], a[c[i]]
			}
			permuteGroups(groups, g+1, emit)
			c[i]++
			i = 0
		} else {
			c[i] = 0
			i++
		}
	}
	// Heap's algorithm leaves an odd-length group permuted; restore ascending order
	sortOrbs(a)
}

func sortOrbs(a []orb) {
	for i := 1; i < len(a); i++ {
		for j := i; j > 0 && a[j].so < a[j-1].so; j-- {
			a[j], a[j-1] = a[j-1], a[j]
		}
	}
}
