package selfenergy

// satspace.go — the satellite configuration spaces the all-order (Σ(∞)) resolvent lives on.
// Ported from ../ADC/self_energy/constanti/constanti/masop.f STATE (lines 348-416).
//
// One builder serves both blocks, exactly as the reference does — the 2h1p (ionization) and
// 2p1h (affinity) spaces differ ONLY in which orbital set supplies the pair (k,l) and which
// supplies the single index j:
//
//	IAB = 1 (2h1p): k,l occupied (two holes), j virtual (one particle)
//	IAB = 2 (2p1h): k,l virtual  (two particles), j occupied (one hole)
//
// The pair is restricted to k's *list position* <= l's within one irrep (masop.f:389
// `IF(KSYM.EQ.LSYM) MAXK = NL`), and the config is kept only when
// sym(j) ⊗ sym(k) ⊗ sym(l) == the target irrep. Each spatial config carries two intermediate-spin
// functions (S=0 and S=1) unless k == l, in which case only S=0 exists (masop.f:392).
//
// The reference bit-packs (j,k,l,maxS) into a REAL*8 (pack8/upk8s); we just use a struct.

// iab distinguishes the two satellite blocks. The values match the reference's IAB so the
// ported formulas can be read against the Fortran directly.
type iab int

const (
	iab2h1p iab = 1 // ionization block
	iab2p1h iab = 2 // affinity block
)

// other is the reference's IBA = 3-IAB: the complementary space, from which j is drawn.
func (b iab) other() iab { return 3 - b }

// satConf is one spatial satellite configuration |j k l>, with maxS spin functions.
// j, k, l are absolute 0-based orbital indices.
type satConf struct {
	j, k, l int
	maxS    int // 2, or 1 when k == l
	off     int // index of this config's first spin function in the spin-resolved space
}

// satSpace is one (irrep, block) satellite space, spin-resolved.
type satSpace struct {
	blk   iab
	sym   int       // target irrep
	confs []satConf // spatial configurations, in the reference's enumeration order
	dim   int       // spin-resolved dimension (the reference's NAB21 / NDIM)
}

// pairSet returns the orbital list a block draws its (k,l) pair from, per irrep.
func (e *engine) pairSet(b iab) [][]int {
	if b == iab2h1p {
		return e.occs // two holes
	}
	return e.virs // two particles
}

// buildSatSpace enumerates the satellite configurations of one irrep for one block.
//
// The enumeration order follows the reference: LSYM outer over irreps, KSYM <= LSYM, then JSYM;
// inside a block, l over its irrep's list, k over the same list up to l's position when the two
// irreps coincide, and j innermost. Order matters only for reproducing the reference's sparse
// traversal (and hence its floating-point accumulation order) — the density itself is invariant.
func (e *engine) buildSatSpace(b iab, sym int) *satSpace {
	pair := e.pairSet(b)
	single := e.pairSet(b.other())

	sp := &satSpace{blk: b, sym: sym}
	for lSym := range e.nsym {
		for kSym := 0; kSym <= lSym; kSym++ {
			for jSym := range e.nsym {
				if jSym^kSym^lSym != sym {
					continue
				}
				for li, l := range pair[lSym] {
					// k runs the same list only up to l's position when both are in one irrep.
					kMax := len(pair[kSym])
					if kSym == lSym {
						kMax = li + 1
					}
					for ki := 0; ki < kMax; ki++ {
						k := pair[kSym][ki]
						maxS := 2
						if k == l {
							maxS = 1
						}
						for _, j := range single[jSym] {
							sp.confs = append(sp.confs, satConf{j: j, k: k, l: l, maxS: maxS, off: sp.dim})
							sp.dim += maxS
						}
					}
				}
			}
		}
	}
	return sp
}
