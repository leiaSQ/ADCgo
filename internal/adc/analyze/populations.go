package analyze

import (
	"math"

	"adcgo/internal/adc/backend"
	"adcgo/internal/adc/dip"
	"adcgo/internal/adc/mo"
)

// U-transform normalization constants (eqs. 3c, 3d).
var (
	sqrt2   = math.Sqrt2
	sqrt1_2 = math.Sqrt(0.5)
)

// PopEngine computes atom-resolved two-hole populations for one (symmetry, spin)
// sector via the Tarantelli U-transform (JCP 94 (1991) 523, eqs. 3a–4/8; the
// reference implementation is adc2_dip_analyzer.cpp:97-345).
//
// It precomputes the AO-pair transform U (eq. 3) and the AO overlap-pair metric
// O (eq. 4); per state it forms Y = U·c_2h and the per-AO-pair charge
// Q_pq = Y_pq·(O·Y)_pq (eq. 8), then folds those into one-site (A⁻²) and
// two-site (A⁻¹B⁻¹) atomic weights. Since Uᵀ·O·U = I, the weights sum to the
// state's pole strength / 100.
type PopEngine struct {
	nao       int
	main      int
	U         backend.Mat // naopair × main
	O         backend.Mat // naopair × naopair
	groupName []string
	aoGroup   []int // group index per AO
}

// triIdx is the packed index of the AO pair (a,b) with a >= b.
func triIdx(a, b int) int { return a*(a+1)/2 + b }

func pairIdx(p, q int) int {
	if p >= q {
		return triIdx(p, q)
	}
	return triIdx(q, p)
}

// NewPopEngine builds the U/O matrices for the sector sp using sidecar md. The
// transforms are small and host-resident, so the population step needs no
// accelerated backend.
func NewPopEngine(sp *dip.Space, md *mo.Data) *PopEngine {
	main := sp.MainBlockSize()
	nao := md.NAO
	naopair := nao * (nao + 1) / 2

	fact1, fact2 := 1.0, 1.0 // singlet
	if sp.Spin == dip.Triplet {
		fact1, fact2 = -1.0, 0.0
	}
	C := md.C

	// U (eq. 3a-3d): column c is the main config (i,j) = Occ[0],Occ[1].
	U := backend.NewMat(naopair, main)
	for c := range main {
		i, j := sp.Configs[c].Occ[0], sp.Configs[c].Occ[1]
		for p := range nao {
			for q := 0; q <= p; q++ {
				var u float64
				switch {
				case i != j && p != q:
					u = C.At(p, i)*C.At(q, j) + fact1*C.At(q, i)*C.At(p, j)
				case i != j && p == q:
					u = C.At(p, i) * C.At(p, j) * fact2
				case i == j && p != q:
					u = sqrt2 * C.At(p, i) * C.At(q, i) * fact2
				default: // i == j && p == q
					u = sqrt1_2 * C.At(p, i) * C.At(p, i) * fact2
				}
				U.Set(triIdx(p, q), c, u)
			}
		}
	}

	// O (eq. 4): symmetric overlap-pair metric.
	S := md.S
	O := backend.NewMat(naopair, naopair)
	for p := range nao {
		for q := 0; q <= p; q++ {
			kpq := triIdx(p, q)
			for r := range nao {
				for s := 0; s <= r; s++ {
					O.Set(kpq, triIdx(r, s), S.At(p, r)*S.At(q, s)+fact1*S.At(p, s)*S.At(q, r))
				}
			}
		}
	}

	return &PopEngine{nao: nao, main: main, U: U, O: O,
		groupName: md.AtomNames, aoGroup: md.AOAtom}
}

// Compute returns the atom-resolved two-hole population for a state whose 2h
// (main-space) coefficients are mainVec (length MainBlockSize).
func (pe *PopEngine) Compute(mainVec []float64) *Pop {
	Y := pe.U.MulVec(mainVec)
	OY := pe.O.MulVec(Y)

	pop := &Pop{OneSite: map[string]float64{}, TwoSite: map[string]float64{}}
	for p := range pe.nao {
		for q := 0; q <= p; q++ {
			gp, gq := pe.aoGroup[p], pe.aoGroup[q]
			if gp < 0 || gq < 0 {
				continue
			}
			k := triIdx(p, q)
			val := Y[k] * OY[k]
			if gp == gq {
				pop.OneSite[pe.groupName[gp]] += val
			} else {
				a, b := gp, gq
				if a > b {
					a, b = b, a
				}
				pop.TwoSite[pe.groupName[a]+"/"+pe.groupName[b]] += val
			}
		}
	}
	return pop
}

// Sum returns the total two-hole population (one-site + two-site), which equals
// the state's pole strength / 100 to numerical precision.
func (p *Pop) Sum() float64 {
	var s float64
	for _, v := range p.OneSite {
		s += v
	}
	for _, v := range p.TwoSite {
		s += v
	}
	return s
}
