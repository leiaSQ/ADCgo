package fano

// decompose.go — the ordering decomposition of a decay amplitude.
//
// A multi-step decay amplitude reaches P through several routes that the
// Kramers-Heisenberg picture calls time orderings: straight from the main configuration,
// through excited intermediates of the (k+1)h1p class, through (k+2)h2p relaxation. In
// the Fano picture they are the blocks of the decaying state, Phi = sum_B Phi_B with Phi_B the components
// of Phi on the Q rows of block B, and since the coupling is linear in Phi,
//
//	g = P(H - E) Phi = sum_B g_B,   g_B = P(H - E) Phi_B,
//
// exactly (the -E term vanishes on P under scheme A, see Coupling). Imaging each g_B gives
// the width that route alone would carry; imaging g_B + g_B' and subtracting both gives
// their interference. The decomposition itself is exact; only the imaging of each piece
// carries the Stieltjes error.

import (
	"fmt"
	"slices"

	"github.com/leiaSQ/ADCgo/backend"
)

// ClassBlocks groups the rows of sp by excitation class, named as ClassWeights names them
// ("4h", "5h1p", "6h2p", ...).
func ClassBlocks(sp Space) map[string][]int {
	if sp.Size() == 0 || sp.MainBlockSize() == 0 {
		return nil
	}
	k := len(sp.Holes(0, nil))
	out := map[string][]int{}
	buf := make([]int, 0, 8)
	for r := range sp.Size() {
		n := len(sp.Holes(r, buf[:0]))
		name := fmt.Sprintf("%dh", n)
		if n > k {
			name = fmt.Sprintf("%dh%dp", n, n-k)
		}
		out[name] = append(out[name], r)
	}
	return out
}

// Decompose returns g_B = P(H - E)Phi_B for every block of Q rows (indices into the
// restricted Q space), from one parent application per block. The blocks must be
// disjoint; rows of Phi in no block are left out of every g_B, so sum_B g_B = g only when
// the blocks cover Q (ClassBlocks does).
func Decompose(mx Applier, part *Partition, phi Discrete, blocks map[string][]int, be backend.Backend) (map[string][]float64, error) {
	seen := make([]bool, len(phi.Vec))
	out := map[string][]float64{}
	names := make([]string, 0, len(blocks))
	for name := range blocks {
		names = append(names, name)
	}
	slices.Sort(names)
	for _, name := range names {
		piece := make([]float64, len(phi.Vec))
		for _, r := range blocks[name] {
			if r < 0 || r >= len(phi.Vec) {
				return nil, fmt.Errorf("fano: block %q row %d outside Q (%d rows)", name, r, len(phi.Vec))
			}
			if seen[r] {
				return nil, fmt.Errorf("fano: Q row %d is in more than one block", r)
			}
			seen[r] = true
			piece[r] = phi.Vec[r]
		}
		gb, err := Coupling(mx, part, Discrete{Energy: phi.Energy, Vec: piece, Weight: phi.Weight,
			Root: phi.Root, Rows: phi.Rows}, be)
		if err != nil {
			return nil, fmt.Errorf("fano: block %q: %w", name, err)
		}
		out[name] = gb
	}
	return out, nil
}
