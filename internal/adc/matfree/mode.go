// Package matfree holds the small, backend-agnostic policy shared by the ADC
// solvers (sip, dip) for deciding whether an operator block is assembled densely
// or applied matrix-free (recompute its elements each mat-vec instead of storing
// a dense copy). The mechanism trades resident memory for per-mat-vec recompute;
// it is what lets the dominant blocks of a large sector run on a machine that
// could not hold the materialized operator.
package matfree

// Mode selects whether the large ADC coupling/satellite blocks are assembled
// densely or applied matrix-free. Off (default) = always dense; On = always
// matrix-free; Auto = matrix-free when a block's dense size exceeds the budget.
type Mode int

const (
	Off Mode = iota
	Auto
	On
)

// Decide applies the mode to one block, given the block's dense size in bytes, the
// Auto per-block budget in bytes, and whether the backend supports a matrix-free
// path for that block. An unsupported backend always falls back to dense (the
// correct, always-available path).
func Decide(mode Mode, denseBytes, budgetBytes int64, supported bool) bool {
	if !supported {
		return false
	}
	switch mode {
	case On:
		return true
	case Auto:
		return denseBytes > budgetBytes
	default:
		return false
	}
}
