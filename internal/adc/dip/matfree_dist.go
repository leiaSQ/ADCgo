package dip

import "github.com/leiaSQ/ADCgo/internal/adc/backend"

// matfree_dist.go — matrix-free 3h1p↔3h1p satellite apply under the row-partitioned (-mgpu)
// backend. Composes the matrix-free satellite region with the distributed Mode-B solver so a
// whole-band DIP sector fits a multi-device node: the dense main/coupling blocks and the Krylov
// panels stay partitioned across the devices (distBackend.GemmMat), and the satellite region —
// the multi-TB memory hog that made -mgpu materialize densely — is recomputed instead of stored.
//
// Because the per-scalar apply reads arbitrary input rows (a candidate column can live on any
// partition), it runs as gather-apply-scatter: the full input panel is gathered to host
// (backend.Download), the satellite contribution is recomputed on the host with the same
// per-scalar kernel the single-node path uses (satScalarPlan.applyHost), and the result is
// scatter-added back into the partitioned output (PanelScatterAdd.AddPanel). This removes the
// resident-operator ceiling (the point) with the satellite compute on the host; a per-partition
// on-device apply (each device recomputing only its output band) is the performance follow-up.
// TestSatelliteMatFreeDistributedEqualsDense validates it over gonum sub-backends.

// newSatelliteMatFreeDistributed builds the gather-apply-scatter satellite applier over a
// row-partitioned backend. The plan is backend-independent (built from the space + block
// physics); only the gather (Download) and scatter (AddPanel) touch the distribution.
func (mx *Matrix) newSatelliteMatFreeDistributed(pg backend.PanelScatterAdd) matFreePart {
	plan := mx.buildSatScalarPlan()
	n := mx.sp.Size()
	apply := func(in, out backend.BlockView) {
		cols := in.Cols
		xfull := mx.be.Download(in.V) // full n×cols column-major host panel
		yfull := make([]float64, n*cols)
		plan.applyHost(xfull, yfull, cols, n, n)
		pg.AddPanel(out.V, yfull)
	}
	return matFreePart{apply: apply, release: func() {}}
}
