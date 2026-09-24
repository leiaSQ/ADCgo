// lowmem.go — a limited-memory (short-recurrence) block-Lanczos driver for the ADC
// secular problem. Unlike Solve, which keeps the whole n×maxdim Krylov basis resident so
// every new block can be reorthogonalized against all of it, SolveLowMem keeps only three
// n×block panels live at a time (previous / current / next block). That is the difference
// between the ~25–36 TB basis a full-band production DIP solve would need and the ~0.4–0.6 TB
// (block = main) or a few GB (small block) it needs here. It mirrors theADCcode's DIP
// diagonalizer (../ADC/analysis/adc_diagonalizer.cpp, ../ADC/libLanczos/lanczos_engine.h).
//
// Two modes, chosen by the block width (Options.LowMemBlock):
//
//   - Mode B (block == MainBlockSize, the default; needs a SatelliteOperator): the faithful
//     theADCcode port. The start block is the full main-space unit vectors, and after the
//     first two blocks the operator's satellite↔satellite sub-block is applied (Tarantelli's
//     subspace iteration). That gate forces every later Lanczos vector to zero main-space
//     weight, so a projected eigenvector's top `main` rows equal its true main-space
//     components — recovered, without the basis, by the banded eigensolver bandSymDiagFast
//     (bandeig.go), which materializes only the 2·band top/bottom eigenvector rows. Resident
//     cost ~3·n·main, a fat-memory CPU node.
//
//   - Mode A (0 < block < main): a small start block that fits a single GPU (~3·n·block).
//     No gate; the projected matrix is small (dim = blocks·block), so it is diagonalized
//     densely with be.SymEig, and the main-space components are recovered by back-transforming
//     the retained main×dim host slice of the basis (Qmain·s). The trade-off, per
//     docs/dip_lowmem_lanczos.md, is that a small start block only reaches the states with
//     overlap in its Krylov space — the strongest lines, not a guaranteed-complete band.
//
// Both drop full reorthogonalization (only the two live blocks are reorthogonalized against),
// so both admit Lanczos ghosts, filtered afterwards by Result.Spurious (the reference's
// spur_thresh = 1e-9 main-space-weight test). The recurrence reuses the package's
// orthBlock (two-pass CGS2 + rank-revealing Gram-QR with deflation), whose R factor is the
// off-diagonal β block of the projected matrix.

package lanczos

import (
	"fmt"
	"math"
	"os"
	"time"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/parallel"
)

// block records one accepted Lanczos block: its global column offset, its size (after
// deflation), the symmetric diagonal projection α = Qᵀ·M·Q (size×size, column-major), and
// the off-diagonal β = R factor coupling it to the previous block (size×prevSize,
// row-major backend.Mat). These are assembled into the projected matrix once the run ends.
type lmBlock struct {
	off   int
	size  int
	alpha []float64   // α = Q_thisᵀ·M·Q_this (size×size, column-major)
	beta  backend.Mat // β_this = Q_thisᵀ·M·Q_prev (this.size × prev.size); zero Rows for the first block
}

// SolveLowMem runs the limited-memory block-Lanczos driver (see the file comment). It
// returns the same Result shape as Solve, with spurious (ghost) roots already dropped.
func SolveLowMem(op Operator, be backend.Backend, opts Options) Result {
	n := op.Size()
	main := op.MainBlockSize()
	opts = opts.normalize(n)
	var tm Timing
	if n == 0 || main == 0 {
		return Result{MainVecs: backend.NewMat(main, 0), Timing: tm}
	}

	// Block width: default (0) → main (Mode B); otherwise clamp to [1, main].
	b := opts.LowMemBlock
	if b <= 0 || b > main {
		b = main
	}
	sat, gateOK := op.(SatelliteOperator)
	modeB := b == main && gateOK

	// Checkpointing covers Mode B only. Mode A additionally retains hostQ — the WHOLE basis on the
	// host, for its full reorthogonalization — and serializing that is exactly what the low-memory
	// driver exists to avoid; the resumable-state argument in lowmem_checkpoint.go does not hold
	// for it. Refuse loudly rather than accept the flag and silently never write a file: a run that
	// believes it is checkpointing and is not is strictly worse than one that knows it is not.
	cp := opts.Checkpoint
	if cp != nil && cp.Path != "" && !modeB {
		why := fmt.Sprintf("block width %d != main %d (Mode A)", b, main)
		if b == main && !gateOK {
			why = "the operator does not implement SatelliteOperator, so the Mode B gate is off"
		}
		panic("lanczos: SolveLowMem checkpointing requires Mode B, but " + why +
			"; run with -lowmem-block 0 or drop -checkpoint")
	}

	// Subspace bound: MaxBlocks blocks of at most b columns, capped at MaxDim/Size.
	maxdim := min(opts.MaxDim, opts.MaxBlocks*b, n)
	maxdim = max(maxdim, b)

	// Resident panels: [prev|cur] compacted contiguously in pc (≤ 2b wide), the work panel
	// (M·Q_cur, then Q_next), and a move scratch. Plus small orthogonalization scratch.
	pcBuf := be.Alloc(n * 2 * b)
	workBuf := be.Alloc(n * b)
	tmpBuf := be.Alloc(n * b)
	abuf := be.Alloc(b * b)     // α = curᵀ·W
	gbuf := be.Alloc(b * b)     // blockOrth Gram
	pbuf := be.Alloc(2 * b * b) // cgs2 projection coefficients (≤2b × b)
	defer be.Free(pcBuf)
	defer be.Free(workBuf)
	defer be.Free(tmpBuf)
	defer be.Free(abuf)
	defer be.Free(gbuf)
	defer be.Free(pbuf)
	pc := backend.BlockView{V: pcBuf, Rows: n, Cols: 2 * b, Ld: n}

	// Mode A retains the main-space slice of every basis vector on the host (main × dim),
	// the cheap ingredient the dense back-transform Qmain·s needs.
	var qmain []float64 // column-major main×dim, grown block by block
	appendQmain := func(panel backend.BlockView, cols int) {
		if modeB {
			return
		}
		// Download only the top `main` rows of each column (main floats), not the whole
		// n-tall column — same pattern as Solve's main-space back-transform (lanczos.go:469).
		for c := range cols {
			qmain = append(qmain, be.Download(panel.Col(c).Slice(0, main))...)
		}
	}

	// Mode A additionally retains the full basis on the host and reorthogonalizes each new
	// block against all of it (streaming one block back to the device at a time). Only three
	// blocks are ever resident on the compute backend, so a GPU whose memory cannot hold the
	// basis still runs; the accuracy is that of full-reorthogonalization Lanczos (unlike the
	// gated Mode B, a small start block has no subspace-iteration structure to keep the short
	// recurrence clean, so full reorth against host RAM is what keeps the Ritz pairs and pole
	// strengths exact rather than ghost-ridden). Host-RAM-bounded, not device-bounded.
	var hostQ [][]float64 // each entry is one block's n×size, column-major
	appendHostQ := func(view backend.BlockView) {
		if modeB {
			return
		}
		hostQ = append(hostQ, append([]float64(nil), be.Download(view.V)...))
	}
	var blocks []lmBlock
	rPrev, rCur, dim, iter0 := 0, b, b, 0

	// Resume from a checkpoint if one is present and matches this problem; otherwise seed the
	// recurrence. A resumed run restores the [prev|cur] window, the accumulated α/β and the block
	// scalars, then re-enters the loop at the saved index and redoes that one iteration
	// (lowmem_checkpoint.go explains why that is the consistent re-entry point).
	lmCkpt := cp != nil && cp.Path != "" && modeB
	resumed := false
	if lmCkpt {
		if st := loadLowMem(be, cp.Path, pc, n, main, b, maxdim, opts.MaxBlocks, modeB, opts.DeflTol); st != nil {
			blocks = st.Blocks
			rPrev, rCur, dim, iter0 = st.RPrev, st.RCur, st.Dim, st.Iter
			tm = st.Timing // so -profile totals span the whole daisychain, not just this generation
			resumed = true
		}
	}
	if !resumed {
		// Start block: main-space Cartesian units e_0..e_{b-1} (same seed as Solve; for Mode B
		// b = main so the start block spans the whole main space), or the rows
		// Options.StartRows names.
		start := startBlock(n, b, opts.StartRows, opts.StartVecs)
		up := be.Upload(start)
		be.Copy(pc.ColRange(0, b).V, up)
		be.Free(up)
		blocks = []lmBlock{{off: 0, size: b, beta: backend.NewMat(0, 0)}}
		appendQmain(pc, b)
		appendHostQ(pc.ColRange(0, b))
	}

	// betaExtra is the β to the (unbuilt) block after the last accepted one, used only for
	// the Ritz residual (the reference's trailing block); populated when the loop stops.
	var betaExtra backend.Mat

	for iter := iter0; ; iter++ {
		// Checkpoint hook. The state here — the compacted [prev|cur] window, the accumulated α/β
		// and the block scalars — fully re-does iteration `iter`: α for the current block is not
		// written until below, and pc is not shifted until the very end, so this is the one point
		// in the loop where the panel and (rPrev, rCur) describe each other. Skip iter0 itself: it
		// was just loaded. Predicate copied from Solve (lanczos.go:424) so the two cannot drift.
		if lmCkpt {
			stop := cp.stopRequested()
			if due := cp.Every > 0 && iter != iter0 && iter%cp.Every == 0; stop || due {
				if err := saveLowMem(be, cp.Path, pc, blocks, n, main, b, maxdim, opts.MaxBlocks,
					modeB, opts.DeflTol, dim, rPrev, rCur, iter, tm); err != nil {
					// Never swallow this the way Solve does (`_ = saveKrylov`, lanczos.go:426):
					// a silently failing write leaves a multi-day run with no restart point,
					// which is the exact failure this checkpointing exists to prevent.
					fmt.Fprintf(os.Stderr, "adcgo: lowmem checkpoint at block %d FAILED: %v "+
						"(no restart point from here)\n", iter, err)
				}
				if stop {
					return Result{Interrupted: true, Timing: tm}
				}
			}
		}

		cur := pc.ColRange(rPrev, rPrev+rCur) // current block within pc (prev occupies [0,rPrev))
		w := backend.BlockView{V: workBuf, Rows: n, Cols: rCur, Ld: n}

		// W = M·Q_cur. Tarantelli gate (Mode B): full operator for the first two blocks,
		// satellite↔satellite only thereafter.
		t0 := time.Now()
		if modeB && iter >= 2 {
			sat.ApplyBlockSatellite(w, cur)
		} else {
			op.ApplyBlock(w, cur)
		}
		tm.Apply += time.Since(t0)

		// α = Q_curᵀ·W (the diagonal projection block), recorded on the current block.
		t0 = time.Now()
		aview := backend.BlockView{V: abuf, Rows: rCur, Cols: rCur, Ld: rCur}
		be.Gemm(true, false, 1, cur, w, 0, aview)
		alpha := be.Download(aview.V) // column-major rCur×rCur
		blocks[len(blocks)-1].alpha = alpha
		tm.Proj += time.Since(t0)

		// orth orthogonalizes a candidate panel and returns its rank and R (= the β coupling).
		// Mode B: local reorth against the two live blocks [prev|cur] (two-pass CGS2 removes
		// the Q_cur·α and Q_prev·β_prevᵀ recurrence terms). Mode A: full reorth against the
		// host-retained basis. Both then orthonormalize with rank-revealing deflation.
		orth := func(ve backend.BlockView) (int, backend.Mat) {
			if modeB {
				return orthBlock(be, pc.ColRange(0, rPrev+rCur), ve, pbuf, 2*b, gbuf, opts.DeflTol)
			}
			return reorthFull(be, hostQ, ve, n, b, pbuf, gbuf, opts.DeflTol)
		}

		// Stop once the basis holds MaxBlocks blocks (matching `-blocks`) or fills the space.
		if iter+1 >= opts.MaxBlocks || dim >= maxdim {
			t0 = time.Now()
			be.Copy(tmpBuf, workBuf)
			ve := backend.BlockView{V: tmpBuf, Rows: n, Cols: rCur, Ld: n}
			_, betaExtra = orth(ve)
			tm.Orth += time.Since(t0)
			break
		}

		// Candidate next block: orthogonalize W and orthonormalize; R is the off-diagonal β
		// coupling cur→next.
		t0 = time.Now()
		rank, r := orth(w)
		tm.Orth += time.Since(t0)
		if rank == 0 {
			betaExtra = r // invariant subspace: residuals vanish
			break
		}
		if dim+rank > maxdim {
			rank = maxdim - dim
		}

		// Advance the window: pc becomes [cur | next], compacted at the front. Move cur to
		// tmp first so the left-shift never overlaps a device copy.
		be.Copy(tmpBuf, cur.V)
		be.Copy(pc.ColRange(0, rCur).V, tmpBuf)
		be.Copy(pc.ColRange(rCur, rCur+rank).V, w.ColRange(0, rank).V)

		appendQmain(pc.ColRange(rCur, rCur+rank), rank)
		appendHostQ(pc.ColRange(rCur, rCur+rank))
		blocks = append(blocks, lmBlock{off: dim, size: rank, beta: r})
		rPrev, rCur = rCur, rank
		dim += rank

		if opts.Progress != nil {
			opts.Progress(iter, dim, rank, tm)
		}
	}

	// Assemble the projected matrix T from the block α/β and diagonalize.
	tEig := time.Now()
	theta, topVecs, botVecs, sDense := diagProjected(be, blocks, dim, b, main, modeB)
	tm.Eig = time.Since(tEig)

	// Recover main-space components and pole strengths.
	tBack := time.Now()
	res := packLowMem(theta, topVecs, botVecs, sDense, qmain, blocks, betaExtra, dim, main, modeB)
	tm.Back = time.Since(tBack)
	res.Timing = tm

	// The solve completed; drop any checkpoint so a later rerun of the same job starts fresh
	// rather than resuming a finished computation (mirrors lanczos.go's removeCheckpoint).
	if lmCkpt {
		removeLowMemCheckpoint(cp.Path)
	}

	// Drop Lanczos ghosts: roots with essentially zero main-space weight (spur_thresh=1e-9).
	return filterSpurious(res)
}

// reorthFull orthogonalizes the candidate panel v against the entire host-retained basis
// hostQ (Mode A), then orthonormalizes it with rank-revealing deflation. It streams one
// block back to the compute backend at a time (be.Upload/Free), so only the candidate and a
// single basis block are resident beyond hostQ's host storage. Two sweeps of block
// Gram–Schmidt (each cgs2 is itself two passes) restore orthogonality to O(eps). Returns the
// surviving column count and the R factor such that the incoming v equals q·R.
func reorthFull(be backend.Backend, hostQ [][]float64, v backend.BlockView, n, b int, pbuf, gbuf backend.Vector, deflTol float64) (int, backend.Mat) {
	for range 2 {
		for _, blk := range hostQ {
			sz := len(blk) / n
			up := be.Upload(blk)
			bv := backend.BlockView{V: up, Rows: n, Cols: sz, Ld: n}
			cgs2(be, bv, v, pbuf, b)
			be.Free(up)
		}
	}
	rank, r, _ := blockOrth(be, v, gbuf, deflTol)
	return rank, r
}

// diagProjected builds the block-tridiagonal projected matrix T from the accepted blocks and
// diagonalizes it. Mode B returns the 2·band top/bottom eigenvector slices via the banded
// solver (topVecs/botVecs, each main-relevant rows × dim); Mode A returns the full dense
// eigenvectors sDense (dim×dim) instead. theta is ascending in both.
func diagProjected(be backend.Backend, blocks []lmBlock, dim, b, main int, modeB bool) (theta []float64, topVecs, botVecs [][]float64, sDense backend.Mat) {
	if modeB {
		band := max(min(2*b-1, dim-1), 0)
		lastSize := 0
		if len(blocks) > 0 {
			lastSize = blocks[len(blocks)-1].size
		}
		// The banded solver returns only the first and last `band` rows of each eigenvector, and
		// the caller needs the top `main` rows (main space) plus the last block's rows (Ritz
		// residual). The dim-1 cap on the bandwidth means that at a SMALL Krylov width those
		// slices can be shorter than what is needed — at dim = 1 band is 0, so nm = 0 and the
		// split below reads an empty column — so solve densely and slice the full eigenvectors
		// instead. dim is tiny exactly when this triggers, so the dense solve is cheap.
		//
		// Both edges here have shipped: MaxBlocks = 1 with main > 1 gave a bottom slice one row
		// short (index -1, job 14787917), and a one-dimensional main space gave band = 0. The
		// block sweep in cmd/adcgo covers both.
		if band < main || band < lastSize {
			T := backend.NewMat(dim, dim)
			fillDense(T, blocks)
			theta, sFull := be.SymEig(T)
			bw := min(dim, max(main, lastSize))
			top := make([][]float64, main)
			for r := range main {
				top[r] = make([]float64, dim)
				for k := range dim {
					top[r][k] = sFull.At(r, k)
				}
			}
			bot := make([][]float64, bw)
			for r := range bw {
				bot[r] = make([]float64, dim)
				for k := range dim {
					bot[r][k] = sFull.At(dim-bw+r, k)
				}
			}
			return theta, top, bot, backend.Mat{}
		}
		bs := newBandStorage(dim, band)
		fillBand(bs, blocks)
		theta, z := bandSymDiagFast(bs)
		nm := 2 * band
		// Split z (nm×dim column-major) into the top `main` rows and the bottom `size_last`
		// rows per eigenvector. Callers only need the top `main` (main space) and the whole
		// bottom band (for the residual); expose both as row-major [row][k].
		top := make([][]float64, main)
		for r := range main {
			top[r] = make([]float64, dim)
		}
		bot := make([][]float64, band)
		for r := range band {
			bot[r] = make([]float64, dim)
		}
		for k := range dim {
			col := z[k*nm : k*nm+nm]
			for r := range main {
				top[r][k] = col[r]
			}
			for r := range band {
				bot[r][k] = col[band+r]
			}
		}
		return theta, top, bot, backend.Mat{}
	}
	// Mode A: dense T, dense SymEig.
	T := backend.NewMat(dim, dim)
	fillDense(T, blocks)
	theta, s := be.SymEig(T)
	return theta, nil, nil, s
}

// fillBand scatters the block α/β scalars into the banded storage bs. α_bk contributes the
// within-block entries T[off+p][off+q]=α[p][q] (p≥q); β_bk contributes the cross-block
// entries T[off_bk+q][off_prev+p]=β[q][p] (β = this.size × prev.size).
func fillBand(bs bandStorage, blocks []lmBlock) {
	for bi := range blocks {
		blk := blocks[bi]
		a := blk.alpha // column-major size×size
		s := blk.size
		for q := range s {
			for p := q; p < s; p++ {
				bs.set(blk.off+q, p-q, a[q*s+p]) // T[off+p][off+q]
			}
		}
		if blk.beta.Rows > 0 {
			prev := blocks[bi-1]
			for q := range blk.beta.Rows { // this.size
				for p := range blk.beta.Cols { // prev.size
					i := prev.size + q - (p) // (blk.off+q) - (prev.off+p), prev.off+prev.size=blk.off
					bs.set(prev.off+p, i, blk.beta.At(q, p))
				}
			}
		}
	}
}

// fillDense scatters the same block α/β scalars into a dense symmetric matrix (Mode A).
func fillDense(T backend.Mat, blocks []lmBlock) {
	for bi := range blocks {
		blk := blocks[bi]
		a := blk.alpha
		s := blk.size
		for q := range s {
			for p := range s {
				T.Set(blk.off+p, blk.off+q, a[q*s+p])
			}
		}
		if blk.beta.Rows > 0 {
			prev := blocks[bi-1]
			for q := range blk.beta.Rows {
				for p := range blk.beta.Cols {
					v := blk.beta.At(q, p)
					T.Set(blk.off+q, prev.off+p, v)
					T.Set(prev.off+p, blk.off+q, v)
				}
			}
		}
	}
}

// packLowMem assembles the Result: eigenvalues, main-space vectors, pole strengths, and a
// best-effort Ritz residual. Mode B reads main components straight from the banded solver's
// top slice; Mode A back-transforms the retained host basis slice (Qmain·s).
func packLowMem(theta []float64, topVecs, botVecs [][]float64, sDense backend.Mat, qmain []float64, blocks []lmBlock, betaExtra backend.Mat, dim, main int, modeB bool) Result {
	mainVecs := backend.NewMat(main, dim)
	if modeB {
		// topVecs is `main` independent row slices, so this transpose re-bases the pointer on
		// every element: dim·main scattered loads. At the DIP singlet block width (main = 1711)
		// with -blocks 200 that is 342,200 × 1711 ≈ 5.9e8 loads/stores, one-time but on a single
		// core as written. Every cell mainVecs[r,k] is written exactly once from topVecs[r][k] —
		// no accumulator, nothing reassociated — so splitting the k range is bit-for-bit the
		// serial result; only the order of independent stores changes.
		//
		// Chunks, not Rows: the split is over k, and Chunks has no serial floor (parallel.Rows
		// falls back to a serial walk below 2*GOMAXPROCS) and gives each worker a fixed
		// contiguous k range, so the schedule cannot vary run to run.
		parallel.Chunks(dim, parallel.ChunkWorkers(dim), func(_, k0, k1 int) {
			for k := k0; k < k1; k++ {
				for r := range main {
					mainVecs.Set(r, k, topVecs[r][k])
				}
			}
		})
	} else {
		// Qmain is column-major main×dim; MainVecs = Qmain · sDense.
		qm := backend.Mat{Rows: main, Cols: dim, Data: make([]float64, main*dim)}
		for c := range dim {
			for r := range main {
				qm.Set(r, c, qmain[c*main+r])
			}
		}
		mainVecs = backend.MatMul(qm, sDense)
	}

	ps := make([]float64, dim)
	// A reduction, so the partition is by WHOLE columns: acc for a given k is summed over
	// r = 0..main-1 in ascending order inside one goroutine, exactly as before, and ps[k] is
	// therefore byte-for-byte the serial value. Nothing is reassociated across workers — only
	// independent columns move between cores. Chunks' static contiguous ranges are what
	// guarantee a column is never split; parallel.Rows would additionally have run serially
	// below 2*GOMAXPROCS. dim·main is the same ≈5.9e8 multiply-adds as the transpose above.
	parallel.Chunks(dim, parallel.ChunkWorkers(dim), func(_, k0, k1 int) {
		for k := k0; k < k1; k++ {
			var acc float64
			for r := range main {
				v := mainVecs.At(r, k)
				acc += v * v
			}
			ps[k] = 100 * acc
		}
	})

	// Ritz residual ‖β_extra · (eigenvector's last-block components)‖. Mode A takes the
	// last-block rows from the dense eigenvectors; Mode B from the tail of the banded
	// solver's bottom slice.
	residual := make([]float64, dim)
	last := blocks[len(blocks)-1]
	if betaExtra.Rows > 0 {
		for k := range dim {
			var acc float64
			for rr := range betaExtra.Rows {
				var v float64
				for t := range last.size {
					var comp float64
					if modeB {
						// Global row of this last-block component. The banded solver returns only
						// the first and last `band` rows of each eigenvector, so read it from
						// whichever slice actually covers that row.
						//
						// The bottom slice normally does: it holds rows [dim-band, dim) and
						// band = min(2b-1, dim-1) >= last.size whenever two or more blocks were
						// accepted. With exactly ONE accepted block the cap bites — dim ==
						// last.size == main and band == dim-1 — leaving the bottom slice one row
						// short and indexing it at -1 (the mgpu smoke, job 14787917, hit exactly
						// this at -blocks 1). In that case the TOP slice spans the whole vector,
						// because main == dim, so the component is there instead.
						g := last.off + t
						if lo := dim - len(botVecs); g >= lo {
							comp = botVecs[g-lo][k]
						} else if g < len(topVecs) {
							comp = topVecs[g][k]
						} else {
							panic(fmt.Sprintf("lanczos: Ritz residual needs projected row %d of %d, "+
								"but Mode B retained only the top %d and bottom %d rows",
								g, dim, len(topVecs), len(botVecs)))
						}
					} else {
						comp = sDense.At(last.off+t, k)
					}
					v += betaExtra.At(rr, t) * comp
				}
				acc += v * v
			}
			residual[k] = math.Sqrt(acc)
		}
	}

	return Result{Values: theta, PS: ps, MainVecs: mainVecs, Residual: residual}
}

// filterSpurious drops the Lanczos ghosts (zero main-space weight) from a Result, keeping
// every field aligned. It is the load-bearing use of Result.Spurious the design note
// describes: under the short recurrence, unlike full-reorth Solve, ghosts do arise.
func filterSpurious(r Result) Result {
	const spurThresh = 1e-9
	keep := make([]int, 0, len(r.Values))
	for k := range r.Values {
		if !r.Spurious(k, spurThresh) {
			keep = append(keep, k)
		}
	}
	if len(keep) == len(r.Values) {
		return r
	}
	out := Result{
		Values:   make([]float64, len(keep)),
		PS:       make([]float64, len(keep)),
		MainVecs: backend.NewMat(r.MainVecs.Rows, len(keep)),
		Timing:   r.Timing,
	}
	if len(r.Residual) == len(r.Values) {
		out.Residual = make([]float64, len(keep))
	}
	for i, k := range keep {
		out.Values[i] = r.Values[k]
		out.PS[i] = r.PS[k]
		for row := range r.MainVecs.Rows {
			out.MainVecs.Set(row, i, r.MainVecs.At(row, k))
		}
		if out.Residual != nil {
			out.Residual[i] = r.Residual[k]
		}
	}
	return out
}
