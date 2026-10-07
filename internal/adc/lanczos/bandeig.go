// bandeig.go — a banded real-symmetric eigensolver that returns, for each eigenvalue, only a
// caller-chosen strip of top and bottom rows of its eigenvector. It is a Go port of
// Tarantelli's bnd2td + tddiag (../ADC/libLanczos/{bnd2td,tddiag}.f, wrapped by
// band_sym_diag_fast in lanczos_util.cpp).
//
// Why not LAPACK: the short-recurrence low-memory Lanczos driver (lowmem.go, Mode B)
// discards the Krylov basis, so it cannot back-transform full eigenvectors. But it does
// not need them: the first `band` Lanczos vectors are the main-space unit vectors, so the
// top rows of a projected eigenvector ARE its main-space components (its pole strength), and
// the bottom rows are the residual-tail slice. bnd2td/tddiag accumulate exactly and only those
// rows through the band→tridiagonal→QL rotations, making the eigenvector cost O(dim*rows)
// instead of the O(dim*dim) a full LAPACK dsbev would spend —
// which for the production block width (band ≈ 1700, dim ≈ 10^5) is the difference between a
// few GB and ~900 GB. The eigenvalue reduction itself is O(dim^2 * band); it runs once per
// sector solve, off the mat-vec hot path.
//
// The port keeps the reference's Fortran control flow (including the underflow rescale and the
// arithmetic-IF band-width dispatch) statement for statement, so it can be checked against the
// Fortran; the indexing helpers below translate the 1-based column-major a(n,mb) / z storage
// into flat Go slices.
//
// It deviates from the Fortran in two ways, both of which leave the arithmetic bit-identical
// and both of which exist because the serial reduction is ~23 days of work at the production
// shape (dim = 308000, band = 3079):
//
//   - how many eigenvector rows are accumulated is a parameter (bandEigOpts) rather than
//     2*band, because Mode B needs only `main` top rows and the last block's bottom rows;
//   - every write to the eigenvector accumulator is deferred onto a tape and replayed across
//     row blocks in parallel (zTape), which is a loop interchange over independent rows.
//
// TestBandSymDiagFastBitExactVsReference pins both against a frozen copy of the serial code.
package lanczos

import (
	"fmt"
	"math"
	"os"
	"sort"
	"time"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/parallel"
)

// bandStorage holds the projected banded matrix in the short-recurrence driver's native
// layout: dim columns, each of band+1 entries. col j occupies data[j*(band+1) : (j+1)*(band+1)],
// with entry 0 the diagonal and entry i (1..band) the coupling between Lanczos vectors j
// and j+i (the i-th sub-diagonal). This is exactly the `subdiags[col][0..band]` accumulation
// of theADCcode's LanczosEngine and of lowmem.go.
type bandStorage struct {
	data []float64 // dim*(band+1)
	dim  int
	band int
}

// newBandStorage allocates a zeroed banded matrix for dim Lanczos vectors and half-bandwidth
// band (= the block width).
func newBandStorage(dim, band int) bandStorage {
	return bandStorage{data: make([]float64, dim*(band+1)), dim: dim, band: band}
}

// set stores the i-th band entry (i=0 diagonal) of column col.
func (b bandStorage) set(col, i int, v float64) { b.data[col*(b.band+1)+i] = v }

// at returns the i-th band entry (i=0 diagonal) of column col.
func (b bandStorage) at(col, i int) float64 { return b.data[col*(b.band+1)+i] }

// bandEigOpts selects which rows of each eigenvector the solver accumulates, and how the
// accumulation is threaded.
//
// topRows/botRows are deliberately NOT the bandwidth. band sizes the matrix — fillBand
// (lowmem.go) writes band entries up to prev.size+lastSize-1 and bandStorage.set is
// unchecked — whereas topRows/botRows size only z, the passive eigenvector accumulator that
// the reduction writes and never reads back. Mode B needs the first `main` rows (main-space
// components) and the last block's rows (Ritz residual), which at band = 2*b-1 is about half
// of the 2*band rows the symmetric wrapper accumulates. Asking for only those halves all z
// work and halves z's footprint (15.2 GB -> 7.6 GB at the production shape).
type bandEigOpts struct {
	topRows int // accumulate eigenvector rows [0,topRows)
	botRows int // and rows [dim-botRows,dim)
	workers int // replay workers; 0 = the measured default (see bandEigDefaultWorkers), 1 = serial
	batch   int // deferred-op batch size; 0 = zBatchDefault

	// accel, when non-nil, replays the deferred eigenvector rotations on a device instead of on
	// the host. The band reduction that records them stays on the host either way — it is a
	// sequential chase that synchronizes far too often for kernel launches. See
	// backend.BandEigKernels and backend/bandeig_kernels.cu.
	accel backend.BandEigKernels

	// ckpt, when enabled, lets the band reduction save its state at a column boundary and resume
	// there in a later process. See bandeig_checkpoint.go for why the reduction needs its own.
	ckpt *bandEigCkpt

	// blockRun, when > 0, replays runs of chained QL rotations as one dense orthogonal block instead
	// of one rotation at a time, capped at this many rotations per block (zBlockDefault is the
	// measured starting point). It is OFF by default because it reassociates the arithmetic: the
	// result is correct to roundoff but no longer bit-identical to the reference, which every other
	// guarantee in this file is. See bandeig_blocked.go.
	blockRun int

	// resumeZ, when non-nil, is installed as the accumulator instead of the identity strips, so a
	// transform applied BEFORE this solve is carried into it. That is how stage 1 of a two-stage
	// reduction hands over Eᵀ·U1 (see bandreduce.go); reseeding here would discard it.
	resumeZ []float64

	// aGrain overrides the band-matrix replay's grain; 0 means aGrainDefault.
	//
	// It exists for testability, and it is the only way the parallel band-matrix replay can be
	// tested at all. A chase carries ~4*(dim-k) updates, so reaching the default grain on two
	// workers needs dim in the tens of thousands — a solve of minutes, not the seconds a test can
	// spend. Lowering the grain forces the same partition at a shape a test can afford.
	aGrain int

	// progress, when non-nil, is called every bandEigProgressEvery columns of the reduction with
	// the column reached, the total, and the elapsed reduction time. See Options.EigenProgress.
	progress func(col, cols int, elapsed time.Duration)
}

// rows is the number of eigenvector rows accumulated per eigenvalue.
func (o bandEigOpts) rows() int { return o.topRows + o.botRows }

// bandSymDiagFast diagonalizes the banded matrix bs and returns the ascending eigenvalues
// (length dim) together with a (2*band)×dim column-major slice z: column i is eigenvector
// i, rows [0,band) its top (main-space) components, rows [band,2*band) its bottom
// (residual-tail) components. Port of band_sym_diag_fast (lanczos_util.cpp:105).
//
// For band == 0 (no satellites) it is a plain diagonal read; z is then the 0×dim empty
// slice and every eigenvalue is its own diagonal entry.
//
// This is the symmetric 2*band convenience form, kept because it is what the Fortran and the
// C++ wrapper expose and what the tests compare against. It repacks the solver's padded z
// into a tight 2*band leading dimension, so production goes through bandSymDiagFastOpts
// instead and asks for only the rows it needs.
func bandSymDiagFast(bs bandStorage) (evals []float64, z []float64) {
	nm := 2 * bs.band
	evals, zp, ld, _ := bandSymDiagFastOpts(bs, bandEigOpts{topRows: bs.band, botRows: bs.band})
	if ld == nm {
		return evals, zp
	}
	z = make([]float64, nm*bs.dim)
	for k := range bs.dim {
		copy(z[k*nm:(k+1)*nm], zp[k*ld:k*ld+nm])
	}
	return evals, z
}

// bandSymDiagFastOpts diagonalizes bs, accumulating only the eigenvector rows o asks for.
// It returns the ascending eigenvalues, the column-major accumulator z and its leading
// dimension ld: column k is eigenvector k, z[k*ld+r] for r in [0,topRows) its top rows and
// z[k*ld+topRows+r] for r in [0,botRows) its bottom rows. ld is returned rather than implied
// because it is padded (see zAccum) — never assume ld == topRows+botRows.
// interrupted is true only when a checkpointing reduction was asked to stop; the other three
// results are then meaningless and the caller must not use them.
func bandSymDiagFastOpts(bs bandStorage, o bandEigOpts) (evals []float64, z []float64, ld int, interrupted bool) {
	dim, band := bs.dim, bs.band
	if dim == 0 {
		return nil, nil, 0, false
	}
	b1 := band + 1

	// The band matrix in the Fortran's a(n=dim, mb=b1) indexing but stored ROW-major:
	// A(row,col) 1-based at af[(row-1)*b1 + col-1]. See bnd2td for why the layout is
	// transposed relative to the Fortran.
	//
	// lanczos_util.cpp:114-116 writes a[i+j+dim*(b1-i-1)] = mat[i+j*b1] for i in [0,b1),
	// j in [0,dim-i), which is A(i+j+1, b1-i) = bs.at(j,i): band entry i of column j is matrix
	// row i+j+1, band column b1-i. Walking matrix rows r = i+j instead of band columns i makes
	// the writes contiguous within a row and lets each worker own whole af rows.
	af := make([]float64, b1*dim)
	parallel.Chunks(dim, parallel.ChunkWorkers(dim), func(_, lo, hi int) {
		for r := lo; r < hi; r++ {
			row := af[r*b1 : (r+1)*b1]
			hi := min(b1-1, r)
			for i := 0; i <= hi; i++ {
				row[b1-1-i] = bs.at(r-i, i)
			}
		}
	})

	d := make([]float64, dim)
	e := make([]float64, dim)
	// e2 holds the squared off-diagonals. The Fortran computes it for callers that bisect on
	// the tridiagonal; tddiag does not use it and neither do we. Kept so bnd2td stays a
	// faithful port, allocated once and never read.
	e2 := make([]float64, dim)

	// One pool for both parallel regions of the solve. Two would spin against each other, and
	// spinning workers are only free while their cores have nothing else to do.
	//
	// Which is also why a solve too small to use them gets none. The workers spin between rounds,
	// so on a solve that finishes in microseconds they are pure cost: with a pool per solve the
	// low-memory driver's own test suite went from 4.4 s to 40 s at 2.6 cores busy, all of it idle
	// spinning. A chase only splits at dim >= 2*aGrainDefault and a flush only splits at
	// rows >= 2*zRowMin, so below both there is nothing for a worker to do anyway. An explicit
	// o.workers is honoured regardless — a test that asks for seven workers gets seven.
	workers := o.workers
	if workers <= 0 {
		workers = bandEigDefaultWorkers
	}
	pool := parallel.NewFixedPool(workers)
	defer pool.Close()

	// Resume the band reduction if a usable checkpoint is there. A fingerprint mismatch or a
	// truncated file is reported and then ignored: starting the reduction over costs time, whereas
	// resuming onto the wrong matrix would produce a plausible wrong answer.
	var (
		fp     uint64
		resume *beCkptState
	)
	if o.ckpt.enabled() {
		fp = bandEigFingerprint(bs)
		st, err := readBandEigCheckpoint(o.ckpt.path)
		switch {
		case err != nil:
			fmt.Fprintf(os.Stderr, "adcgo: banded eigensolve checkpoint %s unusable (%v); "+
				"starting the reduction from scratch\n", o.ckpt.path, err)
		case st == nil:
			// No checkpoint: a fresh reduction, the common case.
		case !st.matches(dim, b1, o.topRows, o.botRows, ((o.rows()+zRowAlign-1)/zRowAlign)*zRowAlign, fp):
			fmt.Fprintf(os.Stderr, "adcgo: banded eigensolve checkpoint %s does not match this "+
				"problem (dim=%d mb=%d column=%d); starting the reduction from scratch\n",
				o.ckpt.path, st.Dim, st.Mb, st.K)
		default:
			resume = st
			fmt.Fprintf(os.Stderr, "adcgo: resuming the banded eigensolve at column %d of %d\n",
				st.K, max(dim-2, 0))
		}
	}

	// bnd2td seeds (or restores) the accumulator and, if a device was requested, uploads it there;
	// from that point the host z is stale until the download below.
	za := newZAccum(dim, o, pool)
	if bnd2td(za, dim, b1, af, d, e, e2, pool, o.aGrain, o.progress, o.ckpt, fp, resume, 0) {
		// Asked to stop; the reduction saved its state. Release the device and report upward so the
		// driver can exit "resume needed" rather than return a half-reduced spectrum.
		if za.dev != nil {
			za.dev.Free()
			za.dev = nil
		}
		return nil, nil, 0, true
	}
	if l := tddiag(za, dim, d, e); l != 0 {
		// tddiag gave up after 30 QL iterations on root l, leaving d[l-1..] unconverged. The
		// Fortran returns this in ierr and the C++ wrapper checked it; discarding it meant a
		// multi-day solve could return silently wrong eigenvalues. It is not fatal — roots
		// below l did converge — so report it and let the caller see the rest.
		fmt.Fprintf(os.Stderr, "adcgo: banded eigensolve did not converge at root %d of %d "+
			"(30 QL iterations); eigenvalues from index %d upward are unreliable\n", l, dim, l-1)
	}
	// Covers tddiag's early returns; the sort already flushed on the converged path.
	za.flush()
	if za.dev != nil {
		za.dev.Download(za.z)
		za.dev.Free()
		za.dev = nil
	}
	if za.tape.n != 0 {
		panic("lanczos: banded eigensolve returned with deferred eigenvector ops pending")
	}
	// The reduction finished, so its restart point is not only useless but actively harmful: a rerun
	// of the same job would resume a completed computation.
	if o.ckpt.enabled() {
		removeBandEigCheckpoint(o.ckpt.path)
	}
	return d, za.z, za.ld, false
}

// ---------------------------------------------------------------------------
// The eigenvector accumulator
// ---------------------------------------------------------------------------

const (
	// zRowAlign is the row granularity of the accumulator: z's leading dimension is rounded up
	// to a multiple of it and the replay's row blocks start on multiples of it, so every block
	// boundary is 64-byte aligned in every column. Without it two workers writing adjacent row
	// blocks would ping-pong one cache line ~10^11 times, which is the single largest hazard
	// in the parallel replay.
	zRowAlign = 8

	// zRowMin is the fewest rows a replay worker may own, expressed to the pool as a grain of
	// zRowMin/zRowAlign row groups. Below it the barrier costs more than the work, and
	// BenchmarkZReplayBlock shows a 32-row block already running at ~60% of peak throughput
	// because the inner loops get too short to amortize the per-op dispatch. At the production
	// shape (3080 accumulated rows) it permits 96 workers, so it never binds on a 64-core node;
	// it binds on the small shapes the tests use, where it correctly keeps the replay serial.
	zRowMin = 32

	// bandEigDefaultWorkers is ONE, and that is a measured result rather than an oversight.
	//
	// Everything needed to run the eigenvector and band-matrix replays in parallel is here and is
	// tested bit-exact (zTape, aTape, parallel.FixedPool). It is off by default because at the
	// production half-bandwidth it makes the solve SLOWER. Measured on a 64-core Helix node,
	// band = 3079 with dim = 4620 (scripts/helix/eigensolver_tier2.sbatch):
	//
	//	frozen serial reference                      462.1 s   1.00x
	//	narrowed rows + transposed layout, 1 worker   139.1 s   3.32x
	//	the same at 16 workers                        196.9 s   2.35x
	//
	// So the whole gain comes from accumulating fewer eigenvector rows and from the row-major band
	// matrix, and threading costs 1.4x on top. band = 3079 puts the accumulated rows at 3080, which
	// is exactly what production carries, so that half of the result transfers directly; candidate
	// causes are the 15 workers spinning through the long serial stretches, the NUMA placement of a
	// z that one goroutine first-touched, and clock throttling from 16 busy cores instead of one.
	//
	// What is NOT measured is the band-matrix replay at production's chase length. At dim/band =
	// 1.5 a chase is one or two steps and never reaches aGrainDefault, so the A-side parallel path
	// did not execute in the run above; at production's dim/band = 100 a chase carries ~1.2e6
	// updates and would use all 16. No benchmark reproduces both that chase length and that
	// bandwidth without also reproducing production's cost, so the only way to settle it is a
	// production run with -eig-workers set.
	//
	// Raise this only with a measurement attached. The tests cover every worker count either way.
	bandEigDefaultWorkers = 1

	// zBatchDefault is how many deferred ops accumulate before a replay. It only places
	// barriers — it does not change any worker's access sequence — so anything above a few
	// hundred captures all the cache reuse there is; 65536 keeps the tape at 1.4 MB (L2/L3
	// resident) while amortizing the ~30 us spawn/join over 5-10 ms of work per worker.
	zBatchDefault = 1 << 16
)

// bandEigProgressEvery is how often the reduction reports a column. Its outer loop runs dim times
// and each pass costs O(dim*band), so at the production shape a pass is ~10 ms and 1024 of them is a
// line every few minutes — frequent enough to extrapolate a finish time from, rare enough that the
// log stays readable over hours.
//
// A var rather than a const only so TestSolveLowMemSurfacesReductionInterrupt can shrink it and arm
// a stop request from inside the reduction; nothing at runtime changes it. Same device as
// lmPanelChunkBytes.
var bandEigProgressEvery = 1024

// Deferred op kinds. Each corresponds to exactly one Fortran loop over z's rows.
const (
	zOpRotA  uint8 = iota // bnd2td.f s2 < 0.5 rotation of columns j-1, j
	zOpRotB               // bnd2td.f s2 >= 0.5 rotation of columns j-1, j
	zOpRotTD              // tddiag.f QL Givens rotation of columns i, i+1
	zOpScale              // multiply one column by a scalar
)

// zTape records the pending row operations in struct-of-arrays form: 21 bytes per op, no
// allocation in the hot path. col holds the second (higher) column of a rotation, whose
// partner is always col-1 for zOpRotA/zOpRotB and col+1 for zOpRotTD, so one index suffices.
//
// WHY THIS IS SOUND. z is a write-only accumulator for the whole reduction: every write to it
// is a seed, one of the four row ops below, or the sort's column swap, and no scalar in either
// algorithm is ever read out of z — b1v, b2 and s2 come from the band matrix and d; c, s, p, g
// and h come from d and e. So the spine can run to completion recording the op stream, and the
// stream can then be replayed row block by row block.
//
// Replaying by row blocks is a LOOP INTERCHANGE, not a reassociation: for any fixed row the
// ops still execute in their original order with their original operand order, and distinct
// rows never interact. That is why the result is bit-identical rather than merely close, and
// why TestBandSymDiagFastBitExactVsReference can compare with != across worker counts.
type zTape struct {
	kind []uint8
	col  []int32
	f1   []float64
	f2   []float64
	n    int

	// seg holds the boundaries between runs of ops that COMMUTE with each other, as offsets into the
	// arrays above. A run is [seg[i], seg[i+1]).
	//
	// This exists because the replay has far more parallelism than one op at a time, and the device
	// kernel's own comment used to deny it. Inside one bnd2td chase, j advances by m1 >= 2 and each
	// step records a single rotation on columns (j-1, j), so consecutive ops touch DISJOINT column
	// pairs: they commute, and a device grid can run them concurrently instead of serially. That is
	// ~dim/band of them per chase, which is the occupancy the one-op-at-a-time kernel lacks (it runs
	// ~96 warps on a 132-SM GPU and is latency-bound at 4% of HBM).
	//
	// tddiag's QL rotations are the opposite case — (i, i+1) then (i-1, i) share a column — so they
	// get a segment each and gain nothing. Only bnd2td's share benefits.
	//
	// The host replay ignores both entirely: it walks the ops in order, which is always valid.
	seg []int32
	// segPar says, per segment, whether its ops may be applied CONCURRENTLY. This flag is the whole
	// safety of the scheme and must default to false: tddiag's QL rotations are chained — (i, i+1)
	// then (i-1, i) share a column — so treating an unmarked tape as one commuting run would silently
	// apply them out of order. Only the band reduction sets it.
	segPar []bool
}

// zAccum is the eigenvector accumulator z: a rows×n column-major array written by the band→
// tridiagonal reduction and the QL sweep, and read by neither.
//
// Layout: column j (1-based, as in the Fortran) occupies z[(j-1)*ld : (j-1)*ld+rows]; entries
// [0,top) are eigenvector rows [0,top) of the original matrix and entries [top,top+bot) are
// its last bot rows. ld >= rows; the padding is never touched.
type zAccum struct {
	z    []float64
	ld   int // padded leading dimension; ALWAYS index with this, never with rows
	rows int // top+bot, the live entries per column
	top  int
	bot  int
	n    int

	// preset is bandEigOpts.resumeZ: seed installs it instead of the identity strips.
	preset []float64

	// blockRun is bandEigOpts.blockRun; 0 replays every rotation individually.
	blockRun int

	// commuting is the state setCommuting maintains: whether ops recorded now may be applied
	// concurrently. False by default, which is the safe answer.
	commuting bool

	tape   zTape
	groups int // ceil(rows/zRowAlign), the units the replay partitions
	pool   *parallel.FixedPool

	// accel builds dev, and dev is the device-resident accumulator once seed has run. While dev is
	// set, z on the host is STALE: every mutation goes to the device and the host copy is only
	// refreshed by an explicit download.
	accel backend.BandEigKernels
	dev   backend.BandEigZ
}

// newZAccum allocates the accumulator for n columns with the rows o requests.
func newZAccum(n int, o bandEigOpts, pool *parallel.FixedPool) *zAccum {
	if o.topRows < 0 || o.botRows < 0 {
		panic("lanczos: bandEigOpts row counts must be non-negative")
	}
	rows := o.rows()
	// Each strip has to fit, but they need not be disjoint: at 2*band > dim the symmetric form
	// tracks matrix rows [0,band) and [dim-band,dim), which overlap, and the reference does the
	// same — it simply carries some matrix rows in two z rows. Requiring topRows+botRows <= n
	// would reject that, and the production-band benchmark (dim 4620, band 3079) is exactly it.
	if o.topRows > n || o.botRows > n {
		panic("lanczos: bandEigOpts asks for a strip taller than the dimension")
	}
	ld := 0
	if rows > 0 {
		ld = ((rows + zRowAlign - 1) / zRowAlign) * zRowAlign
	}
	batch := o.batch
	if batch <= 0 {
		batch = zBatchDefault
	}
	return &zAccum{
		z:        make([]float64, ld*n),
		ld:       ld,
		rows:     rows,
		top:      o.topRows,
		bot:      o.botRows,
		n:        n,
		groups:   (rows + zRowAlign - 1) / zRowAlign,
		pool:     pool,
		accel:    o.accel,
		preset:   o.resumeZ,
		blockRun: o.blockRun,
		tape: zTape{
			kind: make([]uint8, batch),
			col:  make([]int32, batch),
			f1:   make([]float64, batch),
			f2:   make([]float64, batch),
		},
	}
}

// seed writes the two identity strips, as bnd2td.f does: eigenvector row r in column r+1 for
// the top strip, and in column n-bot+r+1 for the bottom.
func (za *zAccum) seed() {
	if za.preset != nil {
		// A transform from an earlier stage is already accumulated; continue from it rather than
		// overwriting it with the identity strips.
		za.restore(za.preset)
		return
	}
	for j := 1; j <= za.top; j++ {
		za.z[(j-1)*za.ld+(j-1)] = 1
	}
	for j := 1; j <= za.bot; j++ {
		za.z[(za.n-za.bot+j-1)*za.ld+(za.top+j-1)] = 1
	}
	za.activate()
}

// restore installs a checkpointed accumulator in place of the identity strips, for a resumed
// reduction. Like seed it ends by handing the values to the device, and for the same reason.
func (za *zAccum) restore(z []float64) {
	copy(za.z, z)
	za.activate()
}

// activate hands the host accumulator to the device, if one was requested.
//
// This is called at the END of seed/restore, not at construction: it is the moment the host copy
// first holds the values the device must start from. Uploading any earlier ships an all-zero
// accumulator and every rotation afterwards runs on zeros — which is exactly what the first run of
// TestGPUBandEigReplayParity reported, every element 0 against a non-zero host.
func (za *zAccum) activate() {
	if za.accel != nil && za.rows > 0 {
		za.dev = za.accel.NewBandEigZ(za.z, za.rows, za.ld, za.n)
	}
}

// closeSeg ends the current segment, recording whether its ops may be applied concurrently.
func (za *zAccum) closeSeg() {
	if za.rows == 0 || za.tape.n == 0 {
		return
	}
	last := 0
	if n := len(za.tape.seg); n > 0 {
		last = int(za.tape.seg[n-1])
	}
	if za.tape.n == last {
		return // nothing new since the last boundary
	}
	za.tape.seg = append(za.tape.seg, int32(za.tape.n))
	za.tape.segPar = append(za.tape.segPar, za.commuting)
}

// mark ends the current run. Called by the band reduction at each chase boundary, where the
// disjointness argument above stops holding: the next chase's rotations can share columns with this
// one's. A flush implicitly ends a run too.
func (za *zAccum) mark() { za.closeSeg() }

// setCommuting declares whether the ops recorded from here on commute with each other. Changing it
// closes the current segment, so a tape may carry both kinds — which it does, since the band
// reduction's ops can still be pending when the QL sweep starts.
func (za *zAccum) setCommuting(v bool) {
	if za.commuting != v {
		za.closeSeg()
		za.commuting = v
	}
}

// segment is one run of recorded ops, with whether it may be replayed concurrently.
type segment struct {
	lo, hi int
	par    bool
}

// segments returns the runs currently on the tape. A run is concurrent only if it was explicitly
// declared so; anything else is sequential, which is always correct.
func (za *zAccum) segments() []segment {
	var out []segment
	lo := 0
	for i, b := range za.tape.seg {
		if int(b) > lo {
			out = append(out, segment{lo, int(b), za.tape.segPar[i]})
			lo = int(b)
		}
	}
	if lo < za.tape.n {
		out = append(out, segment{lo, za.tape.n, za.commuting})
	}
	return out
}

// record appends one op, flushing first if the tape is full.
func (za *zAccum) record(kind uint8, col int, f1, f2 float64) {
	if za.rows == 0 {
		return
	}
	if za.tape.n == len(za.tape.kind) {
		za.flush()
	}
	i := za.tape.n
	za.tape.kind[i] = kind
	za.tape.col[i] = int32(col)
	za.tape.f1[i] = f1
	za.tape.f2[i] = f2
	za.tape.n = i + 1
}

// rotA defers bnd2td.f's s2 < 0.5 rotation of columns j-1 and j.
func (za *zAccum) rotA(j int, b1v, b2 float64) { za.record(zOpRotA, j, b1v, b2) }

// rotB defers bnd2td.f's s2 >= 0.5 rotation of columns j-1 and j.
func (za *zAccum) rotB(j int, b1v, b2 float64) { za.record(zOpRotB, j, b1v, b2) }

// rotTD defers tddiag.f's QL Givens rotation of columns i and i+1.
func (za *zAccum) rotTD(i int, c, s float64) { za.record(zOpRotTD, i, c, s) }

// scale defers multiplying column j by f. It has to be an op rather than a separate pass:
// bnd2td's underflow rescale of column j must stay ordered between the rotations around it.
func (za *zAccum) scale(j int, f float64) { za.record(zOpScale, j, f, 0) }

// scaleAll defers multiplying column j by e[j-1] for j = 2..n (bnd2td.f label 800).
func (za *zAccum) scaleAll(e []float64) {
	for j := 2; j <= za.n; j++ {
		za.scale(j, e[j-1])
	}
}

// swap exchanges columns i and k for the eigenvalue sort. The caller must have flushed.
func (za *zAccum) swap(i, k int) {
	if za.dev != nil {
		za.dev.SwapCols(i, k)
		return
	}
	ci, ck := za.col(i), za.col(k)
	for l := range ci {
		ci[l], ck[l] = ck[l], ci[l]
	}
}

// col returns the live entries of column j (1-based).
func (za *zAccum) col(j int) []float64 {
	o := (j - 1) * za.ld
	return za.z[o : o+za.rows : o+za.rows]
}

// flush applies every pending op to z and empties the tape. The work is split over contiguous,
// zRowAlign-aligned row blocks; each block's arithmetic is independent of every other's, so
// the partition cannot change the result. workers == 1 takes the same path with a single
// block, so the serial and parallel arms differ only in where the block boundaries fall.
func (za *zAccum) flush() {
	if za.tape.n == 0 {
		return
	}
	// Close the trailing run before anything consumes the boundaries.
	za.closeSeg()
	if za.dev != nil {
		// The device replays each run separately: a concurrent run gets a 2-D grid over
		// (rows x ops), a sequential one the old one-op-at-a-time loop. Launches go on one stream,
		// so they are ordered by stream semantics and need no barrier between them.
		za.dev.Replay(za.tape.kind, za.tape.col, za.tape.f1, za.tape.f2, za.tape.n,
			za.tape.seg, za.tape.segPar)
	} else {
		za.pool.Run(za.groups, zRowMin/zRowAlign, func(glo, ghi int) {
			lo, hi := glo*zRowAlign, min(ghi*zRowAlign, za.rows)
			if lo < hi {
				za.replay(lo, hi)
			}
		})
	}
	za.tape.n = 0
	za.tape.seg = za.tape.seg[:0]
	za.tape.segPar = za.tape.segPar[:0]
}

// replay applies the whole tape to rows [lo,hi) of every column it touches.
//
// The expressions are transcribed operand for operand from bnd2td.f and tddiag.f. Do not
// simplify them: `-b1v*cj1[l] + cj[l]` must not become `cj[l] - b1v*cj1[l]`, and no rounding
// barrier belongs here either — there is no summation to reassociate, so operand order and FMA
// fusion are the only things that can move a bit, and
// TestBandSymDiagFastBitExactVsReference is what catches either.
func (za *zAccum) replay(lo, hi int) {
	t := &za.tape
	z, ld, m := za.z, za.ld, hi-lo
	// Scratch for the blocked path, allocated once per flush per row block rather than per run.
	var gbuf, wbuf []float64
	if za.blockRun > 0 {
		gbuf = make([]float64, (za.blockRun+1)*(za.blockRun+1))
		wbuf = make([]float64, m*(za.blockRun+1))
	}
	// Slicing the SoA arrays to t.n once hoists their bounds checks out of the op loop, and the
	// three-index slices below let the compiler drop the per-element checks: it can see that
	// both operand slices have length exactly m, so `for l := range cj` cannot run off cj1.
	kinds, cols, f1s, f2s := t.kind[:t.n], t.col[:t.n], t.f1[:t.n], t.f2[:t.n]
	for o := 0; o < t.n; o++ {
		// Blocked back-transform: a run of chained QL rotations becomes one dense orthogonal block
		// applied as a matrix product. Worth it only for a long enough run — below zBlockMin the
		// block's extra flops are not repaid by the halved traffic.
		if za.blockRun > 0 {
			if run := runOfChainedRotTD(kinds, cols, o, t.n, za.blockRun); run >= zBlockMin {
				col0, span := givensBlockInto(gbuf, cols[o:o+run], f1s[o:o+run], f2s[o:o+run])
				applyGivensBlock(z, ld, lo, hi, col0, span, gbuf[:span*span], wbuf)
				o += run - 1
				continue
			}
		}
		kind := kinds[o]
		p := (int(cols[o])-1)*ld + lo
		f1, f2 := f1s[o], f2s[o]
		switch kind {
		case zOpRotA:
			q := p - ld
			cj1, cj := z[q:q+m:q+m], z[p:p+m:p+m]
			for l := range cj {
				u := cj1[l] + f2*cj[l]
				cj[l] = -f1*cj1[l] + cj[l]
				cj1[l] = u
			}
		case zOpRotB:
			q := p - ld
			cj1, cj := z[q:q+m:q+m], z[p:p+m:p+m]
			for l := range cj {
				u := f2*cj1[l] + cj[l]
				cj[l] = -cj1[l] + f1*cj[l]
				cj1[l] = u
			}
		case zOpRotTD:
			q := p + ld
			ci, ci1 := z[p:p+m:p+m], z[q:q+m:q+m]
			for k := range ci {
				hh := ci1[k]
				ci1[k] = f2*ci[k] + f1*hh
				ci[k] = f1*ci[k] - f2*hh
			}
		case zOpScale:
			cj := z[p : p+m : p+m]
			for l := range cj {
				cj[l] = f1 * cj[l]
			}
		}
	}
}

// ---------------------------------------------------------------------------
// The band-matrix deferral
// ---------------------------------------------------------------------------

// Deferred band-matrix run kinds. Row/Col name which walk the run came from and A/B which of
// bnd2td's two rotation forms produced it.
const (
	aOpRowA uint8 = iota // bnd2td.f's ugl..j2 walk along matrix rows j-1 and j, s2 < 0.5 form
	aOpRowB              // the same walk, s2 >= 0.5 form
	aOpColA              // bnd2td.f's 2..maxl walk down matrix rows j+1.., s2 < 0.5 form
	aOpColB              // the same walk, s2 >= 0.5 form
)

// aGrainDefault is the fewest band-matrix updates a replay worker is given. The chase this serves
// carries ~4*(n-k) updates, so at the production shape the big chases hand 64 workers ~2e4 each
// while the short chases near k = n fall below 2*aGrainDefault and run on the caller with no
// barrier at all — which is what keeps ~9.5e8 rounds affordable at all.
const aGrainDefault = 4096

// aTape defers one chase's row and column walks so they can be replayed in parallel.
//
// WHY THIS IS SOUND — and this is the part to re-derive before touching it, because the failure
// mode is a silently wrong eigenvector row rather than a crash. Inside one chase (k and r fixed,
// j stepping by m1):
//
//	step j's row walk writes matrix rows j-1 and j;
//	step j's column walk writes matrix rows j+1 .. j+maxl-1;
//	the next step's scalars read row j+m1-1 at band columns 1 and 2, and its row walk reads that
//	  same row at columns 2..mb-1.
//
// Row j+m1-1 is written by the column walk at l == maxl and by nothing else, so that ONE
// iteration is peeled back onto the spine. Everything else each step writes lands in rows no
// later step of the chase reads or writes — the steps' row sets are disjoint — so the remaining
// runs are mutually independent and independent of the spine, and replaying them in any order
// over any partition reproduces the serial result exactly.
//
// The flush is per chase and cannot be coarsened: chases run with r descending, and chase r-1's
// row walk reads rows j-2 and j-1 that chase r's column walk wrote. Batching whole chases would
// also let chase r's scalars read A(j,m1) that chase r+1's row walk had not yet written.
type aTape struct {
	kind []uint8
	p0   []int   // flat offset of the run's first operand
	cnt  []int32 // iterations
	f1   []float64
	f2   []float64
	pre  []int // pre[i] = iterations before run i; len(kind)+1 entries after flush

	a     []float64
	mb    int
	grain int
	pool  *parallel.FixedPool
}

func newATape(a []float64, mb, grain int, pool *parallel.FixedPool) *aTape {
	if grain <= 0 {
		grain = aGrainDefault
	}
	return &aTape{a: a, mb: mb, grain: grain, pool: pool}
}

// add records one run. The slices grow to the longest chase on the first few calls and are then
// reused for the rest of the reduction.
func (at *aTape) add(kind uint8, p0, cnt int, f1, f2 float64) {
	if cnt <= 0 {
		return
	}
	at.kind = append(at.kind, kind)
	at.p0 = append(at.p0, p0)
	at.cnt = append(at.cnt, int32(cnt))
	at.f1 = append(at.f1, f1)
	at.f2 = append(at.f2, f2)
}

// flush replays the chase's runs and empties the tape.
func (at *aTape) flush() {
	if len(at.kind) == 0 {
		return
	}
	at.pre = append(at.pre[:0], 0)
	s := 0
	for _, c := range at.cnt {
		s += int(c)
		at.pre = append(at.pre, s)
	}
	at.pool.Run(s, at.grain, at.replay)
	at.kind, at.p0, at.cnt = at.kind[:0], at.p0[:0], at.cnt[:0]
	at.f1, at.f2 = at.f1[:0], at.f2[:0]
}

// replay applies the iterations in the flattened range [g0,g1), which spans whole and partial
// runs. Partitioning the flattened iteration space rather than the runs matters because the
// first run of a chase is short (r-1 iterations against m1-1 for the rest).
//
// The expressions are transcribed operand for operand from bnd2td.f. See replay on zAccum.
func (at *aTape) replay(g0, g1 int) {
	a, mb := at.a, at.mb
	r := sort.Search(len(at.cnt), func(i int) bool { return at.pre[i+1] > g0 })
	for ; r < len(at.cnt) && at.pre[r] < g1; r++ {
		i0 := max(g0-at.pre[r], 0)
		i1 := min(g1-at.pre[r], int(at.cnt[r]))
		f1, f2 := at.f1[r], at.f2[r]
		switch at.kind[r] {
		case aOpRowA:
			// i2 rises by one per iteration and the two matrix rows are fixed, so both operands
			// walk contiguously: A(j,i2) at p and A(j-1,i2+1) at p-mb+1.
			for p, end := at.p0[r]+i0, at.p0[r]+i1; p < end; p++ {
				q := p - mb + 1
				u := a[q] + f2*a[p]
				a[p] = -f1*a[q] + a[p]
				a[q] = u
			}
		case aOpRowB:
			for p, end := at.p0[r]+i0, at.p0[r]+i1; p < end; p++ {
				q := p - mb + 1
				u2 := f2*a[q] + a[p]
				a[p] = -a[q] + f1*a[p]
				a[q] = u2
			}
		case aOpColA:
			// The matrix row rises and the band column falls, so A(i1,i2) advances by mb-1 per
			// iteration and A(i1,i2+1) is its immediate neighbour.
			for i, p := i0, at.p0[r]+i0*(mb-1); i < i1; i, p = i+1, p+mb-1 {
				u := a[p] + f2*a[p+1]
				a[p+1] = -f1*a[p] + a[p+1]
				a[p] = u
			}
		case aOpColB:
			for i, p := i0, at.p0[r]+i0*(mb-1); i < i1; i, p = i+1, p+mb-1 {
				u2 := f2*a[p] + a[p+1]
				a[p+1] = -a[p] + f1*a[p+1]
				a[p] = u2
			}
		}
	}
}

// bnd2td reduces the real-symmetric band matrix a(n,mb) (mb = band+1 stored diagonals,
// column-major, diagonal in the last band-column) to tridiagonal form (d, e, e2), while
// deferring every write to the eigenvector accumulator za. Port of ../ADC/libLanczos/bnd2td.f.
// All indices below mirror the Fortran 1-based arithmetic.
//
// LAYOUT. a holds the Fortran's a(n,mb) TRANSPOSED — A(i,j) at a[(i-1)*mb + j-1], one matrix
// row per contiguous run of mb entries. A rotation of a symmetric band matrix always touches
// one matrix row and one matrix column, so exactly one of the two inner walks below is
// non-contiguous whichever layout is chosen; the only question is whether the bad stride is n
// or mb, and at the production shape mb = 3080 against n = 308000. Transposed, the row walk
// becomes unit stride (24 KB, L1/L2 resident) and the column walk drops from a 2.46 MB stride
// across all 7.6 GB of a to a 24 KB stride across 74 MB. The arithmetic is untouched — this is
// pure re-indexing, and TestBandSymDiagFastBitExactVsReference holds it to that.
//
// The z rotations do not appear inline: they go onto za's tape and are replayed across row
// blocks, which is what makes them parallel without changing a bit. See zTape. Everything else
// here is a sequential spine — the k, r and j-chase loops carry g and ugl, and the tridiagonal
// assembly carries u — except the two inner walks over the band matrix itself, which go onto a
// second tape and are replayed once per chase. See aTape for why that is legal.
func bnd2td(za *zAccum, n, mb int, a, d, e, e2 []float64, pool *parallel.FixedPool, aGrain int,
	progress func(col, cols int, elapsed time.Duration), ckpt *bandEigCkpt, fp uint64,
	resume *beCkptState, stopBand int) (interrupted bool) {
	const (
		half   = 0.5
		two    = 2.0
		dmin   = 5.421010862427522e-20  // 2^-64
		dminrt = 2.3283064365386963e-10 // 2^-32
	)
	// A(i,j), 1-based like Fortran, over the transposed layout. Used for the scalar and cold
	// accesses; the four hot inner loops walk a with running offsets instead, because their
	// index progressions are affine and a multiply per access is otherwise ~40% of the spine.
	A := func(i, j int) float64 { return a[(i-1)*mb+(j-1)] }
	setA := func(i, j int, v float64) { a[(i-1)*mb+(j-1)] = v }

	// k0 is the last column already reduced: 0 on a fresh run, the checkpointed column on a resumed
	// one. On resume a, d and z come from the checkpoint instead of being initialized, which is why
	// neither the d fill nor the seed may run.
	k0 := 0
	if resume != nil {
		k0 = resume.K
		copy(a, resume.A)
		copy(d, resume.D)
		za.restore(resume.Z)
	} else {
		for j := 1; j <= n; j++ {
			d[j-1] = 1
		}
		za.seed()
	}

	at := newATape(a, mb, aGrain, pool)

	// The rotations recorded inside a chase commute (see zTape.segPar); mark() closes each chase.
	// Cleared before label 800 so nothing downstream — in particular tddiag's chained QL rotations —
	// inherits the claim.
	za.setCommuting(true)
	defer za.setCommuting(false)

	// The reduction's progress is reported in columns of the outer loop below, whose bound is n-2.
	// The same total is used for the completion call, so a log does not switch denominators
	// halfway through.
	tStart := time.Now()
	cols := max(n-2, 0)
	report := func(col int) {
		if progress != nil {
			progress(col, cols, time.Since(tStart))
		}
	}
	defer func() { report(cols) }()

	m1 := mb - 1
	switch {
	case m1 < 1: // m1-1 < 0  → label 900: diagonal only
		for j := 1; j <= n; j++ {
			d[j-1] = A(j, mb)
			e[j-1] = 0
			e2[j-1] = 0
		}
		return
	case m1 == 1: // m1-1 == 0 → label 800 directly (already tridiagonal)
	default: // m1 > 1 → general band reduction (label 70), then fall through to 800
		n2 := n - 2
		lastSave := time.Now()
		for k := k0 + 1; k <= n2; k++ {
			maxr := min(m1, n-k)
			// Iteration r annihilates sub-diagonal r: g = A(k+r, mb-r) is the element at matrix row
			// k+r, column k. r descends from maxr to 2, leaving only sub-diagonal 1 — tridiagonal.
			// Stopping at r = stopBand+1 therefore leaves half-bandwidth stopBand instead, which is
			// what makes a two-stage reduction possible with this same chase. stopBand <= 1 is
			// tridiagonal, i.e. the original behaviour.
			rEnd := maxr
			if stopBand > 1 {
				rEnd = min(maxr, maxr+1-stopBand)
			}
			for r1 := 2; r1 <= rEnd; r1++ {
				r := maxr + 2 - r1
				kr := k + r
				mr := mb - r
				g := A(kr, mr)
				setA(kr-1, 1, A(kr-1, mr+1))
				ugl := k
				for j := kr; j <= n; j += m1 {
					j1 := j - 1
					j2 := j1 - 1
					if g == 0 {
						break
					}
					b1v := A(j1, 1) / g
					b2 := b1v * d[j1-1] / d[j-1]
					s2 := 1.0 / (1.0 + b1v*b2)
					if s2 < half {
						b1v = g / A(j1, 1)
						b2 = b1v * d[j-1] / d[j1-1]
						c2 := 1.0 - s2
						d[j1-1] = c2 * d[j1-1]
						d[j-1] = c2 * d[j-1]
						f1 := two * A(j, m1)
						f2 := b1v * A(j1, mb)
						setA(j, m1, -b2*(b1v*A(j, m1)-A(j, mb))-f2+A(j, m1))
						setA(j1, mb, b2*(b2*A(j, mb)+f1)+A(j1, mb))
						setA(j, mb, b1v*(f2-f1)+A(j, mb))
						at.add(aOpRowA, (j-1)*mb+mb-j+ugl-1, j2-ugl+1, b1v, b2)
						ugl = j
						setA(j1, 1, A(j1, 1)+b2*g)
						if j != n {
							maxl := min(m1, n-j1)
							// l == maxl writes matrix row j1+maxl at band columns 1 and 2, which the
							// next chase step reads. It is the chase's only internal dependency, so
							// it stays on the spine and the rest is deferred.
							if maxl >= 2 {
								p0 := (j1+1)*mb + mb - 3
								at.add(aOpColA, p0, maxl-2, b1v, b2)
								p := p0 + (maxl-2)*(mb-1)
								u := a[p] + b2*a[p+1]
								a[p+1] = -b1v*a[p] + a[p+1]
								a[p] = u
							}
							i1 := j + m1
							if i1 <= n {
								g = b2 * A(i1, 1)
							}
						}
						za.rotA(j, b1v, b2)
					} else {
						u := d[j1-1]
						d[j1-1] = s2 * d[j-1]
						d[j-1] = s2 * u
						f1 := two * A(j, m1)
						f2 := b1v * A(j, mb)
						uu := b1v*(f2-f1) + A(j1, mb)
						setA(j, m1, b2*(b1v*A(j, m1)-A(j1, mb))+f2-A(j, m1))
						setA(j1, mb, b2*(b2*A(j1, mb)+f1)+A(j, mb))
						setA(j, mb, uu)
						at.add(aOpRowB, (j-1)*mb+mb-j+ugl-1, j2-ugl+1, b1v, b2)
						ugl = j
						setA(j1, 1, b2*A(j1, 1)+g)
						if j != n {
							maxl := min(m1, n-j1)
							if maxl >= 2 {
								p0 := (j1+1)*mb + mb - 3
								at.add(aOpColB, p0, maxl-2, b1v, b2)
								p := p0 + (maxl-2)*(mb-1)
								u2 := b2*a[p] + a[p+1]
								a[p+1] = -a[p] + b1v*a[p+1]
								a[p] = u2
							}
							i1 := j + m1
							if i1 <= n {
								g = A(i1, 1)
								setA(i1, 1, b1v*A(i1, 1))
							}
						}
						za.rotB(j, b1v, b2)
					}
				}
				// ONE CHASE DONE — and note the indentation: this is after the `for j` loop, not inside
				// it. It sat one level deeper until TestZTapeSegmentsCommute reported one op per
				// segment, which is what a per-STEP flush looks like. Correctness was never affected
				// (flushing more often is always safe), but the band-matrix tape never held more than a
				// single step's runs, so there was essentially nothing for its parallel replay to
				// partition — which is the real reason that path never engaged, not the grain.
				//
				// The next chase reads rows this one wrote, so the band-matrix runs must land now.
				at.flush()
				// The eigenvector rotations need not land here: they commute within a chase (j advances
				// by m1 >= 2, so consecutive rotations touch disjoint column pairs) but not across one,
				// so this only records the boundary for a concurrent replay.
				za.mark()
			}
			if k%bandEigProgressEvery == 0 {
				report(k)
			}
			// Checkpoint hook. The end of a k iteration is the only clean boundary in this routine:
			// a, d, z and k fully describe the state here, while inside a chase the rotation scalars
			// are carried across the j walk. See bandeig_checkpoint.go.
			if ckpt.enabled() {
				stop := ckpt.stopRequested()
				due := ckpt.interval > 0 && time.Since(lastSave) >= ckpt.interval
				if stop || due {
					// Bring the accumulator up to date before writing it. TWO things make it stale
					// here, and the first cost a failed TestBandEigCheckpointResumeIsBitExact:
					//
					//  1. the eigenvector tape is flushed when it FILLS or at the final sort, not per
					//     chase, so at an arbitrary column boundary it still holds pending rotations.
					//     Saving z without applying them silently drops every one, and the resumed run
					//     produced correct eigenvalues (which come from a and d) with wrong
					//     eigenvectors — the failure mode hardest to notice downstream.
					//  2. the host copy is only a mirror while a device owns the live one.
					za.flush()
					if za.dev != nil {
						za.dev.Download(za.z)
					}
					if err := saveBandEigReduction(ckpt.path, k, mb, a, d, za, fp); err != nil {
						// Never swallowed: a silently failing write leaves a multi-day reduction with
						// no restart point, which is the entire failure this checkpointing prevents.
						fmt.Fprintf(os.Stderr, "adcgo: banded eigensolve checkpoint at column %d of "+
							"%d FAILED: %v (no restart point from here)\n", k, n2, err)
					} else {
						fmt.Fprintf(os.Stderr, "adcgo: banded eigensolve checkpoint at column %d of %d\n",
							k, n2)
					}
					lastSave = time.Now()
					if stop {
						return true
					}
				}
			}
			// Periodic underflow rescale every 64 columns (bnd2td.f:124-145).
			if k%64 == 0 {
				// MARKS ARE LOAD-BEARING. The scale ops below touch pairwise-distinct columns, so
				// they commute among themselves — but not with the NEXT chase's rotations, which act
				// on those same columns. Nothing between the last chase's mark and here closes the
				// run, so without these two the rescale's ops share a run with the following chase's
				// and the concurrent device replay applies them simultaneously. That is not a small
				// error: TestGPUBandEigReplayParity saw device 1.5e+109 against host -1.4e-16 at
				// dim=1000, band=10, and passed at dim=40/200/260 only because the rescale never
				// fired there.
				za.mark()
				for j := k; j <= n; j++ {
					if d[j-1] >= dmin {
						continue
					}
					maxl := max(1, mb+1-j)
					for l := maxl; l <= m1; l++ {
						setA(j, l, dminrt*A(j, l))
					}
					if j != n {
						maxl2 := min(m1, n-j)
						for l := 1; l <= maxl2; l++ {
							i1 := j + l
							i2 := mb - l
							setA(i1, i2, dminrt*A(i1, i2))
						}
					}
					za.scale(j, dminrt)
					setA(j, mb, dmin*A(j, mb))
					d[j-1] = d[j-1] / dmin
				}
				za.mark()
			}
		}
	}

	if stopBand > 1 {
		// Stopped early: the caller wants the partially reduced band matrix, which is still in the
		// chase's SCALED representation — true A[i][j] = sqrt(d_i*d_j) * a(i,j), the same relation
		// label 800 below applies to the tridiagonal. unscaleBand does that conversion; nothing here
		// may touch a, d or z first.
		return false
	}

	// Label 800: form the tridiagonal (d, e, e2) and scale z.
	for j := 2; j <= n; j++ {
		e[j-1] = math.Sqrt(d[j-1])
	}
	za.scaleAll(e)
	u := 1.0
	for j := 2; j <= n; j++ {
		setA(j, m1, u*e[j-1]*A(j, m1))
		u = e[j-1]
		e2[j-1] = A(j, m1) * A(j, m1)
		setA(j, mb, d[j-1]*A(j, mb))
		d[j-1] = A(j, mb)
		e[j-1] = A(j, m1)
	}
	d[0] = A(1, mb)
	e[0] = 0
	e2[0] = 0
	return false
}

// tddiag is the QL-with-implicit-shifts tridiagonal eigensolver (modified EISPACK tql2),
// deferring the same rotations onto za. On exit d holds the ascending eigenvalues and za the
// correspondingly permuted partial eigenvectors. Port of ../ADC/libLanczos/tddiag.f. Returns
// the index of a non-converged root, or 0 on success.
//
// No flush is needed between bnd2td and here: bnd2td's tridiagonal assembly reads only a and d,
// and the QL sweep below reads only d and e, so the pending op stream crosses the boundary
// untouched. The sort at the end does read z, and flushes first.
func tddiag(za *zAccum, n int, d, e []float64) int {
	const (
		machep = 2.22045e-16
		mxiter = 30
	)
	if n == 1 {
		return 0
	}
	for i := 2; i <= n; i++ {
		e[i-2] = e[i-1]
	}
	f := 0.0
	b := 0.0
	e[n-1] = 0
	for l := 1; l <= n; l++ {
		j := 0
		h := machep * (math.Abs(d[l-1]) + math.Abs(e[l-1]))
		if b < h {
			b = h
		}
		// Look for a small sub-diagonal element (finds m; only its use as p=d(m) below matters).
		m := l
		for ; m <= n; m++ {
			if math.Abs(e[m-1]) <= b {
				break
			}
		}
		for math.Abs(e[l-1]) > b && j < mxiter {
			j++
			l1 := l + 1
			g := d[l-1]
			p := (d[l1-1] - g) / (e[l-1] * 2)
			r := math.Sqrt(p*p + 1)
			d[l-1] = e[l-1] / (p + math.Copysign(r, p))
			h = g - d[l-1]
			for i := l1; i <= n; i++ {
				d[i-1] -= h
			}
			f += h
			p = d[m-1]
			c := 1.0
			s := 0.0
			mml := m - l
			for ii := 1; ii <= mml; ii++ {
				i := m - ii
				g = c * e[i-1]
				h = c * p
				if math.Abs(p) >= math.Abs(e[i-1]) {
					c = e[i-1] / p
					r = math.Sqrt(c*c + 1)
					e[i] = s * p * r
					s = c / r
					c = 1 / r
				} else {
					c = p / e[i-1]
					r = math.Sqrt(c*c + 1)
					e[i] = s * e[i-1] * r
					s = 1 / r
					c = c * s
				}
				p = c*d[i-1] - s*g
				d[i] = h + s*(c*g+s*d[i-1])
				za.rotTD(i, c, s)
			}
			e[l-1] = s * p
			d[l-1] = c * p
		}
		if j == mxiter {
			return l
		}
		d[l-1] += f
	}

	// Ascending selection sort of eigenvalues, carrying the z columns along. This reads z, so
	// every deferred rotation has to have landed first.
	//
	// The sort is left exactly as the Fortran has it. It is selection sort WITH SWAPS, so it is
	// not a stable permutation: replacing it with a library sort reorders eigenvectors whenever
	// two eigenvalues come out bit-equal, which is exactly what Lanczos ghost pairs produce. It
	// is also a rounding error of the total cost.
	za.flush()
	for ii := 2; ii <= n; ii++ {
		i := ii - 1
		k := i
		p := d[i-1]
		for jj := ii; jj <= n; jj++ {
			if d[jj-1] >= p {
				continue
			}
			k = jj
			p = d[jj-1]
		}
		if k == i {
			continue
		}
		d[k-1] = d[i-1]
		d[i-1] = p
		za.swap(i, k)
	}
	return 0
}
