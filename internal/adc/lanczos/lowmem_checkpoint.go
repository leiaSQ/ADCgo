package lanczos

// lowmem_checkpoint.go — checkpoint/restore of SolveLowMem's short-recurrence state, so a
// multi-day DIP solve survives a process death (a walltime kill, or the cudaErrorLaunchFailure
// that cost job 14040960 its whole 1 d 15 h) and resumes in a successor job.
//
// This is a SIBLING of checkpoint.go, not a reuse of it, because the two drivers keep different
// state. Solve retains the entire orthonormal Krylov basis, so its checkpoint is that basis plus
// the projected T. SolveLowMem deliberately throws the basis away and keeps only a three-panel
// window, so what has to be saved is:
//
//   - the live [prev|cur] columns of the pc panel (rPrev+rCur of them) — the recurrence state,
//   - the accumulated per-block α/β (the `blocks` slice) — which is the projected matrix T, i.e.
//     the ANSWER, so a checkpoint doubles as a recoverable partial spectrum,
//   - the block-window scalars rPrev/rCur/dim/iter.
//
// docs/dip_lowmem_lanczos.md:129-135 anticipated exactly this shape ("checkpoint the
// short-recurrence state Q_{j-1}, Q_j, current α/β, iter").
//
// Mode A is NOT supported: it additionally retains hostQ, the whole basis on the host, for full
// reorthogonalization, and writing that out is precisely what the low-memory driver exists to
// avoid. SolveLowMem rejects the combination rather than silently ignoring the checkpoint.
//
// Format: magic, a fixed little-endian int64 header (version, guards, progress, counts), the
// per-block descriptors, the per-block α/β payloads, then the panel LAST so the one large payload
// is a single sequential streamed write. Writes are atomic (tmp + fsync + rename). Unlike
// checkpoint.go there is no ".bak" generation: at the production system's 274 GB panel a second copy costs half
// a terabyte to guard against a window the atomic rename has already closed.

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"time"

	"github.com/leiaSQ/ADCgo/backend"
)

const (
	lmCkptMagic   = "ADCGOLM1" // 8 bytes; identifies a low-memory (short-recurrence) checkpoint
	lmCkptVersion = 1
	lmHeaderInts  = 18 // int64s after the magic (see writeLowMemCheckpoint)
)

// lmPanelChunkBytes bounds the host staging buffer used to stream the pc panel to and from the
// backend. It is a BYTE budget, not a column count, because the two drivers differ by four orders
// of magnitude in n: checkpoint.go's ckptBasisChunkCols=256 is ~1.1 GB per chunk at SIP's
// n≈518k but would be 20.5 GB at DIP's n≈10M, which is exactly the kind of oversized transient
// that drove the SIP solve's RSS to the cgroup cap (jobs 14040959 / 14075367).
//
// A var rather than a const only so the chunk-boundary test can shrink it; nothing at runtime
// changes it.
var lmPanelChunkBytes = 1 << 30

// lmPanelChunkCols is how many n-tall columns fit the staging budget (at least one, so a single
// column taller than the budget still works — it just exceeds it, which is unavoidable).
//
// At the production system's n=10,014,483 a column is 80.1 MB, so this yields 13 columns (~1.04 GB) per chunk.
// At SIP's n≈518k it yields 259, reassuringly close to the 256 that checkpoint.go's
// ckptBasisChunkCols was hand-tuned to — which is why the byte-budget form is trustworthy for both.
func lmPanelChunkCols(n int) int {
	if n <= 0 {
		return 1
	}
	return max(lmPanelChunkBytes/(n*8), 1)
}

// lmCkptState is SolveLowMem's resumable snapshot, captured at the TOP of the iteration loop —
// before ApplyBlock, before α is recorded, and before the window shift. The in-flight iteration is
// simply redone on resume, which is what makes the continuation bit-reproducible on the same
// backend (the same property Solve's checkpoint relies on).
type lmCkptState struct {
	// Guards: must match the current problem/config or the checkpoint is rejected. DeflTol is one
	// of them because the rank-revealing deflation in orthBlock decides each block's surviving
	// column count from it, so a different tolerance yields a different — and silently
	// incompatible — block structure from the same operator.
	N, Main, B, Maxdim, MaxBlocks int
	ModeB                         bool
	DeflTol                       float64
	// Progress.
	Dim, RPrev, RCur, Iter int
	// Timing accumulated so far, carried across the chain so -profile totals describe the whole
	// solve rather than only the generation that happened to finish it.
	Timing Timing
	// Payload.
	Blocks []lmBlock
	Panel  []float64 // N*(RPrev+RCur), column-major; nil when the caller streams it instead
}

// panelCols is the number of live columns in the saved window.
func (s *lmCkptState) panelCols() int { return s.RPrev + s.RCur }

// matches guards a loaded checkpoint against the current problem and its own internal consistency.
// A mismatch (different FCIDUMP/active space/-blocks/-lowmem-block, or a truncated file) makes
// SolveLowMem ignore it and start fresh rather than resume something incompatible.
func (s *lmCkptState) matches(n, main, b, maxdim, maxBlocks int, modeB bool, deflTol float64) bool {
	if s.N != n || s.Main != main || s.B != b || s.Maxdim != maxdim ||
		s.MaxBlocks != maxBlocks || s.ModeB != modeB || s.DeflTol != deflTol {
		return false
	}
	if s.RPrev < 0 || s.RCur <= 0 || s.panelCols() > 2*b || s.Dim <= 0 || s.Dim > maxdim {
		return false
	}
	if len(s.Blocks) == 0 || s.Iter < 0 {
		return false
	}
	// The block list must account for exactly Dim columns, with the offsets the loop maintains.
	total := 0
	for i, blk := range s.Blocks {
		if blk.size <= 0 || blk.off != total {
			return false
		}
		// β couples this block to the previous one, so its column count is the previous block's
		// size. Its ROW count is deliberately not checked against blk.size: the rank clamp at
		// lowmem.go:193-195 shrinks `size` after orth already produced R, so on the run that
		// truncates at MaxDim beta.Rows legitimately exceeds it.
		if i == 0 {
			if blk.beta.Rows != 0 {
				return false
			}
		} else if blk.beta.Cols != s.Blocks[i-1].size {
			return false
		}
		if len(blk.beta.Data) != blk.beta.Rows*blk.beta.Cols {
			return false
		}
		// Every block carries α = Q_iᵀ·M·Q_i EXCEPT the last: it is appended with α unset and
		// receives it partway through the next iteration (lowmem.go:160), which is precisely the
		// iteration a checkpoint re-does. A file whose last block already has α was written
		// somewhere other than the top of the loop and is not a valid re-entry point.
		wantAlpha := blk.size * blk.size
		if i == len(s.Blocks)-1 {
			wantAlpha = 0
		}
		if len(blk.alpha) != wantAlpha {
			return false
		}
		total += blk.size
	}
	// One accepted block per completed iteration, plus the start block.
	return total == s.Dim && len(s.Blocks) == s.Iter+1 &&
		s.Blocks[len(s.Blocks)-1].size == s.RCur &&
		(len(s.Blocks) == 1) == (s.RPrev == 0)
}

// writeLowMemCheckpoint serializes s to path atomically (tmp + fsync + rename).
//
// panelLen is the declared panel element count and writePanel streams those elements; passing nil
// writes s.Panel directly (the in-memory form the round-trip test uses). saveLowMem streams off
// the backend in chunks so the host never materializes the whole panel — at production scale that is
// 274 GB, and distBackend.Download on the full panel would allocate all of it in one slice.
func writeLowMemCheckpoint(path string, s *lmCkptState, panelLen int, writePanel func(io.Writer) error) error {
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	w := bufio.NewWriterSize(f, 1<<22)

	fail := func(e error) error {
		f.Close()
		os.Remove(tmp)
		return e
	}
	if _, err := w.WriteString(lmCkptMagic); err != nil {
		return fail(err)
	}
	modeB := int64(0)
	if s.ModeB {
		modeB = 1
	}
	hdr := [lmHeaderInts]int64{
		lmCkptVersion,
		int64(s.N), int64(s.Main), int64(s.B), int64(s.Maxdim), int64(s.MaxBlocks), modeB,
		int64(math.Float64bits(s.DeflTol)),
		int64(s.Dim), int64(s.RPrev), int64(s.RCur), int64(s.Iter),
		int64(len(s.Blocks)),
		int64(s.Timing.Apply), int64(s.Timing.Orth), int64(s.Timing.Proj),
		int64(s.Timing.Eig), int64(s.Timing.Back),
	}
	for _, v := range hdr {
		if err := binary.Write(w, binary.LittleEndian, v); err != nil {
			return fail(err)
		}
	}
	// Block descriptors first, so a reader can size every payload before touching it.
	for _, blk := range s.Blocks {
		d := [5]int64{int64(blk.off), int64(blk.size), int64(len(blk.alpha)),
			int64(blk.beta.Rows), int64(blk.beta.Cols)}
		for _, v := range d {
			if err := binary.Write(w, binary.LittleEndian, v); err != nil {
				return fail(err)
			}
		}
	}
	// Block payloads: α (column-major size×size) then β (row-major size×prevSize).
	for _, blk := range s.Blocks {
		if _, err := w.Write(floatBytes(blk.alpha)); err != nil {
			return fail(err)
		}
		if _, err := w.Write(floatBytes(blk.beta.Data)); err != nil {
			return fail(err)
		}
	}
	// Panel last: the one large sequential payload.
	if err := binary.Write(w, binary.LittleEndian, int64(panelLen)); err != nil {
		return fail(err)
	}
	if writePanel != nil {
		if err := writePanel(w); err != nil {
			return fail(err)
		}
	} else if _, err := w.Write(floatBytes(s.Panel)); err != nil {
		return fail(err)
	}

	if err := w.Flush(); err != nil {
		return fail(err)
	}
	if err := f.Sync(); err != nil {
		return fail(err)
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, path)
}

// readLowMemCheckpoint deserializes a checkpoint. It returns (nil, nil) when the file does not
// exist (a fresh run) and a non-nil error on a corrupt or truncated one.
//
// readPanel, when non-nil, is called with the reader positioned at the panel payload and the
// element count, so the caller can stream it straight onto the backend without a host copy of the
// whole thing. When nil the panel is read into s.Panel.
func readLowMemCheckpoint(path string, readPanel func(r io.Reader, elems int) error) (*lmCkptState, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()
	r := bufio.NewReaderSize(f, 1<<22)

	magic := make([]byte, len(lmCkptMagic))
	if _, err := io.ReadFull(r, magic); err != nil {
		return nil, err
	}
	if string(magic) != lmCkptMagic {
		return nil, fmt.Errorf("lowmem checkpoint %s: bad magic", path)
	}
	var hdr [lmHeaderInts]int64
	for i := range hdr {
		if err := binary.Read(r, binary.LittleEndian, &hdr[i]); err != nil {
			return nil, err
		}
	}
	if hdr[0] != lmCkptVersion {
		return nil, fmt.Errorf("lowmem checkpoint %s: version %d != %d", path, hdr[0], lmCkptVersion)
	}
	s := &lmCkptState{
		N: int(hdr[1]), Main: int(hdr[2]), B: int(hdr[3]), Maxdim: int(hdr[4]),
		MaxBlocks: int(hdr[5]), ModeB: hdr[6] == 1,
		DeflTol: math.Float64frombits(uint64(hdr[7])),
		Dim:     int(hdr[8]), RPrev: int(hdr[9]), RCur: int(hdr[10]), Iter: int(hdr[11]),
		Timing: Timing{
			Apply: time.Duration(hdr[13]), Orth: time.Duration(hdr[14]),
			Proj: time.Duration(hdr[15]), Eig: time.Duration(hdr[16]),
			Back: time.Duration(hdr[17]),
		},
	}
	nblk := int(hdr[12])
	if nblk < 0 || nblk > s.Maxdim+1 {
		return nil, fmt.Errorf("lowmem checkpoint %s: implausible block count %d", path, nblk)
	}

	type desc struct{ off, size, alphaLen, betaRows, betaCols int }
	descs := make([]desc, nblk)
	for i := range descs {
		var d [5]int64
		for j := range d {
			if err := binary.Read(r, binary.LittleEndian, &d[j]); err != nil {
				return nil, err
			}
		}
		descs[i] = desc{int(d[0]), int(d[1]), int(d[2]), int(d[3]), int(d[4])}
	}
	s.Blocks = make([]lmBlock, nblk)
	for i, d := range descs {
		blk := lmBlock{off: d.off, size: d.size}
		blk.alpha = make([]float64, d.alphaLen)
		if _, err := io.ReadFull(r, floatBytes(blk.alpha)); err != nil {
			return nil, err
		}
		blk.beta = backend.Mat{Rows: d.betaRows, Cols: d.betaCols,
			Data: make([]float64, d.betaRows*d.betaCols)}
		if _, err := io.ReadFull(r, floatBytes(blk.beta.Data)); err != nil {
			return nil, err
		}
		s.Blocks[i] = blk
	}

	var panelLen int64
	if err := binary.Read(r, binary.LittleEndian, &panelLen); err != nil {
		return nil, err
	}
	if want := int64(s.N) * int64(s.panelCols()); panelLen != want {
		return nil, fmt.Errorf("lowmem checkpoint %s: panel length %d != n*(rPrev+rCur) = %d",
			path, panelLen, want)
	}
	if readPanel != nil {
		if err := readPanel(r, int(panelLen)); err != nil {
			return nil, err
		}
	} else {
		s.Panel = make([]float64, panelLen)
		if _, err := io.ReadFull(r, floatBytes(s.Panel)); err != nil {
			return nil, err
		}
	}
	return s, nil
}

// LowMemCheckpointable reports why SolveLowMem cannot checkpoint this configuration, or nil if it
// can. Only Mode B — block width == MainBlockSize() with an operator that implements
// SatelliteOperator — is resumable: Mode A additionally retains the whole basis on the host for
// its full reorthogonalization, which is the very object checkpointing exists to avoid writing.
//
// It is exported so the command line can reject the combination with a clear message before a
// multi-day job starts, rather than reaching SolveLowMem's panic backstop. The rule lives here, in
// one place, so the driver and the CLI cannot drift apart.
func LowMemCheckpointable(op Operator, opts Options) error {
	main := op.MainBlockSize()
	b := opts.LowMemBlock
	if b <= 0 || b > main {
		b = main
	}
	if b != main {
		return fmt.Errorf("lowmem checkpointing needs Mode B (block == main): block width %d != main %d; "+
			"Mode A retains the full basis on the host and is not resumable", b, main)
	}
	if _, ok := op.(SatelliteOperator); !ok {
		return fmt.Errorf("lowmem checkpointing needs Mode B, but %T does not implement "+
			"SatelliteOperator, so the Tarantelli gate is off", op)
	}
	return nil
}

// removeLowMemCheckpoint deletes a checkpoint and its in-progress sibling, called when a solve
// completes so a later rerun of the same job does not resume a finished computation.
func removeLowMemCheckpoint(path string) {
	for _, p := range []string{path, path + ".tmp"} {
		_ = os.Remove(p)
	}
}

// saveLowMem streams SolveLowMem's state into a checkpoint. The pc panel's live columns are
// contiguous (column-major, leading dimension n), so any column range of them is contiguous too —
// which is what lets this be chunked, on both gpuBackend and the row-partitioned distBackend
// (distVec.Slice resolves a column range per partition).
//
// One reusable staging buffer serves every chunk: be.Download returns a FRESH slice per call, so
// without it the bytes allocated per checkpoint would scale with the panel and the run's total
// allocation with panel×checkpoints — the same churn that OOM-killed the production SIP solve.
// Backends that implement BufferedDownloader avoid the copy entirely; the rest fall back to
// Download, which is correct, just more allocating.
func saveLowMem(be backend.Backend, path string, pc backend.BlockView, blocks []lmBlock,
	n, main, b, maxdim, maxBlocks int, modeB bool, deflTol float64,
	dim, rPrev, rCur, iter int, tm Timing) error {

	cols := rPrev + rCur
	cw := min(lmPanelChunkCols(n), cols)

	bd, buffered := be.(backend.BufferedDownloader)
	var stage []float64
	if buffered {
		stage = make([]float64, n*cw)
	}
	writePanel := func(w io.Writer) error {
		for c0 := 0; c0 < cols; c0 += cw {
			width := min(cw, cols-c0)
			col := pc.ColRange(c0, c0+width).V
			var chunk []float64
			if buffered {
				chunk = stage[:n*width]
				bd.DownloadInto(chunk, col)
			} else {
				chunk = be.Download(col) // length n*width
			}
			if _, err := w.Write(floatBytes(chunk)); err != nil {
				return err
			}
		}
		return nil
	}

	return writeLowMemCheckpoint(path, &lmCkptState{
		N: n, Main: main, B: b, Maxdim: maxdim, MaxBlocks: maxBlocks, ModeB: modeB,
		DeflTol: deflTol,
		Dim:     dim, RPrev: rPrev, RCur: rCur, Iter: iter, Timing: tm, Blocks: blocks,
	}, n*cols, writePanel)
}

// loadLowMem reads path, restores the pc panel onto the backend in chunks, and returns the state.
// It returns nil when there is no checkpoint, when the file is unreadable, or when the guard
// rejects it — in every case the caller starts fresh.
//
// The panel is restored into pc's leading `RPrev+RCur` columns, which is exactly the layout the
// loop expects: `cur` is pc.ColRange(rPrev, rPrev+rCur) and orthogonalization runs against
// pc.ColRange(0, rPrev+rCur).
func loadLowMem(be backend.Backend, path string, pc backend.BlockView,
	n, main, b, maxdim, maxBlocks int, modeB bool, deflTol float64) *lmCkptState {

	// Two passes: the first reads the guard fields only (skipping the panel), so a checkpoint for
	// a different problem is rejected before any of its panel touches the device.
	head, err := readLowMemCheckpoint(path, func(io.Reader, int) error { return nil })
	if err != nil || head == nil || !head.matches(n, main, b, maxdim, maxBlocks, modeB, deflTol) {
		return nil
	}

	cols := head.panelCols()
	cw := min(lmPanelChunkCols(n), cols)
	stage := make([]float64, n*cw)
	c0 := 0
	restore := func(r io.Reader, elems int) error {
		for c0 < cols {
			width := min(cw, cols-c0)
			buf := stage[:n*width]
			if _, err := io.ReadFull(r, floatBytes(buf)); err != nil {
				return err
			}
			up := be.Upload(buf)
			be.Copy(pc.ColRange(c0, c0+width).V, up)
			be.Free(up)
			c0 += width
		}
		return nil
	}
	s, err := readLowMemCheckpoint(path, restore)
	if err != nil || s == nil {
		return nil
	}
	return s
}
