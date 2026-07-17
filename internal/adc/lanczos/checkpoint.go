package lanczos

// Checkpoint/restore of Solve's block-Krylov state, so a long solve can survive a process
// death (typically a SLURM walltime kill) and resume in a successor job. See Solve for where
// the hooks fire.
//
// Why this is near-pure serialization: Solve keeps the whole orthonormal Krylov basis
// resident for the entire run (full reorthogonalization + deflation), so the resumable state
// is just that basis, the projected block-tridiagonal T, and a few scalars. Nothing is
// recomputed on resume beyond the single in-flight block, so the continuation is
// bit-reproducible on the same backend.
//
// Format: a fixed little-endian header (magic, version, guard dims, progress scalars, blob
// lengths) followed by the raw float64 payloads. The blobs are written via an unsafe byte
// view rather than element-wise binary.Write — the basis alone is tens of GB — which assumes
// a little-endian host (every Helix node is x86-64). Writes are atomic: a sibling ".tmp" is
// fsynced and renamed over the target, with the previous generation kept as ".bak".

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"sync/atomic"
	"unsafe"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
)

const (
	ckptMagic   = "ADCGOLZ1" // 8 bytes; identifies a lanczos checkpoint
	ckptVersion = 1
)

// Checkpoint configures periodic and signal-driven serialization of Solve's Krylov state.
// A nil *Checkpoint (or an empty Path) disables checkpointing entirely, leaving Solve's
// behavior bit-for-bit unchanged.
type Checkpoint struct {
	Path  string       // file to save/restore; "" disables checkpointing
	Every int          // save every Every blocks (<=0 → save only when Stop fires)
	Stop  *atomic.Bool // when set true (e.g. by a SIGUSR1 handler), Solve saves at the next
	// block boundary and returns early with Result.Interrupted = true. May be nil.
}

// stopRequested reports whether an external signal has asked Solve to checkpoint and stop.
func (c *Checkpoint) stopRequested() bool { return c != nil && c.Stop != nil && c.Stop.Load() }

// ckptState is the resumable snapshot, captured at the top of Solve's iteration loop (before
// the current block is projected). rNext is intentionally omitted: it is recomputed on the
// first resumed iteration.
type ckptState struct {
	// Guard: must match the current problem/config or the checkpoint is rejected.
	N, Main, Maxdim, MaxBlocks int
	// Progress.
	Dim, BlkStart, BlkSize, Iter int
	// Payload.
	Basis []float64 // N*Dim, column-major (the first Dim basis columns)
	T     []float64 // Dim*Dim, row-major (leading block of the Maxdim×Maxdim projection)
}

// matches guards a loaded checkpoint against the current problem and its own internal
// consistency; a mismatch (different FCIDUMP/active space/-blocks, or a truncated file) makes
// Solve ignore it and start fresh.
func (s *ckptState) matches(n, main, maxdim, maxBlocks int) bool {
	return s.N == n && s.Main == main && s.Maxdim == maxdim && s.MaxBlocks == maxBlocks &&
		s.Dim >= main && s.Dim <= maxdim &&
		len(s.Basis) == n*s.Dim && len(s.T) == s.Dim*s.Dim
}

// floatBytes returns a byte view aliasing f's backing array (no copy). Little-endian host.
func floatBytes(f []float64) []byte {
	if len(f) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&f[0])), len(f)*8)
}

// writeCheckpoint serializes s to path atomically (tmp + fsync + rename), keeping the prior
// file as ".bak".
func writeCheckpoint(path string, s *ckptState) error {
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
	if _, err := w.WriteString(ckptMagic); err != nil {
		return fail(err)
	}
	hdr := []int64{
		ckptVersion,
		int64(s.N), int64(s.Main), int64(s.Maxdim), int64(s.MaxBlocks),
		int64(s.Dim), int64(s.BlkStart), int64(s.BlkSize), int64(s.Iter),
		int64(len(s.Basis)), int64(len(s.T)),
	}
	for _, v := range hdr {
		if err := binary.Write(w, binary.LittleEndian, v); err != nil {
			return fail(err)
		}
	}
	if _, err := w.Write(floatBytes(s.Basis)); err != nil {
		return fail(err)
	}
	if _, err := w.Write(floatBytes(s.T)); err != nil {
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
	if _, err := os.Stat(path); err == nil {
		_ = os.Rename(path, path+".bak")
	}
	return os.Rename(tmp, path)
}

// readCheckpoint deserializes a checkpoint. It returns (nil, nil) when the file does not
// exist (a fresh run), and a non-nil error on a corrupt/truncated file.
func readCheckpoint(path string) (*ckptState, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()
	r := bufio.NewReaderSize(f, 1<<22)

	magic := make([]byte, len(ckptMagic))
	if _, err := io.ReadFull(r, magic); err != nil {
		return nil, err
	}
	if string(magic) != ckptMagic {
		return nil, fmt.Errorf("lanczos checkpoint %s: bad magic", path)
	}
	var hdr [11]int64
	for i := range hdr {
		if err := binary.Read(r, binary.LittleEndian, &hdr[i]); err != nil {
			return nil, err
		}
	}
	if hdr[0] != ckptVersion {
		return nil, fmt.Errorf("lanczos checkpoint %s: version %d != %d", path, hdr[0], ckptVersion)
	}
	s := &ckptState{
		N: int(hdr[1]), Main: int(hdr[2]), Maxdim: int(hdr[3]), MaxBlocks: int(hdr[4]),
		Dim: int(hdr[5]), BlkStart: int(hdr[6]), BlkSize: int(hdr[7]), Iter: int(hdr[8]),
	}
	s.Basis = make([]float64, hdr[9])
	if _, err := io.ReadFull(r, floatBytes(s.Basis)); err != nil {
		return nil, err
	}
	s.T = make([]float64, hdr[10])
	if _, err := io.ReadFull(r, floatBytes(s.T)); err != nil {
		return nil, err
	}
	return s, nil
}

// loadResumable reads path, falling back to its ".bak" if the primary is missing or corrupt,
// and returns a state only if it passes the guard. Any non-matching or unreadable checkpoint
// yields (nil) so the caller starts fresh.
func loadResumable(path string, n, main, maxdim, maxBlocks int) *ckptState {
	for _, p := range []string{path, path + ".bak"} {
		if s, err := readCheckpoint(p); err == nil && s != nil && s.matches(n, main, maxdim, maxBlocks) {
			return s
		}
	}
	return nil
}

// removeCheckpoint deletes a checkpoint and its siblings, called when a solve completes so a
// later rerun of the same job does not resume a finished computation.
func removeCheckpoint(path string) {
	for _, p := range []string{path, path + ".bak", path + ".tmp"} {
		_ = os.Remove(p)
	}
}

// saveKrylov pulls the first dim basis columns off the backend and writes a checkpoint. The
// basis columns are contiguous (column-major, leading dimension n), so the first dim columns
// are exactly the first n*dim elements; only they are transferred, not the full n*maxdim
// panel. T's populated leading dim×dim block is copied out of the maxdim-wide host matrix.
func saveKrylov(be backend.Backend, path string, basis backend.BlockView, t backend.Mat,
	n, main, maxdim, maxBlocks, dim, blkStart, blkSize, iter int) error {
	basisHost := be.Download(basis.ColRange(0, dim).V) // length n*dim
	tHost := make([]float64, dim*dim)
	for i := 0; i < dim; i++ {
		copy(tHost[i*dim:(i+1)*dim], t.Data[i*maxdim:i*maxdim+dim])
	}
	return writeCheckpoint(path, &ckptState{
		N: n, Main: main, Maxdim: maxdim, MaxBlocks: maxBlocks,
		Dim: dim, BlkStart: blkStart, BlkSize: blkSize, Iter: iter,
		Basis: basisHost, T: tHost,
	})
}
