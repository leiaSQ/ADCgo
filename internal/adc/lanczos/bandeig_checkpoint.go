package lanczos

// bandeig_checkpoint.go — checkpoint/restore of the projected banded eigensolver's band→tridiagonal
// reduction, so a multi-day Mode B eigensolve survives a walltime kill and resumes in a successor
// job.
//
// WHY THIS EXISTS, in numbers. Everything else in this tree checkpoints the Krylov phase; the
// reduction that follows it was a single uninterruptible call. Measured at the production
// half-bandwidth and extrapolated to the melanin DIP shape (dim 308000, band 3079), that call is
// ~144 h serial — longer than the 120 h walltime it runs under. With no restart point inside it,
// every generation of the daisychain restored the Krylov state, ground ~114 h into the reduction,
// was killed, and started over: the chain could not converge, and each generation burned five days
// of eight H200s to reach exactly the point the previous one reached. Job 14834817 spent 4 d 19 h
// doing that before this existed.
//
// This is a THIRD sibling of checkpoint.go (Solve) and lowmem_checkpoint.go (SolveLowMem), not a
// reuse of either, because the state is completely different: not a Krylov basis or a panel window,
// but the partially reduced band matrix, the diagonal scaling, the eigenvector accumulator, and the
// column the outer loop reached.
//
// WHERE THE BOUNDARY IS. bnd2td's outer k loop is the only clean one. At the end of a k iteration
// (after the periodic underflow rescale) the live state is exactly {a, d, z, k}: g and ugl are
// per-chase and re-derived, the deferred-op tapes are flushed per chase, and e/e2 are not written
// until the tridiagonal assembly after the loop. Inside a chase there is no such point — the
// rotation scalars are carried in registers across the j walk.
//
// Format: magic, a fixed little-endian int64 header, then d, then a, then z — the two large
// payloads last and contiguous, so the write is sequential. Atomic (tmp + fsync + rename). No
// ".bak" generation: at production the payload is ~15 GB and a second copy would cost 30 GB to
// guard a window the atomic rename has already closed, the same trade lowmem_checkpoint.go makes.

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"hash/crc64"
	"io"
	"os"
	"sync/atomic"
	"time"
)

// bandEigCkptInterval is how often the band reduction saves, in wall time.
//
// The payload is a and z, ~15 GB at the production shape, so a save costs seconds of sequential
// GPFS write. Thirty minutes bounds the work a kill can destroy at half an hour while keeping the
// I/O under 1% of a run that takes days. It is wall time rather than a column stride because the
// cost of one column is proportional to (dim-k): a fixed stride would fire minutes apart at the
// start of the reduction and seconds apart at the end.
const bandEigCkptInterval = 30 * time.Minute

const (
	beCkptMagic   = "ADCGOBE1" // 8 bytes; identifies a banded-eigensolve reduction checkpoint
	beCkptVersion = 1
	beHeaderInts  = 9 // int64s after the magic (see writeBandEigCheckpoint)
)

// beCkptTable is the CRC-64 table used to fingerprint the input band matrix. ECMA rather than ISO
// for no reason beyond matching Go's own documented example; only self-consistency matters.
var beCkptTable = crc64.MakeTable(crc64.ECMA)

// bandEigCkpt is the reduction's checkpoint configuration, threaded in through bandEigOpts.
//
// interval is wall time, not a column count, and deliberately so: the work in one column of k is
// proportional to (dim-k), so the loop crawls through early columns and races through late ones. A
// fixed column stride would put the checkpoints minutes apart at the start and seconds apart at the
// end, while what actually needs bounding is how much wall time a kill can cost.
type bandEigCkpt struct {
	path     string        // file to save/restore; "" disables checkpointing
	interval time.Duration // save at the first k boundary this long after the last save; <=0 → only on stop
	stop     *atomic.Bool  // when set (e.g. by a SIGUSR1 handler), save at the next k boundary and give up
}

func (c *bandEigCkpt) enabled() bool { return c != nil && c.path != "" }

// stopRequested reports whether an external signal has asked the reduction to save and stop. Same
// shape as Checkpoint.stopRequested, so the two cannot drift in meaning.
func (c *bandEigCkpt) stopRequested() bool { return c != nil && c.stop != nil && c.stop.Load() }

// beCkptState is the resumable snapshot of bnd2td, captured at the end of a k iteration.
type beCkptState struct {
	// Guard: must match the current problem or the checkpoint is rejected. Fingerprint is a CRC-64
	// of the ORIGINAL band matrix, which is what makes resuming onto a different Krylov state
	// impossible — dim and mb alone would not notice a different matrix of the same shape, and the
	// result would be silently wrong rather than obviously broken.
	Dim, Mb, Top, Bot, Ld int
	Fingerprint           uint64
	// Progress: the last completed column of bnd2td's outer loop.
	K int
	// Payload.
	D []float64 // Dim
	A []float64 // Mb*Dim, the partially reduced band matrix
	Z []float64 // Ld*Dim, the eigenvector accumulator
}

// matches guards a loaded checkpoint against the current problem and its own internal consistency.
// A mismatch makes the solve start the reduction from scratch rather than trust it.
func (s *beCkptState) matches(dim, mb, top, bot, ld int, fp uint64) bool {
	return s.Dim == dim && s.Mb == mb && s.Top == top && s.Bot == bot && s.Ld == ld &&
		s.Fingerprint == fp &&
		s.K >= 1 && s.K <= dim &&
		len(s.D) == dim && len(s.A) == mb*dim && len(s.Z) == ld*dim
}

// bandEigFingerprint is the CRC-64 of the input band matrix. It is computed once per solve, before
// the reduction overwrites the matrix in place, which is why it has to be taken from bandStorage
// rather than recomputed from `a` at resume time.
func bandEigFingerprint(bs bandStorage) uint64 {
	return crc64.Checksum(floatBytes(bs.data), beCkptTable)
}

// writeBandEigCheckpoint serializes s to path atomically (tmp + fsync + rename).
func writeBandEigCheckpoint(path string, s *beCkptState) error {
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
	if _, err := w.WriteString(beCkptMagic); err != nil {
		return fail(err)
	}
	hdr := [beHeaderInts]int64{
		beCkptVersion,
		int64(s.Dim), int64(s.Mb), int64(s.Top), int64(s.Bot), int64(s.Ld),
		int64(s.Fingerprint),
		int64(s.K),
		int64(len(s.Z)),
	}
	for _, v := range hdr {
		if err := binary.Write(w, binary.LittleEndian, v); err != nil {
			return fail(err)
		}
	}
	for _, p := range [][]float64{s.D, s.A, s.Z} {
		if len(p) == 0 {
			continue
		}
		if _, err := w.Write(floatBytes(p)); err != nil {
			return fail(err)
		}
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

// readBandEigCheckpoint deserializes a checkpoint. It returns (nil, nil) when the file does not
// exist — a fresh run, the overwhelmingly common case — and a non-nil error on a corrupt or
// truncated one, which the caller reports and then ignores rather than treating as fatal.
func readBandEigCheckpoint(path string) (*beCkptState, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()
	r := bufio.NewReaderSize(f, 1<<22)

	magic := make([]byte, len(beCkptMagic))
	if _, err := io.ReadFull(r, magic); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if string(magic) != beCkptMagic {
		return nil, fmt.Errorf("not a banded-eigensolve checkpoint (magic %q)", magic)
	}
	var hdr [beHeaderInts]int64
	for i := range hdr {
		if err := binary.Read(r, binary.LittleEndian, &hdr[i]); err != nil {
			return nil, fmt.Errorf("read header word %d: %w", i, err)
		}
	}
	if hdr[0] != beCkptVersion {
		return nil, fmt.Errorf("checkpoint version %d, want %d", hdr[0], beCkptVersion)
	}
	s := &beCkptState{
		Dim: int(hdr[1]), Mb: int(hdr[2]), Top: int(hdr[3]), Bot: int(hdr[4]), Ld: int(hdr[5]),
		Fingerprint: uint64(hdr[6]),
		K:           int(hdr[7]),
	}
	zLen := int(hdr[8])
	if s.Dim <= 0 || s.Mb <= 0 || s.Ld < 0 || zLen < 0 {
		return nil, fmt.Errorf("implausible header: dim=%d mb=%d ld=%d len(z)=%d", s.Dim, s.Mb, s.Ld, zLen)
	}
	s.D = make([]float64, s.Dim)
	s.A = make([]float64, s.Mb*s.Dim)
	s.Z = make([]float64, zLen)
	for _, p := range []struct {
		name string
		buf  []float64
	}{{"d", s.D}, {"a", s.A}, {"z", s.Z}} {
		if len(p.buf) == 0 {
			continue
		}
		if _, err := io.ReadFull(r, floatBytes(p.buf)); err != nil {
			return nil, fmt.Errorf("read %s payload: %w", p.name, err)
		}
	}
	return s, nil
}

// removeBandEigCheckpoint deletes a checkpoint and its in-progress sibling, called when the
// reduction completes so a later rerun of the same job does not resume a finished computation.
func removeBandEigCheckpoint(path string) {
	for _, p := range []string{path, path + ".tmp"} {
		_ = os.Remove(p)
	}
}

// saveBandEigReduction writes the reduction's state at the end of column k.
//
// za.z is the host copy, which is STALE whenever a device accumulator holds the live one, so the
// caller must have downloaded it first — see bnd2td's checkpoint hook. Getting that wrong would
// checkpoint a z frozen at its seed values and resume into silent nonsense, which is why the
// download is not optional and not deferred.
func saveBandEigReduction(path string, k int, mb int, a, d []float64, za *zAccum, fp uint64) error {
	return writeBandEigCheckpoint(path, &beCkptState{
		Dim: za.n, Mb: mb, Top: za.top, Bot: za.bot, Ld: za.ld,
		Fingerprint: fp,
		K:           k,
		D:           d,
		A:           a,
		Z:           za.z,
	})
}
