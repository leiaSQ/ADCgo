package main

// mainblock_cache.go — persistence for the assembled SIP 1h/1h main block.
//
// WHY. The main block is 58×58 for the production system — 26 KiB — and job 14551670 measured it taking
// **8 h 16 m 27 s** to build, against 3 m 55 s for the 58×518,056 coupling beside it and 3 s for
// the whole matrix-free satellite region. It is small in DIMENSION, not in cost: every element is
// c11 → c11_3sums, four five-deep a,b,l,c,d sums that are O(nvir⁴·nocc) each, and the production system is C1 so
// none of the symmetry filters prune anything. 1,711 lower-triangle elements × ~6.2e10 innermost
// iterations = ~1.06e14 iterations for 26 KiB of output.
//
// Nothing covered it. `-checkpoint` protects only lanczos.Solve's Krylov state, which is not
// reached until the whole operator is assembled, so every generation of the daisychain paid the
// 8 h again — out of a 120 h allocation, and again for each generation after that. This is the
// same trade sigma_cache.go makes for Σ(∞) (78 h for 351 KB) and the file mirrors it deliberately.
//
// Only the main block is cached. The coupling costs 4 minutes and would add 229 MiB; the
// satellite region is matrix-free and has no resident bytes at all. Caching the 26 KiB that costs
// 8 h is where the whole benefit is.
//
// The guard is conservative for the same reason Σ's is: a WRONG main block is not a crash, it
// silently shifts every ionization line and still looks publishable. Anything that could change
// the block invalidates the cache — the ADC order, the sector's irrep and spin multiplicity, the
// orbital-space dimensions, the WERT3 flag, a hash of the orbital energies, a hash of the static
// self-energy actually subtracted into the block, and the FCIDUMP's size and mtime. A miss costs
// a rebuild; a false hit costs a wrong spectrum, so the trade is not symmetric.

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"hash/fnv"
	"io"
	"math"
	"os"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/sip"
)

const (
	mainBlockCacheMagic   = "ADCGOMB1"
	mainBlockCacheVersion = 1
)

// mainBlockCacheKey fingerprints everything the assembled 1h/1h block depends on. Two runs
// agreeing on all of it must produce the same matrix; any disagreement rebuilds.
type mainBlockCacheKey struct {
	Order       int
	Sym         int
	Norb, Nocc  int
	Main        int // 1h main-space dimension
	N           int // full sector dimension — distinguishes spaces the other fields would not
	Wert3       bool
	EpsHash     uint64 // FNV-1a over the orbital energies
	SigmaHash   uint64 // FNV-1a over Σ_ij across the main space (0 when Σ is off)
	CoreHash    uint64 // FNV-1a over the CVS core-orbital list (0 when not CVS)
	FcidumpSize int64
	FcidumpMod  int64 // UnixNano
}

// mainBlockCacheKeyFor builds the key. Σ is hashed by VALUE over exactly the indices the block
// subtracts it at, rather than trusting that the Σ cache key upstream agreed — Σ enters the main
// block directly, so a mismatch there is a mismatch here.
func mainBlockCacheKeyFor(order, sym, norb, nocc, main, n int, wert3 bool, core []int,
	eps []float64, sigma func(i, j int) float64, fcidump string) mainBlockCacheKey {
	hashFloats := func(vals func(emit func(float64))) uint64 {
		h := fnv.New64a()
		var buf [8]byte
		vals(func(v float64) {
			binary.LittleEndian.PutUint64(buf[:], math.Float64bits(v))
			_, _ = h.Write(buf[:])
		})
		return h.Sum64()
	}
	k := mainBlockCacheKey{
		Order: order, Sym: sym,
		Norb: norb, Nocc: nocc, Main: main, N: n, Wert3: wert3,
		EpsHash: hashFloats(func(emit func(float64)) {
			for _, e := range eps {
				emit(e)
			}
		}),
	}
	if sigma != nil {
		k.SigmaHash = hashFloats(func(emit func(float64)) {
			for i := range nocc {
				for j := range nocc {
					emit(sigma(i, j))
				}
			}
		})
	}
	if len(core) > 0 {
		h := fnv.New64a()
		var buf [8]byte
		for _, c := range core {
			binary.LittleEndian.PutUint64(buf[:], uint64(c))
			_, _ = h.Write(buf[:])
		}
		k.CoreHash = h.Sum64()
	}
	if fi, err := os.Stat(fcidump); err == nil {
		k.FcidumpSize = fi.Size()
		k.FcidumpMod = fi.ModTime().UnixNano()
	}
	return k
}

func (k mainBlockCacheKey) equal(o mainBlockCacheKey) bool { return k == o }

func (k mainBlockCacheKey) header(rows, cols int) []int64 {
	w3 := int64(0)
	if k.Wert3 {
		w3 = 1
	}
	return []int64{
		mainBlockCacheVersion,
		int64(k.Order), int64(k.Sym),
		int64(k.Norb), int64(k.Nocc), int64(k.Main), int64(k.N), w3,
		int64(k.EpsHash), int64(k.SigmaHash), int64(k.CoreHash),
		k.FcidumpSize, k.FcidumpMod,
		int64(rows), int64(cols),
	}
}

// writeMainBlockCache persists the block under key, atomically (tmp + fsync + rename).
func writeMainBlockCache(path string, k mainBlockCacheKey, m backend.Mat) error {
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	w := bufio.NewWriterSize(f, 1<<16)
	fail := func(e error) error {
		f.Close()
		os.Remove(tmp)
		return e
	}
	if _, err := w.WriteString(mainBlockCacheMagic); err != nil {
		return fail(err)
	}
	for _, v := range k.header(m.Rows, m.Cols) {
		if err := binary.Write(w, binary.LittleEndian, v); err != nil {
			return fail(err)
		}
	}
	if err := binary.Write(w, binary.LittleEndian, m.Data); err != nil {
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

// readMainBlockCache returns the cached block when the file exists AND its key matches. It
// returns (zero Mat, false, nil) for "no usable cache" — a missing file, a corrupt one, or a
// stale key are all the same to the caller, which just rebuilds.
func readMainBlockCache(path string, want mainBlockCacheKey) (backend.Mat, bool, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return backend.Mat{}, false, nil
		}
		return backend.Mat{}, false, err
	}
	defer f.Close()
	r := bufio.NewReaderSize(f, 1<<16)

	magic := make([]byte, len(mainBlockCacheMagic))
	if _, err := io.ReadFull(r, magic); err != nil {
		return backend.Mat{}, false, err
	}
	if string(magic) != mainBlockCacheMagic {
		return backend.Mat{}, false, fmt.Errorf("main-block cache %s: bad magic", path)
	}
	var hdr [15]int64
	for i := range hdr {
		if err := binary.Read(r, binary.LittleEndian, &hdr[i]); err != nil {
			return backend.Mat{}, false, err
		}
	}
	if hdr[0] != mainBlockCacheVersion {
		return backend.Mat{}, false, nil // a different format generation: rebuild rather than guess
	}
	rows, cols := int(hdr[13]), int(hdr[14])
	if rows < 0 || cols < 0 || rows > 1<<20 || cols > 1<<20 {
		return backend.Mat{}, false, fmt.Errorf("main-block cache %s: implausible dims %dx%d", path, rows, cols)
	}
	got := mainBlockCacheKey{
		Order: int(hdr[1]), Sym: int(hdr[2]),
		Norb: int(hdr[3]), Nocc: int(hdr[4]), Main: int(hdr[5]), N: int(hdr[6]), Wert3: hdr[7] == 1,
		EpsHash: uint64(hdr[8]), SigmaHash: uint64(hdr[9]), CoreHash: uint64(hdr[10]),
		FcidumpSize: hdr[11], FcidumpMod: hdr[12],
	}
	if !got.equal(want) {
		return backend.Mat{}, false, nil // stale: different orbital space, order, sector or Σ
	}
	m := backend.NewMat(rows, cols)
	if err := binary.Read(r, binary.LittleEndian, m.Data); err != nil {
		return backend.Mat{}, false, err
	}
	return m, true, nil
}

// installMainBlockCache wires the persistent 1h/1h cache onto a SIP matrix, unless -mainblock-cache
// is "off". The path is per-sector (order, multiplicity, irrep) because -sym all solves several
// sectors from one process and each has its own block.
//
// Every failure mode is non-fatal and loud: an unusable file rebuilds, and a failed write warns,
// because the whole point is that the NEXT generation does not spend eight hours on 26 KiB.
func installMainBlockCache(mx *sip.Matrix, sp *sip.Space, eps []float64, order int, cfg sipConfig) {
	base := cfg.mainCache
	if base == "off" {
		return
	}
	if base == "auto" || base == "" {
		base = cfg.fcidumpPath + ".mainblock"
	}
	path := fmt.Sprintf("%s.o%d.i%d.cache", base, order, sp.Sym)
	key := mainBlockCacheKeyFor(order, sp.Sym, sp.Norb, sp.Nocc,
		sp.MainBlockSize(), sp.Size(), cfg.wert3, cfg.core, eps, cfg.sig, cfg.fcidumpPath)

	load := func() (backend.Mat, bool) {
		m, ok, err := readMainBlockCache(path, key)
		switch {
		case err != nil:
			fmt.Fprintf(os.Stderr, "adcgo: sip sym=%d main block: ignoring unusable cache %s: %v\n",
				sp.Sym, path, err)
			return backend.Mat{}, false
		case ok:
			fmt.Fprintf(os.Stderr, "adcgo: sip sym=%d main block: loaded from %s — skipped the rebuild\n",
				sp.Sym, path)
			return m, true
		}
		return backend.Mat{}, false
	}
	save := func(m backend.Mat) {
		if err := writeMainBlockCache(path, key, m); err != nil {
			fmt.Fprintf(os.Stderr, "adcgo: sip sym=%d main block: WARNING could not cache to %s: %v "+
				"(the next run will rebuild it from scratch)\n", sp.Sym, path, err)
			return
		}
		fmt.Fprintf(os.Stderr, "adcgo: sip sym=%d main block: cached to %s\n", sp.Sym, path)
	}
	mx.SetMainBlockCache(load, save)
}
