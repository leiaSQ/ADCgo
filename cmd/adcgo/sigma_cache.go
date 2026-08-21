package main

// sigma_cache.go — persistence for the static self-energy.
//
// WHY. Σ(∞) is the all-order resolvent resummation, and at production scale it is the single most
// expensive phase of a SIP run by a wide margin: the production system's took 3 d 06 h 19 m (job 14134491,
// 2026-08-06T09:05 → 2026-08-09T15:24). It is also rebuilt from scratch by every process, so a
// daisychained run that is walltime-killed before its solver checkpoints pays those 78 h again in
// the next generation — out of a 120 h allocation, leaving ~42 h for the actual solve, and again
// for every generation after that. Nothing in the checkpoint machinery covers it: -checkpoint only
// protects lanczos.Solve's Krylov state, which is not even reached until Σ is finished.
//
// The absurdity is the size. Σ is n×n over the correlated orbital space — 212² float64 = 351 KB
// for the production system. Three hundred kilobytes stands between a chain that converges and one that spends
// most of its allocation recomputing the same matrix.
//
// The guard is deliberately conservative, because a WRONG Σ is not a crash: it silently shifts
// every main line by ~0.2–0.35 eV, which looks entirely publishable. Anything that could change Σ
// invalidates the cache — the scheme and its two iteration knobs, the orbital-space dimensions, a
// hash of the orbital energies, and the FCIDUMP's size and modification time. A miss costs a
// rebuild; a false hit costs a wrong spectrum, so the trade is not symmetric.

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"hash/fnv"
	"io"
	"math"
	"os"

	"github.com/leiaSQ/ADCgo/internal/adc/selfenergy"
)

const (
	sigmaCacheMagic   = "ADCGOSIG"
	sigmaCacheVersion = 1
)

// sigmaCacheKey fingerprints everything Σ depends on. Two runs agreeing on all of it must produce
// the same matrix; any disagreement rebuilds.
type sigmaCacheKey struct {
	Scheme      string
	Norb, Nocc  int
	MaxIt       int
	Akrit       float64
	EpsHash     uint64 // FNV-1a over the orbital energies — the cheap proxy for "same orbital space"
	FcidumpSize int64
	FcidumpMod  int64 // UnixNano
}

// sigmaCacheKeyFor builds the key. A missing/unstatable FCIDUMP yields zeroed file fields, which
// simply makes the key weaker, never wrong: the eps hash and the dimensions still have to match.
func sigmaCacheKeyFor(scheme string, norb, nocc, maxIt int, akrit float64, eps []float64, fcidump string) sigmaCacheKey {
	h := fnv.New64a()
	var buf [8]byte
	for _, e := range eps {
		binary.LittleEndian.PutUint64(buf[:], math.Float64bits(e))
		_, _ = h.Write(buf[:])
	}
	k := sigmaCacheKey{
		Scheme: scheme, Norb: norb, Nocc: nocc, MaxIt: maxIt, Akrit: akrit,
		EpsHash: h.Sum64(),
	}
	if fi, err := os.Stat(fcidump); err == nil {
		k.FcidumpSize = fi.Size()
		k.FcidumpMod = fi.ModTime().UnixNano()
	}
	return k
}

func (k sigmaCacheKey) equal(o sigmaCacheKey) bool { return k == o }

// writeSigmaCache persists Σ under key, atomically (tmp + fsync + rename).
func writeSigmaCache(path string, k sigmaCacheKey, s *selfenergy.Sigma) error {
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
	if _, err := w.WriteString(sigmaCacheMagic); err != nil {
		return fail(err)
	}
	hdr := []int64{
		sigmaCacheVersion,
		int64(k.Norb), int64(k.Nocc), int64(k.MaxIt),
		int64(math.Float64bits(k.Akrit)), int64(k.EpsHash),
		k.FcidumpSize, k.FcidumpMod,
		int64(len(k.Scheme)), int64(s.N()),
	}
	for _, v := range hdr {
		if err := binary.Write(w, binary.LittleEndian, v); err != nil {
			return fail(err)
		}
	}
	if _, err := w.WriteString(k.Scheme); err != nil {
		return fail(err)
	}
	d := s.Data()
	if err := binary.Write(w, binary.LittleEndian, d); err != nil {
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

// readSigmaCache returns the cached Σ when the file exists AND its key matches. It returns
// (nil, nil) for "no usable cache" — a missing file, a corrupt one, or a key mismatch are all the
// same to the caller, which just rebuilds.
func readSigmaCache(path string, want sigmaCacheKey) (*selfenergy.Sigma, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()
	r := bufio.NewReaderSize(f, 1<<16)

	magic := make([]byte, len(sigmaCacheMagic))
	if _, err := io.ReadFull(r, magic); err != nil {
		return nil, err
	}
	if string(magic) != sigmaCacheMagic {
		return nil, fmt.Errorf("sigma cache %s: bad magic", path)
	}
	var hdr [10]int64
	for i := range hdr {
		if err := binary.Read(r, binary.LittleEndian, &hdr[i]); err != nil {
			return nil, err
		}
	}
	if hdr[0] != sigmaCacheVersion {
		return nil, nil // a different format generation: rebuild rather than guess
	}
	schemeLen, n := int(hdr[8]), int(hdr[9])
	if schemeLen < 0 || schemeLen > 64 || n < 0 || n > 1<<20 {
		return nil, fmt.Errorf("sigma cache %s: implausible header", path)
	}
	scheme := make([]byte, schemeLen)
	if _, err := io.ReadFull(r, scheme); err != nil {
		return nil, err
	}
	got := sigmaCacheKey{
		Scheme: string(scheme),
		Norb:   int(hdr[1]), Nocc: int(hdr[2]), MaxIt: int(hdr[3]),
		Akrit:       math.Float64frombits(uint64(hdr[4])),
		EpsHash:     uint64(hdr[5]),
		FcidumpSize: hdr[6], FcidumpMod: hdr[7],
	}
	if !got.equal(want) {
		return nil, nil // stale: different orbital space, scheme or tuning
	}
	d := make([]float64, n*n)
	if err := binary.Read(r, binary.LittleEndian, d); err != nil {
		return nil, err
	}
	return selfenergy.FromMatrix(n, d)
}
