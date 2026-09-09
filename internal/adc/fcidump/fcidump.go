// Package fcidump reads MO-basis integrals from a standard FCIDUMP file.
//
// FCIDUMP is the integral-ingestion contract for ADCgo: SCF and integral
// evaluation are delegated to an external code (e.g. pyscf's
// pyscf.tools.fcidump.from_scf), which writes the one-electron integrals
// h_pq, the two-electron integrals (pq|rs) in chemist notation, the core
// energy, and the electron/orbital counts. Orbital energies are NOT stored in
// FCIDUMP; for a canonical HF reference they are reconstructed from the Fock
// diagonal (see package mp).
package fcidump

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"os"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"unicode/utf8"
	"unsafe"
)

// Data holds the parsed MO-basis integrals and metadata.
//
// Two-electron integrals are stored fully expanded (8-fold permutation
// symmetry applied) in a dense NORB^4 array using chemist notation (pq|rs);
// this is memory-cheap at the small NORB of the M0 spike and keeps access O(1).
// A blocked, symmetry-aware store replaces this for larger cases (see plan).
type Data struct {
	NORB   int
	NELEC  int
	MS2    int
	ISYM   int
	OrbSym []int // point-group label per orbital (all 1 when symmetry is off)

	Ecore float64   // core energy: nuclear repulsion (+ frozen-core, if any)
	h     []float64 // NORB*NORB one-electron integrals h_pq (symmetric)
	eri   []float64 // NORB^4 two-electron integrals (pq|rs), chemist notation
}

// OneE returns the one-electron integral h_pq (0-based indices).
func (d *Data) OneE(p, q int) float64 { return d.h[p*d.NORB+q] }

// TwoE returns the two-electron integral (pq|rs) in chemist notation (0-based).
func (d *Data) TwoE(p, q, r, s int) float64 {
	n := d.NORB
	return d.eri[((p*n+q)*n+r)*n+s]
}

var headerIntRe = func(key string) *regexp.Regexp {
	return regexp.MustCompile(`(?i)\b` + key + `\s*=\s*(-?\d+)`)
}

// ReadFile parses the FCIDUMP at path.
func ReadFile(path string) (*Data, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return Read(f)
}

// Read parses a FCIDUMP from r.
//
// The header is consumed line by line; the body is then streamed as fixed-size byte
// blocks, parsed concurrently and stored in file order (see readBody). This replaces
// buffering the whole body into a []string and then parsing it on one core: for
// the production system (22.4 GB FCIDUMP, ~252 million data lines) that intermediate cost more
// than the 16.2 GB ERI array it was filling, roughly doubling peak RSS, at the start
// of every single job.
//
// The result is bit-identical to that serial parse, which matters because redundant
// lines in a real FCIDUMP disagree in the last ulp — see readBody.
func Read(r io.Reader) (*Data, error) {
	br := bufio.NewReaderSize(r, blockBytes)

	hstr, inHeader, err := readHeader(br)
	if err != nil {
		return nil, err
	}
	if !inHeader {
		return nil, fmt.Errorf("fcidump: no &FCI header found")
	}

	d := &Data{}
	norb, ok := parseInt(hstr, "NORB")
	if !ok {
		return nil, fmt.Errorf("fcidump: NORB not found in header")
	}
	nelec, ok := parseInt(hstr, "NELEC")
	if !ok {
		return nil, fmt.Errorf("fcidump: NELEC not found in header")
	}
	d.NORB = norb
	d.NELEC = nelec
	d.MS2, _ = parseInt(hstr, "MS2")
	d.ISYM, _ = parseInt(hstr, "ISYM")
	d.OrbSym = parseOrbSym(hstr, norb)

	n := norb
	d.h = make([]float64, n*n)
	d.eri = make([]float64, n*n*n*n)

	if err := d.readBody(br); err != nil {
		return nil, err
	}
	return d, nil
}

// readHeader consumes br up to and including the &END/$END// terminator, returning
// the joined header text and whether an &FCI/$FCI opener was ever seen. When there
// is no terminator it consumes the whole stream, which leaves the body empty — the
// same outcome the previous single-pass scanner produced for that malformed input.
func readHeader(br *bufio.Reader) (string, bool, error) {
	var header strings.Builder
	inHeader := false
	for {
		line, err := br.ReadString('\n')
		if err != nil && line == "" {
			if err == io.EOF {
				return header.String(), inHeader, nil
			}
			return "", false, err
		}
		t := strings.TrimSpace(line)
		up := strings.ToUpper(t)
		switch {
		case !inHeader:
			if strings.HasPrefix(up, "&FCI") || strings.HasPrefix(up, "$FCI") {
				inHeader = true
				header.WriteString(" " + t)
			}
		case up == "&END" || up == "$END" || up == "/" || strings.HasSuffix(up, "&END"):
			return header.String(), true, nil
		default:
			header.WriteString(" " + t)
		}
		if err != nil {
			if err == io.EOF {
				return header.String(), inHeader, nil
			}
			return "", false, err
		}
	}
}

// blockBytes is the size of one raw byte block handed to a parser goroutine.
//
// At 512 KiB a block is ~8k FCIDUMP lines: large enough that the channel handoff
// and the ordered handover below are noise against the parse, small enough that
// the whole pipeline's in-flight set stays bounded at roughly (workers+4) *
// (blockBytes + records) — about 110 MB on a 128-core node — instead of the
// 22.4 GB of line strings the old whole-body []string held for the production system.
const blockBytes = 512 << 10

// rec is one parsed data line: its value and its four raw 1-based FCIDUMP
// indices, i.e. everything the store step needs and nothing that refers back to
// the byte buffer the line came from.
type rec struct {
	v          float64
	i, j, k, l int
}

// blockBuf is one unit of work travelling through the pipeline, and the unit of
// recycling: raw bytes in, records out, plus its own completion channel so no
// per-block allocation is needed. Exactly one goroutine owns a blockBuf at a
// time (reader -> parser -> applier -> pool), so its fields need no locking.
type blockBuf struct {
	raw  []byte // backing allocation, reused across blocks
	data []byte // the whole lines of this block, a prefix of raw
	recs []rec  // parse output, grows to the block's line count and stays there
	done chan error
}

// readBody parses the data lines remaining in br into d's integral arrays through
// a three-stage pipeline: one reader goroutine slices the stream into whole-line
// blocks, GOMAXPROCS parser goroutines turn block bytes into records, and one
// applier goroutine stores those records in strict file order.
//
// Only the parse is parallel, and that is deliberate. The tempting claim is that
// every line writes disjoint slots of h/eri by assignment, so the stores could run
// concurrently too — but that is FALSE for real FCIDUMPs. pyscf writes the
// permutation-redundant orderings as separate lines: in testdata/h2o.fcidump 21783
// of the 23731 distinct two-electron integrals appear twice, as (pq|rs) and again
// as (rs|pq), and in every one of those pairs the two printed values differ in the
// last ulp. The two lines expand to the SAME eight slots, so which one lands last
// decides the stored bits, and the serial parser's answer is "the later line in the
// file". Storing concurrently would pick a winner by scheduling and make the ERI —
// and every energy computed from it — differ run to run. The store therefore stays
// on one goroutine, replaying exactly the serial sequence, which also makes Ecore
// (last (0,0,0,0) line wins) and the reported parse error (the first malformed line
// in file order) fall out for free.
//
// This is still the win the production-scale numbers ask for: the ~252 million lines of
// strings.Fields + ParseFloat + 4x Atoi, measured at ~460 ns/line, move off the
// critical path and overlap the ~340 ns/line of stores, and the 22.4 GB of buffered
// line strings (which more than doubled peak RSS over the 16.2 GB ERI array they were
// filling) are gone.
//
// The parallel package's helpers do not fit here: Rows/HeavyRows/Chunks all partition
// a known [0,n), and the block count is only discovered while reading.
func (d *Data) readBody(br *bufio.Reader) error {
	workers := runtime.GOMAXPROCS(0)
	if workers < 1 {
		workers = 1
	}
	// One buffer per worker plus a little run-ahead for the reader. This is the only
	// thing bounding the pipeline's memory, and it also throttles the reader to the
	// applier's pace, which is what keeps a 22.4 GB file from piling up in RAM.
	nbuf := workers + 4

	pool := make(chan *blockBuf, nbuf)
	for range nbuf {
		// raw is left nil and allocated on first use: a small FCIDUMP (the test
		// dumps, a spike case) must not pay for a 128-core node's whole pool.
		pool <- &blockBuf{done: make(chan error, 1)}
	}
	work := make(chan *blockBuf, workers)
	// order carries the same blocks in file order. It can never block the reader:
	// only nbuf buffers exist and the reader is holding one of them.
	order := make(chan *blockBuf, nbuf)

	var stop atomic.Bool
	var wg sync.WaitGroup
	for range workers {
		wg.Go(func() {
			for b := range work {
				b.done <- parseBlock(b)
			}
		})
	}

	var applyErr error
	var applier sync.WaitGroup
	applier.Go(func() {
		for b := range order {
			err := <-b.done
			switch {
			case applyErr != nil: // already failed: just drain and recycle
			case err != nil:
				applyErr = err
				stop.Store(true)
			default:
				d.apply(b.recs)
			}
			pool <- b
		}
	})

	readErr := readBlocks(br, pool, work, order, &stop)
	close(work)
	close(order)
	wg.Wait()
	applier.Wait()

	// Read errors outrank parse errors, as in the previous two-pass parser: it
	// scanned the whole file (and returned any read error) before parsing a line.
	if readErr != nil {
		return readErr
	}
	return applyErr
}

// readBlocks slices br into whole-line blocks and feeds them to the parsers (work)
// and, in file order, to the applier (order). A partial trailing line is carried
// into the next block.
func readBlocks(br *bufio.Reader, pool chan *blockBuf, work, order chan<- *blockBuf, stop *atomic.Bool) error {
	var carry []byte
	for {
		if stop.Load() {
			return nil // a line has already failed to parse; nothing later matters
		}
		b := <-pool
		if cap(b.raw) < blockBytes {
			b.raw = make([]byte, 0, blockBytes)
		}
		buf := append(b.raw[:0], carry...)
		if cap(buf)-len(buf) < blockBytes/2 {
			// Only reachable for a single line longer than half a block; grow so the
			// read below always makes progress.
			grown := make([]byte, len(buf), len(buf)+blockBytes)
			copy(grown, buf)
			buf = grown
		}
		n, err := io.ReadFull(br, buf[len(buf):cap(buf)])
		buf = buf[:len(buf)+n]
		b.raw = buf
		if err != nil && err != io.EOF && err != io.ErrUnexpectedEOF {
			pool <- b
			return err
		}
		if err != nil {
			// End of stream: the tail is the final line, terminated or not.
			b.data = buf
			order <- b
			work <- b
			return nil
		}
		cut := bytes.LastIndexByte(buf, '\n')
		if cut < 0 {
			carry = append(carry[:0], buf...)
			pool <- b
			continue
		}
		carry = append(carry[:0], buf[cut+1:]...)
		b.data = buf[:cut+1]
		order <- b
		work <- b
	}
}

// parseBlock turns b.data into b.recs, stopping at the first malformed line exactly
// as the serial loop did. It only reads text and writes b's own record slice, so it
// is safe to run on every block at once.
func parseBlock(b *blockBuf) error {
	recs := b.recs[:0]
	data := b.data
	var f [5][]byte
	for len(data) > 0 {
		var line []byte
		if i := bytes.IndexByte(data, '\n'); i >= 0 {
			line, data = data[:i], data[i+1:]
		} else {
			line, data = data, nil
		}
		if !fields5(line, &f) {
			continue // the serial parser's `len(fields) < 5` skip (blank lines too)
		}
		val, err := parseFloatField(f[0])
		if err != nil {
			b.recs = recs
			return fmt.Errorf("fcidump: bad value %q: %w", string(f[0]), err)
		}
		var idx [4]int
		for k := range 4 {
			v, err := parseIntField(f[k+1])
			if err != nil {
				b.recs = recs
				return fmt.Errorf("fcidump: bad index %q: %w", string(f[k+1]), err)
			}
			idx[k] = v
		}
		recs = append(recs, rec{v: val, i: idx[0], j: idx[1], k: idx[2], l: idx[3]})
	}
	b.recs = recs
	return nil
}

// apply stores one block's records. This is the serial parser's store loop verbatim,
// run on one goroutine over blocks in file order, which is what makes the result
// bit-identical to the old whole-file serial parse even where redundant lines write
// the same slots with values that differ in the last ulp.
func (d *Data) apply(recs []rec) {
	for _, r := range recs {
		switch {
		case r.i == 0 && r.j == 0 && r.k == 0 && r.l == 0:
			d.Ecore = r.v
		case r.k == 0 && r.l == 0:
			// one-electron h_ij (indices 1-based, i>=j); symmetric.
			d.setOne(r.i-1, r.j-1, r.v)
		default:
			d.setTwo(r.i-1, r.j-1, r.k-1, r.l-1, r.v)
		}
	}
}

// fields5 slices out the first five whitespace-separated fields of line, reporting
// false when there are fewer than five. It reproduces strings.Fields' split while
// allocating nothing: at the production system's ~252 million lines the old strings.Fields built
// well over a billion short-lived strings, all of it GC pressure on the critical
// path of every job.
func fields5(line []byte, f *[5][]byte) bool {
	i := 0
	for n := range 5 {
		for i < len(line) && isSpaceASCII(line[i]) {
			i++
		}
		if i >= len(line) {
			return false
		}
		start := i
		for i < len(line) && !isSpaceASCII(line[i]) {
			if line[i] >= utf8.RuneSelf {
				// A non-ASCII byte may be a Unicode space that strings.Fields would
				// have split on; hand the line to the real thing so the split stays
				// identical. Never happens for numeric FCIDUMP data.
				return fields5Slow(line, f)
			}
			i++
		}
		f[n] = line[start:i]
	}
	return true
}

func fields5Slow(line []byte, f *[5][]byte) bool {
	fs := strings.Fields(string(line))
	if len(fs) < 5 {
		return false
	}
	for k := range 5 {
		f[k] = []byte(fs[k])
	}
	return true
}

// isSpaceASCII is unicode.IsSpace restricted to ASCII, the set strings.Fields uses
// for single-byte runes.
func isSpaceASCII(c byte) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r'
}

// parseFloatField / parseIntField parse a field in place. The fast path views the
// block buffer as a string without copying; on failure they re-parse from an owned
// copy, because strconv's *NumError retains the string it was handed and the block
// buffer is about to be recycled under it.
func parseFloatField(b []byte) (float64, error) {
	v, err := strconv.ParseFloat(unsafeString(b), 64)
	if err != nil {
		return strconv.ParseFloat(string(b), 64)
	}
	return v, nil
}

func parseIntField(b []byte) (int, error) {
	v, err := strconv.Atoi(unsafeString(b))
	if err != nil {
		return strconv.Atoi(string(b))
	}
	return v, nil
}

// unsafeString views b as a string. Safe here because the callers are strconv
// parsers, which neither mutate nor retain the string on their success path, and b
// is not written while they run.
func unsafeString(b []byte) string {
	if len(b) == 0 {
		return ""
	}
	return unsafe.String(&b[0], len(b))
}

func (d *Data) setOne(p, q int, v float64) {
	n := d.NORB
	d.h[p*n+q] = v
	d.h[q*n+p] = v
}

// setTwo stores (pq|rs) into all 8 permutation-equivalent slots.
func (d *Data) setTwo(p, q, r, s int, v float64) {
	n := d.NORB
	set := func(a, b, c, e int) { d.eri[((a*n+b)*n+c)*n+e] = v }
	set(p, q, r, s)
	set(q, p, r, s)
	set(p, q, s, r)
	set(q, p, s, r)
	set(r, s, p, q)
	set(s, r, p, q)
	set(r, s, q, p)
	set(s, r, q, p)
}

func parseInt(header, key string) (int, bool) {
	m := headerIntRe(key).FindStringSubmatch(header)
	if m == nil {
		return 0, false
	}
	v, err := strconv.Atoi(m[1])
	if err != nil {
		return 0, false
	}
	return v, true
}

// parseOrbSym reads the ORBSYM=... comma list following the key, up to the next
// KEY= token or the end of the header. Returns nil if absent.
func parseOrbSym(header string, norb int) []int {
	loc := regexp.MustCompile(`(?i)\bORBSYM\s*=`).FindStringIndex(header)
	if loc == nil {
		return nil
	}
	rest := header[loc[1]:]
	// Stop at the next "WORD=" (another namelist key) or the &END terminator.
	if stop := regexp.MustCompile(`(?i)[A-Z_]+\s*=|&END|\$END`).FindStringIndex(rest); stop != nil {
		rest = rest[:stop[0]]
	}
	syms := make([]int, 0, norb)
	for _, tok := range strings.FieldsFunc(rest, func(r rune) bool { return r == ',' || r == ' ' || r == '\t' }) {
		if v, err := strconv.Atoi(strings.TrimSpace(tok)); err == nil {
			syms = append(syms, v)
		}
	}
	if len(syms) == 0 {
		return nil
	}
	return syms
}
