// Command dimsprobe reports, per (spin, irrep) sector, the DIP-ADC(2) configuration
// space dimensions and the size of the assembled block-sparse operator.
//
// The operator nnz is the memory-traffic unit of one mat-vec (every apply streams
// all of it) and the sizing input for a device-residency check, so this is also the
// calibration front-end for backend dispatch.
//
//	go run ./cmd/dimsprobe <file.fcidump> [label] [-nnz] [-blocks N]
//	go run ./cmd/dimsprobe <file.fcidump> -sip -order 22 -fano-init 0
//
// -nnz assembles each sector's operator, which is expensive for large cases.
package main

import (
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/dip"
	"github.com/leiaSQ/ADCgo/internal/adc/fano"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
	"github.com/leiaSQ/ADCgo/internal/adc/sip"
)

const gb = 1 << 30

// parseCore parses a comma-separated list of 0-based core orbital indices (the
// dimsprobe mirror of cmd/adcgo's -core flag, for CVS SIP-ADC(4) sizing).
func parseCore(s string) []int {
	var out []int
	for _, f := range strings.Split(s, ",") {
		if f = strings.TrimSpace(f); f == "" {
			continue
		}
		v, err := strconv.Atoi(f)
		if err != nil || v < 0 {
			fmt.Fprintf(os.Stderr, "dimsprobe: bad -core orbital %q\n", f)
			os.Exit(2)
		}
		out = append(out, v)
	}
	return out
}

// probeSIP sizes the SIP (single-ionization) sectors, incl. CVS Dyson ADC(4) with
// its 1h/2h1p/3h2p space, so the block-Lanczos memory of a `-sip -order 4 -core …`
// run can be gated before committing. Lanczos never forms the dense n×n operator,
// so only the Krylov basis B (n·dim·8) and projected T (dim²·8) are reported.
func probeSIP(d *fcidump.Data, nocc, order, nblocks int, core []int, vacancy int) {
	total, maxDim := 0, 0
	for sym := 0; sym < 8; sym++ {
		var sp *sip.Space
		switch order {
		case 4:
			sp = sip.NewSpace4(nocc, d.NORB, d.OrbSym, sym, core)
		case sip.Order22:
			sp = sip.NewSpace22(nocc, d.NORB, d.OrbSym, sym)
		default:
			sp = sip.NewSpace(nocc, d.NORB, d.OrbSym, sym)
		}
		if sp.MainBlockSize() == 0 {
			continue // no 1h configurations in this irrep (CVS: no core hole here)
		}
		n := sp.Size()
		b := sp.MainBlockSize()
		n3h2p := len(sp.Sat3)
		n2h1p := n - b - n3h2p
		total += n
		if n > maxDim {
			maxDim = n
		}
		dim := lanczos.SubspaceDim(n, b, lanczos.Options{MaxBlocks: nblocks})
		// Assembled-operator estimate. Order 4 stores the 2h1p×3h2p coupling densely
		// (the dominant block); the 3h2p diagonal is a vector (~n3·8, negligible). Order
		// 2/3 stores the 2h1p×2h1p satellite block densely. opMF is the residency with
		// -matfree on: the 2h1p×3h2p (and, with -matfree, the 2h1p×2h1p) blocks are
		// recomputed each mat-vec instead of stored, leaving only the ~n3·8 diagonal.
		coupling := float64(n2h1p) * float64(n3h2p) * 8 / gb
		sat2 := float64(n2h1p) * float64(n2h1p) * 8 / gb
		diag3 := float64(n3h2p) * 8 / gb
		sat3 := float64(n3h2p) * float64(n3h2p) * 8 / gb
		var opGB, opMF float64
		switch order {
		case 4:
			opGB = coupling + diag3 + sat2
			opMF = diag3 // both coupling blocks matrix-free
		case sip.Order22:
			// ADC(2,2): the 3h2p/3h2p block is n3² for variants x and f (diagonal for m),
			// and the 2h1p/3h2p coupling n2·n3. Both go matrix-free, leaving the dense
			// 2h1p/2h1p block and, for variant m, the 3h2p diagonal vector.
			opGB = sat2 + coupling + sat3
			opMF = sat2 + diag3
		default:
			opGB = sat2
			opMF = sat2
		}
		fmt.Printf("   irrep=%d  dim=%-9d 1h=%-3d 2h1p=%-6d 3h2p=%-9d krylov=%-6d (%3.0f%% of n)  B=%7.3f GB  T=%6.3f GB  op~%8.3f GB (matfree %.3f)\n",
			sym, n, b, n2h1p, n3h2p, dim, 100*float64(dim)/float64(n),
			float64(n)*float64(dim)*8/gb, float64(dim)*float64(dim)*8/gb, opGB, opMF)

		// The Fano Q/P split, when a vacancy is named: it is what actually gets solved,
		// and each half is a fraction of the sector above.
		if vacancy >= 0 && vacancy < nocc && (d.OrbSym == nil || d.OrbSym[vacancy]-1 == sym) {
			sel, err := fano.NewHoleLocalization(fano.AnyHole, []int{vacancy}, "")
			if err == nil {
				part := fano.NewPartition(sp, sel)
				fmt.Printf("     fano vacancy %d: Q=%-9d P=%-9d (Q main %d); dense P = %.3g GB, "+
					"P krylov at %d blocks x %d = %.3g GB\n",
					vacancy, part.QSize(), part.PSize(), part.QMain,
					float64(part.PSize())*float64(part.PSize())*8/gb,
					nblocks, 64, float64(part.PSize())*float64(nblocks*64)*8/gb)
			}
		}
	}
	fmt.Printf("   TOTAL dim=%d  largest sector=%d\n\n", total, maxDim)
}

func main() {
	withNNZ := flag.Bool("nnz", false, "assemble each sector's operator and report its nnz (slow)")
	nblocks := flag.Int("blocks", 200, "block-Lanczos iteration count, for the subspace-size estimate")
	doSIP := flag.Bool("sip", false, "size SIP (single-ionization) sectors instead of DIP")
	order := flag.Int("order", 3, "SIP ADC order (2, 3, 4=CVS Dyson ADC(4), or 22=ADC(2,2))")
	coreFlag := flag.String("core", "", "CVS core orbitals for -order 4 (comma-separated 0-based)")
	vacancy := flag.Int("fano-init", -1, "also report the Fano Q/P split for this vacancy orbital (0-based occupied); -1 = skip")
	flag.Parse()

	if flag.NArg() < 1 {
		fmt.Fprintln(os.Stderr, "usage: dimsprobe <file.fcidump> [label] [-nnz] [-blocks N]")
		os.Exit(2)
	}
	path := flag.Arg(0)
	label := path
	if flag.NArg() > 1 {
		label = flag.Arg(1)
	}

	d, err := fcidump.ReadFile(path)
	if err != nil {
		fmt.Fprintln(os.Stderr, "dimsprobe:", err)
		os.Exit(1)
	}
	nocc := mp.NOcc(d)
	norb := float64(d.NORB)
	fmt.Printf("%-16s norb=%-4d nocc=%-3d nvir=%-4d  ERI(NORB^4)=%.3f GB\n",
		label, d.NORB, nocc, d.NORB-nocc, norb*norb*norb*norb*8/gb)

	if *doSIP {
		core := parseCore(*coreFlag)
		if *order == 4 && len(core) == 0 {
			fmt.Fprintln(os.Stderr, "dimsprobe: -order 4 (CVS) requires -core (e.g. -core 0)")
			os.Exit(2)
		}
		probeSIP(d, nocc, *order, *nblocks, core, *vacancy)
		return
	}

	var be backend.Backend
	var ints *integrals.Store
	var eps []float64
	if *withNNZ {
		if be, err = backend.New("gonum"); err != nil {
			fmt.Fprintln(os.Stderr, "dimsprobe:", err)
			os.Exit(1)
		}
		eps = mp.OrbitalEnergies(d, nocc)
		ints = integrals.New(d, nocc, d.OrbSym)
	}

	total, maxDim := 0, 0
	for _, spin := range []dip.Spin{dip.Singlet, dip.Triplet} {
		name := "singlet"
		if spin == dip.Triplet {
			name = "triplet"
		}
		for sym := 0; sym < 8; sym++ {
			sp := dip.NewSpace(nocc, d.NORB, d.OrbSym, sym, spin)
			n := sp.Size()
			if n == 0 {
				continue
			}
			b := sp.MainBlockSize()
			total += n
			if n > maxDim {
				maxDim = n
			}

			// Block-Lanczos subspace: the basis holds one block of width b per
			// iteration, capped at the full space. Ask the solver rather than
			// re-deriving it, so the estimate cannot drift from what Solve builds.
			dim := lanczos.SubspaceDim(n, b, lanczos.Options{MaxBlocks: *nblocks})
			fmt.Printf("   %s irrep=%d  dim=%-7d 2h=%-5d 3h1p=%-8d dense=%6.3f GB  krylov=%-6d (%3.0f%% of n)  B=%5.2f GB  T=%5.2f GB",
				name, sym, n, b, n-b, float64(n)*float64(n)*8/gb,
				dim, 100*float64(dim)/float64(n),
				float64(n)*float64(dim)*8/gb, float64(dim)*float64(dim)*8/gb)

			if *withNNZ {
				mx := dip.New(sp, ints, eps, be)
				nnz, nb := mx.OperatorNNZ()
				dens := float64(nnz) / (float64(n) * float64(n))
				fmt.Printf("  op=%5.2f GB (%d blocks, %.1f%% dense)",
					float64(nnz)*8/gb, nb, 100*dens)
			}
			fmt.Println()
		}
	}
	fmt.Printf("   TOTAL dim=%d  largest sector=%d (dense %.3f GB)\n\n",
		total, maxDim, float64(maxDim)*float64(maxDim)*8/gb)
}
