// Command dimsprobe reports, per (spin, irrep) sector, the DIP-ADC(2) configuration
// space dimensions and the size of the assembled block-sparse operator.
//
// The operator nnz is the memory-traffic unit of one mat-vec (every apply streams
// all of it) and the sizing input for a device-residency check, so this is also the
// calibration front-end for backend dispatch.
//
//	go run ./cmd/dimsprobe <file.fcidump> [label] [-nnz] [-blocks N]
//
// -nnz assembles each sector's operator, which is expensive for large cases.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/dip"
	"github.com/leiaSQ/ADCgo/internal/adc/fcidump"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
	"github.com/leiaSQ/ADCgo/internal/adc/mp"
)

const gb = 1 << 30

func main() {
	withNNZ := flag.Bool("nnz", false, "assemble each sector's operator and report its nnz (slow)")
	nblocks := flag.Int("blocks", 200, "block-Lanczos iteration count, for the subspace-size estimate")
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
