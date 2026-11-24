package main

import (
	"fmt"
	"time"

	"github.com/leiaSQ/ADCgo/backend"
	"github.com/leiaSQ/ADCgo/internal/adc/integrals"
	"github.com/leiaSQ/ADCgo/internal/adc/lanczos"
	"github.com/leiaSQ/ADCgo/internal/adc/sip"
)

// solveSIPSpace solves one SIP configuration space and returns the result together with
// the still-live operator (the caller must Release it).
func solveSIPSpace(ch *chooser, label string, sp *sip.Space, ints *integrals.Store, eps []float64, order int, cfg sipConfig) (lanczos.Result, *sip.Matrix, error) {
	lopts := lanczos.Options{MaxBlocks: cfg.blocks}
	// Checkpoint the block-Krylov build so a walltime kill can resume in a successor job.
	// Per-sector path (irrep-keyed) so concurrent sectors never share a file. Lanczos only —
	// the dense/Davidson drivers hold no resumable Krylov basis.
	if cfg.solver == "lanczos" && cfg.ckpt != "" {
		lopts.Checkpoint = &lanczos.Checkpoint{
			Path:  fmt.Sprintf("%s.i%d", cfg.ckpt, sp.Sym),
			Every: cfg.ckptEvery,
			Stop:  cfg.stop,
		}
	}
	// Same trigger as the DIP twin (dipLanczosOpts), and for the same reason, which this path
	// learned the hard way: production job 14717237 ran a full 200-block solve in 11 h
	// and printed not one line between "entering the solver" and the final JSON, so whether it
	// had reached its block budget could only be established afterwards by matching Slurm's
	// recorded 4.9 TB of checkpoint writes against saveKrylov's per-block size.
	if cfg.profile || cfg.ckpt != "" {
		lopts.Progress = progressReporter(fmt.Sprintf("sip irrep=%d", sp.Sym+1))
	}
	davOpts := davidsonOpts(cfg.nroots, cfg.maxdavsp, cfg.maxdavit, cfg.convthr, false)
	n, b := sp.Size(), sp.MainBlockSize()
	subspace := lanczos.SubspaceDim(n, b, lopts)
	if cfg.solver == "davidson" {
		subspace = lanczos.DavidsonSubspaceDim(n, davOpts)
	}

	var be backend.Backend
	if cfg.solver == "dense" {
		be = ch.pickDense(label, n)
	} else {
		be = ch.pickLanczos(label, n, b, subspace,
			func(cand backend.Backend) time.Duration {
				m := sip.New(sp, ints, eps, order, cand)
				m.SetMatFree(cfg.matFree, cfg.matFreeBudget)
				defer m.Release()
				return timeApplyBlock(cand, n, b, m.ApplyBlock)
			})
	}
	mx := sip.New(sp, ints, eps, order, be)
	mx.SetMatFree(cfg.matFree, cfg.matFreeBudget)
	if cfg.sig != nil {
		mx.SetStaticSelfEnergy(cfg.sig)
	}
	installMainBlockCache(mx, sp, eps, order, cfg)

	// Pre-flight device-memory guard, the SIP twin of solveDIPSector's. It matters more here
	// than the DIP call site suggests: pickLanczos' own fits() check is skipped whenever a
	// single backend is pinned (chooser.single), which is the normal production case
	// (-backend cuda), so without this an oversized sector reached a raw cudaMalloc panic.
	//
	// Resident = the block-sparse operator (exact, honouring the matrix-free gates) + the
	// Krylov basis. The basis term is the dominant one for SIP and the reason -matfree alone
	// does not make a sector fit: full-reorth Solve keeps the whole n×subspace basis, so a
	// missing -matfree shows up in the operator term and an over-large -blocks in the basis
	// term, and the message below distinguishes them.
	if cfg.solver != "dense" {
		need := mx.OperatorResidentBytes() + uint64(n)*uint64(subspace)*8
		if err := ch.checkDeviceFit(label, be, need); err != nil {
			mx.Release()
			return lanczos.Result{}, nil, err
		}
	}

	var res lanczos.Result
	switch cfg.solver {
	case "dense":
		res = lanczos.SolveDense(mx, be)
	case "lanczos":
		res = lanczos.Solve(mx, be, lopts)
	case "davidson":
		res = lanczos.SolveDavidson(mx, be, davOpts)
	default:
		mx.Release()
		return lanczos.Result{}, nil, fmt.Errorf("unknown solver %q (want lanczos, davidson or dense)", cfg.solver)
	}
	if res.Interrupted {
		// A stop signal checkpointed and bailed out mid-build; propagate so main exits with
		// the "resume needed" code rather than emitting a partial spectrum.
		return res, mx, errInterrupted
	}
	if cfg.profile {
		reportTiming(label, sp.Size(), sp.MainBlockSize(), res.Timing)
	}
	return res, mx, nil
}
