package dip

import (
	"unsafe"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
)

// matfree_device.go — the DEVICE (CUDA) matrix-free applier for the 3h1p↔3h1p satellite region.
//
// It is the GPU twin of the host per-scalar applier (satscalar.go): it uploads the same config
// struct-of-arrays plus the flat ERI tensor and orbital data once, then each mat-vec launches the
// DIP satellite kernel (backend/adc2dip_kernels.cu) — one thread per output 3h1p row — so the
// multi-TB satellite block never occupies VRAM. The kernel's element evaluation is a line-for-line
// transcription of satelem.go, pinned on the host by TestSatelliteScalarMatchesDense; this Go side
// only marshals the plan into the flat int32/float64 device buffers the kernel indexes.
//
// The flattening (satDeviceSoA) is deliberately separated from the upload (satDeviceBufs): the
// -mgpu per-device path (matfree_dist.go) uploads the SAME host arrays to every device, so
// building them once and uploading N times keeps the two paths bit-identical by construction.
//
// Compiles on every build (it calls the backend.DeviceKernels interface, not CUDA directly); it is
// only reached when the backend implements DeviceKernels, i.e. the cuda backend. Parity on real
// hardware is TestSatelliteMatFreeDeviceParity (matfree_cuda_test.go).

// satDeviceSoA is the host-side flattening of a satScalarPlan into the flat arrays the CUDA
// kernel indexes. It holds no device state, so one instance serves any number of devices.
type satDeviceSoA struct {
	rTyp              []int32
	rGrp, rPart, rVir []int32

	jO0, jO1, jSt, jVoff, jNv, jVir      []int32
	iO0, iO1, iO2, iSt, iVoff, iNv, iVir []int32

	eri  []float64
	eps  []float64
	osym []int32

	nsat, njii, nijk        int
	main, norb, parts, spin int
}

// buildSatDeviceSoA flattens the per-scalar plan for the kernel. The ERI layout
// eri[((a·n+b)·n+c)·n+d] = Eri(a,b,c,d) must match dd_eri in adc2dip_kernels.cu.
func (mx *Matrix) buildSatDeviceSoA(p *satScalarPlan) *satDeviceSoA {
	sp := mx.sp
	norb := sp.Norb
	s := &satDeviceSoA{
		rGrp: p.rGrp, rPart: p.rPart, rVir: p.rVir,
		nsat: len(p.rTyp), njii: len(p.jCfg), nijk: len(p.iCfg),
		main: p.main, norb: norb, parts: p.parts,
	}

	s.rTyp = make([]int32, len(p.rTyp))
	for i, v := range p.rTyp {
		s.rTyp[i] = int32(v)
	}

	// JII group SoA + flat virtuals.
	s.jO0, s.jO1 = make([]int32, s.njii), make([]int32, s.njii)
	s.jSt, s.jVoff, s.jNv = make([]int32, s.njii), make([]int32, s.njii), make([]int32, s.njii)
	for g := range s.njii {
		s.jO0[g], s.jO1[g] = int32(p.jCfg[g].Occ[0]), int32(p.jCfg[g].Occ[1])
		s.jSt[g], s.jVoff[g], s.jNv[g] = int32(p.jSt[g]), int32(len(s.jVir)), int32(len(p.jVir[g]))
		for _, orb := range p.jVir[g] {
			s.jVir = append(s.jVir, int32(orb))
		}
	}

	// IJK group SoA + flat virtuals.
	s.iO0, s.iO1, s.iO2 = make([]int32, s.nijk), make([]int32, s.nijk), make([]int32, s.nijk)
	s.iSt, s.iVoff, s.iNv = make([]int32, s.nijk), make([]int32, s.nijk), make([]int32, s.nijk)
	for g := range s.nijk {
		s.iO0[g], s.iO1[g], s.iO2[g] = int32(p.iCfg[g].Occ[0]), int32(p.iCfg[g].Occ[1]), int32(p.iCfg[g].Occ[2])
		s.iSt[g], s.iVoff[g], s.iNv[g] = int32(p.iSt[g]), int32(len(s.iVir)), int32(len(p.iVir[g]))
		for _, orb := range p.iVir[g] {
			s.iVir = append(s.iVir, int32(orb))
		}
	}

	s.eri = make([]float64, norb*norb*norb*norb)
	for a := range norb {
		for b := range norb {
			for c := range norb {
				base := ((a*norb+b)*norb + c) * norb
				for d := range norb {
					s.eri[base+d] = mx.ints.Eri(a, b, c, d)
				}
			}
		}
	}
	s.osym = make([]int32, norb)
	for o := range norb {
		s.osym[o] = int32(mx.ints.OrbIrrep(o))
	}
	s.eps = mx.eps

	if sp.Spin == Triplet {
		s.spin = 1
	}
	return s
}

// satDeviceBufs is one device's uploaded copy of a satDeviceSoA. Pointers live until free().
type satDeviceBufs struct {
	dk backend.DeviceKernels

	rTyp, rGrp, rPart, rVir              unsafe.Pointer
	jO0, jO1, jSt, jVoff, jNv, jVir      unsafe.Pointer
	iO0, iO1, iO2, iSt, iVoff, iNv, iVir unsafe.Pointer
	eri, eps, osym                       unsafe.Pointer
}

// uploadSatSoA uploads the flattened plan to one device.
func uploadSatSoA(dk backend.DeviceKernels, s *satDeviceSoA) *satDeviceBufs {
	return &satDeviceBufs{
		dk:   dk,
		rTyp: dk.UploadInts(s.rTyp), rGrp: dk.UploadInts(s.rGrp),
		rPart: dk.UploadInts(s.rPart), rVir: dk.UploadInts(s.rVir),
		jO0: dk.UploadInts(s.jO0), jO1: dk.UploadInts(s.jO1), jSt: dk.UploadInts(s.jSt),
		jVoff: dk.UploadInts(s.jVoff), jNv: dk.UploadInts(s.jNv), jVir: dk.UploadInts(s.jVir),
		iO0: dk.UploadInts(s.iO0), iO1: dk.UploadInts(s.iO1), iO2: dk.UploadInts(s.iO2),
		iSt: dk.UploadInts(s.iSt), iVoff: dk.UploadInts(s.iVoff), iNv: dk.UploadInts(s.iNv),
		iVir: dk.UploadInts(s.iVir),
		eri:  dk.DeviceERI(s.eri), eps: dk.UploadFloats(s.eps), osym: dk.UploadInts(s.osym),
	}
}

func (b *satDeviceBufs) free() {
	for _, p := range []unsafe.Pointer{
		b.rTyp, b.rGrp, b.rPart, b.rVir,
		b.jO0, b.jO1, b.jSt, b.jVoff, b.jNv, b.jVir,
		b.iO0, b.iO1, b.iO2, b.iSt, b.iVoff, b.iNv, b.iVir,
		b.eri, b.eps, b.osym,
	} {
		b.dk.FreeDev(p)
	}
}

// args assembles the launch arguments for a row band. in must be FULL HEIGHT (its rows span the
// whole sector) because candidate columns are global; out is the panel this device writes, whose
// rows start at global row outRowOff.
func (b *satDeviceBufs) args(s *satDeviceSoA, in, out backend.BlockView, rowLo, rowHi, outRowOff int) backend.DipSatArgs {
	return backend.DipSatArgs{
		Nsat: s.nsat, Njii: s.njii, Nijk: s.nijk, B: in.Cols, LdIn: in.Ld, LdOut: out.Ld,
		MainOff: s.main, Norb: s.norb, Parts: s.parts, Spin: s.spin,
		RowLo: rowLo, RowHi: rowHi, OutRowOff: outRowOff,
		RTyp: b.rTyp, RGrp: b.rGrp, RPart: b.rPart, RVir: b.rVir,
		JO0: b.jO0, JO1: b.jO1, JSt: b.jSt, JVoff: b.jVoff, JNv: b.jNv, JVir: b.jVir,
		IO0: b.iO0, IO1: b.iO1, IO2: b.iO2, ISt: b.iSt, IVoff: b.iVoff, INv: b.iNv, IVir: b.iVir,
		ERI: b.eri, Eps: b.eps, OrbSym: b.osym, In: in.V, Out: out.V,
	}
}

// newSatelliteMatFreeDevice builds the single-device CUDA satellite applier over dk: the whole
// 3h1p band on one device, input and output both the full resident panels.
func (mx *Matrix) newSatelliteMatFreeDevice(dk backend.DeviceKernels) matFreePart {
	s := mx.buildSatDeviceSoA(mx.buildSatScalarPlan())
	bufs := uploadSatSoA(dk, s)

	apply := func(in, out backend.BlockView) {
		// Whole band on one device: every 3h1p row, output panel at global row offset 0.
		dk.DipSatApply(bufs.args(s, in, out, 0, s.nsat, 0))
	}
	return matFreePart{apply: apply, release: bufs.free}
}
