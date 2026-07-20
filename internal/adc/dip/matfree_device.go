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
// Compiles on every build (it calls the backend.DeviceKernels interface, not CUDA directly); it is
// only reached when the backend implements DeviceKernels, i.e. the cuda backend. Parity on real
// hardware is TestSatelliteMatFreeDeviceParity (matfree_cuda_test.go).

// newSatelliteMatFreeDevice builds the CUDA satellite applier over dk. It reuses the per-scalar
// plan's group/row layout, flattens it to the device SoA the kernel expects, uploads the ERI
// tensor + orbital energies + irreps, and returns a matFreePart whose apply launches the kernel.
func (mx *Matrix) newSatelliteMatFreeDevice(dk backend.DeviceKernels) matFreePart {
	sp := mx.sp
	norb := sp.Norb
	p := mx.buildSatScalarPlan()

	// Per-row SoA (int32).
	rTyp := make([]int32, len(p.rTyp))
	for i, v := range p.rTyp {
		rTyp[i] = int32(v)
	}
	// p.rGrp/rPart/rVir are already []int32.

	// JII group SoA + flat virtuals.
	njii := len(p.jCfg)
	jO0, jO1 := make([]int32, njii), make([]int32, njii)
	jSt, jVoff, jNv := make([]int32, njii), make([]int32, njii), make([]int32, njii)
	var jVir []int32
	for g := range njii {
		jO0[g], jO1[g] = int32(p.jCfg[g].Occ[0]), int32(p.jCfg[g].Occ[1])
		jSt[g], jVoff[g], jNv[g] = int32(p.jSt[g]), int32(len(jVir)), int32(len(p.jVir[g]))
		for _, orb := range p.jVir[g] {
			jVir = append(jVir, int32(orb))
		}
	}

	// IJK group SoA + flat virtuals.
	nijk := len(p.iCfg)
	iO0, iO1, iO2 := make([]int32, nijk), make([]int32, nijk), make([]int32, nijk)
	iSt, iVoff, iNv := make([]int32, nijk), make([]int32, nijk), make([]int32, nijk)
	var iVir []int32
	for g := range nijk {
		iO0[g], iO1[g], iO2[g] = int32(p.iCfg[g].Occ[0]), int32(p.iCfg[g].Occ[1]), int32(p.iCfg[g].Occ[2])
		iSt[g], iVoff[g], iNv[g] = int32(p.iSt[g]), int32(len(iVir)), int32(len(p.iVir[g]))
		for _, orb := range p.iVir[g] {
			iVir = append(iVir, int32(orb))
		}
	}

	// Flat ERI tensor: eri[((a·n+b)·n+c)·n+d] = Eri(a,b,c,d) (row-major; matches dd_eri in
	// adc2dip_kernels.cu). Irreps as int32 (the kernel's dSym test reads osym[absOrbital]).
	eri := make([]float64, norb*norb*norb*norb)
	for a := range norb {
		for b := range norb {
			for c := range norb {
				base := ((a*norb+b)*norb + c) * norb
				for d := range norb {
					eri[base+d] = mx.ints.Eri(a, b, c, d)
				}
			}
		}
	}
	osym := make([]int32, norb)
	for o := range norb {
		osym[o] = int32(mx.ints.OrbIrrep(o))
	}

	spin := 0
	if sp.Spin == Triplet {
		spin = 1
	}

	// Upload everything once; the pointers live until release().
	dRTyp, dRGrp, dRPart, dRVir := dk.UploadInts(rTyp), dk.UploadInts(p.rGrp), dk.UploadInts(p.rPart), dk.UploadInts(p.rVir)
	dJO0, dJO1, dJSt, dJVoff, dJNv, dJVir := dk.UploadInts(jO0), dk.UploadInts(jO1), dk.UploadInts(jSt), dk.UploadInts(jVoff), dk.UploadInts(jNv), dk.UploadInts(jVir)
	dIO0, dIO1, dIO2 := dk.UploadInts(iO0), dk.UploadInts(iO1), dk.UploadInts(iO2)
	dISt, dIVoff, dINv, dIVir := dk.UploadInts(iSt), dk.UploadInts(iVoff), dk.UploadInts(iNv), dk.UploadInts(iVir)
	dERI, dEps, dOsym := dk.DeviceERI(eri), dk.UploadFloats(mx.eps), dk.UploadInts(osym)

	apply := func(in, out backend.BlockView) {
		dk.DipSatApply(backend.DipSatArgs{
			Nsat: len(p.rTyp), Njii: njii, Nijk: nijk, B: in.Cols, LdIn: in.Ld, LdOut: out.Ld,
			MainOff: p.main, Norb: norb, Parts: p.parts, Spin: spin,
			RTyp: dRTyp, RGrp: dRGrp, RPart: dRPart, RVir: dRVir,
			JO0: dJO0, JO1: dJO1, JSt: dJSt, JVoff: dJVoff, JNv: dJNv, JVir: dJVir,
			IO0: dIO0, IO1: dIO1, IO2: dIO2, ISt: dISt, IVoff: dIVoff, INv: dINv, IVir: dIVir,
			ERI: dERI, Eps: dEps, OrbSym: dOsym, In: in.V, Out: out.V,
		})
	}
	release := func() {
		for _, ptr := range []unsafe.Pointer{
			dRTyp, dRGrp, dRPart, dRVir,
			dJO0, dJO1, dJSt, dJVoff, dJNv, dJVir,
			dIO0, dIO1, dIO2, dISt, dIVoff, dINv, dIVir,
			dERI, dEps, dOsym,
		} {
			dk.FreeDev(ptr)
		}
	}
	return matFreePart{apply: apply, release: release}
}
