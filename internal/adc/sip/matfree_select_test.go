package sip

import (
	"testing"
	"unsafe"

	"github.com/leiaSQ/ADCgo/internal/adc/backend"
)

// stubDeviceBackend reproduces the shape of the real CUDA backend for selection tests: it
// embeds Gonum — exactly as backend.gpuBackend does (backend/gpu_device.go) — so it satisfies
// backend.HostData through the promoted Gonum.HostSlice, while ALSO satisfying
// backend.DeviceKernels. That combination is what made the order-4 c22 selection pick the host
// applier on a real GPU and panic on HostSlice(devVec); reproducing it here catches a
// regression without needing a GPU allocation. The kernel methods are never called by the
// selection logic under test, so they are inert.
type stubDeviceBackend struct{ backend.Gonum }

func (stubDeviceBackend) SetCoeff1([]float64)                   {}
func (stubDeviceBackend) DeviceERI([]float64) unsafe.Pointer    { return nil }
func (stubDeviceBackend) UploadInts([]int32) unsafe.Pointer     { return nil }
func (stubDeviceBackend) UploadFloats([]float64) unsafe.Pointer { return nil }
func (stubDeviceBackend) FreeDev(unsafe.Pointer)                {}
func (stubDeviceBackend) DevPtr(backend.Vector) unsafe.Pointer  { return nil }
func (stubDeviceBackend) Wert2Apply(backend.Wert2Args)          {}
func (stubDeviceBackend) C22Apply(backend.C22Args)              {}
func (stubDeviceBackend) DipSatApply(backend.DipSatArgs)        {}

// Compile-time proof the stub really does present both capabilities — if either assertion
// breaks, the test below stops exercising the condition it exists for.
var (
	_ backend.DeviceKernels = stubDeviceBackend{}
	_ backend.HostData      = stubDeviceBackend{}
)

// TestMatFreeC22RejectsDeviceBackend pins the ordering fix in matFreeC22: there is no device
// c22elem4 kernel, so a DeviceKernels backend must fall back to the dense order-4 satellite
// block even under -matfree on. Without the DeviceKernels-first test, the embedded Gonum makes
// the HostData assertion succeed and newC22MatFree is selected on a device backend, which
// panics on its first HostSlice of a device vector.
func TestMatFreeC22RejectsDeviceBackend(t *testing.T) {
	dev := &Matrix{be: stubDeviceBackend{}}
	dev.SetMatFree(MatFreeOn, 0)
	if dev.matFreeC22(1 << 40) {
		t.Fatal("matFreeC22 = true on a DeviceKernels backend; must be dense (no device c22elem4 kernel)")
	}

	// The guard must be specific to device backends, not a blanket false: a genuine host
	// backend still takes the matrix-free path.
	host := &Matrix{be: backend.Gonum{}}
	host.SetMatFree(MatFreeOn, 0)
	if !host.matFreeC22(1 << 40) {
		t.Fatal("matFreeC22 = false on a host backend; the host matrix-free path regressed")
	}
}

// TestMatFreeC22O3AcceptsDeviceBackend guards the order-3 twin, which DOES have a device kernel
// (adc4_kernels.cu c22_apply) and so must stay enabled on a DeviceKernels backend — the melanin
// SIP-ADC(3) run depends on it. This is the case that must NOT be swept up by the order-4 fix.
func TestMatFreeC22O3AcceptsDeviceBackend(t *testing.T) {
	dev := &Matrix{be: stubDeviceBackend{}}
	dev.SetMatFree(MatFreeOn, 0)
	if !dev.matFreeC22O3(1 << 40) {
		t.Fatal("matFreeC22O3 = false on a DeviceKernels backend; the order-3 device c22 path regressed")
	}
}
