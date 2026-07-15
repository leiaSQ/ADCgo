//go:build cuda

// Multi-GPU fan-out test: verifies NewAll builds one backend per visible device, each
// pinned to a distinct physical device, and that identical work run concurrently on all
// of them agrees — i.e. the per-device contexts are independent and do not cross-
// contaminate. Skips cleanly on a single-GPU box or CPU-only CI (DeviceCount < 2).
package backend

import (
	"math"
	"sync"
	"testing"
)

func TestMultiDeviceFanOut(t *testing.T) {
	n := DeviceCount("cuda")
	if n < 2 {
		t.Skipf("multi-GPU test needs >=2 CUDA devices, have %d", n)
	}
	bes, err := NewAll("cuda", 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(bes) != n {
		t.Fatalf("NewAll returned %d backends, want %d", len(bes), n)
	}

	// Each backend must be pinned to a distinct device.
	seen := map[int]bool{}
	for _, be := range bes {
		di, ok := be.(interface{ DeviceIndex() int })
		if !ok {
			t.Fatal("cuda backend does not expose DeviceIndex()")
		}
		if seen[di.DeviceIndex()] {
			t.Fatalf("duplicate device index %d across NewAll backends", di.DeviceIndex())
		}
		seen[di.DeviceIndex()] = true
	}

	// The same vector uploaded and reduced on every device concurrently must give the
	// same norm — proving the device contexts are independent (a shared handle/context
	// would fail with CUBLAS_STATUS_INTERNAL_ERROR or a wrong answer under concurrency).
	host := make(Vec, 1<<14)
	var want float64
	for i := range host {
		host[i] = float64(i%13) - 6
		want += host[i] * host[i]
	}
	want = math.Sqrt(want)

	got := make([]float64, len(bes))
	var wg sync.WaitGroup
	for i, be := range bes {
		wg.Add(1)
		go func(i int, be Backend) {
			defer wg.Done()
			v := be.Upload(host)
			defer be.Free(v)
			got[i] = be.Nrm2(v)
		}(i, be)
	}
	wg.Wait()

	for i, g := range got {
		if math.Abs(g-want) > 1e-9*want {
			t.Errorf("device %d: Nrm2 = %.12g, want %.12g", i, g, want)
		}
	}
}

// TestMultiDeviceCap checks the maxDevices cap on NewAll.
func TestMultiDeviceCap(t *testing.T) {
	if DeviceCount("cuda") < 2 {
		t.Skip("needs >=2 CUDA devices")
	}
	bes, err := NewAll("cuda", 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(bes) != 1 {
		t.Fatalf("NewAll(cuda, 1) returned %d backends, want 1", len(bes))
	}
}
