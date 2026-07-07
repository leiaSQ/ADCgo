package dip

import "testing"

// H2O/cc-pVDZ (symmetry off): nocc=5, norb=24, nvir=19.
const (
	testNocc = 5
	testNorb = 24
	testNvir = 19
)

func TestConfigCountsSinglet(t *testing.T) {
	s := NewSpace(testNocc, testNorb, nil, 0, Singlet)

	// |ii> = 5, |ij> (i>j) = C(5,2)=10 → main block 15.
	if s.BeginIJ != 5 {
		t.Errorf("BeginIJ=%d want 5", s.BeginIJ)
	}
	if s.MainBlockSize() != 15 {
		t.Errorf("main block=%d want 15", s.MainBlockSize())
	}
	// |jiir>: 5*4=20 groups × 19 virtuals = 380.
	jiirDim := s.BeginIJK - s.BeginJII
	if jiirDim != 20*testNvir {
		t.Errorf("|jiir> dim=%d want %d", jiirDim, 20*testNvir)
	}
	if len(s.JII) != 20 {
		t.Errorf("JII groups=%d want 20", len(s.JII))
	}
	// |ijkr,T>: C(5,3)=10 groups × mult(2) × 19 = 380.
	ijkrDim := s.Size() - s.BeginIJK
	if ijkrDim != 10*2*testNvir {
		t.Errorf("|ijkr> dim=%d want %d", ijkrDim, 10*2*testNvir)
	}
	if len(s.IJK) != 10 {
		t.Errorf("IJK groups=%d want 10", len(s.IJK))
	}
	if s.Size() != 15+380+380 {
		t.Errorf("total size=%d want 775", s.Size())
	}
}

func TestConfigCountsTriplet(t *testing.T) {
	s := NewSpace(testNocc, testNorb, nil, 0, Triplet)

	// No |ii>; |ij| (i>j) = 10 → main block 10.
	if s.BeginIJ != 0 {
		t.Errorf("BeginIJ=%d want 0", s.BeginIJ)
	}
	if s.MainBlockSize() != 10 {
		t.Errorf("main block=%d want 10", s.MainBlockSize())
	}
	// |ijkr,T>: 10 groups × mult(3) × 19 = 570.
	ijkrDim := s.Size() - s.BeginIJK
	if ijkrDim != 10*3*testNvir {
		t.Errorf("|ijkr> dim=%d want %d", ijkrDim, 10*3*testNvir)
	}
	if s.Size() != 10+380+570 {
		t.Errorf("total size=%d want 960", s.Size())
	}
}

// TestGroupBoundaries checks the group-start arrays are consistent strides.
func TestGroupBoundaries(t *testing.T) {
	s := NewSpace(testNocc, testNorb, nil, 0, Singlet)

	if s.JII[0] != s.BeginJII {
		t.Errorf("JII[0]=%d want BeginJII=%d", s.JII[0], s.BeginJII)
	}
	if s.IJK[0] != s.BeginIJK {
		t.Errorf("IJK[0]=%d want BeginIJK=%d", s.IJK[0], s.BeginIJK)
	}
	// Each type-I group spans exactly nvir rows.
	for m := range s.JII {
		end := s.BeginIJK
		if m+1 < len(s.JII) {
			end = s.JII[m+1]
		}
		if end-s.JII[m] != testNvir {
			t.Fatalf("JII group %d stride=%d want %d", m, end-s.JII[m], testNvir)
		}
	}
	// Each type-II group spans mult*nvir rows.
	for m := range s.IJK {
		end := s.Size()
		if m+1 < len(s.IJK) {
			end = s.IJK[m+1]
		}
		if end-s.IJK[m] != s.Mult*testNvir {
			t.Fatalf("IJK group %d stride=%d want %d", m, end-s.IJK[m], s.Mult*testNvir)
		}
	}
}
