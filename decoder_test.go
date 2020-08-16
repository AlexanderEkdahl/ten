package ten

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"testing"
)

func TestDecoderFloat32(t *testing.T) {
	buf := bytes.NewBuffer(tensorFloat32.Bytes)
	dec := NewDecoder(buf)

	tensorData, shape, _, err := dec.Decode()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(shape) != len(tensorFloat32.Shape) {
		t.Fatalf("incorrect shape length, got: %d, expected: %d", len(shape), len(tensorFloat32.Shape))
	}
	for i, s := range shape {
		if s != tensorFloat32.Shape[i] {
			t.Fatalf("incorrect shape, got: %d, expected: %d(index %d)", s, tensorFloat32.Shape[i], i)
		}
	}
	tensorDataFloat32, ok := tensorData.([]float32)
	if !ok {
		t.Fatalf("wrong type, got: %T, expected: []float32", tensorDataFloat32)
	}
	if len(tensorDataFloat32) != len(tensorFloat32.Data.([]float32)) {
		t.Fatalf("incorrect tensor data length, got: %d, expected: %d", len(tensorDataFloat32), len(tensorFloat32.Data.([]float32)))
	}
	for i, d := range tensorDataFloat32 {
		if d != tensorFloat32.Data.([]float32)[i] {
			t.Fatalf("incorrect tensor data, got: %f, expected: %f(index %d)", d, tensorFloat32.Data.([]float32)[i], i)
		}
	}

	_, _, _, err = dec.Decode()
	if err != io.EOF {
		t.Fatalf("unexpected error, got: %v, expected: %v", err, io.EOF)
	}
}

func TestDecoderInt8(t *testing.T) {
	buf := bytes.NewBuffer(tensorInt8.Bytes)
	dec := NewDecoder(buf)

	tensorData, shape, _, err := dec.Decode()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(shape) != len(tensorInt8.Shape) {
		t.Fatalf("incorrect shape length, got: %d, expected: %d", len(shape), len(tensorInt8.Shape))
	}
	for i, s := range shape {
		if s != tensorInt8.Shape[i] {
			t.Fatalf("incorrect shape, got: %d, expected: %d(index %d)", s, tensorInt8.Shape[i], i)
		}
	}
	tensorDataInt8, ok := tensorData.([]int8)
	if !ok {
		t.Fatalf("wrong type, got: %T, expected: []int8", tensorDataInt8)
	}
	if len(tensorDataInt8) != len(tensorInt8.Data.([]int8)) {
		t.Fatalf("incorrect tensor data length, got: %d, expected: %d", len(tensorDataInt8), len(tensorInt8.Data.([]int8)))
	}
	for i, d := range tensorDataInt8 {
		if d != tensorInt8.Data.([]int8)[i] {
			t.Fatalf("incorrect tensor data, got: %d, expected: %d(index %d)", d, tensorInt8.Data.([]int8)[i], i)
		}
	}

	_, _, _, err = dec.Decode()
	if err != io.EOF {
		t.Fatalf("unexpected error, got: %v, expected: %v", err, io.EOF)
	}
}

func TestDecoderInvalid(t *testing.T) {
	buf := bytes.NewBuffer([]byte{0x54, 0x65, 0x6e, 0x42, 0x69, 0x6e, 0x7e, 0x24})
	dec := NewDecoder(buf)

	_, _, _, err := dec.Decode()
	if !errors.Is(err, ErrMagicNumberMismatch) {
		t.Fatalf("expected error to be %v, got: %v", ErrMagicNumberMismatch, err)
	}
}

func BenchmarkDecoder(b *testing.B) {
	sizes := []int{1e2, 5e2, 1e3, 1e4}

	for _, size := range sizes {
		var buf bytes.Buffer
		enc := NewEncoder(&buf)
		tensorData := make([]float32, size)
		enc.Encode(tensorData, []int{0}, "")

		b.Run(fmt.Sprintf("%d", size), func(b *testing.B) {
			b.SetBytes(int64(binary.Size(tensorData)))

			for i := 0; i < b.N; i++ {
				buf := bytes.NewBuffer(buf.Bytes())
				NewDecoder(buf).Decode()
			}
		})
	}
}
