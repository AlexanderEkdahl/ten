package ten

import (
	"encoding/binary"
	"fmt"
	"io"
)

// MagicNumber is the magic number before every chunk.
var MagicNumber = []byte{0x7e, 0x54, 0x65, 0x6e, 0x42, 0x69, 0x6e, 0x7e}

// Encoding errors
var (
	ErrTooManyDimensions = fmt.Errorf("too many dimensions")
	ErrInfoTooLong       = fmt.Errorf("info can not exceed 8 bytes")
)

// An Encoder writes tensors to an output stream.
type Encoder struct {
	w io.Writer
}

// NewEncoder returns a new encoder that writes to w.
func NewEncoder(w io.Writer) *Encoder {
	return &Encoder{w}
}

// Encode writes the tensor encoding of t to the stream along with a
// custom info header.
func (e *Encoder) Encode(tensorData interface{}, shape []int, info string) error {
	t := dataType(tensorData)
	if t == nil {
		return fmt.Errorf("unsupported data type: %T", tensorData)
	}

	if len(shape) >= 10 {
		return ErrTooManyDimensions
	}

	infoBytes := []byte(info)
	if len(infoBytes) > 8 {
		return ErrInfoTooLong
	}

	if err := e.writeHeader(t, shape, infoBytes); err != nil {
		return fmt.Errorf("encoding chunk header: %w", err)
	}

	if err := e.writeTensorData(tensorData); err != nil {
		return err
	}

	return nil
}

func dataType(tensorData interface{}) []byte {
	switch tensorData.(type) {
	case []float32:
		return []byte("f4")
	case []float64:
		return []byte("f8")
	case []int8:
		return []byte("i1")
	case []int16:
		return []byte("i2")
	case []int32:
		return []byte("i4")
	case []int64:
		return []byte("i8")
	case []uint8:
		return []byte("u1")
	case []uint16:
		return []byte("u2")
	case []uint32:
		return []byte("u4")
	case []uint64:
		return []byte("u8")
	default:
		return nil
	}
}

func (e *Encoder) writeHeader(t []byte, shape []int, info []byte) error {
	if _, err := e.w.Write(MagicNumber); err != nil {
		return err
	}

	size := int64(8 * (3 + len(shape)))
	if err := binary.Write(e.w, binary.LittleEndian, size); err != nil {
		return err
	}

	if _, err := e.w.Write(str(t)); err != nil {
		return err
	}

	if _, err := e.w.Write(str(info)); err != nil {
		return err
	}

	if err := binary.Write(e.w, binary.LittleEndian, int64(len(shape))); err != nil {
		return err
	}

	for _, s := range shape {
		if err := binary.Write(e.w, binary.LittleEndian, int64(s)); err != nil {
			return err
		}
	}

	if _, err := e.w.Write(padding[:alignment(size)]); err != nil {
		return err
	}

	return nil
}

func (e *Encoder) writeTensorData(tensorData interface{}) error {
	if _, err := e.w.Write(MagicNumber); err != nil {
		return err
	}

	size := int64(binary.Size(tensorData))
	if err := binary.Write(e.w, binary.LittleEndian, size); err != nil {
		return err
	}

	if err := binary.Write(e.w, binary.LittleEndian, tensorData); err != nil {
		return err
	}

	if _, err := e.w.Write(padding[:alignment(size)]); err != nil {
		return err
	}

	return nil
}

func str(s []byte) []byte {
	b := make([]byte, 8)

	for i, sx := range s {
		b[i] = sx
	}

	return b
}

// avoid allocation by re-using padding
var padding = make([]byte, 64)

// alignment find the next 64 byte boundary
func alignment(n int64) int64 {
	return ((n + 63) & -64) - n
}
