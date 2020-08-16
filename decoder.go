package ten

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"io/ioutil"
)

// Decoding errors
var (
	ErrMagicNumberMismatch     = fmt.Errorf("magic number mismatch")
	ErrNegativeLength          = fmt.Errorf("negative length")
	ErrNegativeDimensions      = fmt.Errorf("negative dimensions")
	ErrDecodingUnsupportedType = fmt.Errorf("unsupported data type")
)

// A Decoder reads and decodes tensor data from an input stream.
type Decoder struct {
	r io.Reader
}

// NewDecoder returns a new decoder that reads from r.
func NewDecoder(r io.Reader) *Decoder {
	return &Decoder{r}
}

// Decode reads the next ten-encoded tensor from its input.
func (d *Decoder) Decode() (tensorData interface{}, shape []int, info string, err error) {
	/*
		Decoding chunks can be further optimized to reduce allocations by
		reusing buffers between invocations of Decode.
	*/
	buf := make([]byte, 8)

	if _, err := io.ReadFull(d.r, buf); err != nil {
		return nil, nil, "", err
	}

	if !bytes.Equal(buf, MagicNumber) {
		return nil, nil, "", ErrMagicNumberMismatch
	}

	var len int64
	if err := binary.Read(d.r, binary.LittleEndian, &len); err != nil {
		return nil, nil, "", err
	} else if len < 0 {
		return nil, nil, "", ErrNegativeLength
	}

	dataType := make([]byte, 8)
	if _, err := io.ReadFull(d.r, dataType); err != nil {
		return nil, nil, "", err
	}

	infoBytes := make([]byte, 8)
	if _, err := io.ReadFull(d.r, infoBytes); err != nil {
		return nil, nil, "", err
	}

	var dimensions int64
	if err := binary.Read(d.r, binary.LittleEndian, &dimensions); err != nil {
		return nil, nil, "", err
	} else if dimensions < 0 {
		return nil, nil, "", ErrNegativeDimensions
	}

	shape = make([]int, int(dimensions))
	for i := 0; i < int(dimensions); i++ {
		var dim int64
		if err := binary.Read(d.r, binary.LittleEndian, &dim); err != nil {
			return nil, nil, "", err
		} else if dim < 0 {
			return nil, nil, "", ErrNegativeDimensions
		}
		shape[i] = int(dim)
	}

	if _, err := io.CopyN(ioutil.Discard, d.r, int64(alignment(len))); err != nil {
		return nil, nil, "", err
	}

	if _, err := io.ReadFull(d.r, buf); err != nil {
		return nil, nil, "", err
	}

	if !bytes.Equal(buf, MagicNumber) {
		return nil, nil, "", ErrMagicNumberMismatch
	}

	if err := binary.Read(d.r, binary.LittleEndian, &len); err != nil {
		return nil, nil, "", err
	} else if len < 0 {
		return nil, nil, "", ErrNegativeLength
	}

	tensorData, err = d.readTensorData(dataType, len)
	if err != nil {
		return nil, nil, "", err
	}

	if _, err := io.CopyN(ioutil.Discard, d.r, int64(alignment(len))); err != nil {
		return nil, nil, "", err
	}

	return tensorData, shape, string(bytes.Trim(infoBytes, "\x00")), nil
}

func (d *Decoder) readTensorData(dataType []byte, bytes int64) (interface{}, error) {
	/*
		Possible improvements to the performance of reading tensor data:

		- Re-use a shared buffer for tensor data before converting to the target type.
		- Write conversion in assembler to avoid bounds check and possibly use SIMD.
	*/

	switch string(dataType[:2]) {
	case "f4":
		tensorData := make([]float32, bytes/4)
		if err := binary.Read(d.r, binary.LittleEndian, tensorData); err != nil {
			return nil, err
		}

		return tensorData, nil
	case "f8":
		tensorData := make([]float64, bytes/8)
		if err := binary.Read(d.r, binary.LittleEndian, tensorData); err != nil {
			return nil, err
		}

		return tensorData, nil
	case "i1":
		tensorData := make([]int8, bytes/1)
		if err := binary.Read(d.r, binary.LittleEndian, tensorData); err != nil {
			return nil, err
		}

		return tensorData, nil
	case "i2":
		tensorData := make([]int16, bytes/2)
		if err := binary.Read(d.r, binary.LittleEndian, tensorData); err != nil {
			return nil, err
		}

		return tensorData, nil
	case "i4":
		tensorData := make([]int32, bytes/4)
		if err := binary.Read(d.r, binary.LittleEndian, tensorData); err != nil {
			return nil, err
		}

		return tensorData, nil
	case "i8":
		tensorData := make([]int64, bytes/8)
		if err := binary.Read(d.r, binary.LittleEndian, tensorData); err != nil {
			return nil, err
		}

		return tensorData, nil
	case "u1":
		tensorData := make([]uint8, bytes/1)
		if err := binary.Read(d.r, binary.LittleEndian, tensorData); err != nil {
			return nil, err
		}

		return tensorData, nil
	case "u2":
		tensorData := make([]uint16, bytes/2)
		if err := binary.Read(d.r, binary.LittleEndian, tensorData); err != nil {
			return nil, err
		}

		return tensorData, nil
	case "u4":
		tensorData := make([]uint32, bytes/4)
		if err := binary.Read(d.r, binary.LittleEndian, tensorData); err != nil {
			return nil, err
		}

		return tensorData, nil
	case "u8":
		tensorData := make([]uint64, bytes/8)
		if err := binary.Read(d.r, binary.LittleEndian, tensorData); err != nil {
			return nil, err
		}

		return tensorData, nil
	default:
		return nil, ErrDecodingUnsupportedType
	}
}
