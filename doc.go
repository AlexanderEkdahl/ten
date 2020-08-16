/*

Package ten provides efficient binary encoding for tensors. The format is
8 byte aligned and can be used directly for computations when transmitted,
say, via RDMA. The format is supported by WebDataset with the `.ten`
filename extension. It is also used by Tensorcom, Tensorcom RDMA, and can
be used for fast tensor storage with LMDB and in disk files (which can be
memory mapped).

Data is encoded as a series of chunks:
	- magic number (int64)
	- length in bytes (int64)
	- bytes (multiple of 64 bytes long)

Arrays are a header chunk followed by a data chunk.
Header chunks have the following structure:

	- dtype (int64)
	- 8 byte array name
	- ndim (int64)
	- dim[0]
	- dim[1]
	- ...

*/
package ten
