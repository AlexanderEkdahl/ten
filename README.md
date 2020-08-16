# ten

Efficient binary encoding for tensors. The format is supported by
[WebDataset](https://github.com/tmbdev/webdataset) with the `.ten` filename extension.

The reference implemention developed in Python by [tmbdev](https://github.com/tmbdev) at Nvidia can be found [here](https://github.com/tmbdev/webdataset/blob/master/webdataset/tenbin.py)

`ten` uses Go modules and can be installed by running:

```
go get github.com/AlexanderEkdahl/ten
```

This package uses no external dependencies outside of the Go standard library.

## Usage

``` go
// Writes encoded tensor to w
e := NewEncoder(w)
e.Encode([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, []int{3, 3}, "policy")

// Reads encoded tensor from r
d := NewDecoder(r)
tensorData, shape, info, err := d.Decode()
```

`float16` is not supported due to missing support in Go ([#32022](https://github.com/golang/go/issues/32022)).

## Benchmarks

```
$ go test -bench .
goos: linux
goarch: amd64
pkg: github.com/AlexanderEkdahl/ten
BenchmarkDecoder/100-16         	  531340	      2107 ns/op	 189.88 MB/s
BenchmarkDecoder/500-16         	  224413	      5368 ns/op	 372.55 MB/s
BenchmarkDecoder/1000-16        	  120788	      9430 ns/op	 424.20 MB/s
BenchmarkDecoder/10000-16       	   15444	     78619 ns/op	 508.79 MB/s
BenchmarkEncoder/100-16         	  920988	      1231 ns/op	 324.97 MB/s
BenchmarkEncoder/500-16         	  324014	      3880 ns/op	 515.49 MB/s
BenchmarkEncoder/1000-16        	  159812	      7117 ns/op	 562.01 MB/s
BenchmarkEncoder/10000-16       	   18616	     64677 ns/op	 618.46 MB/s
```
