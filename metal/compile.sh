xcrun -sdk macosx metal -c fft-big.metal -o fft-big.air
xcrun -sdk macosx metallib fft-big.air -o fft-big.metallib