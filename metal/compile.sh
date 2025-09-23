xcrun -sdk macosx metal -c fft.metal -o fft.air
xcrun -sdk macosx metallib fft.air -o fft.metallib

xcrun -sdk macosx metal -c fft-big.metal -o fft-big.air
xcrun -sdk macosx metallib fft-big.air -o fft-big.metallib