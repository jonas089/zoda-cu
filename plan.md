Hi Claude, this project includes an inefficient, CPU implementation of the 
ZODA protocol (zero overhead data availability with Reed solomon codes). 
You can study the CPU implementation that uses NTT over Bigint prime field 
initially, to get an idea of how this works and what has been built yet.

You can use the current implementation as your spec. Your goal is to 
implement a CUDA NTT over the babybear field and use it to 
hardware-accelerate our ZODA implementation (this should be a HUGE 
speedup).


You are expected to get rid of BitUint / Bigint arithmetic and use u64 / u32 (babybear works with u64 so should be fine).

Summary:

- write a CUDA NTT for our new babybear poly in u64 (efficient limbs 
optimized for NVIDIA GPUs) - use it to accelerate our ZODA implementation 
(should be a very significant gain)


Target hardware:
I have an RTX 5090 (main goal), but would be nice if it also worked on older cards like RTX 3060 or perhps even 970 (970 is completely optional)

