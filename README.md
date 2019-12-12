# Training cifar-10 in under 11 seconds

We use 8 V100 Nvidia GPUs to train cifar-10 in under 11 seconds using mini-batches of 2048 elements,
with 256 elements per GPU.
Our code modifies David C. Pages's 
[bag of tricks](https://github.com/davidcpage/cifar10-fast/blob/master/bag_of_tricks.ipynb) 
implementation to take advantage of Nvidia's NCCL through pytorch's distributed training framework.

In our modifications we remove the exponential moving average model and extend the code to use multiple GPUs, 
but otherwise maintain the rest of the original strategies. 

We distribute the computation using data parallelism, that is: we maintain a copy of the full model in each GPU and process a subset of 
 the mini-batch in each. After every backward pass we average the 8 resulting gradient estimates to produce the 
 final gradient estimate. 
 At the start of training we partition the dataset into 8 subsets and assign one subset to each worker. We maintain 
 the same partition for all epochs. 

## Faster training with larger batches

We execute 50 runs and achieve an accuracy of 94% (or more) in 41 of them, with a maximum training time of 
10.31 (seconds) mean training time of 7.81 (seconds), minimum accuracy of 93.74% and median accuracy of 94.19%.
A log of the output is available in [run_log.txt](run_log.txt)

### Reproducing our results 
#### Hardware
These results were generated on a platform with:
- 8 NVIDIA Tesla V100-SMX2 GPUs witn 32 GB of memory
- An Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz with 20 cores

#### Dependencies
We use python 3.6 the remaining dependencies are listed in requirements.txt

#### To Run
- Navigate to large_batch_cifar_10_fast 
- execute `bash ./train_cifar_parallel.sh` the log will be printed onto STDOUT and the file timing_log.tsv will contain 
the per-epoch accuracies and run-times of the first replica as measured by worker 0

## DAWNBench 
 Note that DAWNBench timings do not include validation time, as in 
 [this FAQ](https://github.com/stanford-futuredata/dawn-bench-entries), 
 but do include initial preprocessing.
 
## License
This sample code is released under the [LICENSE](LICENSE) terms.

