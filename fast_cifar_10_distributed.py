#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#

from utils import *
from functools import partial
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse

# We use pytorch's distributed package with NCCL for inter-gpu communication
# Define the process group
dist.init_process_group(
    backend='nccl',
    init_method='env://'
)

# Set variables for the local worker to determine its rank and the world size
rank = dist.get_rank()
is_rank0 = rank == 0
world_size = dist.get_world_size()

# Assign a device for this worker.
device = torch.device(
    "cuda:{}".format(rank) if torch.cuda.is_available() else "cpu"
)
torch.random.manual_seed(rank)
torch.backends.cudnn.benchmark = True


def run_benchmark(lr_scaler=1.0,
                  lr_end_fraction=0.1,
                  epochs=16,
                  batch_size=512,
                  ema_epochs=2,
                  n_runs=1,
                  warmup_fraction=5):

    # Wait for GPUS to be initialized
    torch.cuda.synchronize()

    # Download the dataset
    dataset = cifar10(root='./data/') # downloads dataset

    # Start timing all processes together
    dist.barrier()
    timer = Timer(synch=torch.cuda.synchronize)
    # Copy the dataset to the GPUs
    dataset = map_nested(to(device), dataset)
    dist.barrier()
    timer()
    data_transfer_time = timer.total_time
    if rank == 0:
        print(f"Uploaded data to GPUs {data_transfer_time:.3f}s")

    # Select a shard of the training dataset for this worker, and select all of the validation dataset
    selector = list(range(rank, len(dataset['train']['data']), world_size))
    dataset = {'train': {'data': dataset['train']['data'][selector], 'targets':dataset['train']['targets'][selector]},
                         'valid': dataset['valid']}

    # Upload the mean and standard deviations to the GPU
    mean, std = [torch.tensor(x, device=device, dtype=torch.float16) for x in (CIFAR10_MEAN, CIFAR10_STD)]

    train_set = preprocess(dataset['train'], [partial(pad, border=4), transpose,
                                              partial(normalise, mean=mean, std=std), to(torch.float16)])
    valid_set = preprocess(dataset['valid'], [transpose,
                                              partial(normalise, mean=mean, std=std), to(torch.float16)])

    train_batches = partial(
        Batches,
        dataset=train_set,
        shuffle=True,
        drop_last=True,
        max_options=200,
        device=device
    )

    valid_batches = partial(
        Batches,
        dataset=valid_set,
        shuffle=False,
        drop_last=False,
        device=device
    )

    # Data pre-processing
    dist.barrier()
    timer()
    eigen_values, eigen_vectors = compute_patch_whitening_statistics(train_set)
    timer(update_total=False)  # We do not count the data pre-processing time

    # Run the training process n_runs times
    logs = []
    for run in range(n_runs):

        # Network construction
        # Architecture
        channels = {'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512}

        input_whitening_net = build_network(
            channels=channels, extra_layers=(), res_layers=('layer1', 'layer3'),
            conv_pool_block=conv_pool_block_pre, prep_block=partial(whitening_block,
                                                                    eigen_values=eigen_values,
                                                                    eigen_vectors=eigen_vectors),
            scale=1 / 16,
            types={
                nn.ReLU: partial(nn.CELU, 0.3),
                BatchNorm: partial(GhostBatchNorm, num_splits=16, weight=False)
            }

        )

        # Model to evaluate after the distributed model is trained
        local_eval_model = Network(input_whitening_net, label_smoothing_loss(0.2)).half().to(device)

        # Distributed model to train by all workers
        distributed_model = Network(input_whitening_net, label_smoothing_loss(0.2)).half().to(device)
        is_bias = group_by_key(('bias' in k, v) for k, v in trainable_params(distributed_model).items())
        loss = distributed_model.loss

        # Make sure all workers start timing here
        dist.barrier()
        timer = Timer(torch.cuda.synchronize)

        # Wrap with distributed data parallel, this introduces hooks to execute all-reduce upon back propagation
        distributed_model = DDP(distributed_model, device_ids=[rank])

        if is_rank0:
            # Save the model in rank 0 to initialize all the others
            with open('initialized.model', 'wb') as f:
                torch.save(distributed_model.state_dict(), f)

        dist.barrier()
        with open('initialized.model', 'rb') as f:
            distributed_model.load_state_dict(torch.load(f))

        # Data iterators
        transforms = (Crop(32, 32), FlipLR())
        tbatches = train_batches(batch_size, transforms)
        train_batch_count = len(tbatches)
        vbatches = valid_batches(batch_size)

        # Construct the learning rate, weight decay and momentum schedules.
        opt_params = {'lr': lr_schedule(
            [0, epochs / warmup_fraction, epochs - ema_epochs],
            [0.0, lr_scaler * 1.0, lr_scaler * lr_end_fraction],
            batch_size, train_batch_count
        ),
            'weight_decay': Const(5e-4 * lr_scaler * batch_size), 'momentum': Const(0.9)}

        opt_params_bias = {'lr': lr_schedule(
            [0, epochs / warmup_fraction, epochs - ema_epochs],
            [0.0, lr_scaler * 1.0 * 64, lr_scaler * lr_end_fraction * 64],
            batch_size, train_batch_count
        ),
            'weight_decay': Const(5e-4 * lr_scaler * batch_size / 64), 'momentum': Const(0.9)}

        opt = SGDOpt(
            weight_param_schedule=opt_params,
            bias_param_schedule=opt_params_bias,
            weight_params=is_bias[False],
            bias_params=is_bias[True]
        )

        # Train the network
        distributed_model.train(True)
        epochs_log = []
        for epoch in range(epochs):
            activations_log = []
            for tb in tbatches:
                # Forward step
                out = loss(distributed_model(tb))
                distributed_model.zero_grad()
                out['loss'].sum().backward() 
                opt.step()

                # Log activations
                activations_log.append(('loss', out['loss'].detach()))
                activations_log.append(('acc', out['acc'].detach()))

            # Compute the average over the activation logs for the last epoch
            res = map_values((lambda xs: to_numpy(torch.cat(xs)).astype(np.float)), group_by_key(activations_log))
            train_summary = mean_members(res)
            timer()

            # Evaluate the model
            # Copy the weights to the local model
            model_dict = {k[7:]: v for k, v in distributed_model.state_dict().items()}
            local_eval_model.load_state_dict(model_dict)
            valid_summary = eval_on_batches(local_eval_model, loss, vbatches)
            timer(update_total=False)
            time_to_epoch_end = timer.total_time + data_transfer_time
            epochs_log.append(
                {
                    'valid': valid_summary,
                    'train': train_summary,
                    'time': time_to_epoch_end
                }
            )

        # Wait until all models finished training
        dist.barrier()
        timer()

        # Print output
        if is_rank0:
            print("Train acc {:.3f} loss {:.3f}, validation acc {:.3f} loss {:.3f} wall time {:3.3f}s".format(
                train_summary['acc'], train_summary['loss'],
                valid_summary['acc'], valid_summary['loss'],
                timer.total_time + data_transfer_time
            ))

            if run == 0:
                save_log_to_tsv(epochs_log, path='timing_log.tsv')
                # Save the model
                torch.save(local_eval_model.state_dict(), 'replica_0_model')

        logs.append(
            {
                'tain_acc': train_summary['acc'],
                'tain_loss': train_summary['loss'],
                'valid_acc': valid_summary['acc'],
                'valid_loss': valid_summary['loss'],
                'time': timer.total_time
            }
        )

        dist.barrier()

    # Compute the average accuracies and training times
    times = [d['time'] for d in logs]
    accuracies = [d['valid_acc'] for d in logs]
    if is_rank0:
        print("Maximum training time {} median {}".format(np.max(times), np.median(times)))
        print("Lowest accuracy {} median {}".format(np.min(accuracies), np.median(accuracies)))
        print("{} runs reached 0.94 out of {}".format(
            np.count_nonzero(
                np.array(accuracies) >= 0.94
            ),
            n_runs
        ))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_scaler', type=float, default=1.5,
                        help='Multiplicative scaling factor for the learning rate schedule')
    parser.add_argument('--lr_scaler_end_fraction', type=float, default=0.1,
                        help='Fraction of the peak learning rate used for the final step')
    parser.add_argument('--epochs', type=int, default=18,
                        help='Total number of training epochs')
    parser.add_argument('--warmup_fraction', type=float, default=5,
                        help='Inverse of fraction of the epochs used to reach the peak learning rate')
    parser.add_argument('--ema_ep', type=float, default=2,
                        help='Number of epochs (at the end of training) '
                             'where the learing rate is to be maintained constant')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Per GPU batch size')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of replicas')
    args = parser.parse_args()
    run_benchmark(
        lr_scaler=args.lr_scaler,
        lr_end_fraction=args.lr_scaler_end_fraction,
        epochs=args.epochs,
        ema_epochs=args.ema_ep,
        n_runs=args.runs,
        batch_size=args.batch_size,
        warmup_fraction=args.warmup_fraction
    )
