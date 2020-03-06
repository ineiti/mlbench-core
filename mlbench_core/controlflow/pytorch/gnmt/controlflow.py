from mlbench_core.utils.pytorch.helpers import iterate_dataloader


def get_model_output(input, target, model, batch_first):
    src, src_lengths = input
    tgt, tgt_lengths = target

    if batch_first:
        try:
            output = model(src.long(), src_lengths.long(), tgt[:, :-1])
        except ValueError as e:
            print(src, src_lengths)
        tgt_labels = tgt[:, 1:]
    else:
        output = model(src.long(), src_lengths.long(), tgt[:-1])
        tgt_labels = tgt[1:]

    return output, tgt_labels


def compute_criterion(output, tgt_labels, criterion, batch_first):
    if batch_first:
        T, B = output.size(1), output.size(0)
    else:
        T, B = output.size(0), output.size(1)
    return criterion(output.view(T * B, -1),
                     tgt_labels.contiguous().view(-1))


def train_round(
        dataloader,
        model,
        optimizer,
        loss_function,
        metrics,
        scheduler,
        dtype,
        batch_first,
        schedule_per="epoch",
        transform_target_type=None,
        use_cuda=False,
        max_batch_per_epoch=None,
        tracker=None,
):
    model.train()

    if tracker:
        tracker.train()

    data_iter = iterate_dataloader(
        dataloader, dtype, max_batch_per_epoch, use_cuda, transform_target_type
    )

    num_batches_per_device_train = len(dataloader)

    # if schedule_per == "epoch":
    #     scheduler.step()

    for batch_idx, (data, target) in enumerate(data_iter):

        if tracker:
            tracker.batch_start()

        # if schedule_per == "batch":
        #     scheduler.step()

        # Clear gradients in the optimizer.
        optimizer.zero_grad()
        if tracker:
            tracker.record_batch_step("init")

        # Compute the output
        output, tgt_labels = get_model_output(data, target, model, batch_first)

        if tracker:
            tracker.record_batch_step("fwd_pass")

        # Compute the loss
        loss = compute_criterion(output, tgt_labels, loss_function, batch_first)

        if tracker:
            tracker.record_batch_step("comp_loss")

        # TODO: if AMP optimizer, do not call backward on loss
        # Backprop
        loss.backward()
        if tracker:
            tracker.record_batch_step("backprop")

        # Aggregate gradients/parameters from all workers and apply updates
        # to model
        optimizer.step()
        if tracker:
            tracker.record_batch_step("opt_step")

            tracker.batch_end()
        #
        # _record_train_batch_stats(
        #     batch_idx,
        #     loss.item(),
        #     output,
        #     target,
        #     metrics,
        #     tracker,
        #     num_batches_per_device_train,
        # )
