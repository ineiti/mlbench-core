import torch
from apex import amp
from mlbench_core.optim.pytorch import FP16Optimizer, FP32Optimizer, AMPOptimizer
from mlbench_core.lr_scheduler.pytorch.lr import ExponentialWarmupMultiStepLR


def build_fp_optimizer(
    model, math, opt_config, grad_clip, loss_scaling, scheduler_config, iters
):
    params = model.parameters()
    opt_name = opt_config.pop("optimizer")
    optimizer = torch.optim.__dict__[opt_name](params, **opt_config)

    # Create a learning rate scheduler for an optimizer
    scheduler = ExponentialWarmupMultiStepLR(optimizer, iters, **scheduler_config)

    if math == "manual_fp16":
        fp_optimizer = FP16Optimizer(
            model=model,
            optimizer=None,
            grad_clip=grad_clip,
            loss_scale=loss_scaling["init_scale"],
            dls_upscale_interval=loss_scaling["upscale_interval"],
        )
        params = fp_optimizer.fp32_params
        optimizer = torch.optim.__dict__[opt_name](params, **opt_config)
        fp_optimizer.set_optimizer(optimizer)

    elif math == "fp32":
        fp_optimizer = FP32Optimizer(
            model=model, optimizer=optimizer, grad_clip=grad_clip
        )

    elif math == "fp16":
        model, optimizer = amp.initialize(
            model,
            optimizer,
            cast_model_outputs=torch.float16,
            keep_batchnorm_fp32=False,
            opt_level="O2",
        )

        fp_optimizer = AMPOptimizer(
            model,
            optimizer,
            grad_clip=grad_clip,
            loss_scale=loss_scaling["init_scale"],
            dls_upscale_interval=loss_scaling["upscale_interval"],
        )
    else:
        return NotImplementedError()

    return fp_optimizer, scheduler, model
