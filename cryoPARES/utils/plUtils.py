import pytorch_lightning
from packaging import version

is_pylig2 = version.parse(pytorch_lightning.__version__) >= version.parse("2.0.0")


def GET_PL_PLUGIN():
    # return None
    from lightning_fabric.plugins.environments import LightningEnvironment
    #LightningEnvironment() is required to avoid it to automatically use SLURM, which requires configuring other stuff
    return LightningEnvironment()


def GET_PL_STRATEGY(dev_count):
    if dev_count > 1:
        return "ddp_find_unused_parameters_false"
    else:
        if is_pylig2:
            return "auto"
        else:
            return None

def GET_PL_NUM_NODES():
    return 1


