# default configure
class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def config():
    # configuration for PIG
    cfg = AttrDict()

    cfg.sigma = 1.0
    #########参数设置#########
    cfg.lamda = 632.8e-6     #wavelength 637
    cfg.z2 = 31   #propagate_distance 10.77
    cfg.pix = 2*4.65e-3  / cfg.sigma  #pix size4.5

    cfg.m = 200
    cfg.n =  200
    cfg.r = 90
    cfg.delta_r = 1/(cfg.lamda *50/(cfg.m * cfg.pix))

    cfg.k = 500

    return cfg