from yacs.config import CfgNode as CN


def get_default_cfg():
    EPN_cfg = CN()
    EPN_cfg.model = CN()
    EPN_cfg.MODEL = CN()
    EPN_cfg.train_lr = CN()
    EPN_cfg.train_loss = CN()

    EPN_cfg.model.model = "enc_so3net"
    EPN_cfg.model.input_num = 1024
    EPN_cfg.model.output_num = 32
    EPN_cfg.model.search_radius = 0.4
    EPN_cfg.model.normalize_input = False
    EPN_cfg.model.dropout_rate = 0.0
    EPN_cfg.model.init_method = "xavier"
    EPN_cfg.model.kpconv = False
    EPN_cfg.model.kanchor = 60
    EPN_cfg.model.normals = False
    EPN_cfg.model.flag = "rotation"
    EPN_cfg.model.representation = "quat"
    EPN_cfg.model.pooling_method = "max"

    EPN_cfg.MODEL.num_in_channels = 1
    EPN_cfg.MODEL.num_mid_channels = 1
    EPN_cfg.MODEL.num_channels_R = 1

    EPN_cfg.train_lr.init_lr = 1e-3
    EPN_cfg.train_lr.lr_type = "exp_decay"
    EPN_cfg.train_lr.decay_rate = 0.5
    EPN_cfg.train_lr.decay_step = 10000

    EPN_cfg.train_loss.loss_type = "soft"
    EPN_cfg.train_loss.attention_loss_type = "no_reg"
    EPN_cfg.train_loss.margin = 1.0
    EPN_cfg.train_loss.temperature = 3
    EPN_cfg.train_loss.attention_margin = 1.0
    EPN_cfg.train_loss.attention_pretrain_step = 3000
    EPN_cfg.train_loss.equi_alpha = 0.0
    EPN_cfg.train_loss.eval = False

    EPN_cfg.t_method_type = 2

    return EPN_cfg
