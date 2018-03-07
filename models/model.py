import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import functools
from torch.optim import lr_scheduler


###############################################################################
# Functions
###############################################################################


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1.0)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1.0)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='xavier'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [%s] is not implemented' % init_type)


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def get_norm_layer(layer_type='instance'):
    if layer_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif layer_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif layer_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError(
            'normalization layer [%s] is not found' % layer_type)
    return norm_layer


def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(
            nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError(
            'nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer


def define_g(input_nc, output_nc, nz, ngf,
             which_model_net_g='unet_128', norm='batch', nl='relu',
             use_dropout=False, init_type='xavier', gpu_ids=list(), where_add='input', upsample='bilinear'):
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(layer_type=norm)
    nl_layer = get_non_linearity(layer_type=nl)
    # upsample = 'bilinear'
    if use_gpu:
        assert torch.cuda.is_available()

    if nz == 0:
        where_add = 'input'

    if which_model_net_g == 'unet_128' and where_add == 'input':
        net_g = GUNetAddInput(input_nc, output_nc, nz, 7, ngf, norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout, upsample=upsample)
    elif which_model_net_g == 'unet_256' and where_add == 'input':
        net_g = GUNetAddInput(input_nc, output_nc, nz, 8, ngf, norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout, upsample=upsample)
    elif which_model_net_g == 'unet_128' and where_add == 'all':
        net_g = GUNetAddAll(input_nc, output_nc, nz, 7, ngf, norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout, upsample=upsample)
    elif which_model_net_g == 'unet_256' and where_add == 'all':
        net_g = GUNetAddAll(input_nc, output_nc, nz, 8, ngf, norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout, upsample=upsample)
    else:
        raise NotImplementedError(
            'Generator model name [%s] is not recognized' % which_model_net_g)

    if len(gpu_ids) > 0:
        net_g.cuda(gpu_ids[0])
    init_weights(net_g, init_type=init_type)
    return net_g


def define_d(input_nc, ndf, which_model_net_d,
             norm='batch', use_sigmoid=False, init_type='xavier', num_d_s=1, gpu_ids=list()):
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(layer_type=norm)
    nl = 'lrelu'  # use leaky relu for D
    nl_layer = get_non_linearity(layer_type=nl)
    if use_gpu:
        assert (torch.cuda.is_available())
    if which_model_net_d == 'basic_128':
        net_d = DNLayers(input_nc, ndf, n_layers=2, norm_layer=norm_layer, nl_layer=nl_layer, use_sigmoid=use_sigmoid)
    elif which_model_net_d == 'basic_256':
        net_d = DNLayers(input_nc, ndf, n_layers=3, norm_layer=norm_layer, nl_layer=nl_layer, use_sigmoid=use_sigmoid)
    elif which_model_net_d == 'basic_128_multi':
        net_d = DNLayersMulti(input_nc=input_nc, ndf=ndf, n_layers=2, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids, num_d=num_d_s)
    elif which_model_net_d == 'basic_256_multi':
        net_d = DNLayersMulti(input_nc=input_nc, ndf=ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids, num_d=num_d_s)
    else:
        raise NotImplementedError(
            'Discriminator model name [%s] is not recognized' % which_model_net_d)
    if use_gpu:
        net_d.cuda(gpu_ids[0])
    init_weights(net_d, init_type=init_type)
    return net_d


def define_e(input_nc, output_nc, ndf, which_model_net_e,
             norm='batch', init_type='xavier', gpu_ids=list(), vae_like=False):
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(layer_type=norm)
    nl = 'lrelu'  # use leaky relu for E
    nl_layer = get_non_linearity(layer_type=nl)
    if use_gpu:
        assert (torch.cuda.is_available())
    if which_model_net_e == 'resnet_128':
        net_e = EResnet(input_nc, output_nc, ndf, n_blocks=4, norm_layer=norm_layer, nl_layer=nl_layer, vae_like=vae_like)
    elif which_model_net_e == 'resnet_256':
        net_e = EResnet(input_nc, output_nc, ndf, n_blocks=5, norm_layer=norm_layer, nl_layer=nl_layer, vae_like=vae_like)
    elif which_model_net_e == 'conv_128':
        net_e = ENLayers(input_nc, output_nc, ndf, n_layers=4, norm_layer=norm_layer, nl_layer=nl_layer, vae_like=vae_like)
    elif which_model_net_e == 'conv_256':
        net_e = ENLayers(input_nc, output_nc, ndf, n_layers=5, norm_layer=norm_layer, nl_layer=nl_layer, vae_like=vae_like)
    else:
        raise NotImplementedError(
            'Encoder model name [%s] is not recognized' % which_model_net_e)
    if use_gpu:
        net_e.cuda(gpu_ids[0])
    init_weights(net_e, init_type=init_type)
    return net_e


class ListModule(object):
    # should work with all kind of module
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(
                self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))


class DNLayersMulti(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3,
                 norm_layer=nn.BatchNorm2d, use_sigmoid=False, num_d=1):
        super(DNLayersMulti, self).__init__()
        # st()
        self.num_D = num_d
        if num_d == 1:
            layers = self.get_layers(
                input_nc, ndf, n_layers, norm_layer, use_sigmoid)
            self.model = nn.Sequential(*layers)
        else:
            self.model = ListModule(self, 'model')
            layers = self.get_layers(
                input_nc, ndf, n_layers, norm_layer, use_sigmoid)
            self.model.append(nn.Sequential(*layers))
            self.down = nn.AvgPool2d(3, stride=2, padding=[
                1, 1], count_include_pad=False)
            for i in range(num_d - 1):
                ndf = int(round(ndf / (2 ** (i + 1))))
                layers = self.get_layers(
                    input_nc, ndf, n_layers, norm_layer, use_sigmoid)
                self.model.append(nn.Sequential(*layers))

    @staticmethod
    def get_layers(input_nc, ndf=64, n_layers=3,
                   norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        kw = 4
        pad = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=pad), nn.LeakyReLU(0.2, True)]

        nf_multi = 1
        for n in range(1, n_layers):
            nf_multi_prev = nf_multi
            nf_multi = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_multi_prev, ndf * nf_multi, kernel_size=kw, stride=2, padding=pad),
                norm_layer(ndf * nf_multi),
                nn.LeakyReLU(0.2, True)
            ]

        nf_multi_prev = nf_multi
        nf_multi = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_multi_prev, ndf * nf_multi, kernel_size=kw, stride=1, padding=pad),
            norm_layer(ndf * nf_multi),
            nn.LeakyReLU(0.2, True)]

        sequence += [nn.Conv2d(ndf * nf_multi, 1, kernel_size=kw, stride=1, padding=pad)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        return sequence

    def parallel_forward(self, model, data):
        if self.gpu_ids and isinstance(data.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(model, data, self.gpu_ids)
        else:
            return model(data)

    def forward(self, data):
        if self.num_D == 1:
            return self.parallel_forward(self.model, data)
        result = []
        down = data
        for i in range(self.num_D):
            result.append(self.parallel_forward(self.model[i], down))
            if i != self.num_D - 1:
                down = self.parallel_forward(self.down, down)
        return result


# Defines the conv discriminator with the specified arguments.
class GNLayers(nn.Module):
    def __init__(self, output_nc=3, nz=100, ngf=64, n_layers=3, norm_layer=None, nl_layer=None):
        super(GNLayers, self).__init__()

        kw, s, pad = 4, 2, 1
        sequence = [nn.ConvTranspose2d(nz, ngf * 4, kernel_size=kw, stride=1, padding=0, bias=True)]
        if norm_layer is not None:
            sequence += [norm_layer(ngf * 4)]

        sequence += [nl_layer()]

        nf_multi = 4
        for n in range(n_layers, 0, -1):
            nf_mult_prev = nf_multi
            nf_multi = min(n, 4)
            sequence += [nn.ConvTranspose2d(ngf * nf_mult_prev, ngf * nf_multi, kernel_size=kw, stride=s, padding=pad, bias=True)]
            if norm_layer is not None:
                sequence += [norm_layer(ngf * nf_multi)]
            sequence += [nl_layer()]

        sequence += [nn.ConvTranspose2d(ngf, output_nc, kernel_size=4, stride=s, padding=pad, bias=True)]
        sequence += [nn.Tanh()]

        self.model = nn.Sequential(*sequence)

    def forward(self, data):
        if len(self.gpu_ids) and isinstance(data.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, data, self.gpu_ids)
        else:
            return self.model(data)


class DNLayers(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=None, nl_layer=None, use_sigmoid=False):
        super(DNLayers, self).__init__()

        kw, pad, use_bias = 4, 1, True
        # st()
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=pad, bias=use_bias), nl_layer()]

        nf_multi = 1
        for n in range(1, n_layers):
            nf_multi_prev = nf_multi
            nf_multi = min(2 ** n, 8)
            sequence += [nn.Conv2d(ndf * nf_multi_prev, ndf * nf_multi, kernel_size=kw, stride=2, padding=pad, bias=use_bias)]
            if norm_layer is not None:
                sequence += [norm_layer(ndf * nf_multi)]
            sequence += [nl_layer()]

        nf_multi_prev = nf_multi
        nf_multi = min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_multi_prev, ndf * nf_multi, kernel_size=kw, stride=1, padding=pad, bias=use_bias)]
        if norm_layer is not None:
            sequence += [norm_layer(ndf * nf_multi)]
        sequence += [nl_layer()]
        sequence += [nn.Conv2d(ndf * nf_multi, 1, kernel_size=4, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, data):
        output = self.model(data)
        return output


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################
class RecLoss(object):
    def __init__(self, use_L2=True):
        super(RecLoss, self).__init__()
        self.use_L2 = use_L2

    def __call__(self, input, target, batch_mean=True):
        if self.use_L2:
            diff = (input - target) ** 2
        else:
            diff = torch.abs(input - target)
        if batch_mean:
            return torch.mean(diff)
        else:
            return torch.mean(torch.mean(torch.mean(diff, dim=1), dim=2), dim=3)


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(object):
    def __init__(self, mse_loss=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if mse_loss:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, data, target_is_real):
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != data.numel()))
            if create_label:
                real_tensor = self.Tensor(data.size()).fill_(self.real_label)
                self.real_label_var = Variable(
                    real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != data.numel()))
            if create_label:
                fake_tensor = self.Tensor(data.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(
                    fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, inputs, target_is_real):
        # if input is a list
        loss = 0.0
        all_losses = []
        for data in inputs:
            target_tensor = self.get_target_tensor(data, target_is_real)
            loss_input = self.loss(data, target_tensor)
            loss = loss + loss_input
            all_losses.append(loss_input)
        return loss, all_losses


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class GUNetAddInput(nn.Module):
    def __init__(self, input_nc, output_nc, nz, num_downs, ngf=64,
                 norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic'):
        super(GUNetAddInput, self).__init__()
        self.nz = nz
        # currently support only input_nc == output_nc
        # assert(input_nc == output_nc)
        max_nchn = 8
        # construct unet structure
        u_net_block = UNetBlock(ngf * max_nchn, ngf * max_nchn, ngf * max_nchn,
                                innermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        for i in range(num_downs - 5):
            u_net_block = UNetBlock(ngf * max_nchn, ngf * max_nchn, ngf * max_nchn, u_net_block,
                                    norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout, upsample=upsample)
        u_net_block = UNetBlock(ngf * 4, ngf * 4, ngf * max_nchn, u_net_block,
                                norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        u_net_block = UNetBlock(ngf * 2, ngf * 2, ngf * 4, u_net_block,
                                norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        u_net_block = UNetBlock(ngf, ngf, ngf * 2, u_net_block,
                                norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        u_net_block = UNetBlock(input_nc + nz, output_nc, ngf, u_net_block,
                                outermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)

        self.model = u_net_block

    def forward(self, x, z=None):
        if self.nz > 0:
            z_img = z.view(z.size(0), z.size(1), 1, 1).expand(
                z.size(0), z.size(1), x.size(2), x.size(3))
            x_with_z = torch.cat([x, z_img], 1)
        else:
            x_with_z = x  # no z

        return self.model(x_with_z)


def upsample_layer(inplanes, outplanes, upsample='basic'):
    # padding_type = 'zero'
    if upsample == 'basic':
        up_conv = [nn.ConvTranspose2d(
            inplanes, outplanes, kernel_size=4, stride=2, padding=1)]
    elif upsample == 'bilinear':
        up_conv = [nn.Upsample(scale_factor=2, mode='bilinear'),
                   nn.ReflectionPad2d(1),
                   nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0)]
    else:
        raise NotImplementedError(
            'upsample layer [%s] not implemented' % upsample)
    return up_conv


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UNetBlock(nn.Module):
    def __init__(self, input_nc, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic', padding_type='zero'):
        super(UNetBlock, self).__init__()
        self.outermost = outermost
        p = 0
        down_conv = []
        if padding_type == 'reflect':
            down_conv += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            down_conv += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)
        down_conv += [nn.Conv2d(input_nc, inner_nc,
                               kernel_size=4, stride=2, padding=p)]
        # downsample is different from upsample
        down_relu = nn.LeakyReLU(0.2, True)
        down_norm = norm_layer(inner_nc) if norm_layer is not None else None
        up_relu = nl_layer()
        up_norm = norm_layer(outer_nc) if norm_layer is not None else None

        if outermost:
            up_conv = upsample_layer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = down_conv
            up = [up_relu] + up_conv + [nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            up_conv = upsample_layer(
                inner_nc, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [down_relu] + down_conv
            up = [up_relu] + up_conv
            if up_norm is not None:
                up += [up_norm]
            model = down + up
        else:
            up_conv = upsample_layer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [down_relu] + down_conv
            if down_norm is not None:
                down += [down_norm]
            up = [up_relu] + up_conv
            if up_norm is not None:
                up += [up_norm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)


# two usage cases, depend on kw and padw
def upsample_conv(inplanes, outplanes, kw, padw):
    sequence = []
    sequence += [nn.Upsample(scale_factor=2, mode='nearest')]
    sequence += [nn.Conv2d(inplanes, outplanes, kernel_size=kw,
                           stride=1, padding=padw, bias=True)]
    return nn.Sequential(*sequence)


def mean_pool_conv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    sequence += [nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True)]
    return nn.Sequential(*sequence)


def conv_mean_pool(inplanes, outplanes):
    sequence = []
    sequence += [conv3x3(inplanes, outplanes)]
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)


class BasicBlockUp(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlockUp, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [upsample_conv(inplanes, outplanes, kw=3, padw=1)]
        if norm_layer is not None:
            layers += [norm_layer(outplanes)]
        layers += [conv3x3(outplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = upsample_conv(inplanes, outplanes, kw=1, padw=0)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out


class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlock, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [conv3x3(inplanes, inplanes)]
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [conv_mean_pool(inplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = mean_pool_conv(inplanes, outplanes)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out


class EResnet(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, ndf=64, n_blocks=4,
                 norm_layer=None, nl_layer=None, vae_like=False):
        super(EResnet, self).__init__()
        self.vae_like = vae_like
        max_ndf = 4
        conv_layers = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=True)]
        for n in range(1, n_blocks):
            input_ndf = ndf * min(max_ndf, n)
            output_ndf = ndf * min(max_ndf, n + 1)
            conv_layers += [BasicBlock(input_ndf, output_ndf, norm_layer, nl_layer)]

        conv_layers += [nl_layer(), nn.AvgPool2d(8)]
        if vae_like:
            self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
            self.fcVar = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1)
        output = self.fc(conv_flat)
        if self.vae_like:
            output_var = self.fcVar(conv_flat)
            return output, output_var
        else:
            return output


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class GUNetAddAll(nn.Module):
    def __init__(self, input_nc, output_nc, nz, num_downs, ngf=64, norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic'):
        super(GUNetAddAll, self).__init__()
        self.nz = nz
        # construct unet structure
        u_net_block = UNetBlockWithZ(ngf * 8, ngf * 8, ngf * 8, nz, None, innermost=True,
                                     norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        u_net_block = UNetBlockWithZ(ngf * 8, ngf * 8, ngf * 8, nz, u_net_block,
                                     norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout,
                                     upsample=upsample)
        for i in range(num_downs - 6):
            u_net_block = UNetBlockWithZ(ngf * 8, ngf * 8, ngf * 8, nz, u_net_block,
                                         norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout,
                                         upsample=upsample)
        u_net_block = UNetBlockWithZ(ngf * 4, ngf * 4, ngf * 8, nz, u_net_block,
                                     norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        u_net_block = UNetBlockWithZ(ngf * 2, ngf * 2, ngf * 4, nz, u_net_block,
                                     norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        u_net_block = UNetBlockWithZ(
            ngf, ngf, ngf * 2, nz, u_net_block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        u_net_block = UNetBlockWithZ(input_nc, output_nc, ngf, nz, u_net_block,
                                     outermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        self.model = u_net_block

    def forward(self, x, z):
        return self.model(x, z)


class UNetBlockWithZ(nn.Module):
    def __init__(self, input_nc, outer_nc, inner_nc, nz=0,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic', padding_type='zero'):
        super(UNetBlockWithZ, self).__init__()
        p = 0
        down_conv = []
        if padding_type == 'reflect':
            down_conv += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            down_conv += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)

        self.outermost = outermost
        self.innermost = innermost
        self.nz = nz
        input_nc = input_nc + nz
        down_conv += [nn.Conv2d(input_nc, inner_nc,
                                kernel_size=4, stride=2, padding=p)]
        # downsample is different from upsample
        down_relu = nn.LeakyReLU(0.2, True)
        up_relu = nl_layer()

        if outermost:
            up_conv = upsample_layer(inplanes=inner_nc * 2, outplanes=outer_nc, upsample=upsample)
            down = down_conv
            up = [up_relu] + up_conv + [nn.Tanh()]
        elif innermost:
            up_conv = upsample_layer(inner_nc, outer_nc, upsample=upsample)
            down = [down_relu] + down_conv
            up = [up_relu] + up_conv
            if norm_layer is not None:
                up += [norm_layer(outer_nc)]
        else:
            up_conv = upsample_layer(inner_nc * 2, outer_nc, upsample=upsample)
            down = [down_relu] + down_conv
            if norm_layer is not None:
                down += [norm_layer(inner_nc)]
            up = [up_relu] + up_conv

            if norm_layer is not None:
                up += [norm_layer(outer_nc)]

            if use_dropout:
                up += [nn.Dropout(0.5)]

        self.down = nn.Sequential(*down)
        self.submodule = submodule
        self.up = nn.Sequential(*up)

    def forward(self, x, z):
        if self.nz > 0:
            z_img = z.view(z.size(0), z.size(1), 1, 1).expand(
                z.size(0), z.size(1), x.size(2), x.size(3))
            x_and_z = torch.cat([x, z_img], 1)
        else:
            x_and_z = x

        if self.outermost:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return self.up(x2)
        elif self.innermost:
            x1 = self.up(self.down(x_and_z))
            return torch.cat([x1, x], 1)
        else:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return torch.cat([self.up(x2), x], 1)


class ENLayers(nn.Module):
    def __init__(self, input_nc, output_nc=1, ndf=64, n_layers=3, norm_layer=None, nl_layer=None, vae_like=False):
        super(ENLayers, self).__init__()
        self.vaeLike = vae_like

        kw, pad = 4, 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=pad), nl_layer()]

        nf_multi = 1
        for n in range(1, n_layers):
            nf_multi_prev = nf_multi
            nf_multi = min(2 ** n, 4)
            sequence += [nn.Conv2d(ndf * nf_multi_prev, ndf * nf_multi, kernel_size=kw, stride=2, padding=pad)]

            if norm_layer is not None:
                sequence += [norm_layer(ndf * nf_multi)]

            sequence += [nl_layer()]

        sequence += [nn.AvgPool2d(kernel_size=8)]
        self.conv = nn.Sequential(*sequence)
        self.fc = nn.Sequential(*[nn.Linear(ndf * nf_multi, output_nc)])

        if vae_like:
            self.fcVar = nn.Sequential(*[nn.Linear(ndf * nf_multi, output_nc)])

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1)
        output = self.fc(conv_flat)
        if self.vaeLike:
            output_var = self.fcVar(conv_flat)
            return output, output_var
        return output
