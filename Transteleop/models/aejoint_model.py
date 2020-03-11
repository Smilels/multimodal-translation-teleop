import torch
from .base_model import BaseModel
from . import networks


class AejointModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        if is_train:
            parser.add_argument('--lambda_Joint', type=float, default=10.0, help='weight for joint loss')

        return parser

    def __init__(self, opt):
        """Initialize the AejointModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L1', 'G_L2']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G']
        else:  # during test time, only load G
            self.model_names = ['G']

        if opt.stn:
            self.stn_A = None
        self.bottleneck = None
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.load_size, opt.fc_embedding,
                                      opt.g_embed, opt.norm, not opt.no_dropout, opt.init_type,
                                      opt.init_gain, self.gpu_ids, stn=opt.stn,
                                      demo_mode=opt.demo)

        if self.isTrain:
            # define loss functions
            # if opt.l2loss:
            #     self.criterionL1 = torch.nn.MSELoss()
            # else:
            #     self.criterionL1 = torch.nn.L1Loss()
            self.criterionL1 = networks.L1Loss(opt.keypoints_factor).to(self.device)
            self.criterionL2 = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        if self.opt.demo:
            self.real_A = input.to(self.device)
        else:
            self.real_A = input['A' if AtoB else 'B'].to(self.device)
            self.real_B = input['B' if AtoB else 'A'].to(self.device)
            self.image_paths = input['A_paths' if AtoB else 'B_paths']
            self.label = input['joints'].to(self.device)
            self.uv = input['uv'].to(self.device)
        self.joint_upper_range = torch.tensor([0.349, 1.571, 1.571, 1.571, 0.785, 0.349, 1.571, 1.571,
                                               1.571, 0.349, 1.571, 1.571, 1.571, 0.349, 1.571, 1.571,
                                               1.571, 1.047, 1.222, 0.209, 0.524, 1.571]).to(self.device)
        self.joint_lower_range = torch.tensor([-0.349, 0, 0, 0, 0, -0.349, 0, 0, 0, -0.349, 0, 0, 0,
                                               -0.349, 0, 0, 0, -1.047, 0, -0.209, -0.524, 0]).to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.opt.demo:
            self.bottleneck = self.netG(self.real_A)
        else:
            if self.opt.stn:
                self.fake_B, self.bottleneck, self.stn_A = self.netG(self.real_A)  # G(A)
            else:
                self.fake_B, self.bottleneck, _ = self.netG(self.real_A)  # G(A)

    def backward_G(self, epoch):
        """Calculate L1 reconstruction loss and L2 joint loss """
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B, self.uv) * self.opt.lambda_L1
        # self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.bottleneck = self.bottleneck * (self.joint_upper_range - self.joint_lower_range) + self.joint_lower_range
        self.loss_G_L2 = self.criterionL2(self.bottleneck, self.label) * self.opt.lambda_Joint
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_L1 + self.loss_G_L2
        self.loss_G.backward()

    def optimize_parameters(self, epoch):
        self.forward()  # compute fake images: G(A)
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G(epoch)  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights
