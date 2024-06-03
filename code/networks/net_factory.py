from networks.unet import UNet, UNets_NoA_2d


def net_factory(args,net_type="unet", in_chns=1, class_num=4, mode = "train"):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()    
    elif net_type == 'unets_noa' and mode == 'train':
        net = UNets_NoA_2d(in_chns=in_chns, class_num=class_num).cuda()    
    elif net_type == 'unets_noa' and mode == 'test':
        net = UNets_NoA_2d(in_chns=in_chns, class_num=class_num).cuda()
    return net
