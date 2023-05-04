import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from time import time
import argparse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import tqdm
from dataloader.dataset import GevLoader,FastecLoader,DrerealLoader,GevrealLoader
import torch
from Networks.SelfUnroll import SelfUnroll_plus
import cv2
import numpy as np
from visualization import event_show, flow_to_image
from tensorboardX import SummaryWriter
import warnings
from pathlib import Path
import lpips
from my_util import get_logger, mkdir
warnings.filterwarnings('ignore')
if __name__ == "__main__":
    logger = get_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--Dataset", type=str, default='Fastec',
                        help="dataset name")
    parser.add_argument('--RSpath',
                        type=str,
                        default='/home/wyg/Documents/wyg/fastec_rs_train', help='the path of dataset')
    parser.add_argument('--batch_sz',
                        type=int,
                        default=1,
                        help='batch size used for testing')
    parser.add_argument('--num_bins', type=int, default=16, help='the number of events divided')
    parser.add_argument('--num_workers',
                        type=int,
                        default=0,
                        help='number of workers')
    parser.add_argument("--test_unroll_path",
                        type=str,
                        default='/home/wyg/Documents/wyg/exp/checkpoints/fastplus.pth',
                        help="path to load model")
    parser.add_argument("--model_path", type=str, default="../experiment/", help="model saving path")
    parser.add_argument("--result_path", type=str, default="../result/fast/",
                        help="path to save result")
    parser.add_argument("--target", type=float, default=0.50,
                        help="timestamp(0/1 represents the exposure time of first/last scanline) of target GS frame")

    opts = parser.parse_args()
    mkdir(opts.result_path)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if opts.Dataset =='Fastec':
        dataset_provider = FastecLoader(opts.RSpath, opts.target, opts.num_bins)
    elif opts.Dataset =='Gev':
        dataset_provider = GevLoader(opts.RSpath, opts.target, opts.num_bins)
    elif opts.Dataset =='Gevreal':
        dataset_provider = GevrealLoader(opts.RSpath, opts.target, opts.num_bins)
    elif opts.Dataset =='Drereal':
        dataset_provider = DrerealLoader(opts.RSpath, opts.target, opts.num_bins)
    else:
        raise ValueError(f'Unknown dataset: {opts.Dataset}')

    test_data = dataset_provider.get_test_dataset()
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=opts.num_workers,
                                               pin_memory=True,
                                               drop_last=True)

    last_time = time()

    unroll_net = SelfUnroll_plus(D=10, inchannels=32)

    unroll_net = unroll_net.cuda()

    logger.info(
        'start testing ! => {} testing samples found in the testing set'.
        format(len(test_data)))
    
    unroll_net.load_state_dict(torch.load(opts.test_unroll_path),
                                strict=False)
    logger.info("Load pretrained network from " + opts.test_unroll_path)

    last_time = time()
    start_test_time = time()
    mkdir(opts.model_path + '/log_test')
    model_path = Path(opts.model_path + '/log_test')
    experiment = 0
    for child in model_path.iterdir():
        if str(child).split('/')[-1].startswith('testexperiment'):
            temp_ex = int(str(child).split('/')[-1][-3:])
            if temp_ex > experiment:
                experiment = temp_ex
    

    exp_path = opts.model_path + '/log_test/' + f'testexperiment{experiment + 1:03d}' + '/'
    test_writer = SummaryWriter(opts.model_path + '/log_test/' + f'testexperiment{experiment + 1:03d}')
    total_iter = 0
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    f = open(exp_path + 'testlogger.txt' , 'w')
    f.write(opts.test_unroll_path+'\n')
    f.write(opts.result_path+'\n')
    current_iter = 0
    
    tar = tqdm.tqdm(total=len(test_loader), position=0, leave=True)
    unroll_net.eval()
    psnr_list = []
    ssim_list = []
    lpips_list = []
    lpips_fn = lpips.LPIPS().to(device)

    flow_psnr_list = []
    flow_ssim_list = []
    flow_lpips_list = []
    sys_psnr_list = []
    sys_ssim_list = []
    sys_lpips_list = []
    eic_psnr_list = []
    eic_ssim_list = []
    eic_lpips_list = []
    with torch.no_grad():
        for _, sample in enumerate(test_loader):
            ##load data from one blur to target need 3 part events :
            # target to blur start，target to blur end ,target exposure
            Orig1 = sample['Orig1'].float().cuda(non_blocking=True)
            Orig2 = sample['Orig2'].float().cuda(non_blocking=True)
            TtoO1_sf = sample['TtoO1_sf'].float().cuda(non_blocking=True)
            TtoO1_ef = sample['TtoO1_ef'].float().cuda(non_blocking=True)
            TtoO2_sf = sample['TtoO2_sf'].float().cuda(non_blocking=True)
            TtoO2_ef = sample['TtoO2_ef'].float().cuda(non_blocking=True)
            Texpf = sample['Texpf'].float().cuda(non_blocking=True)
            B,C,H,W = Orig1.shape
            
            if 'GT' in sample.keys():
                GT = sample['GT'].float().cuda(non_blocking=True)
            else:
                GT = None
            
            
            event1_T2O = torch.cat((TtoO1_sf, TtoO2_sf), 0)
            event2_T2O = torch.cat((TtoO1_ef, TtoO2_ef), 0)
            event3_T2O = torch.cat((Texpf, Texpf), 0)
            Orig = torch.cat((Orig1, Orig2), 0)

            res_img_T_sys, E1, res_img_T_flow, flow1, res_img_T_fusion,res_img_T = unroll_net(event1_T2O, event2_T2O, event3_T2O, Orig)
            res_img_T_fusion_left,res_img_T_fusion_right = torch.chunk(res_img_T_fusion,2,0)
            current_iter += 1
            total_iter += 1
            # tar.update(1)

            res_img_T_sys = torch.clamp(res_img_T_sys, 0, 1)
            res_img_T_fusion = torch.clamp(res_img_T_fusion, 0, 1)
            res_img_T = torch.clamp(res_img_T, 0, 1)
            # res_img_T_flow = torch.clamp(Orig1, 0, 1)
            res_img_T_flow = torch.clamp((res_img_T_fusion[:opts.batch_sz] + res_img_T_fusion[opts.batch_sz:]) / 2, 0,
                                         1)
            Orig1 = torch.clamp(Orig1, 0, 1)
            gt = GT

            for j in range(opts.batch_sz):
                TtoO1_sf_ = event_show(TtoO1_sf[j].permute(1, 2, 0).cpu().detach().numpy())
                TtoO1_ef_ = event_show(TtoO1_ef[j].permute(1, 2, 0).cpu().detach().numpy())
                Texpf_ = event_show(Texpf[j].permute(1, 2, 0).cpu().detach().numpy())
                flow_img1 = flow_to_image(flow1[j, ...].permute(1, 2, 0).cpu().detach().numpy())
                flow_img2 = flow_to_image(flow1[opts.batch_sz + j, ...].permute(1, 2, 0).cpu().detach().numpy())
                evshow = np.concatenate((TtoO1_sf_, Texpf_, flow_img1, flow_img2), 1)

                file_index = sample['file_index'][j].split('/')[-3] + '_' + sample['file_index'][j].split('/')[-1][:-4]
                # show = torch.cat(
                #     (Orig1[j, ...], Orig2[j, ...], res_img_T_flow[j, ...], res_img_T_sys[j, ...], res_img_T_fusion[j],
                #      res_img_T_fusion[opts.batch_sz + j], res_img_T[j, ...],
                #      GT[j, ...]), 1)
                # show = torch.cat(torch.chunk(show, 4, 1), 2)
                # show = show.mul(255).cpu().detach().numpy().astype(np.uint8)
                # show = np.transpose(show, (1, 2, 0))
                # name = opts.result_path + file_index + '-' + f'{int(1000 * opts.target):03d}.png'
                # cv2.imwrite(name, np.concatenate((show[..., [2, 1, 0]], evshow), 0))
                res_img_ = res_img_T[j, ...].cpu().detach().numpy()
                res_img_ = np.transpose(res_img_, (1, 2, 0))
                res_img_T_sys_ = res_img_T_sys[j, ...].cpu().detach().numpy()
                res_img_T_sys_ = np.transpose(res_img_T_sys_, (1, 2, 0))
                res_img_T_flow_ = res_img_T_flow[j, ...].cpu().detach().numpy()
                res_img_T_flow_ = np.transpose(res_img_T_flow_, (1, 2, 0))
                res_img_T_fusion_ = res_img_T_fusion[j, ...].cpu().detach().numpy()
                res_img_T_fusion_ = np.transpose(res_img_T_fusion_, (1, 2, 0))



                reslpips = 0
                syslpips = 0
                flowlpips = 0
                eiclpips = 0
                resssim = 0
                respsnr = 0
                psnr_flow = 0
                ssim_flow = 0
                psnr_sys = 0
                ssim_sys = 0
                psnr_eic = 0
                ssim_eic = 0
                if gt is not None:
                    GT_ = GT[j, ...].cpu().detach().numpy()
                    GT_ = np.transpose(GT_, (1, 2, 0))
                    reslpips = lpips_fn(res_img_T[j, ...], gt[j, ...], normalize=False).item()
                    lpips_list.append(reslpips)
                    flowlpips = lpips_fn(res_img_T_flow[j, ...], gt[j, ...], normalize=False).item()
                    flow_lpips_list.append(flowlpips)
                    syslpips = lpips_fn(res_img_T_sys[j, ...], gt[j, ...], normalize=False).item()
                    sys_lpips_list.append(syslpips)
                    eiclpips = lpips_fn(res_img_T_fusion[j, ...], gt[j, ...], normalize=False).item()
                    eic_lpips_list.append(eiclpips)


                    resssim = ssim(res_img_[..., [2, 1, 0]], GT_[..., [2, 1, 0]], data_range=1, multichannel=True)
                    respsnr = psnr(res_img_[..., [2, 1, 0]], GT_[..., [2, 1, 0]], data_range=1)
                    psnr_flow = psnr(res_img_T_flow_[..., [2, 1, 0]], GT_[..., [2, 1, 0]], data_range=1)
                    ssim_flow = ssim(res_img_T_flow_[..., [2, 1, 0]], GT_[..., [2, 1, 0]], data_range=1, multichannel=True)
                    psnr_sys = psnr(res_img_T_sys_[..., [2, 1, 0]], GT_[..., [2, 1, 0]], data_range=1)
                    ssim_sys = ssim(res_img_T_sys_[..., [2, 1, 0]], GT_[..., [2, 1, 0]], data_range=1, multichannel=True)
                    psnr_eic = psnr(res_img_T_fusion_[..., [2, 1, 0]], GT_[..., [2, 1, 0]], data_range=1)
                    ssim_eic = ssim(res_img_T_fusion_[..., [2, 1, 0]], GT_[..., [2, 1, 0]], data_range=1, multichannel=True)
                    eic_psnr_list.append(psnr_eic)
                    eic_ssim_list.append(ssim_eic)
                    ssim_list.append(resssim)
                    psnr_list.append(respsnr)
                    flow_psnr_list.append(psnr_flow)
                    flow_ssim_list.append(ssim_flow)
                    sys_psnr_list.append(psnr_sys)
                    sys_ssim_list.append(ssim_sys)
                res_img_ = (255. * res_img_).astype(np.uint8)

                name = opts.result_path + file_index + '-' + f'res_{int(1000 * opts.target):03d}.png'
                if 'real' in opts.Dataset:
                    cv2.imwrite(name, res_img_)
                else:
                    cv2.imwrite(name, res_img_[..., [2, 1, 0]])
                # name = opts.result_path + file_index + '-' + f'eic_{int(1000 * opts.target):03d}.png'
                # cv2.imwrite(name, res_img_T_fusion_[..., [2, 1, 0]])
                # GT_ = (255. * GT_).astype(np.uint8)
                # name = opts.result_path + file_index + '-' + f'gt_{int(1000 * opts.target):03d}.png'
                # cv2.imwrite(name, GT_[..., [2, 1, 0]])

            f.write(
                '%s | ssim:%.4f psnr%.4f lpips:%.4f| ssim:%.4f psnr%.4f lpips:%.4f|flow ssim:%.4f psnr%.4f lpips:%.4f|sys ssim:%.4f psnr%.4f lpips:%.4f|\n' % (
                sample['file_index'], resssim, respsnr, reslpips,ssim_eic,psnr_eic,eiclpips, ssim_flow, psnr_flow, flowlpips, ssim_sys, psnr_sys,
                syslpips))
            tar.set_postfix({'psnr':f'{np.array(psnr_list[-opts.batch_sz:]).mean():.4f}','ssim':f'{np.array(ssim_list[-opts.batch_sz:]).mean():.4f}','lpips':f'{np.array(lpips_list[-opts.batch_sz:]).mean():.4f}'})
            tar.update(1)
        # ## save model when loss decreases
        logger.info('---------------- Summary of validate ----------------')
        logger.info(
            'Total: ssim:%.5f psnr%.5f lpips:%.5f| eic ssim:%.5f psnr%.5f lpips:%.5f|flow ssim:%.5f psnr%.5f lpips:%.5f|sys ssim:%.5f psnr%.5f lpips:%.5f|' % (
                np.mean(ssim_list), np.mean(psnr_list), np.mean(lpips_list), np.mean(eic_ssim_list),
                np.mean(eic_psnr_list), np.mean(eic_lpips_list), np.mean(flow_ssim_list),
                np.mean(flow_psnr_list), np.mean(flow_lpips_list), np.mean(sys_ssim_list), np.mean(sys_psnr_list),
                np.mean(sys_lpips_list)))
        f.write('---------------- Summary of test ----------------')
        f.write(
            'Total: ssim:%.5f psnr%.5f lpips:%.5f| eic ssim:%.5f psnr%.5f lpips:%.5f|flow ssim:%.5f psnr%.5f lpips:%.5f|sys ssim:%.5f psnr%.5f lpips:%.5f|' % (
                np.mean(ssim_list), np.mean(psnr_list), np.mean(lpips_list), np.mean(eic_ssim_list),
                np.mean(eic_psnr_list), np.mean(eic_lpips_list), np.mean(flow_ssim_list),
                np.mean(flow_psnr_list), np.mean(flow_lpips_list), np.mean(sys_ssim_list), np.mean(sys_psnr_list),
                np.mean(sys_lpips_list)))
        f.flush()
        f.close()
        torch.cuda.empty_cache()