import torch 
import time 
import argparse 
from fvcore.nn import FlopCountAnalysis, parameter_count
import gc 

from models.KDNA_ResNet18 import get_resnet18 
from models.KDNA_ResNet50 import get_resnet50
from models.KDNA_MobileNetV2 import get_mobilenet_v2, get_mobilenet_v2_1d2 




parser = argparse.ArgumentParser(description='PyTorch ImageNet Metrics')
parser.add_argument('--batch-size', metavar='N', type=int, default=128,
                    help='batch size for measuring latency and throughput')
parser.add_argument('--gpu', action='store_true', help='gpu')
parser.add_argument('--num-threads', type=int, default=None, help='number of threads for cpu')
parser.add_argument('--iter', type=int, default=1000, help='number of iterations')
parser.add_argument('--arch', type=str, default='resnet18', metavar='STR', help='model')
parser.add_argument('--customize', action='store_true', help='whether customize the model')
parser.add_argument('--latency', action='store_true', help='measure latency')
parser.add_argument('--params', action='store_true', help='get params and FLOPs/MACs')

args = parser.parse_args() 


@torch.no_grad()
def measure_latency(images, model, GPU=True, chan_last=False, half=False, num_threads=None, iter=200):
    """
    :param images: b, c, h, w
    :param model: model
    :param GPU: whther use GPU
    :param chan_last: data_format
    :param half: half precision
    :param num_threads: for cpu
    :return:
    """
    print(f"{iter = }")
    if GPU:
        model.cuda()
        model.eval()
        torch.backends.cudnn.benchmark = True

        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        if chan_last:
            images = images.to(memory_format=torch.channels_last)
            model = model.to(memory_format=torch.channels_last)
        if half:
            images = images.half()
            model = model.half()

        for i in range(50):
            model(images)
        torch.cuda.synchronize()
    
        timings = torch.zeros(iter)

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        for i in range(iter):
            starter.record()
            model(images)
            ender.record()
            torch.cuda.synchronize()
            timings[i] = starter.elapsed_time(ender)
            
        throughputs = 1000 * batch_size / timings  # samples per second
        throughput_std, throughput_mean = torch.std_mean(throughputs)
        
        latency_std, latency_mean = torch.std_mean(timings)  # miliseconds per sample (or batch)
        
        print(f"batch_size {batch_size} throughput on GPU {throughput_mean:.1f} \u00B1 {throughput_std:.1f}")
        print(f"batch_size {batch_size} latency on GPU {latency_mean:.3f} \u00B1 {latency_std:.3f} ms")

        return (throughput_mean, throughput_std), (latency_mean, latency_std)
    else:
        model.eval()
        if num_threads is not None:
            torch.set_num_threads(num_threads)

        batch_size = images.shape[0]

        if chan_last:
            images = images.to(memory_format=torch.channels_last)
            model = model.to(memory_format=torch.channels_last)
        if half:
            images = images.half()
            model = model.half()
        for i in range(10):
            model(images)
        
        timings = torch.zeros(iter)
        for i in range(iter):
            tic1 = time.time()
            model(images)
            tic2 = time.time()
            timings[i] = tic2 - tic1
        throughputs = batch_size / timings  # samples per second
        throughput_std, throughput_mean = torch.std_mean(throughputs)
        
        latency_std, latency_mean = torch.std_mean(1000 * timings)  # miliseconds per sample (or batch)
        
        print(f"batch_size {batch_size} throughput on cpu {throughput_mean:.1f} \u00B1 {throughput_std:.1f}")
        print(f"batch_size {batch_size} latency on cpu {latency_mean:.3f} \u00B1 {latency_std:.3f} ms")

        return (throughput_mean, throughput_std), (latency_mean, latency_std)
    
@torch.no_grad()
def get_flops_params(model, input_size=224):
    model = model.to(torch.device("cpu"))
    model.eval()

    tensor = torch.randn(1, 3, input_size, input_size)
    flops = FlopCountAnalysis(model, tensor)
    flops = flops.total() / 1000000.
    print("FVcore FLOPs(M): ", flops)

    params = parameter_count(model)
    params = params[""] / 1000000.
    print("FVcore params(M): ", params)

    return flops, params

 
if __name__ == '__main__':
    images = torch.randn(args.batch_size, 3, 224, 224)

    if args.arch == 'resnet18':
        model = get_resnet18(args.customize)
    elif args.arch == 'resnet50':
        model = get_resnet50(args.customize)
    elif args.arch == 'mobilenet_v2':
        model = get_mobilenet_v2(args.customize)
    elif args.arch == 'mobilenet_v2_1d2':
        model = get_mobilenet_v2_1d2(args.customize)
        
    print(f'{args.arch = }')
    print(f'{args.customize = }')
 
    if args.latency:
        measure_latency(images, model, GPU=args.gpu, chan_last=False, half=False, num_threads=args.num_threads, iter=args.iter)
    if args.params:
        get_flops_params(model)

    # Trigger garbage collection
    gc.collect()

    # Clear PyTorch's CUDA cache
    torch.cuda.empty_cache()