# KernelDNA
**KernelDNA** is a plug-and-play convolution module that enhances model capacity through adaptive kernel specialization while maintaining hardware-friendly efficiency.


## Key Innovations  
✅ **Parameter-Efficient Design**  
- Replaces dense convolutional layers with derived "child" kernels from a shared "parent" kernel  
- Avoids linear parameter growth typical in dynamic convolutions  

✅ **Hardware-Optimized Inference**  
- Decouples adaptation into:  
  - Input-dependent dynamic routing  
  - Pre-trained static modulation  
- Preserves standard convolution's native computational efficiency  

✅ **Enhanced Representation**  
- Achieves input-adaptive kernel adjustments without structural changes  
- Outperforms existing dynamic convolutions in accuracy-efficiency trade-off  

## Performance Highlights  
- State-of-the-art results on image classification & dense prediction tasks  
- Maintains >90% of baseline throughput while improving accuracy  
- Compatible with pre-trained CNNs via adapter-based fine-tuning  

## Installation  
```bash
git clone https://github.com/haiduo/KernelDNA.git
cd KernelDNA
pip install -r requirements

```

## Training on Classification Task
### KernelDNA for ResNet18
```
python main.py --data /path/to/imagenet-1k --ckpt_dir /path/to/save_ckpt --log_dir /path/to/save_log --arch resnet18 --customize --epochs 90 --warmup-epochs 0 --lr 0.1 --wd 1e-4 --batch-size 256
```
### KernelDNA for ResNet50
```
python main.py --data /path/to/imagenet-1k --ckpt_dir /path/to/save_ckpt --log_dir /path/to/save_log --arch resnet50 --customize --epochs 300 --warmup-epochs 20 --lr 0.1 --wd 1e-4 --batch-size 256
```

### KernelDNA for MobileNetV2-1x
```
python main.py --data /path/to/imagenet-1k --ckpt_dir /path/to/save_ckpt --log_dir /path/to/save_log --arch mobilenet_v2 --customize --epochs 150 --warmup-epochs 0 --lr 0.05 --wd 4e-5 --batch-size 256
```

### KernelDNA for MobileNetV2-0.5x
```
python main.py --data /path/to/imagenet-1k --ckpt_dir /path/to/save_ckpt --log_dir /path/to/save_log --arch mobilenet_v2_1d2 --customize --epochs 150 --warmup-epochs 0 --lr 0.05 --wd 4e-5 --batch-size 256
```

### KernelDNA for ConvNeXt-Tiny
```
python main.py --data /path/to/imagenet-1k --ckpt_dir /path/to/save_ckpt --log_dir /path/to/save_log --arch convNeXt-tiny --customize --epochs 150 --warmup-epochs 0 --lr 0.05 --wd 4e-5 --batch-size 256
```

## Training on Dense Prediction Task
### KernelDNA for ResNet18
```
cd detection_and_segmentation
bash mmdetection/tools/dist_train.sh
```


## Evaluation

### KernelDNA ResNet18
```
python main.py --data /path/to/imagenet-1k --evaluate --resume /path/to/ckpt --log_dir /path/to/save_log --arch resnet18 --customize 
```

### KernelDNA ResNet50
```
python main.py --data /path/to/imagenet-1k --evaluate --resume /path/to/ckpt --log_dir /path/to/save_log --arch resnet50 --customize 
```

### KernelDNA MobileNetV2-1x
```
python main.py --data /path/to/imagenet-1k --evaluate --resume /path/to/ckpt --log_dir /path/to/save_log --arch mobilenet_v2 --customize 
```

### KernelDNA MobileNetV2-0.5x
```
python main.py --data /path/to/imagenet-1k --evaluate --resume /path/to/ckpt --log_dir /path/to/save_log --arch mobilenet+_v2_1d2 --customize 
```

### KernelDNA ConvNeXt-Tiny
```
python main.py --data /path/to/imagenet-1k --evaluate --resume /path/to/ckpt --log_dir /path/to/save_log --arch convNeXt-tiny --customize 
```

## Throughput & Latency on GPU

### KernelDNA ResNet18
```
python metrics.py --params --latency --gpu --iter 1000 --batch-size 128 --arch resnet18 --customize 
```

###  KernelDNA ResNet50
```
python metrics.py --params --latency --gpu --iter 1000 --batch-size 128 --arch resnet50 --customize 
```

###  KernelDNA MobileNetV2-1x
```
python metrics.py --params --latency --gpu --iter 1000 --batch-size 128 --arch mobilenet_v2 --customize 
```

###  KernelDNA MobileNetV2-0.5x
```
python metrics.py --params --latency --gpu --iter 1000 --batch-size 128 --arch mobilenet+_v2_1d2 --customize 
```


## Results

For technical details and full experimental results, please check [the paper of KernelDNA](https://arxiv.org/abs/2503.23379).

## Reference

```
@misc{huang2025kerneldnadynamickernelsharing,
      title={KernelDNA: Dynamic Kernel Sharing via Decoupled Naive Adapters}, 
      author={Haiduo Huang and Yadong Zhang and Pengju Ren},
      year={2025},
      eprint={2503.23379},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.23379}, 
}
```