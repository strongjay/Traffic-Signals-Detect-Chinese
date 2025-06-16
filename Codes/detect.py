#Importing necessary libraries.
import argparse
import time
from pathlib import Path

import cv2
import torch    # PyTorch，深度学习框架                        
import torch.backends.cudnn as cudnn    # CUDA加速优化
from numpy import random

# 从YOLOv5项目中导入自定义模块
from models.experimental import attempt_load    # 加载YOLOv5模型
from utils.datasets import LoadStreams, LoadImages  # 加载图像/视频流
from utils.general import ( check_img_size,     # 检查输入图像尺寸
                            non_max_suppression,    # 非极大值抑制（NMS）
                            apply_classifier,   # 应用分类器（可选）
                            scale_coords,   # 调整检测框坐标
                            xyxy2xywh,  # 坐标格式转换
                            strip_optimizer,    # 清理优化器状态
                            set_logging,    # 设置日志
                            increment_path  # 生成递增的保存路径
)
from utils.plots import plot_one_box    # 绘制检测框
from utils.torch_utils import (
                            select_device,  # 选择设备
                            load_classifier,    # 加载分类器
                            time_synchronized   # 计算时间  
)

def detect(save_img=False):
    # 从命令行参数获取输入参数
    source = opt.source  # 输入源（图像/视频/摄像头）
    weights = opt.weights  # 模型权重路径
    view_img = opt.view_img  # 是否实时显示检测结果
    save_txt = opt.save_txt  # 是否保存检测结果（TXT格式）
    imgsz = opt.img_size  # 推理尺寸
    webcam = (
        source.isnumeric() # 摄像头ID（如0）
        or source.endswith('.txt') # 包含多个文件路径的TXT文件
        or source.lower().startswith(('rtsp://', 'rtmp://', 'http://')) # 网络流
    )

    # 设置保存目录（自动递增，避免覆盖）
    save_dir = Path(increment_path(Path("../Results") / opt.name, exist_ok=opt.exist_ok))  # increment run
    # 创建必要的子目录
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True) # 标签目录

    # 初始化
    set_logging()   
    device = select_device(opt.device)  # 选择设备（CPU/GPU）
    half = device.type != 'cpu'  # 是否使用半精度（FP16，仅限CUDA）

    # 加载模型
    model = attempt_load(weights, map_location=device)  # 加载FP32模型
    imgsz = check_img_size(imgsz, s=model.stride.max())  # 检查图像尺寸是否符合模型要求
    if half:
        model.half()  # to FP16   转换为FP16（加速推理）

    # 设置数据加载器
    vid_path, vid_writer = None, None   # 视频路径和写入器
    if webcam:   # 摄像头/视频流模式
        view_img = True
        cudnn.benchmark = True  # 启用CUDA基准优化（固定尺寸加速）
        dataset = LoadStreams(source, img_size=imgsz)   # 加载流
    else:
        view_img = True
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)    # 加载图像/视频

    # 获取类别名称和随机颜色（用于绘制检测框）
    names = model.module.names if hasattr(model, 'module') else model.names  # 类别名称
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]    # 随机颜色

    # 预热模型（初始化推理）
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # 创建空白图像
    _ = model(img.half() if half else img) if device.type != 'cpu' else None    # 运行一次推理


    # 逐帧处理输入数据
    for path, img, im0s, vid_cap in dataset:    #?
        img = torch.from_numpy(img).to(device)  # NumPy数组转Tensor
        img = img.half() if half else img.float()  # 转换为FP16/FP32
        img /= 255.0    # 归一化（0-255 → 0.0-1.0）
        if img.ndimension() == 3:
            img = img.unsqueeze(0)  # 增加批次维度（1x3xHxW）

        # 推理
        t1 = time_synchronized()    # 记录起始时间

        pred = model(img, augment=opt.augment)[0]   # 模型预测
        # Apply NMS
        pred = non_max_suppression( # 非极大值抑制（NMS）
            pred, 
            opt.conf_thres,      # 置信度阈值
            opt.iou_thres,      # IoU阈值
            classes=opt.classes,    # 指定类别
            agnostic=opt.agnostic_nms   # 是否类别无关NMS
        )

        t2 = time_synchronized()   # 记录结束时间

        # 处理检测结果
        for i, det in enumerate(pred):  # 遍历每张图像的检测结果
            if webcam:  # 多摄像头/视频流（批次处理） batch_size >= 1   
                p, s, im0 = Path(path[i]), '%g: ' % i, im0s[i].copy()
            else:   # 单图像/视频
                p, s, im0 = Path(path), '', im0s
            # 设置保存路径
            save_path = str(save_dir / p.name)  # 图像/视频保存路径
            txt_path = str(save_dir / 'labels' / p.stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')   # TXT标签路径
            s += '%gx%g ' % img.shape[2:]  # 打印图像尺寸
            # 坐标归一化增益（用于调整检测框）
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            if len(det):# 如果有检测结果
                # 调整检测框坐标（从推理尺寸→原始尺寸）
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # 打印检测结果（每类数量）
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # 当前类别的检测数
                    s += '%g %ss, ' % (n, names[int(c)])  # 添加到输出字符串

                # 绘制检测框
                for *xyxy, conf, cls in reversed(det):
                    if save_img or view_img:
                        label = '%s %.2f' % (names[int(cls)], conf) # 标签（类别+置信度）
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)  # 画框

            # 打印推理时间 (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))
            # 显示FPS
            try:
                im0 = cv2.putText(im0, "FPS: %.2f" %(1/(t2-t1)), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
            except:
                pass
            
            # 保存结果
            if save_img:
                if dataset.mode == 'images':    # 图像模式
                    cv2.imwrite(save_path, im0)
                else:   
                    if vid_path != save_path:  # 新视频
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # 释放之前的写入器
                        # 初始化视频写入器
                        fourcc = 'mp4v'  # 编码格式
                        fps = vid_cap.get(cv2.CAP_PROP_FPS) # 帧率
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 宽度
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 高度
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

            # Stream live results
            if view_img:
                cv2.imshow("Images", im0)
                if dataset.is_it_web:
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # q to quit
                        raise StopIteration
                else:
                    if dataset.video_flag[0]:
                        if cv2.waitKey(1) & 0xFF == ord('q'):  # q to quit
                            raise StopIteration
                    else:
                        if cv2.waitKey(0) & 0xFF == ord('q'):  # q to quit
                            raise StopIteration
            
            

    if save_txt or save_img:
        print('Results saved to %s' % save_dir)
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='../Model/weights/last.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.50, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
