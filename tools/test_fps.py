import torch
from tqdm import tqdm
from tools.get_config import get_cfg
from Dataset.build import build_testset
from Models.build import build_model
import argparse
import time
import multiprocessing as mp

def parse_args():
    parser = argparse.ArgumentParser(description='Test FPS only')
    parser.add_argument('--gpu_no', default=0, type=int, help='GPU index to use')
    parser.add_argument('--test_batch_size', default=32, type=int)
    parser.add_argument('--cfg', default='Config/polarrcnn_culane_r50.py', type=str)
    parser.add_argument('--weight_path', default='', type=str)
    parser.add_argument('--fps_warmup_batches', default=10, type=int, help='Number of warmup batches')
    parser.add_argument('--max_batches', default=200, type=int, help='Max batches for FPS test')

    # ↓↓↓ 这些是数据集/代码里会访问到的字段（最小必需）↓↓↓
    parser.add_argument('--is_val', default=0, type=int)      # Dataset/base_dataset.py 会用到
    parser.add_argument('--is_view', default=0, type=int)     # 某些路径拼接里可能引用
    parser.add_argument('--result_path', default='./result', type=str)
    parser.add_argument('--view_path', default='./view', type=str)
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = get_cfg(args)
    torch.cuda.set_device(args.gpu_no)

    # 兜底：确保这些属性一定存在（避免别处硬编码访问时再崩）
    for k, v in dict(
        is_val=getattr(cfg, 'is_val', 0),
        is_view=getattr(cfg, 'is_view', 0),
        result_path=getattr(cfg, 'result_path', './result'),
        view_path=getattr(cfg, 'view_path', './view'),
        test_batch_size=getattr(cfg, 'test_batch_size', args.test_batch_size),
    ).items():
        setattr(cfg, k, v)

    # 模型
    net = build_model(cfg)
    net.load_state_dict(torch.load(cfg.weight_path, map_location='cpu'), strict=True)
    net.cuda().eval()

    # 数据集（仅用于形成真实输入尺寸，保证计时可信）
    tsset = build_testset(cfg)
    tsloader = torch.utils.data.DataLoader(
        tsset,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        num_workers=0,  # Windows 下稳定
        drop_last=False,
        collate_fn=tsset.collate_fn
    )

    print(f"\n[INFO] Testset length: {len(tsset)}")
    print(f"[INFO] Start FPS testing with batch_size={cfg.test_batch_size}")

    warmup_batches = args.fps_warmup_batches
    total_imgs = 0
    started = False

    pbar = tqdm(tsloader, desc='Running for FPS')
    for i, (img, _, _) in enumerate(pbar):
        if i >= args.max_batches:
            break
        with torch.no_grad():
            img = img.cuda()

            if not started and i >= warmup_batches:
                torch.cuda.synchronize()
                t_start = time.perf_counter()
                started = True

            _ = net(img)

        if started:
            total_imgs += img.size(0)

    if started:
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_start
        fps = total_imgs / elapsed
        print("\n================ FPS TEST RESULT ================")
        print(f"Warmup batches: {warmup_batches}")
        print(f"Timed images:   {total_imgs}")
        print(f"Elapsed (s):    {elapsed:.4f}")
        print(f"Average FPS:    {fps:.2f}")
        print("=================================================\n")
    else:
        print(f"\n[WARNING] 数据量不足，未进入正式计时阶段 (warmup_batches={warmup_batches}).\n")

if __name__ == '__main__':
    mp.freeze_support()
    main()
