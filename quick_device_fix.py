#!/usr/bin/env python3
"""
Quick device compatibility fix
GPU/CPUデバイス互換性の緊急修正
"""

def show_device_issue_solution():
    """デバイス問題の解決方法を表示"""
    print("🔧 GPU/CPU Tensor Type Mismatch Fix")
    print("=" * 60)
    
    print("\n🚨 Error Analysis:")
    print("   Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor)")
    print("   → Model is on CPU, but input data is sent to GPU")
    
    print("\n💡 Immediate Solutions:")
    
    print("\n1️⃣ Force CPU mode:")
    print("   python -m src.predictor.api.inference \\")
    print("       --config-name inference \\")
    print("       model.device=cpu \\")
    print("       io.video=datasets/test/video_input2.mp4 \\")
    print("       io.output=outputs/ball/hrnet.mp4 \\")
    print("       model.model_path=third_party/WASB_SBDT/pretrained_weights/wasb_tennis_best.pth.tar")
    
    print("\n2️⃣ Use smaller batch size for GPU:")
    print("   python -m src.predictor.api.inference \\")
    print("       --config-name inference \\")
    print("       model.device=cuda \\")
    print("       pipeline=memory_efficient \\")
    print("       pipeline.batch_size=1 \\")
    print("       io.video=datasets/test/video_input2.mp4 \\")
    print("       io.output=outputs/ball/hrnet.mp4 \\")
    print("       model.model_path=third_party/WASB_SBDT/pretrained_weights/wasb_tennis_best.pth.tar")
    
    print("\n3️⃣ Debug mode with CPU:")
    print("   python -m src.predictor.api.inference \\")
    print("       --config-name inference \\")
    print("       model.device=cpu \\")
    print("       pipeline=debug \\")
    print("       system.log_level=DEBUG \\")
    print("       io.video=datasets/test/video_input2.mp4 \\")
    print("       io.output=outputs/ball/hrnet.mp4 \\")
    print("       model.model_path=third_party/WASB_SBDT/pretrained_weights/wasb_tennis_best.pth.tar")
    
    print("\n📋 Root Cause:")
    print("   - Model type detection may be incorrect")
    print("   - Model and input tensor device placement mismatch")
    print("   - WASB model may need specific loading procedure")
    
    print("\n🔍 To Debug Further:")
    print("   1. Check model file format: file third_party/WASB_SBDT/pretrained_weights/wasb_tennis_best.pth.tar")
    print("   2. Verify model type: Should be 'wasb_sbdt' not 'lite_tracknet'")
    print("   3. Test with CPU first to isolate GPU issues")
    
    print("\n⚡ Quick Test Command (Most Likely to Work):")
    print("   python -m src.predictor.api.inference \\")
    print("       --config-name inference \\")
    print("       model.device=cpu \\")
    print("       model.type=wasb_sbdt \\")
    print("       pipeline.batch_size=1 \\")
    print("       io.video=datasets/test/video_input2.mp4 \\")
    print("       io.output=outputs/ball/hrnet_cpu.mp4 \\")
    print("       model.model_path=third_party/WASB_SBDT/pretrained_weights/wasb_tennis_best.pth.tar")

def check_model_files():
    """利用可能なモデルファイルを確認"""
    print("\n📁 Checking available model files...")
    
    import os
    import glob
    
    # WASB model files
    wasb_dir = "third_party/WASB_SBDT/pretrained_weights/"
    if os.path.exists(wasb_dir):
        wasb_files = glob.glob(f"{wasb_dir}*")
        print(f"\n   WASB-SBDT models in {wasb_dir}:")
        for f in wasb_files:
            size = os.path.getsize(f) / (1024*1024) if os.path.exists(f) else 0
            print(f"     - {os.path.basename(f)} ({size:.1f} MB)")
    else:
        print(f"   ❌ WASB directory not found: {wasb_dir}")
    
    # Checkpoint files
    checkpoint_dirs = ["checkpoints/", "checkpoints/ball/", "models/"]
    print("\n   Lightning checkpoint files (.ckpt):")
    found_ckpt = False
    
    for ckpt_dir in checkpoint_dirs:
        if os.path.exists(ckpt_dir):
            ckpt_files = glob.glob(f"{ckpt_dir}**/*.ckpt", recursive=True)
            for f in ckpt_files:
                size = os.path.getsize(f) / (1024*1024) if os.path.exists(f) else 0
                print(f"     - {f} ({size:.1f} MB)")
                found_ckpt = True
    
    if not found_ckpt:
        print("     (No .ckpt files found)")

def main():
    """メイン実行"""
    show_device_issue_solution()
    check_model_files()
    
    print("\n" + "=" * 60)
    print("🎯 Recommendation: Start with CPU mode to verify functionality")
    print("   Then gradually move to GPU with proper device settings")

if __name__ == '__main__':
    main()