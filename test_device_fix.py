#!/usr/bin/env python3
"""
Test device fix for WASB-SBDT model
WASBモデルのデバイス修正テスト
"""

import os
import sys

def test_device_fix():
    """デバイス修正のテスト"""
    print("🔧 Testing WASB-SBDT Device Fix")
    print("=" * 60)
    
    # 修正内容の確認
    print("📋 Checking applied fixes:")
    
    # 1. WASB-SBDT __init__.py の修正確認
    wasb_init_file = "third_party/WASB_SBDT/src/__init__.py"
    if os.path.exists(wasb_init_file):
        with open(wasb_init_file, 'r') as f:
            content = f.read()
            
        if 'model = model.to(device)' in content:
            print("   ✅ WASB __init__.py: Added model.to(device)")
        else:
            print("   ❌ WASB __init__.py: Missing model.to(device)")
            
        if 'torch.load(model_path, map_location=device)' in content:
            print("   ✅ WASB __init__.py: Added map_location for weight loading")
        else:
            print("   ❌ WASB __init__.py: Missing map_location for weight loading")
            
        if 'model.eval()' in content:
            print("   ✅ WASB __init__.py: Added model.eval()")
        else:
            print("   ❌ WASB __init__.py: Missing model.eval()")
    else:
        print("   ❌ WASB __init__.py: File not found")
    
    # 2. WASBSBDTDetector の修正確認
    detector_file = "src/predictor/ball/wasb_sbdt_detector.py"
    if os.path.exists(detector_file):
        with open(detector_file, 'r') as f:
            content = f.read()
            
        if 'self.model = self.model.to(self.device)' in content:
            print("   ✅ WASBSBDTDetector: Added redundant device placement")
        else:
            print("   ❌ WASBSBDTDetector: Missing redundant device placement")
            
        if 'batch_tensor.to(self.device)' in content:
            print("   ✅ WASBSBDTDetector: Has input tensor device placement")
        else:
            print("   ❌ WASBSBDTDetector: Missing input tensor device placement")
    else:
        print("   ❌ WASBSBDTDetector: File not found")

def show_test_commands():
    """テストコマンドの表示"""
    print("\n🧪 Test Commands:")
    
    print("\n1️⃣ CPU Test (Should work now):")
    print("   python -m src.predictor.api.inference \\")
    print("       --config-name inference \\")
    print("       model.type=wasb_sbdt \\")
    print("       model.model_path=third_party/WASB_SBDT/pretrained_weights/wasb_tennis_best.pth.tar \\")
    print("       model.device=cpu \\")
    print("       pipeline.batch_size=1 \\")
    print("       pipeline.num_workers=1 \\")
    print("       io.video=datasets/test/video_input2.mp4 \\")
    print("       io.output=outputs/ball/wasb_cpu_fixed.mp4")
    
    print("\n2️⃣ GPU Test (Should work now):")
    print("   python -m src.predictor.api.inference \\")
    print("       --config-name inference \\")
    print("       model.type=wasb_sbdt \\")
    print("       model.model_path=third_party/WASB_SBDT/pretrained_weights/wasb_tennis_best.pth.tar \\")
    print("       model.device=cuda \\")
    print("       pipeline.batch_size=2 \\")
    print("       pipeline.num_workers=2 \\")
    print("       io.video=datasets/test/video_input2.mp4 \\")
    print("       io.output=outputs/ball/wasb_gpu_fixed.mp4")
    
    print("\n3️⃣ Debug Test:")
    print("   python -m src.predictor.api.inference \\")
    print("       --config-name inference \\")
    print("       model.type=wasb_sbdt \\")
    print("       model.model_path=third_party/WASB_SBDT/pretrained_weights/wasb_tennis_best.pth.tar \\")
    print("       model.device=cpu \\")
    print("       pipeline=debug \\")
    print("       system.log_level=DEBUG \\")
    print("       io.video=datasets/test/video_input2.mp4 \\")
    print("       io.output=outputs/ball/wasb_debug.mp4")

def show_expected_behavior():
    """期待される動作の説明"""
    print("\n📋 Expected Behavior After Fix:")
    
    print("\n✅ What should work now:")
    print("   - Model loading with proper device placement")
    print("   - Weight loading with map_location=device")
    print("   - Input tensors moved to correct device")
    print("   - Both CPU and GPU modes supported")
    print("   - No 'Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor)' errors")
    
    print("\n⚠️ What to watch for:")
    print("   - Weight loading warnings (if checkpoint format is unexpected)")
    print("   - Memory usage (HRNet can be memory intensive)")
    print("   - Processing speed (WASB is heavier than LiteTrackNet)")
    
    print("\n🎯 Success indicators:")
    print("   - 'Successfully loaded weights from ... on [device]' message")
    print("   - No tensor device mismatch errors")
    print("   - Output video generated successfully")

def main():
    """メイン実行"""
    test_device_fix()
    show_test_commands()
    show_expected_behavior()
    
    print("\n" + "=" * 60)
    print("🎉 Device fix applied! Try the test commands above.")
    print("   Start with CPU test, then try GPU if successful.")

if __name__ == '__main__':
    main()