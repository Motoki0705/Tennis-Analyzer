#!/usr/bin/env python3
"""
Pipeline debugging test script
パイプライン問題のデバッグテストスクリプト
"""

import os
import sys

def show_debug_fixes():
    """適用されたデバッグ修正の確認"""
    print("🔧 Applied Debug Fixes Verification")
    print("=" * 60)
    
    # frame_number修正の確認
    async_file = "src/predictor/pipeline/async_processor.py"
    if os.path.exists(async_file):
        with open(async_file, 'r') as f:
            content = f.read()
            
        if "frame_number = metadata.get('frame_number')" in content:
            print("   ✅ AsyncProcessor: Fixed frame_number KeyError")
        else:
            print("   ❌ AsyncProcessor: frame_number fix not applied")
            
        if "traceback.format_exc()" in content:
            print("   ✅ AsyncProcessor: Enhanced error logging")
        else:
            print("   ❌ AsyncProcessor: Enhanced error logging not applied")
    else:
        print("   ❌ AsyncProcessor: File not found")
    
    # 新しいパイプライン設定の確認
    wasb_config = "configs/infer/pipeline/wasb_optimized.yaml"
    if os.path.exists(wasb_config):
        print("   ✅ WASB optimized pipeline config created")
    else:
        print("   ❌ WASB optimized pipeline config missing")

def show_recommended_test_sequence():
    """推奨テスト順序"""
    print("\n🧪 Recommended Test Sequence")
    print("=" * 60)
    
    print("\n📋 Step 1: Debug Mode (Most Information)")
    print("python -m src.predictor.api.inference \\")
    print("    --config-name inference \\")
    print("    model.type=wasb_sbdt \\")
    print("    model.model_path=third_party/WASB_SBDT/pretrained_weights/wasb_tennis_best.pth.tar \\")
    print("    model.device=cpu \\")
    print("    pipeline=debug \\")
    print("    system.log_level=DEBUG \\")
    print("    io.video=datasets/test/video_input2.mp4 \\")
    print("    io.output=outputs/ball/debug_step1.mp4")
    
    print("\n📋 Step 2: WASB Optimized Settings")
    print("python -m src.predictor.api.inference \\")
    print("    --config-name inference \\")
    print("    model.type=wasb_sbdt \\")
    print("    model.model_path=third_party/WASB_SBDT/pretrained_weights/wasb_tennis_best.pth.tar \\")
    print("    model.device=cpu \\")
    print("    pipeline=wasb_optimized \\")
    print("    io.video=datasets/test/video_input2.mp4 \\")
    print("    io.output=outputs/ball/debug_step2.mp4")
    
    print("\n📋 Step 3: Memory Efficient CPU")
    print("python -m src.predictor.api.inference \\")
    print("    --config-name inference \\")
    print("    model.type=wasb_sbdt \\")
    print("    model.model_path=third_party/WASB_SBDT/pretrained_weights/wasb_tennis_best.pth.tar \\")
    print("    model.device=cpu \\")
    print("    pipeline=memory_efficient \\")
    print("    io.video=datasets/test/video_input2.mp4 \\")
    print("    io.output=outputs/ball/debug_step3.mp4")
    
    print("\n📋 Step 4: GPU Test (After CPU Success)")
    print("python -m src.predictor.api.inference \\")
    print("    --config-name inference \\")
    print("    model.type=wasb_sbdt \\")
    print("    model.model_path=third_party/WASB_SBDT/pretrained_weights/wasb_tennis_best.pth.tar \\")
    print("    model.device=cuda \\")
    print("    pipeline=wasb_optimized \\")
    print("    io.video=datasets/test/video_input2.mp4 \\")
    print("    io.output=outputs/ball/debug_step4.mp4")
    
    print("\n📋 Step 5: LiteTrackNet Comparison")
    print("python -m src.predictor.api.inference \\")
    print("    --config-name inference \\")
    print("    model.type=lite_tracknet \\")
    print("    model.model_path=checkpoints/ball/lit_lite_tracknet/best_model.ckpt \\")
    print("    model.device=cuda \\")
    print("    pipeline=high_performance \\")
    print("    io.video=datasets/test/video_input2.mp4 \\")
    print("    io.output=outputs/ball/debug_step5_lite.mp4")

def show_troubleshooting_guide():
    """トラブルシューティングガイド"""
    print("\n🛠️ Troubleshooting Guide")
    print("=" * 60)
    
    print("\n🚨 If you see 'frame_number' errors:")
    print("   ✅ Should be fixed now with frame_id fallback")
    print("   📝 Check that frame_id is in format 'frame_XXXXXX'")
    
    print("\n🚨 If you see empty 'Batch inference error:'")
    print("   ✅ Should show detailed traceback now")
    print("   📝 Look for 'Batch inference traceback:' in logs")
    
    print("\n🚨 If you see queue overflow warnings:")
    print("   💡 Reduce batch_size: pipeline.batch_size=1")
    print("   💡 Reduce workers: pipeline.num_workers=1") 
    print("   💡 Reduce queue: pipeline.queue_size=5")
    
    print("\n🚨 If you see thread failures:")
    print("   💡 Use debug mode: pipeline=debug")
    print("   💡 Use CPU mode: model.device=cpu")
    print("   💡 Enable detailed logging: system.log_level=DEBUG")
    
    print("\n🚨 If processing stalls:")
    print("   💡 Check GPU memory: nvidia-smi")
    print("   💡 Reduce batch size or switch to CPU")
    print("   💡 Use smaller queue sizes")

def show_expected_outputs():
    """期待される出力"""
    print("\n📊 Expected Success Indicators")
    print("=" * 60)
    
    print("\n✅ Successful startup messages:")
    print("   - 'Successfully loaded weights from ... on [device]'")
    print("   - 'HRNet WASB-SBDT detector initialized'")
    print("   - 'Processing video: WIDTHxHEIGHT @ FPS, X frames'")
    print("   - 'Started N processing threads'")
    
    print("\n✅ Successful processing messages:")
    print("   - Progress updates without errors")
    print("   - 'Video processing completed in Xs'")
    print("   - 'Pipeline completed successfully: X frames'")
    print("   - Non-zero frame counts in final summary")
    
    print("\n❌ Warning signs (but may still work):")
    print("   - Occasional queue warnings (normal under load)")
    print("   - 'Warning: Failed to load weights' (using random weights)")
    
    print("\n🚫 Failure indicators:")
    print("   - Repeated 'Batch inference error'")
    print("   - 'Too many errors, stopping processing'")
    print("   - All thread workers failing")
    print("   - Final frame count = 0")

def main():
    """メイン実行"""
    print("🎾 Pipeline Debug Test Guide")
    print("=" * 60)
    
    show_debug_fixes()
    show_recommended_test_sequence()
    show_troubleshooting_guide()
    show_expected_outputs()
    
    print("\n" + "=" * 60)
    print("🎯 Start with Step 1 (debug mode) to get maximum information")
    print("   Then proceed through steps based on results")

if __name__ == '__main__':
    main()