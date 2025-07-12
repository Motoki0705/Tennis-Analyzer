import cv2


def draw_on_frame(frame, tracking_output):
    """フレームにトラッキング結果（ボールの位置とスコア）を描画するヘルパー関数"""
    if tracking_output.get("visi", False) and tracking_output.get("score", 0) > 0.1:
        px, py = int(tracking_output["x"]), int(tracking_output["y"])
        cv2.circle(frame, (px, py), 8, (0, 0, 255), -1)  # 赤い円
        cv2.circle(frame, (px, py), 3, (255, 255, 255), -1)  # 中央の白い点
        score_text = f"Score: {tracking_output['score']:.2f}"
        cv2.putText(frame, score_text, (px + 15, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame
