class Config:
    video_directory = r'C:\Users\kamim\Desktop\vscode\tennis_analyzer\soft_1.mp4'
    frame_directory = r'C:\Users\kamim\Desktop\vscode\tennis_analyzer\frames'
    best_encoder_path = r"best_encoder.pth"
    best_decoder_path = r"best_decoder.pth"
    states = ["unknown", "inplay", "outplay"]
    events = [
            "none",
            "forehand",
            "backhand",
            "firstserve",
            "secondserve",
            "forevolley",
            "backvolley",
            "fore-lowvolley",
            "backlowvolley",
            "forepoatingvolley",
            "backpoatingvolley",
            "smash",
            "block",
            "top"
        ]
    len_states = 3
    len_events = 14
    outplay_range = 50
    input_channels = 3
    input_frame_shape = (360, 720)
    num_blocks = 4
    epochs = 20
    batch_size = 4
    
    