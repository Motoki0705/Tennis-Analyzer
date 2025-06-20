import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  TextField,
  Alert,
  Grid,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Slider,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  Chip,
  Divider,
} from '@mui/material';
import {
  PlayArrow as PlayArrowIcon,
  Pause as PauseIcon,
  ContentCut as ContentCutIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';

interface VideoMetadata {
  filename: string;
  file_path: string;
  duration: number;
  fps: number;
  width: number;
  height: number;
  total_frames: number;
  file_size: number;
  upload_date: string;
}

interface ClipCreationTask {
  task_id: string;
  video_path: string;
  clip_name: string;
  start_time: number;
  end_time: number;
  status: string;
  progress: number;
  error_message?: string;
  created_at: string;
  completed_at?: string;
}

const ClipCreation: React.FC = () => {
  const [videos, setVideos] = useState<VideoMetadata[]>([]);
  const [selectedVideo, setSelectedVideo] = useState<VideoMetadata | null>(null);
  const [clipName, setClipName] = useState('');
  const [startTime, setStartTime] = useState(0);
  const [endTime, setEndTime] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [tasks, setTasks] = useState<ClipCreationTask[]>([]);
  const [creating, setCreating] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);

  const fetchVideos = useCallback(async () => {
    try {
      const response = await fetch('/api/raw_videos');
      const data = await response.json();
      setVideos(data);
    } catch (err) {
      console.error('動画リスト取得エラー:', err);
      setError('動画リストの取得に失敗しました');
    }
  }, []);

  const fetchTasks = useCallback(async () => {
    try {
      const response = await fetch('/api/clip_tasks');
      const data = await response.json();
      setTasks(data);
    } catch (err) {
      console.error('タスクリスト取得エラー:', err);
    }
  }, []);

  useEffect(() => {
    fetchVideos();
    fetchTasks();
  }, [fetchVideos, fetchTasks]);

  useEffect(() => {
    const interval = setInterval(() => {
      fetchTasks();
    }, 2000); // 2秒ごとに更新

    return () => clearInterval(interval);
  }, [fetchTasks]);

  const handleVideoSelect = (video: VideoMetadata) => {
    setSelectedVideo(video);
    setStartTime(0);
    setEndTime(Math.min(video.duration, 30)); // デフォルト30秒
    setCurrentTime(0);
    setClipName(`clip_${video.filename.split('.')[0]}_${Date.now()}`);
    
    if (videoRef.current) {
      videoRef.current.currentTime = 0;
    }
  };

  const handlePlayPause = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const handleSeek = (time: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = time;
      setCurrentTime(time);
    }
  };

  const setCurrentAsStart = () => {
    setStartTime(currentTime);
    if (endTime <= currentTime) {
      setEndTime(Math.min(currentTime + 10, selectedVideo?.duration || 0));
    }
  };

  const setCurrentAsEnd = () => {
    setEndTime(currentTime);
    if (startTime >= currentTime) {
      setStartTime(Math.max(currentTime - 10, 0));
    }
  };

  const handleCreateClip = async () => {
    if (!selectedVideo || !clipName.trim()) {
      setError('動画とクリップ名を選択してください');
      return;
    }

    if (startTime >= endTime) {
      setError('開始時間は終了時間より前である必要があります');
      return;
    }

    setCreating(true);
    setError(null);
    setSuccess(null);

    try {
      const response = await fetch('/api/create_clip', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          video_path: selectedVideo.file_path,
          clip_name: clipName.trim(),
          start_time: startTime,
          end_time: endTime,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'クリップ生成に失敗しました');
      }

      const data = await response.json();
      setSuccess(`クリップ生成を開始しました: ${data.task_id}`);
      
      // タスクリストを更新
      fetchTasks();
      
      // フォームをリセット
      setClipName(`clip_${selectedVideo.filename.split('.')[0]}_${Date.now()}`);
      
    } catch (err: any) {
      console.error('クリップ生成エラー:', err);
      setError(err.message || 'クリップ生成に失敗しました');
    } finally {
      setCreating(false);
    }
  };

  const formatTime = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'success';
      case 'failed': return 'error';
      case 'processing': return 'warning';
      default: return 'default';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'pending': return '待機中';
      case 'processing': return '処理中';
      case 'completed': return '完了';
      case 'failed': return 'エラー';
      default: return status;
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        クリップ生成
      </Typography>

      {/* エラー・成功メッセージ */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {success && (
        <Alert severity="success" sx={{ mb: 2 }} onClose={() => setSuccess(null)}>
          {success}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* 左側: 動画選択とプレビュー */}
        <Grid item xs={12} md={8}>
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                動画選択
              </Typography>
              
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>動画を選択</InputLabel>
                <Select
                  value={selectedVideo?.filename || ''}
                  label="動画を選択"
                  onChange={(e) => {
                    const video = videos.find(v => v.filename === e.target.value);
                    if (video) handleVideoSelect(video);
                  }}
                >
                  {videos.map((video) => (
                    <MenuItem key={video.filename} value={video.filename}>
                      {video.filename} ({formatTime(video.duration)})
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              {selectedVideo && (
                <Box>
                  <video
                    ref={videoRef}
                    src={`/api/raw_video/${selectedVideo.filename}`}
                    onTimeUpdate={handleTimeUpdate}
                    onLoadedMetadata={() => {
                      if (videoRef.current) {
                        setCurrentTime(0);
                      }
                    }}
                    style={{
                      width: '100%',
                      maxHeight: '400px',
                      backgroundColor: '#000',
                    }}
                    controls={false}
                  />
                  
                  <Box sx={{ mt: 2, display: 'flex', alignItems: 'center', gap: 2 }}>
                    <Button
                      variant="contained"
                      onClick={handlePlayPause}
                      startIcon={isPlaying ? <PauseIcon /> : <PlayArrowIcon />}
                    >
                      {isPlaying ? '一時停止' : '再生'}
                    </Button>
                    
                    <Box sx={{ flexGrow: 1 }}>
                      <Slider
                        value={currentTime}
                        max={selectedVideo.duration}
                        onChange={(_, value) => handleSeek(value as number)}
                        valueLabelDisplay="auto"
                        valueLabelFormat={formatTime}
                        step={0.1}
                      />
                    </Box>
                    
                    <Typography variant="body2">
                      {formatTime(currentTime)} / {formatTime(selectedVideo.duration)}
                    </Typography>
                  </Box>
                </Box>
              )}
            </CardContent>
          </Card>

          {/* クリップ範囲設定 */}
          {selectedVideo && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  クリップ範囲設定
                </Typography>
                
                <Grid container spacing={2} sx={{ mb: 2 }}>
                  <Grid item xs={6}>
                    <TextField
                      fullWidth
                      label="開始時間（秒）"
                      type="number"
                      value={startTime.toFixed(1)}
                      onChange={(e) => setStartTime(Math.max(0, parseFloat(e.target.value) || 0))}
                      inputProps={{
                        step: 0.1,
                        min: 0,
                        max: selectedVideo.duration,
                      }}
                    />
                    <Button
                      size="small"
                      onClick={setCurrentAsStart}
                      sx={{ mt: 1 }}
                    >
                      現在時間を開始時間に設定
                    </Button>
                  </Grid>
                  
                  <Grid item xs={6}>
                    <TextField
                      fullWidth
                      label="終了時間（秒）"
                      type="number"
                      value={endTime.toFixed(1)}
                      onChange={(e) => setEndTime(Math.min(selectedVideo.duration, parseFloat(e.target.value) || 0))}
                      inputProps={{
                        step: 0.1,
                        min: 0,
                        max: selectedVideo.duration,
                      }}
                    />
                    <Button
                      size="small"
                      onClick={setCurrentAsEnd}
                      sx={{ mt: 1 }}
                    >
                      現在時間を終了時間に設定
                    </Button>
                  </Grid>
                </Grid>

                <TextField
                  fullWidth
                  label="クリップ名"
                  value={clipName}
                  onChange={(e) => setClipName(e.target.value)}
                  sx={{ mb: 2 }}
                />

                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  クリップ長: {formatTime(endTime - startTime)}
                </Typography>

                <Button
                  variant="contained"
                  startIcon={<ContentCutIcon />}
                  onClick={handleCreateClip}
                  disabled={creating || !clipName.trim() || startTime >= endTime}
                  fullWidth
                >
                  {creating ? 'クリップ生成中...' : 'クリップを生成'}
                </Button>
              </CardContent>
            </Card>
          )}
        </Grid>

        {/* 右側: タスクリスト */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">
                  クリップ生成タスク
                </Typography>
                <Button
                  size="small"
                  startIcon={<RefreshIcon />}
                  onClick={fetchTasks}
                >
                  更新
                </Button>
              </Box>
              
              {tasks.length === 0 ? (
                <Typography color="text.secondary">
                  タスクはありません
                </Typography>
              ) : (
                <List dense>
                  {tasks.map((task, index) => (
                    <React.Fragment key={task.task_id}>
                      <ListItem>
                        <ListItemText
                          primary={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <Typography variant="subtitle2">
                                {task.clip_name}
                              </Typography>
                              <Chip 
                                label={getStatusText(task.status)} 
                                size="small" 
                                color={getStatusColor(task.status) as any}
                              />
                            </Box>
                          }
                          secondary={
                            <Box>
                              <Typography variant="caption">
                                {formatTime(task.start_time)} - {formatTime(task.end_time)}
                              </Typography>
                              {task.status === 'processing' && (
                                <LinearProgress 
                                  variant="determinate" 
                                  value={task.progress * 100} 
                                  sx={{ mt: 1 }}
                                />
                              )}
                              {task.error_message && (
                                <Typography variant="caption" color="error">
                                  {task.error_message}
                                </Typography>
                              )}
                            </Box>
                          }
                        />
                      </ListItem>
                      {index < tasks.length - 1 && <Divider />}
                    </React.Fragment>
                  ))}
                </List>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ClipCreation; 