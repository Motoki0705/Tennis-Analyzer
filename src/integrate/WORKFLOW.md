# Flexible Tennis Analysis Pipeline - Workflow Documentation

## システム全体のアーキテクチャ

```mermaid
graph TB
    subgraph "Configuration Layer"
        YML[flexible_pipeline.yaml]
        CFG[Hydra Config]
    end
    
    subgraph "Core Framework"
        FP[FlexiblePipeline]
        TM[TaskManager]
        DF[DataFlow]
        VIO[VideoIO]
    end
    
    subgraph "Task Layer"
        CT[CourtDetectionTask]
        BT[BallTrackingTask]
        PT[PlayerDetectionTask]
        PST[PoseEstimationTask]
    end
    
    subgraph "Pipeline Modules"
        CM[Court Modules]
        BM[Ball Modules]
        PM[Player Modules]
        PSM[Pose Modules]
    end
    
    YML --> CFG
    CFG --> FP
    FP --> TM
    FP --> DF
    FP --> VIO
    
    TM --> CT
    TM --> BT
    TM --> PT
    TM --> PST
    
    CT --> CM
    BT --> BM
    PT --> PM
    PST --> PSM
```

## データフロー（シングルスレッドモード）

```mermaid
flowchart TD
    Start([開始]) --> LoadConfig[設定読み込み]
    LoadConfig --> InitTasks[タスク初期化]
    InitTasks --> InitVideo[ビデオI/O初期化]
    InitVideo --> ReadFrame[フレーム読み込み]
    
    ReadFrame --> BatchReady{バッチ準備完了?}
    BatchReady -->|No| ReadFrame
    BatchReady -->|Yes| CreatePacket[データパケット作成]
    
    CreatePacket --> ExecutePipeline[パイプライン実行]
    
    subgraph "Pipeline Execution"
        ExecutePipeline --> CourtTask[コート検出]
        CourtTask --> BallTask[ボール追跡]
        BallTask --> PlayerTask[選手検出] 
        PlayerTask --> PoseTask[姿勢推定<br/>※選手検出に依存]
    end
    
    PoseTask --> Visualize[可視化]
    Visualize --> WriteOutput[出力書き込み]
    WriteOutput --> MoreFrames{次のフレーム?}
    
    MoreFrames -->|Yes| ReadFrame
    MoreFrames -->|No| SaveResults[結果保存]
    SaveResults --> End([終了])
```

## データフロー（マルチスレッドモード）

```mermaid
flowchart TD
    subgraph "Worker Threads"
        subgraph "Input Worker"
            VR[Video Reader]
            FB[Frame Batching]
        end
        
        subgraph "Preprocessing Worker"
            PP1[Court Preprocess]
            PP2[Ball Preprocess]
            PP3[Player Preprocess]
            PP4[Pose Preprocess]
        end
        
        subgraph "Inference Worker"
            IF1[Court Inference]
            IF2[Ball Inference]
            IF3[Player Inference]
            IF4[Pose Inference]
        end
        
        subgraph "Postprocessing Worker"
            OP1[Court Postprocess]
            OP2[Ball Postprocess]
            OP3[Player Postprocess]
            OP4[Pose Postprocess]
        end
        
        subgraph "Output Worker"
            VZ[Visualization]
            VW[Video Writer]
            RS[Result Saver]
        end
    end
    
    subgraph "Queue System"
        Q1[Preprocessing Queue]
        Q2[Inference Queue]
        Q3[Postprocessing Queue]
        Q4[Output Queue]
    end
    
    VR --> FB
    FB --> Q1
    Q1 --> PP1
    Q1 --> PP2
    Q1 --> PP3
    Q1 --> PP4
    
    PP1 --> Q2
    PP2 --> Q2
    PP3 --> Q2
    PP4 --> Q2
    
    Q2 --> IF1
    Q2 --> IF2
    Q2 --> IF3
    Q2 --> IF4
    
    IF1 --> Q3
    IF2 --> Q3
    IF3 --> Q3
    IF4 --> Q3
    
    Q3 --> OP1
    Q3 --> OP2
    Q3 --> OP3
    Q3 --> OP4
    
    OP1 --> Q4
    OP2 --> Q4
    OP3 --> Q4
    OP4 --> Q4
    
    Q4 --> VZ
    VZ --> VW
    VZ --> RS
```

## タスク依存関係グラフ

```mermaid
graph LR
    subgraph "Independent Tasks"
        CT[Court Detection]
        BT[Ball Tracking]
        PT[Player Detection]
    end
    
    subgraph "Dependent Tasks"
        PST[Pose Estimation]
    end
    
    PT --> PST
    
    style CT fill:#e1f5fe
    style BT fill:#e1f5fe
    style PT fill:#e1f5fe
    style PST fill:#fff3e0
```

## タスク実行フロー（各タスク共通）

```mermaid
flowchart TD
    TaskStart([タスク開始]) --> Initialize[initialize]
    Initialize --> Preprocess[preprocess]
    
    subgraph "Preprocessing Stage"
        Preprocess --> PrepareData[データ準備]
        PrepareData --> PrepareMetadata[メタデータ準備]
    end
    
    PrepareMetadata --> Inference[inference]
    
    subgraph "Inference Stage"
        Inference --> ModelPredict[モデル推論]
        ModelPredict --> SyncGPU[GPU同期]
    end
    
    SyncGPU --> Postprocess[postprocess]
    
    subgraph "Postprocessing Stage"
        Postprocess --> ProcessResults[結果処理]
        ProcessResults --> CalculateStats[統計計算]
    end
    
    CalculateStats --> Visualize[visualize]
    
    subgraph "Visualization Stage"
        Visualize --> DrawResults[結果描画]
        DrawResults --> AddAnnotations[注釈追加]
    end
    
    AddAnnotations --> TaskEnd([タスク完了])
```

## エラー処理フロー

```mermaid
flowchart TD
    TaskExecution[タスク実行] --> Success{成功?}
    
    Success -->|Yes| NextTask[次のタスク]
    Success -->|No| CheckCritical{クリティカル?}
    
    CheckCritical -->|Yes| AbortPipeline[パイプライン中止]
    CheckCritical -->|No| LogWarning[警告ログ]
    
    LogWarning --> SkipTask[タスクをスキップ]
    SkipTask --> NextTask
    
    AbortPipeline --> CleanupResources[リソース清理]
    CleanupResources --> Error([エラー終了])
    
    NextTask --> AllTasksDone{全タスク完了?}
    AllTasksDone -->|No| TaskExecution
    AllTasksDone -->|Yes| Success([正常終了])
```

## 設定システム

```mermaid
flowchart LR
    subgraph "Configuration Sources"
        YAML[flexible_pipeline.yaml]
        CLI[Command Line Args]
        ENV[Environment Variables]
    end
    
    subgraph "Hydra Processing"
        HC[Hydra Composer]
        OC[OmegaConf]
    end
    
    subgraph "Runtime Configuration"
        TC[Task Configs]
        SC[System Configs]
        VC[Visualization Configs]
    end
    
    YAML --> HC
    CLI --> HC
    ENV --> HC
    
    HC --> OC
    OC --> TC
    OC --> SC
    OC --> VC
    
    TC --> TM[TaskManager]
    SC --> FP[FlexiblePipeline]
    VC --> Tasks[Individual Tasks]
```

## パフォーマンス監視

```mermaid
flowchart TD
    subgraph "Performance Metrics"
        TT[Task Timing]
        QU[Queue Utilization]
        MU[Memory Usage]
        TH[Throughput]
    end
    
    subgraph "Data Collection"
        DC[DataFlow Stats]
        TM[Task Manager Stats]
        QS[Queue Stats]
    end
    
    subgraph "Reporting"
        LG[Live Logging]
        FR[Final Report]
        VZ[Visual Dashboard]
    end
    
    TT --> DC
    QU --> QS
    MU --> TM
    TH --> DC
    
    DC --> LG
    TM --> FR
    QS --> VZ
```

このワークフロー図により、システムの全体像と各コンポーネントの相互作用が明確に理解できます。