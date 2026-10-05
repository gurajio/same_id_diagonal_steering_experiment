(function () {
  "use strict";

  const practiceConditions = [
    {
      id: "C1",
      name: "ベースライン",
      label: "base",
      amplitude: 675,
      width: 70,
      trials: 30
    },
    {
      id: "C2",
      name: "長距離",
      label: "long",
      amplitude: 1350,
      width: 70,
      trials: 30
    },
    {
      id: "C3",
      name: "幅狭",
      label: "narrow",
      amplitude: 675,
      width: 35,
      trials: 30
    },
    {
      id: "C4",
      name: "距離小条件",
      label: "W固定・A短い",
      amplitude: 100,
      width: 100,
      trials: 1
    }
  ];

  function withSteeringId(condition, index) {
    return Object.freeze({
      ...condition,
      order: index + 1,
      steeringId: Number((condition.amplitude / condition.width).toFixed(3))
    });
  }

  const conditions = practiceConditions.map(withSteeringId);
  const postTrialsPerCondition = 25;

  window.SteeringExperimentConfig = Object.freeze({
    appName: "同一ID斜め直線ステアリング課題 実験システム",
    appVersion: "1.0.0",

    practiceTrials: 300,
    postTrials: 100,
    pilotTrials: 300,
    breakInterval: 100,
    manualPauseInterval: 100,
    forcedBreakSeconds: 60,

    experimentModes: Object.freeze({
      main: Object.freeze({
        id: "main",
        name: "本実験",
        description: "反復試行，事後試行まで実施する",
        practiceTrials: 300,
        postTrials: 100,
        hasPostPhase: true,
        forcedBreak: true
      }),
      pilot: Object.freeze({
        id: "pilot",
        name: "予備実験",
        description: "選択した反復条件を300試行ずつ実施する",
        trials: 300,
        hasPostPhase: false,
        forcedBreak: false,
        manualPause: true
      })
    }),

    assignment: Object.freeze({
      method: "manual-condition-toggle",
      description: "選択条件を条件ごとのC番号順または試行単位のランダム順で提示する",
      fallbackConditionId: "C1"
    }),

    errorCriteria: Object.freeze({
      deviationThresholdPx: 24,
      nearGoalProgressThreshold: 0.9
    }),

    display: Object.freeze({
      diagonalAngleDeg: 30,
      marginPx: 88,
      minCorridorWidthPx: 8,
      exactPixels: true,
      maxDisplayScale: 1
    }),

    conditions: Object.freeze(conditions),

    postConditions: Object.freeze(
      conditions.map((condition) =>
        Object.freeze({
          ...condition,
          trials: postTrialsPerCondition,
          phaseLabel: "事後試行"
        })
      )
    ),

    postConditionOrder: Object.freeze({
      method: "balanced-seeded-shuffle",
      description:
        "4条件を25試行ずつ用意し，参加者IDをシードにした固定順で提示する"
    }),

    export: Object.freeze({
      filenamePrefix: "same_id_steering"
    })
  });
})();
