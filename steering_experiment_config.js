(function () {
  "use strict";

  // 各条件の trials が，反復試行・予備試行で提示する回数です。
  const practiceConditions = [
    {
      id: "C1",
      name: "幅広条件",
      label: "A固定・W広い",
      amplitude: 1050,
      width: 30,
      trials: 400
    },
    {
      id: "C2",
      name: "幅狭条件",
      label: "A固定・W狭い",
      amplitude: 1050,
      width: 21,
      trials: 400
    },
    {
      id: "C3",
      name: "距離大条件",
      label: "W固定・A長い",
      amplitude: 1250,
      width: 25,
      trials: 400
    },
    {
      id: "C4",
      name: "距離小条件",
      label: "W固定・A短い",
      amplitude: 875,
      width: 25,
      trials: 400
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
  // 事後試行で各条件を提示する回数です。
  const postTrialsPerCondition = 25;
  const postTrialTotal = conditions.length * postTrialsPerCondition;

  window.SteeringExperimentConfig = Object.freeze({
    appName: "同一ID斜め直線ステアリング課題実験システム",
    appVersion: "1.0.0",

    practiceTrials: 400,
    postTrials: postTrialTotal,
    pilotTrials: 400,
    // 本実験で休憩画面を表示する間隔。0以下なら休憩なし。
    breakInterval: 100,
    // 予備実験で任意停止画面を表示する間隔。0以下なら停止なし。
    manualPauseInterval: 100,
    breakTimerMode: "count-up",

    experimentModes: Object.freeze({
      main: Object.freeze({
        id: "main",
        name: "本実験",
        description: "反復試行から事後試行まで実施する",
        practiceTrials: 400,
        postTrials: postTrialTotal,
        hasPostPhase: true,
        scheduledBreak: true
      }),
      pilot: Object.freeze({
        id: "pilot",
        name: "予備実験",
        description: "選択した反復条件を設定試行数ずつ実施する",
        trials: 400,
        hasPostPhase: false,
        scheduledBreak: false,
        manualPause: true
      })
    }),

    assignment: Object.freeze({
      method: "manual-condition-toggle",
      description: "開始画面で選択した条件をC1からC4の順に提示する",
      fallbackConditionId: "C1"
    }),

    errorCriteria: Object.freeze({
      deviationThresholdPx: 24,
      nearGoalProgressThreshold: 0.9
    }),

    display: Object.freeze({
      diagonalAngleDeg: 30,
      marginPx: 88,
      movementAreaPx: 200,
      minEndpointLengthPx: 40,
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
      description: `${conditions.length}条件を${postTrialsPerCondition}試行ずつ用意し，参加者IDをシードにした固定順で提示する`
    }),

    export: Object.freeze({
      filenamePrefix: "same_id_steering"
    })
  });
})();
