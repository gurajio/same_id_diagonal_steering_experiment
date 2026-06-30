(function () {
  "use strict";

  const practiceConditions = [
    {
      id: "C1",
      name: "幅広条件",
      label: "A固定・W広い",
      amplitude: 1000,
      width: 50,
      trials: 300
    },
    {
      id: "C2",
      name: "幅狭条件",
      label: "A固定・W狭い",
      amplitude: 1000,
      width: 20,
      trials: 300
    },
    {
      id: "C3",
      name: "距離大条件",
      label: "W固定・A長い",
      amplitude: 1500,
      width: 30,
      trials: 300
    },
    {
      id: "C4",
      name: "距離小条件",
      label: "W固定・A短い",
      amplitude: 600,
      width: 30,
      trials: 300
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
        description: "反復試行，事後試行，アンケートまで実施する",
        practiceTrials: 300,
        postTrials: 100,
        hasPostPhase: true,
        hasQuestionnaire: true,
        forcedBreak: true
      }),
      pilot: Object.freeze({
        id: "pilot",
        name: "予備実験",
        description: "選択した反復条件を300試行ずつ実施する",
        trials: 300,
        hasPostPhase: false,
        hasQuestionnaire: false,
        forcedBreak: false,
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
      diagonalAngleDeg: 32,
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
