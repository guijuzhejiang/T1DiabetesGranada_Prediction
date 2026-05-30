The current result is approximately RMSE ≈ 18, while the original research target was RMSE ≤ 9.

At this stage, I only need a clear technical report explaining the work already completed, especially the reasoning behind the preprocessing pipeline, model settings, and methodology choices. After that, we can finalize Milestone 1.

The original research hypothesis was to investigate whether xLSTM, as an improved sequence architecture, could outperform traditional LSTM for 30-minute blood glucose forecasting. Unfortunately, based on the current results, this has not yet been demonstrated.

If you still believe there is a strong technical direction that could significantly improve xLSTM performance — for example through different window construction, normalization strategy, target formulation, feature engineering, or model configuration — please share your recommendations clearly. We may then discuss continuing with specific targeted improvements.

I also need clarification regarding several methodological decisions used in your implementation:

1. Why were duplicate records handled using keep=first?
1. Why was the last input reading excluded from rolling statistics using closed='left'?
1. Why were 2h and 6h rolling windows specifically selected?
1. Why were features such as p5, p95, and TIR selected instead of other variability metrics or slope-based features?
1. Is the evaluation intended for forecasting future values of the same patients, or for generalization to unseen patients?
1. Validation RMSE plateaued around ≈18.9. What is your interpretation of this limitation?
1. Do you believe the limitation mainly comes from:
- the model itself,
- feature engineering,
- normalization,
- preprocessing,
- temporal resolution,
- or the dataset itself?
1. Did you analyze why most patients remained above the RMSE ≤ 9 target?

Please also provide practical suggestions for improving xLSTM performance for blood glucose prediction and explain:

- why xLSTM did not outperform standard LSTM,
- what limitations prevented reaching the target,
- and what changes you would recommend to help xLSTM achieve better performance than LSTM for this task.

I mainly need this report so I can fully understand the methodology and evaluate the research direction scientifically, not only the final RMSE value.