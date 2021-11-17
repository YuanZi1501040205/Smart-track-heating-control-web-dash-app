# Timely Treatment-RL ventilator suggestion demo


## Quick start

- Create conda environment: `conda create -n icu_demo python=3.7`
- Install Dash for UI: `pip install dash`
- Install pandas for data: `pip install pandas`
- Open demo from browser: `http://127.0.0.1:8050/`

- Add vital signal to check current patient state: Heart Rate(HR), Mean blood pressure(MBP), Oxygen saturation(OS), Respiratory rate(RR), Diastolic blood pressure(DBP)

- mute 11/12/2010 17:00:00 box and run submit: LSTM predict Mobility Possibility based on current observation

- After click submit, RL Agent Optimal Ventilator Suggestion will show recommendation level to wear Ventilator in each 1h time slots.

- Chose 11/12/2010 17:00:00 box and click submit, the result shows new possibility of Mobility after taking AI suggestion.