import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objects as go
import pandas as pd
from dash.dependencies import Input, Output
import copy
import numpy as np

colorVal = [
        "#F4EC15",
        "#DAF017",
        "#BBEC19",
        "#9DE81B",
        "#80E41D",
        "#66E01F",
        "#4CDC20",
        "#34D822",
        "#24D249",
        "#25D042",
        "#26CC58",
        "#28C86D",
        "#29C481",
        "#2AC093",
        "#2BBCA4",
        "#2BB5B8",
        "#2C99B4",
        "#2D7EB0",
        "#2D65AC",
        "#2E4EA4",
        "#2E38A4",
        "#3B2FA0",
        "#4E2F9C",
        "#603099",
    ]

app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True


def get_options(list_vitals):
    dict_list = []
    for i in list_vitals:
        dict_list.append({'label': i, 'value': i})

    return dict_list

df = pd.read_csv('data/vitaldata2.csv', index_col=0, parse_dates=True)

df.index = pd.to_datetime(df['Date'])

app.layout = html.Div(
    children=[
        html.Div(className='row',
                 children=[
                    html.Div(className='four columns div-user-controls',
                             children=[
                                 html.H2('ICU Intervention Simulation-Optimal Ventilator'),
                                 html.P('Visualising life vital time series signal: Heart Rate(HR), Mean blood pressure(MBP), Oxygen saturation(OS), Respiratory rate(RR), Diastolic blood pressure(DBP)'),
                                 html.P('Pick one or more vital signals to monitor.'),
                                 html.P('Pick one time to wear a ventilator for this patient to see if this increase the survival possibility.'),
                                 html.Div(
                                     className='div-for-dropdown',
                                     children=[
                                         dcc.Dropdown(id='vitalselector', options=get_options(df['vital'].unique()),
                                                      multi=True, value= [df['vital'].sort_values()[0]],
                                                      style={'backgroundColor': '#1E1E1E'},
                                                      className='vitalselector'
                                                      ),
                                     ],

                                     style={'color': '#1E1E1E'}),

                                 html.H2('Choose ventilator times'),
                                 dcc.Checklist(id='intervention_time',
                                     options=[{'label': '11/12/2010 00:00:00', 'value': '11/12/2010 00:00:00'},
                                              {'label': '11/12/2010 01:00:00', 'value': '11/12/2010 01:00:00'},
                                              {'label': '11/12/2010 02:00:00', 'value': '11/12/2010 02:00:00'},
                                             {'label': '11/12/2010 03:00:00', 'value': '11/12/2010 03:00:00'},
                                              {'label': '11/12/2010 04:00:00', 'value': '11/12/2010 04:00:00'},
                                              {'label': '11/12/2010 05:00:00', 'value': '11/12/2010 05:00:00'},
                                              {'label': '11/12/2010 06:00:00', 'value': '11/12/2010 06:00:00'},
                                              {'label': '11/12/2010 07:00:00', 'value': '11/12/2010 07:00:00'},
                                              {'label': '11/12/2010 08:00:00', 'value': '11/12/2010 08:00:00'},
                                              {'label': '11/12/2010 09:00:00', 'value': '11/12/2010 09:00:00'},
                                              {'label': '11/12/2010 10:00:00', 'value': '11/12/2010 10:00:00'},
                                              {'label': '11/12/2010 11:00:00', 'value': '11/12/2010 11:00:00'},
                                              {'label': '11/12/2010 12:00:00', 'value': '11/12/2010 12:00:00'},
                                              {'label': '11/12/2010 13:00:00', 'value': '11/12/2010 13:00:00'},
                                              {'label': '11/12/2010 14:00:00', 'value': '11/12/2010 14:00:00'},
                                              {'label': '11/12/2010 15:00:00', 'value': '11/12/2010 15:00:00'},
                                              {'label': '11/12/2010 16:00:00', 'value': '11/12/2010 16:00:00'},
                                              {'label': '11/12/2010 17:00:00', 'value': '11/12/2010 17:00:00'},
                                              {'label': '11/12/2010 18:00:00', 'value': '11/12/2010 18:00:00'},
                                              {'label': '11/12/2010 19:00:00', 'value': '11/12/2010 19:00:00'},
                                              {'label': '11/12/2010 20:00:00', 'value': '11/12/2010 20:00:00'},
                                              {'label': '11/12/2010 21:00:00', 'value': '11/12/2010 21:00:00'},
                                              {'label': '11/12/2010 22:00:00', 'value': '11/12/2010 22:00:00'},
                                              {'label': '11/12/2010 23:00:00', 'value': '11/12/2010 23:00:00'},
                                              {'label': '11/13/2010 00:00:00', 'value': '11/13/2010 00:00:00'}
                                              ],
                                     value=['11/12/2010 17:00:00']),

                                 html.Button(id='submit-button-state', n_clicks=0, children='Submit'),
                                 html.Div(id='output-state')
                                ]


                             ),
                    html.Div(className='eight columns div-for-charts bg-grey',
                             children=[
                                 dcc.Graph(id='timeseries', config={'displayModeBar': False}, animate=True),
                                 html.H2('RL Agent Optimal Ventilator Suggestion'),
                                 dcc.Graph(id="histogram"),
                             ])
                           ])
            ]
                    )

input01 = '83.77%'
input02 = '56.25%'
@app.callback(Output('output-state', 'children'),
              [Input('submit-button-state', 'n_clicks'),
               Input('intervention_time', 'value')]
              )

def update_output(n_clicks, intervention_time):
    # intervention_time

    if n_clicks == 1:
        possibility = input01
    elif n_clicks == 2:
        possibility = input02

        # vital_index = list(df.columns).index('vital')
        # value_index = list(df.columns).index('value')
        #
        # for i in len(df[df['Date'].keys() == '11/12/2010 17:00:00']['vital']):
        #     if df[df['Date'].keys() == '11/12/2010 17:00:00']['vital'][i] == 'OS':
        #         df[df['Date'].keys() == '11/12/2010 17:00:00']['value'][i] = 100

        # df = pd.read_csv('data/vitaldata2.csv', index_col=0, parse_dates=True)
        # os_df = df[df['vital'] == 'OS']
        # print('intervention_time: ', intervention_time)
        #
        # final_intervention_time_points = os_df[pd.to_datetime(os_df['Date'].values) == intervention_time[0]].index
        # print('final_intervention_time_points: ', final_intervention_time_points)
        #
        #
        # print('''df['value'][113]: ''', df['value'][114])
        # print('''df.loc[final_intervention_time_points, 'value']: ''', df.loc[final_intervention_time_points, 'value'])
        # df.loc[final_intervention_time_points, 'value'] = 10
        # print('''df['value'][113]: ''', df['value'][114])
        # print('''df.loc[final_intervention_time_points, 'value']: ''', df.loc[final_intervention_time_points, 'value'])
        # df.index = pd.to_datetime(df['Date'])

        # print('expected output: ', df.loc[final_intervention_time_points, 'value'])
    else:
        possibility = 'None'
    return u'''
        Mobility Possibility : {},
    '''.format(possibility)



# df[df['vital'] == 'OS' and df['date']==intervention_time].index
#
# # Callback for timeseries price
# @app.callback(Output('timeseries', 'figure'),
#               [Input('intervention_time', 'value')])
# def intervention_update_graph(intervention_time):
#     # intervention_time = '11/12/2010 17:00:00'
#     #intervention_time = pd.to_datetime(intervention_time)
#     df = pd.read_csv('data/vitaldata2.csv', index_col=0, parse_dates=True)
#     os_df = df[df['vital'] == 'OS']
#     final_intervention_time_points = os_df[pd.to_datetime(os_df['Date'].values) == intervention_time].index
#
#     df['value'][final_intervention_time_points] = 100
#
#     fig = update_graph('OS')
#     return None

# Callback for timeseries price
@app.callback(Output('timeseries', 'figure'),
              [Input('vitalselector', 'value')])
def update_graph(selected_dropdown_value):
    trace1 = []
    df_sub = df
    # print('after update graph expected output: ', df['value'][113])
    for vital in selected_dropdown_value:
        trace1.append(go.Scatter(x=df_sub[df_sub['vital'] == vital].index,
                                 y=df_sub[df_sub['vital'] == vital]['value'],
                                 mode='lines',
                                 opacity=0.7,
                                 name=vital,
                                 textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    print('data: ', data)
    figure = {'data': data,
              'layout': go.Layout(
                  colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                  template='plotly_dark',
                  paper_bgcolor='rgba(0, 0, 0, 0)',
                  plot_bgcolor='rgba(0, 0, 0, 0)',
                  margin={'b': 15},
                  hovermode='x',
                  autosize=True,
                  title={'text': 'vital values', 'font': {'color': 'white'}, 'x': 0.5},
                  xaxis={'range': [df_sub.index.min(), df_sub.index.max()]},
              ),

              }

    return figure


# Update Histogram Figure based on Month, Day and Times Chosen
@app.callback(
    Output("histogram", "figure"),
    [Input("submit-button-state", "n_clicks")],
)
def update_histogram(n_clicks):
    # date_picked = dt.strptime(datePicked, "%Y-%m-%d")
    # monthPicked = date_picked.month - 4
    # dayPicked = date_picked.day - 1
    #
    # [xVal, yVal, colorVal] = get_selection(monthPicked, dayPicked, selection)

    if n_clicks == 1:
        distribution = [0,2,3,0,1,2,2,3,4,5,6,7,6,5,7,8,10,15,12,5,2,1,2,3,5,3,5,3,5,4,5,2,1,2,3,4,5,6,5,5,6,5,4,8,5,7,5]
        distribution = np.array(distribution) / np.array(distribution).sum()

        yVal = list(distribution)
        xVal =list(range(len(yVal)))
    else:
        distribution = np.zeros(48)
        #distribution = np.array(distribution) / np.array(distribution).sum()
        yVal = list(distribution)
        xVal =list(range(len(yVal)))
    layout = go.Layout(
        bargap=0.01,
        bargroupgap=0,
        barmode="group",
        margin=go.layout.Margin(l=20, r=0, t=0, b=50),
        showlegend=False,
        plot_bgcolor="#323130",
        paper_bgcolor="#323130",
        dragmode="select",
        font=dict(color="white"),
        xaxis=dict(
            range=[-0.5, 48.5],
            showgrid=False,
            nticks=48,
            fixedrange=True,
            ticksuffix=":00",
        ),
        yaxis=dict(
            range=[0, max(yVal) + max(yVal) / 4],
            showticklabels=False,
            showgrid=False,
            fixedrange=True,
            rangemode="nonnegative",
            zeroline=False,
        ),
        annotations=[
            dict(
                x=xi,
                y=yi,
                text=str(round(yi, 3)),
                xanchor="center",
                yanchor="bottom",
                showarrow=False,
                font=dict(color="white"),
            )
            for xi, yi in zip(xVal, yVal)
        ],
    )

    return go.Figure(
        data=[
            go.Bar(x=xVal, y=yVal, marker=dict(color=colorVal), hoverinfo="x"),
            go.Scatter(
                opacity=0,
                x=xVal,
                y=yVal,
                hoverinfo="none",
                mode="markers",
                marker=dict(color="rgb(66, 134, 244, 0)", symbol="square", size=48),
                visible=True,
            ),
        ],
        layout=layout,
    )


if __name__ == '__main__':
    app.run_server(debug=True)