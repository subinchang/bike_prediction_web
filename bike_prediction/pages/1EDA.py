import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import calendar
import datetime
import os

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# 페이지 기본 설정
st.set_page_config(
    layout="wide",
)
st.title("⚒️ 전처리 및 EDA ⚒️")

@st.cache_data
def load_data():
    train = pd.read_csv("bike_prediction/kaggle_data/train.csv")
    test = pd.read_csv("bike_prediction/kaggle_data/test.csv")
    return train, test
train, test = load_data()

# train data가 어떻게 생겼는지
st.markdown('#### :blue[0. Check original data 💡]')
def highlight_cols_a(col):
    cols_to_highlight = ["casual", "registered"]
    if col.name in cols_to_highlight:
        return ["background-color: peachpuff"]
    else:
        return ["background-color: None"]
st.write(train.head().style.apply(highlight_cols_a, axis=0))

st.markdown('- 변수 중 casual, registered는 bike를 빌린 사용자가\
            회원인지 아닌지를 나타내는 사후변수이므로 예측 과정에서 사용할 수 있는 변수가 아니다.')

# 데이터 전처리
drop_cols = ['registered', 'casual']
train.drop(drop_cols, axis=1, inplace=True)

# 1. datetime handling
train["Day of week"] = train["datetime"].map(lambda x :  calendar.day_name[datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').date().weekday()])
train["Year"] = train["datetime"].map(lambda x :  datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').year)
train["Month"] = train["datetime"].map(lambda x :  datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').month)
train["Day"] = train["datetime"].map(lambda x :  datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').day)
train["Hour"] = train["datetime"].map(lambda x :  datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').hour)

test["Day of week"] = test["datetime"].map(lambda x :  calendar.day_name[datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').date().weekday()])
test["Year"] = test["datetime"].map(lambda x :  datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').year)
test["Month"] = test["datetime"].map(lambda x :  datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').month)
test["Day"] = test["datetime"].map(lambda x :  datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').day)
test["Hour"] = test["datetime"].map(lambda x :  datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').hour)

st.markdown('')
st.markdown('#### :blue[1. datetime 변환 💡]')
st.markdown('- 아래의 코드를 통해 datetime을 Year, Month, Day, Hour로 변경하였다.\n'
            '- 또한 datetime에서 요일을 별도로 추출하는 코드를 통해 Day of week라는 파생변수를 생성하였다.')
st.code("""
import calendar
import datetime

train["Day of week"] = train["datetime"].map(lambda x :  calendar.day_name[datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').date().weekday()])
train["Year"] = train["datetime"].map(lambda x :  datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').year)
train["Month"] = train["datetime"].map(lambda x :  datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').month)
train["Day"] = train["datetime"].map(lambda x :  datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').day)
train["Hour"] = train["datetime"].map(lambda x :  datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').hour)
""")
st.markdown('- 변환 이후의 data')
def highlight_cols(col):
    cols_to_highlight = ["datetime", "Day of week", "Year", "Month", "Day", "Hour"]
    if col.name in cols_to_highlight:
        return ["background-color: thistle"]
    else:
        return ["background-color: None"]
st.write(train.head().style.apply(highlight_cols, axis=0))

# 2. Month별 관측치 개수 분포
st.markdown('')
st.markdown("#### :blue[2. Month별 관측치 개수의 분포 확인 💡]")
st.markdown('- Year 또는 Month에 따라 관측치 수가 달라지는지 확인하기 위해 월별 관측치 수를 그래프로 확인해보았다.\n'
            '- 최대 22개(2011년 1월: 431개, 2012년 1월: 453개) 차이를 보였으나 5% 정도에 불과하여 특별한 조치 없이 데이터를 그대로 사용하였다.')
month = sorted(train.Month.unique())
x_2011 = list(train.loc[train.Year==2011, "Month"].value_counts().index)
y_2011 = list(train.loc[train.Year==2011, "Month"].value_counts().values)

x_2012 = list(train.loc[train.Year==2012, "Month"].value_counts().index)
y_2012 = list(train.loc[train.Year==2012, "Month"].value_counts().values)

fig = go.Figure()
fig.add_trace(go.Scatter(x=x_2011, y=y_2011, mode="markers", name="2011"))
fig.add_trace(go.Scatter(x=x_2012, y=y_2012, mode="markers", name="2012"))

fig.update_layout(title=dict({"text": "Monthly Record Count Distribution"}))
fig.update_layout(xaxis=dict({"tickvals": month,
                            "ticktext": [calendar.month_abbr[x] for x in month],
                            "title": "Month"}))
fig.update_layout(yaxis=dict({"title": "Count"}))

st.plotly_chart(fig, theme=None, use_container_width=True)


# 3. Day 변수 확인
st.markdown("#### :blue[3. Day 변수 확인 💡]")
st.markdown('- Train set과 Test set의 Day 변수 Unique 값을 살펴보았다.\n'
            '- Train set은 1일부터 19일의 데이터, Test set은 20일부터 말일까지의 데이터만 포함되어 있는 것을 알 수 있다.\n'
            '- 따라서 Day가 전혀 겹치지 않기 때문에 Day 변수는 사용하지 않았다.')
fig = go.Figure()
fig.add_trace(go.Scatter(x=train.Day.unique(), y=[20]*len(train.Hour.unique()), mode="markers", name="train"))
fig.add_trace(go.Scatter(x=test.Day.unique(), y=[10]*len(test.Hour.unique()), mode="markers", name="test"))

fig.update_layout(title=dict({"text": "Days in Train & Test set"}))
fig.update_layout(yaxis=dict({"tickvals": [10, 20],
                            "ticktext": ["test set", "train set"]}))
fig.update_layout(xaxis=dict({"title": "Days"}))
st.plotly_chart(fig, theme=None, use_container_width=True)

# 4. Month별 자전거 대여 건수 추이
st.markdown("#### :blue[4. Month별 자전거 대여 건수의 추이 확인 💡]")

tab1, tab2 = st.tabs(["📈 Total Count", "📉 Average Count"])
with tab1:
    month = sorted(train.Month.unique())
    cnt_2011 = [train.loc[(train.Month==m)&(train.Year==2011) , "count"].sum() for m in month]
    cnt_2012 = [train.loc[(train.Month==m)&(train.Year==2012) , "count"].sum() for m in month]
    cnt_all = [train.loc[train.Month==m, "count"].sum() for m in month]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=month, y=cnt_all, mode="lines", name="2011 & 2012"))
    fig.add_trace(go.Scatter(x=month, y=cnt_2012, mode="lines", name="2012"))
    fig.add_trace(go.Scatter(x=month, y=cnt_2011, mode="lines", name="2011"))

    fig.update_layout(title=dict({"text": "Monthly Rental Count Trend"}))
    fig.update_layout(xaxis=dict({"tickvals": month,
                                "ticktext": [calendar.month_abbr[x] for x in month],
                                "title": "Month"}))
    fig.update_layout(yaxis=dict({"title": "Monthly Rental Count"}))
    st.plotly_chart(fig, theme=None, use_container_width=True)
    st.markdown('- 2011년보다 2012년에 전체적으로 자전거 대여 건수가 많은 것을 확인할 수 있었다.\n'
                '- 따라서 Year를 사용하면 성능은 당연히 향상된다. 실제로 Kaggle에서도 높은 순위를 기록한 경우 Year를 포함한 경우를 다수 확인하였다.\n'
                '- 그러나 이는 out of sample data에는 적용할 수 없기 때문에 성능 향상만을 목적으로 하는 기존 모델과 달리 Year 변수를 모델링에서 제외하였다.\n')
with tab2:
    col1, col2 = st.columns([4,1])
    with col1:
        # 월별 평균 대여 건수
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=month, y=np.array(cnt_2012)//np.array(y_2012), mode="lines+markers", name="2012"))
        fig.add_trace(go.Scatter(x=month, y=np.array(cnt_2011)//np.array(y_2011), mode="lines+markers", name="2011"))
        fig.update_layout(title=dict({"text": "Monthly Rental Average Count"}))
        fig.update_layout(xaxis=dict({"tickvals": month,
                                    "ticktext": [calendar.month_abbr[x] for x in month]}))
        fig.update_layout(yaxis=dict({"title": "Average count"}))
        fig.update_layout(width=800, height=400)
        st.plotly_chart(fig, theme=None, use_container_width=False)

        # 월별 평균 기온
        Temps = [train.loc[train.Month==m, "temp"].mean() for m in month]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=month, y=Temps, mode="lines+markers", name="temperatuew"))
        fig.update_layout(title=dict({"text": "Monthly Average Temperature"}))
        fig.update_layout(xaxis=dict({"tickvals": month,
                                    "ticktext": [calendar.month_abbr[x] for x in month]}))
        fig.update_layout(yaxis=dict({"title": "Temperature"}))
        fig.update_layout(width=800, height=400)
        st.plotly_chart(fig, theme=None, use_container_width=False)
    with col2:
        st.title('')
        st.markdown('- 기온이 따뜻할 수록 자전거  \n 대여량이 늘어나는 것을  \n알 수 있다.\n'
                    '- 그러나 기온이 가장 높은  \n7월에는 자전거 대여 건수가 6, 8월에 비해 떨어지는 것을 확인하였다.')
        st.write(
            """<style>
            [data-testid="stHorizontalBlock"] {
            align-items: center;
            }
            </style>
            """,
            unsafe_allow_html=True)

# 5. season 확인
st.title('')
st.markdown("#### :blue[5. Season 변수 확인 💡]")

months = [list(train.loc[train.season==i+1, "Month"].unique()) for i in range(4)] #각 season에 따라서  month 가 어떻게 있는지
x1, x2, x3, x4 = months[0], months[1], months[2], months[3]

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=x1, y=np.zeros(len(x1)), name="season_1", mode="markers", marker_color="darkgreen"))
fig1.add_trace(go.Scatter(x=x2, y=np.zeros(len(x2))+5, name="season_2", mode="markers", marker_color="darkblue"))
fig1.add_trace(go.Scatter(x=x3, y=np.zeros(len(x3))+10, name="season_3", mode="markers", marker_color="darkred"))
fig1.add_trace(go.Scatter(x=x4, y=np.zeros(len(x4))+15, name="season_4", mode="markers", marker_color="darkorange"))
fig1.update_traces(mode="markers", marker_line_width=1, marker_size=10)

fig1.update_layout(title=dict({"text": "Season & Month"}))
fig1.update_layout(xaxis=dict({"tickvals": np.arange(1, 13, 1),
                            "ticktext": [calendar.month_abbr[x] for x in np.arange(1, 13, 1)],
                            "title": "Month"}))
fig1.update_layout(yaxis=dict({"tickvals": [0, 5, 10, 15],
                            "ticktext": ["season"+str(i) for i in range(1, 5)]}))


# season별 평균 count & 달별 평균 count
y_all = list(train["Month"].value_counts().values)
season_sum = [train.loc[train.season==s, "count"].sum() for s in sorted(train.season.unique()) ]
season_avg = [season_sum[s-1]//train.loc[train.season==s].shape[0] for s in sorted(train.season.unique())]

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=month, y=np.array(cnt_all)//np.array(list(y_all)), mode="lines+markers", name="Month_Avg"))
fig2.add_trace(go.Scatter(x=[2, 5, 8, 11], y=season_avg, mode="markers", name="Season_Avg"))
fig2.add_vrect(x0=1, x1=4, line_width=0, fillcolor="blue", opacity=0.2,
              annotation_text="Season_1", 
              annotation_position="bottom right",
              annotation_font_size=20,
              annotation_font_color="black",
              annotation_font_family="Times New Roman")
fig2.add_vrect(x0=4, x1=7, line_width=0, fillcolor="green", opacity=0.2,
              annotation_text="Season_2", 
              annotation_position="bottom right",
              annotation_font_size=20,
              annotation_font_color="black",
              annotation_font_family="Times New Roman")
fig2.add_vrect(x0=7, x1=10, line_width=0, fillcolor="orange", opacity=0.2,
              annotation_text="Season_3", 
              annotation_position="bottom right",
              annotation_font_size=20,
              annotation_font_color="black",
              annotation_font_family="Times New Roman")
fig2.add_vrect(x0=10, x1=12, line_width=0, fillcolor=" blueviolet", opacity=0.2,
              annotation_text="Season_4", 
              annotation_position="bottom right",
              annotation_font_size=20,
              annotation_font_color="black",
              annotation_font_family="Times New Roman")

fig2.update_layout(title=dict({"text": "Overall Average Count"}))
fig2.update_layout(xaxis=dict({"tickvals": month,
                            "ticktext": [calendar.month_abbr[x] for x in month],
                            "title": "Month"}))
fig2.update_layout(yaxis=dict({"title": "Avg count"}))

tab1, tab2 = st.tabs(["Season & Month", "Season별 Average Count"])
with tab1:
    st.markdown('- season=1이면 1월부터 3월, season=2이면 4월부터 6월로 나타나 있는 것을 원핫인코딩 처리하였다.')
    st.plotly_chart(fig1, theme=None, use_container_width=True)

with tab2:
    st.markdown('- 3개월 동안의 Season 전체 평균이 Monthly 평균과 크게 다르지 않음을 확인할 수 있다.')
    st.plotly_chart(fig2, theme=None, use_container_width=True)
    
# 6. Holiday, Workingday 처리
st.title('')
st.markdown("#### :blue[6. Holiday, Workingday 2개 컬럼을 하나의 컬럼으로 통합 💡]")

st.markdown('- holiday와 workingday 변수는 겹치는 정보가 있어 우선 의미를 이해하고자 했다. holiday는 공휴일 여부를 나타내고, workingday 해석은 다음과 같다.\n'
            '- **workingday=0**\n'
            '   - 주말도 공휴일도 아닌 날\n'
            '- **workingday=1**\n'
            '   - 주말이거나 공휴일인 날\n'
            '- 따라서 의미상으로 workingday와 holiday를 구분하여 다음과 같은 코드를 작성하였다.\n')

st.code("""
train["Day_info"] = np.nan*train.shape[0]
train.loc[train.workingday == 1, "Day_info"] = "working day"
train.loc[(train.workingday == 0)&(train.holiday == 1), "Day_info"] = "weekday&holiday"
train.loc[((train["Day of week"]=='Saturday') | (train["Day of week"]=='Sunday'))&(train.holiday==0),  "Day_info"] = "Weekend"
""")

st.markdown('- workingday = 1 인 경우에는 holiday가 자연히 0이기 때문에 workingday 변수를 그대로 유지하였다.\n'
            '- workingday = 0 인 경우에는\n'
            '   - 평일인데 공휴일인 weekday&holiday와\n'
            '   - 공휴일은 아닌데 일하지 않는 주말인 Weekend로 구분하였다.'
            )
train["Day_info"] = np.nan*train.shape[0]
train.loc[train.workingday == 1, "Day_info"] = "working day"
train.loc[(train.workingday == 0)&(train.holiday == 1), "Day_info"] = "weekday&holiday"
train.loc[((train["Day of week"]=='Saturday') | (train["Day of week"]=='Sunday'))&(train.holiday==0),  "Day_info"] = "Weekend"
st.write('')

col1, col2 = st.columns([1,3])
with col1:
    st.markdown(":orange[**Value counts of Day_info**]")
    st.write(train.Day_info.value_counts())

x = list(train.Day_info.unique())
day_info_cnt = [train.loc[train.Day_info==d, "count"].mean() for d in train.Day_info.unique()]

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=day_info_cnt, mode="lines+markers", name="temperatuew"))

fig.update_layout(title=dict({"text": "Day_info별 Average Rental Count"}))
fig.update_layout(xaxis=dict({"tickvals": x}), font=dict(size=15))
fig.update_layout(yaxis=dict({"title": "Avg count"}))
with col2:
    st.plotly_chart(fig, theme=None, use_container_width=True)

##########################################################
st.markdown("#### :blue[7. Weather 변수 확인 💡]")

x = list(train.weather.unique())
weather_cnt = [train.loc[train.weather==w, "count"].mean() for w in train.weather.unique()]

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=weather_cnt, mode="lines+markers", name="temperatuew"))

fig.update_layout(title=dict({"text": "Weather별 Average Rental Count"}))
fig.update_layout(xaxis=dict({"tickvals": x,
                              "ticktext": ['Clear, Few clouds', 
                                         'Mist, Mist+Cloudy', 
                                         'Light Snow, Light Rain + Thunder', 
                                         'Heavy Rain + Thunder + Mist']}), font=dict(size=15))
fig.update_layout(yaxis=dict({"title": "Avg count"}))
st.plotly_chart(fig, theme=None, use_container_width=True)
st.markdown('- 날씨가 안 좋아질 수록 Average Rental Count가 떨어지는 것을 볼 수 있다.\n'
            '- 그러나 4번째의 Heavy Rain + Thunder 가 아닌 3번째의 Light Rain + Thunder 의 평균 대여 건수가 가장 낮았다는 것은 특징적이다.\n'
            '- weather=4인 행을 추출해보았더니 행이 단 1개만 존재한다는 것을 확인하였다. 해당 행의 데이터는 다음과 같다.')
# st.write(train.loc[train.weather==4])
def highlight_cols_c(col):
    cols_to_highlight = ["weather"]
    if col.name in cols_to_highlight:
        return ["background-color: peachpuff"]
    else:
        return ["background-color: None"]
st.write(train.loc[train.weather==4].style.apply(highlight_cols_c, axis=0))

st.markdown('')
st.markdown("- 따라서 weather=4인 유일한 행의 값을 3으로 바꿔서 weather를 3개의 카테고리 값[good, soso, bad]을 가지는 컬럼으로 변경하였다.")
# weather이 1인것을 good, soso, bad로 카테고리형으로 change
train['weather'] = train['weather'].map({1: 'good', 
                                         2: 'soso', 
                                         3: 'bad'})

test['weather'] = test['weather'].map({1: 'good', 
                                         2: 'soso', 
                                         3: 'bad'})

st.write(train.weather.value_counts())

#############################################################
st.title('')
st.markdown("#### :blue[8. Hour 변수 확인 및 처리 💡]")
x = sorted(list(train.Hour.unique()))
hour_cnt = [train.loc[train.Hour==h, "count"].mean() for h in x]

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=hour_cnt, mode="lines+markers", name="temperatuew"))

fig.update_layout(title=dict({"text": "Hour별 Average Rental Count"}))
fig.update_layout(xaxis=dict({"tickvals": x,
                            "title": "Hour"}), font=dict(size=15))
fig.update_layout(yaxis=dict({"title": "Avg count"}))
st.plotly_chart(fig, theme=None, use_container_width=True)
st.markdown('- 시간대별 대여 건수의 경우, 오전 8시와 오후 17시에서 2개의 peak를 가짐을 확인할 수 있다.\n'
            '- 위 시간대는 일반적으로 통근 및 통학 시간대이기 때문에 대여 건수가 많았던 것으로 해석하였다.')
st.subheader('')
st.markdown('##### 1차원인 Hour를 2차원으로 변환')
st.markdown('- Hour를 그대로 사용하게 될 경우, 00시와 23시는 실제로는 1시간 차이이지만 23시간만큼의 차이가 발생하게 된다.\n'
            '- 또한, 이진 분류 알고리즘에서는 어떤 경우에도 00시와 23시가 같이 묶이는 일이 없게 된다.\n'
            '- Hour를 one hot encoding 할 수는 있지만, 23차원이 늘어나게 되므로 다른 방식을 시도하고자 하였다.\n'
            '- 첫 번째로 시간의 주기성을 반영하기 위해 sin_hour 변수를 생성하였다.\n'
            '- 그러나 sin_hour만 사용할 경우 다른 시간이 같은 값을 갖는 일이 생겨서 두 번째로 cos_hour까지 추가하여 시간의 주기성 및 고유성을 모두 보존하였다.\n')
st.subheader('')
st.markdown('변환에 사용한 코드와, 시간대별로 sin_hour와 cos_hour 값을 나타낸 그림은 아래와 같다.')
st.code("""
train['sin_hour'] = np.sin(2*np.pi*train.Hour/24)
train['cos_hour'] = np.cos(2*np.pi*train.Hour/24)
""")

train['sin_hour'] = np.sin(2*np.pi*train.Hour/24)
train['cos_hour'] = np.cos(2*np.pi*train.Hour/24)
fig = px.scatter(train, x='sin_hour', y='cos_hour',
                 hover_data=["Hour"], width=400, height=400)
st.plotly_chart(fig, theme=None)
