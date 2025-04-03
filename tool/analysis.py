import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import altair as alt
from itertools import combinations
import networkx as nx
from wordcloud import WordCloud
import io
from pgmpy.estimators import PC
import networkx as nx
import os
import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from collections import Counter
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
os.environ["DISABLE_STREAMLIT_WATCHER"] = "1"
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 适用于 Windows
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题
from typing import Dict, Any
"""
学生知识点分析工具代码注释说明

本工具提供多维度的学生知识点分析功能，包含频率分析、共现分析、时序分析、因果推断等模块。
各模块函数均可独立调用，输入数据格式统一，输出包含可视化图表和结构化数据。

主要数据结构说明：
- 原始数据应为包含"knowledge_points"和"timestamp"列的DataFrame
- knowledge_points列格式：列表或逗号分隔字符串，如["知识点A", "知识点B"] 或 "知识点A, 知识点B"
- timestamp列格式：可转换为datetime的时间戳字符串

代码结构：
1. 基础统计分析模块
2. 时间分析模块
3. 知识点共现分析模块
4. 因果推断模块
5. 学习行为分析模块
6. 个性化反馈模块
7. 记忆持久性分析模块
"""
# ---------------------------
# 1. 基础统计分析模块
# ---------------------------
def knowledge_frequency_analysis(data):
    """
    知识点频率统计分析

    参数:
        data: list[list] - 二维列表，每个子列表表示单次提问涉及的知识点
        示例: [["三角形", "向量"], ["向量", "导数"]]

    返回:
        Counter对象 - 各知识点出现频次
        示例: Counter({"向量":2, "三角形":1, "导数":1})

    功能:
        1. 扁平化处理二维知识点列表
        2. 使用Counter统计知识点出现次数
    """
    flat_list = [kp for record in data for kp in record]
    freq = Counter(flat_list)
    return freq

def plot_frequency_table(freq):
    """
    可视化知识点频率表格

    参数:
        freq: Counter对象 - 知识点频率数据

    功能:
        1. 将Counter转换为DataFrame
        2. 使用Streamlit展示排序后的频率表格
        3. 支持动态交互排序
    """
    df_freq = pd.DataFrame({
        '知识点': list(freq.keys()),
        '出现频次': list(freq.values())
    }).sort_values(by='出现频次', ascending=False).reset_index(drop=True)

    st.dataframe(
        df_freq,
        column_config={
            "知识点": st.column_config.TextColumn("知识点"),
            "出现频次": st.column_config.NumberColumn("出现频次", format="%d")
        },
        use_container_width=True,
        hide_index=True
    )

def plot_top_frequency_bar(freq, top_n=10):
    """
    绘制知识点频率TOP10柱状图

    参数:
        freq: Counter对象 - 知识点频率数据
        top_n: 显示前N个知识点

    功能:
        1. 提取TOP N知识点
        2. 使用Altair绘制交互式柱状图
        3. 支持鼠标悬停查看数值
    """
    top_items = dict(sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n])
    df_chart = pd.DataFrame({
        '知识点': list(top_items.keys()),
        '频次': list(top_items.values())
    })

    chart = alt.Chart(df_chart).mark_bar().encode(
        x=alt.X('知识点:N', sort='-y'),
        y=alt.Y('频次:Q'),
        color=alt.Color('知识点:N', legend=None),
        tooltip=['知识点', '频次']
    ).properties(
        title='知识点频率TOP10',
        width=600,
        height=400
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

def plot_top_frequency_pie(freq, top_n=10):
    top_items = dict(sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n])
    fig = px.pie(
        names=list(top_items.keys()),
        values=list(top_items.values()),
        title='知识点频率TOP10占比',
        hole=0.3,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
    st.plotly_chart(fig, use_container_width=True)

def plot_frequency_wordcloud_streamlit(freq):
    try:
        wc = WordCloud(
            font_path="msyh.ttc",  # 根据实际情况调整
            width=800,
            height=400,
            background_color='white',
            max_words=100,
            contour_width=1,
            contour_color='steelblue'
        )
        wc.generate_from_frequencies(freq)
        img = wc.to_image()
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        st.image(buf, caption='知识点词云图', use_column_width=True)
    except Exception as e:
        st.error(f"生成词云图时出错: {e}")
        st.info("检查字体路径是否正确或系统中是否安装了'msyh.ttc'字体。")

# ---------------------------
# 2. 时间分析模块
# ---------------------------

def analyze_daily_knowledge_composition(df):
    """
    每日知识点组成分析

    参数:
        df: 包含timestamp和knowledge_points列的DataFrame

    功能:
        1. 按天聚合知识点数据
        2. 统计每日知识点分布
        3. 绘制堆叠柱状图展示每日知识点组成变化
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date

    daily_data = []
    for date, group in df.groupby('date'):
        day_kps = []
        for kps in group['knowledge_points']:
            # 如果知识点已是列表则直接添加，否则按逗号分割
            if isinstance(kps, list):
                day_kps.extend(kps)
            else:
                day_kps.extend([kp.strip() for kp in kps.split(',')])
        kp_count = Counter(day_kps)
        top_kps = dict(kp_count.most_common(5))
        for kp, count in top_kps.items():
            daily_data.append({'日期': date, '知识点': kp, '频次': count})
    if not daily_data:
        st.warning("没有足够的日期数据进行分析")
        return

    df_daily = pd.DataFrame(daily_data)
    fig = px.bar(
        df_daily,
        x='日期',
        y='频次',
        color='知识点',
        title='每日提问知识点组成',
        barmode='stack'
    )
    fig.update_layout(
        xaxis_title='日期',
        yaxis_title='提问频次',
        legend_title='知识点',
        hovermode='x'
    )
    st.plotly_chart(fig, use_container_width=True,key="unique_chart_daily")

def analyze_time_preference(df):
    """
    学习时间偏好分析

    参数:
        df: 包含timestamp列的DataFrame

    功能:
        1. 将时间划分为上午/下午/晚上三个时段
        2. 统计各时段提问数量分布
        3. 识别学习高峰时段
        4. 生成时间管理建议
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    def get_time_period(hour):
        if 5 <= hour < 12:
            return '上午 (5:00-11:59)'
        elif 12 <= hour < 18:
            return '下午 (12:00-17:59)'
        else:
            return '晚上 (18:00-4:59)'
    df['time_period'] = df['hour'].apply(get_time_period)
    period_counts = df['time_period'].value_counts().reset_index()
    period_counts.columns = ['时间段', '提问数量']
    # 自然语言分析
    most_active_period = period_counts.iloc[0]['时间段']
    hour_counts = df['hour'].value_counts().sort_index()
    peak_hour = hour_counts.idxmax()
    analysis_text = f"""
       ### 📝 时间分析结论：
       - **最活跃时段**：{most_active_period}（占全天提问量的{period_counts.iloc[0]['提问数量'] / len(df) * 100:.1f}%）
       - **提问高峰时刻**：{peak_hour}:00 左右
       - **学习时间分布**：{'均匀' if hour_counts.max() / hour_counts.min() < 2 else '集中'}

       ### 🚀 个性化建议：
       1. 在{peak_hour - 1}-{peak_hour + 1}点的高效时段进行难点学习
       2. 利用{most_active_period}进行知识复盘
       3. 在低活跃时段安排预习性学习
       """
    fig_pie = px.pie(
        period_counts,
        names='时间段',
        values='提问数量',
        title='提问时间段分布',
        color='时间段',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_pie.update_traces(textinfo='percent+label', pull=[0.05, 0, 0])

    hour_counts = df['hour'].value_counts().sort_index().reset_index()
    hour_counts.columns = ['小时', '提问数量']
    fig_hour = px.bar(
        hour_counts,
        x='小时',
        y='提问数量',
        title='各小时提问数量分布',
        labels={'小时': '时间 (小时)', '提问数量': '提问数量'},
        color='小时',
        color_continuous_scale='Viridis'
    )
    fig_hour.add_vrect(x0=5, x1=12, fillcolor="green", opacity=0.1, layer="below", line_width=0, annotation_text="上午")
    fig_hour.add_vrect(x0=12, x1=18, fillcolor="blue", opacity=0.1, layer="below", line_width=0, annotation_text="下午")
    fig_hour.add_vrect(x0=18, x1=24, fillcolor="purple", opacity=0.1, layer="below", line_width=0, annotation_text="晚上")
    fig_hour.add_vrect(x0=0, x1=5, fillcolor="purple", opacity=0.1, layer="below", line_width=0)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        st.plotly_chart(fig_hour, use_container_width=True)
    st.markdown(analysis_text)
# ---------------------------
# 3. 知识点共现分析模块
# ---------------------------
def build_cooccurrence_matrix(data):
    """
       构建知识点共现矩阵

       参数:
           data: list[list] - 二维知识点列表

       返回:
           tuple: (共现矩阵DataFrame, 知识点列表)

       示例:
           输入: [["A","B"], ["B","C"]]
           输出:
               DataFrame:
                   A  B  C
                 A 1  1  0
                 B 1  2  1
                 C 0  1  1
               ["A","B","C"]
       """
    all_knowledge_points = sorted(set([kp for record in data for kp in record]))
    cooccurrence_matrix = pd.DataFrame(0, index=all_knowledge_points, columns=all_knowledge_points)
    for record in data:
        valid_kps = [kp for kp in record if kp]
        for i, kp1 in enumerate(valid_kps):
            for kp2 in valid_kps[i:]:
                cooccurrence_matrix.loc[kp1, kp2] += 1
                if kp1 != kp2:
                    cooccurrence_matrix.loc[kp2, kp1] += 1
    return cooccurrence_matrix, all_knowledge_points


def analyze_knowledge_cooccurrence(data:list):
    """
       知识点共现综合分析

       参数:
           data: list[list] - 二维知识点列表

       功能:
           1. 构建共现矩阵
           2. 识别高频共现对
           3. 检测知识社区
           4. 可视化共现热力图
           5. 生成教学建议
       """
    cooccurrence_matrix, all_kps = build_cooccurrence_matrix(data)

    # 自然语言描述分析
    cooccurrence_pairs = []
    for i, kp1 in enumerate(all_kps):
        for j, kp2 in enumerate(all_kps):
            if i < j and cooccurrence_matrix.loc[kp1, kp2] > 0:
                cooccurrence_pairs.append((kp1, kp2, cooccurrence_matrix.loc[kp1, kp2]))

    top_pairs = sorted(cooccurrence_pairs, key=lambda x: x[2], reverse=True)[:5]

    # 构建知识网络进行社区检测
    G = nx.Graph()
    for kp in all_kps:
        G.add_node(kp)
    for kp1, kp2, count in cooccurrence_pairs:
        if count >= 2:  # 筛选显著共现关系
            G.add_edge(kp1, kp2, weight=count)

    communities = nx.algorithms.community.greedy_modularity_communities(G)

    # 生成自然语言分析
    analysis_text = "### 📝 共现分析结论\n"

    # 高频组合分析
    analysis_text += "#### 高频知识组合：\n"
    for pair in top_pairs:
        analysis_text += f"- **{pair[0]}** 和 **{pair[1]}** 共同出现 {pair[2]} 次（建议加强组合练习）\n"

    # 知识群落分析
    analysis_text += "\n#### 知识模块识别：\n"
    for i, comm in enumerate(communities[:3]):  # 显示前3个主要社区
        analysis_text += f"\n**模块{i + 1}**：{', '.join(list(comm)[:5])}{'等' if len(comm) > 5 else ''}\n"

    # 薄弱环节识别
    degree_centrality = nx.degree_centrality(G)
    low_degree_kps = sorted(degree_centrality.items(), key=lambda x: x[1])[:3]
    analysis_text += "\n#### 需关注知识点：\n"
    for kp, score in low_degree_kps:
        analysis_text += f"- **{kp}**（关联度较低，建议加强与其他知识的联系）\n"

    # 教学建议
    analysis_text += "\n#### 🚀 学习建议：\n"
    analysis_text += "1. 优先掌握高频组合中的核心知识\n"
    analysis_text += "2. 按知识模块进行系统化复习\n"
    analysis_text += "3. 为薄弱知识点设计专项练习\n"

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📊 知识点共现频率表")
        if cooccurrence_pairs:
            df_cooccurrence = pd.DataFrame(cooccurrence_pairs, columns=['知识点1', '知识点2', '共现频次'])
            st.dataframe(df_cooccurrence.sort_values('共现频次', ascending=False),
                         height=400,
                         use_container_width=True)

    with col2:
        st.subheader("🔥 知识点共现热力图")
        fig = px.imshow(cooccurrence_matrix, text_auto=True, aspect="auto")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(analysis_text)

def build_knowledge_network(data, freq, min_cooccurrence=1, max_nodes=20):
    cooccurrence_matrix, all_kps = build_cooccurrence_matrix(data)
    top_kps = [kp for kp, count in freq.most_common(max_nodes)]
    subset_matrix = cooccurrence_matrix.loc[top_kps, top_kps]
    G = nx.Graph()
    for kp in top_kps:
        G.add_node(kp, frequency=freq[kp])
    for i, kp1 in enumerate(top_kps):
        for j, kp2 in enumerate(top_kps):
            if i < j:
                weight = subset_matrix.loc[kp1, kp2]
                if weight >= min_cooccurrence:
                    G.add_edge(kp1, kp2, weight=weight)
    edge_x = []
    edge_y = []
    pos = nx.spring_layout(G, seed=42)
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')
    node_x = []
    node_y = []
    node_text = []
    node_sizes = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_sizes.append(G.nodes[node]['frequency'])
        node_text.append(f"{node}<br>频率: {G.nodes[node]['frequency']}")
    if node_sizes:
        max_size = max(node_sizes)
        node_sizes = [20 + (size / max_size) * 50 for size in node_sizes]
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=list(G.nodes()),
        hovertext=node_text,
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=node_sizes,
            color=[G.nodes[node]['frequency'] for node in G.nodes()],
            colorbar=dict(
                thickness=15,
                title='知识点频率',
                xanchor='left',
                title_side='right'
            ),
            line=dict(width=2)
        )
    )
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='知识点共现网络',
                        titlefont=dict(size=16),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=600
                    ))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### 网络分析结果")
    if len(G.nodes()) > 0:
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        most_central_kp = max(degree_centrality.items(), key=lambda x: x[1])[0]
        most_betweenness_kp = max(betweenness_centrality.items(), key=lambda x: x[1])[0]
        col1, col2 = st.columns(2)
        with col1:
            st.metric("网络中的知识点数量", len(G.nodes()))
            st.metric("网络中的连接关系数量", len(G.edges()))
            st.metric("网络密度", round(nx.density(G), 3))
        with col2:
            st.metric("最核心知识点(度中心性)", most_central_kp)
            st.metric("最重要桥接知识点(中介中心性)", most_betweenness_kp)
            if len(G.nodes()) >= 3:
                communities = nx.community.greedy_modularity_communities(G)
                st.metric("知识点社区数量", len(communities))
    else:
        st.info("网络中没有足够的节点进行分析")
# ---------------------------
# 4. 因果推断与知识图谱模块
# ---------------------------

def preprocess_causal_data(data, max_nodes=50):
    """预处理因果分析数据（增强健壮性）"""
    try:
        # 数据清洗
        clean_data = [
            [kp.strip() for kp in record if kp.strip()]
            for record in data
            if isinstance(record, (list, tuple))
        ]
        clean_data = [record for record in clean_data if record]

        # 频率统计
        freq = Counter(kp for record in clean_data for kp in record)
        top_kps = [kp for kp, _ in freq.most_common(max_nodes)]

        if not top_kps:
            st.error("无有效知识点进行分析")
            return None, None

        # 创建布尔型特征矩阵
        df = pd.DataFrame(False, columns=top_kps, index=range(len(clean_data)))
        for i, record in enumerate(clean_data):
            df.loc[i] = [kp in record for kp in top_kps]

        # 强制列顺序一致性
        df = df.reindex(columns=top_kps)
        return df.astype(bool), top_kps
    except Exception as e:
        st.error(f"数据预处理失败: {str(e)}")
        return None, None


@st.cache_data
def causal_discovery(_df, alpha=0.01, method='pearson'):
    """使用PC算法进行因果发现"""
    """
     因果知识图谱构建

     参数:
         df: 预处理后的布尔型DataFrame，列表示知识点

     返回:
         nx.DiGraph - 因果有向图

     算法:
         使用PC算法进行因果发现:
         1. 构建无向骨架
         2. 定向V型结构
         3. 定向剩余边
     """
    est = PC(_df)
    model = est.estimate(variant="stable",
                         ci_test="chi_square",
                         alpha=alpha,
                         return_type="dag")

    # 确保模型节点名称与列名一致
    model.nodes = _df.columns.tolist()
    return model


def build_causal_knowledge_graph(model, feature_names):
    """构建因果知识图谱（增强错误处理）"""
    try:
        G = nx.DiGraph()

        # 验证节点一致性
        model_nodes = model.nodes  # 正确访问节点属性（非方法调用）
        if not set(model_nodes) == set(feature_names):
            missing = set(feature_names) - set(model_nodes)
            extra = set(model_nodes) - set(feature_names)
            st.error(f"节点不匹配:\n缺失节点: {missing}\n额外节点: {extra}")
            return None

        # 添加节点和边
        G.add_nodes_from(model_nodes)
        G.add_edges_from(model.edges)

        return G
    except AttributeError as e:
        st.error(f"模型结构异常: {str(e)}")
        return None
    except Exception as e:
        st.error(f"图谱构建失败: {str(e)}")
        return None


def plot_causal_graph(G):
    """交互式可视化因果图谱"""
    pos = nx.spring_layout(G, seed=42)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            size=20,
            color='lightblue',
            line=dict(width=2, color='darkblue')))

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='因果知识图谱',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=600
                    ))
    st.plotly_chart(fig, use_container_width=True)


def analyze_causal_relationships(data):
    """因果分析主函数"""
    st.header("🧠 因果知识图谱分析")

    with st.expander("🔍 分析方法说明", expanded=True):
        st.markdown("""
        ### 因果发现流程：
        1. **特征选择**：选取高频知识点（Top30）
        2. **条件独立性检验**：使用卡方检验（α=0.01）
        3. **骨架学习**：构建无向因果骨架
        4. **方向确定**：基于时序数据和V型结构
        5. **图谱构建**：生成有向无环图(DAG)

        ### 教学解读指南：
        - **箭头方向**：表示可能的因果关系（A→B 表示A影响B的掌握）
        - **枢纽节点**：多个入度的知识点可能需要前置强化
        - **孤立节点**：可能需要单独教学模块
        - **长路径**：提示核心知识链条
        """)

    # 参数设置
    col1, col2 = st.columns(2)
    with col1:
        max_nodes = st.slider("最大分析知识点数", 10, 50, 30, key="causal_max_nodes")
    with col2:
        alpha = st.slider("显著性水平α", 0.001, 0.1, 0.01, step=0.005)

    # 因果分析
    with st.spinner('正在进行因果发现...'):
        df_causal, features = preprocess_causal_data(data, max_nodes)
        model = causal_discovery(df_causal, alpha=alpha)
        G = build_causal_knowledge_graph(model, features)

    # 可视化
    plot_causal_graph(G)

    # 结构分析
    st.subheader("📌 关键结构分析")
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())

    causal_chains = []
    for node in G.nodes():
        if in_degrees[node] == 0 and out_degrees[node] > 1:
            causal_chains.append(node)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总知识点数", len(G.nodes()))
        st.metric("源头知识点", len([n for n, d in in_degrees.items() if d == 0]))
    with col2:
        st.metric("平均因果链长度",
                  round(np.mean([len(nx.dag_longest_path(G))]), 1))
        st.metric("关键枢纽点", len(causal_chains))
    with col3:
        st.metric("最大入度", max(in_degrees.values()))
        st.metric("最大出度", max(out_degrees.values()))

    if causal_chains:
        st.markdown(f"**主要因果链起点**: {', '.join(causal_chains[:3])}")
        # 新增自然语言分析
    if G and len(G.nodes) > 0:
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())

        # 关键节点识别
        source_nodes = [n for n, d in in_degrees.items() if d == 0]
        hub_nodes = [n for n, d in out_degrees.items() if d >= 3]

        longest_path = nx.dag_longest_path(G)[:3]  # 取最长因果链的前3个
        analysis_text = f"""
                   ### 📝 因果分析结论：
                   - **知识基础节点**：{', '.join(source_nodes[:3])}
                   - **核心枢纽节点**：{', '.join(hub_nodes[:3])}
                   - **最长因果链**：{'→'.join(longest_path)}（深度 {len(nx.dag_longest_path(G))} 级）

                   ### 🚀 学习路径建议：
                   1. 优先掌握基础节点：{source_nodes[0] if source_nodes else ""}
                   2. 重点突破枢纽节点：{hub_nodes[0] if hub_nodes else ""}
                   3. 按因果链顺序学习：{' -> '.join(longest_path + [""] * (3 - len(longest_path)))}
                   """

        st.markdown(analysis_text)

# ---------------------------
# 5.学习行为分析模块
# ---------------------------

def analyze_learning_sessions(df):
    """
    学习会话分析

    参数:
        df: 包含timestamp列的DataFrame

    返回:
        tuple: (原始DataFrame, 会话统计DataFrame)

    功能:
        1. 根据30分钟无活动划分会话
        2. 统计会话时长、提问数等指标
        3. 分析会话时间分布模式
    """
    # 确保时间戳格式正确
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    # 定义会话间隔阈值（30分钟无活动视为新会话）
    SESSION_THRESHOLD = pd.Timedelta(minutes=30)

    # 计算时间差并标记会话
    df['time_diff'] = df['timestamp'].diff()
    df['new_session'] = (df['time_diff'] > SESSION_THRESHOLD) | (df['time_diff'].isna())
    df['session_id'] = df['new_session'].cumsum()

    # 计算每个会话的统计信息
    session_stats = df.groupby('session_id').agg(
        session_start=('timestamp', 'min'),
        session_end=('timestamp', 'max'),
        session_duration=('timestamp', lambda x: (x.max() - x.min()).total_seconds() / 60),
        questions_count=('timestamp', 'count'),
        unique_knowledge_points=('knowledge_points', lambda x: len(set(kp for sublist in x for kp in sublist)))
    ).reset_index()

    # 添加会话时段分类
    def get_time_category(time):
        hour = time.hour
        if 5 <= hour < 9:
            return '早晨 (5:00-8:59)'
        elif 9 <= hour < 12:
            return '上午 (9:00-11:59)'
        elif 12 <= hour < 14:
            return '中午 (12:00-13:59)'
        elif 14 <= hour < 18:
            return '下午 (14:00-17:59)'
        elif 18 <= hour < 22:
            return '晚上 (18:00-21:59)'
        else:
            return '深夜 (22:00-4:59)'

    session_stats['time_category'] = session_stats['session_start'].apply(get_time_category)

    # 可视化会话分布
    st.subheader("📊 学习会话时段分布")
    time_category_counts = session_stats['time_category'].value_counts().reset_index()
    time_category_counts.columns = ['时段', '会话数量']

    # 对时段按一天的时间顺序排序
    time_order = ['早晨 (5:00-8:59)', '上午 (9:00-11:59)', '中午 (12:00-13:59)',
                  '下午 (14:00-17:59)', '晚上 (18:00-21:59)', '深夜 (22:00-4:59)']
    time_category_counts['时段'] = pd.Categorical(time_category_counts['时段'], categories=time_order, ordered=True)
    time_category_counts = time_category_counts.sort_values('时段')

    fig = px.bar(
        time_category_counts,
        x='时段',
        y='会话数量',
        color='时段',
        title='学习会话时段分布',
        labels={'时段': '时间段', '会话数量': '会话数量'}
    )
    return df, session_stats


def analyze_knowledge_learning_curve(df):
    """
    分析知识点学习曲线，跟踪学习进度和关注点变化

    参数:
        df: 包含timestamp和knowledge_points的DataFrame
    """
    st.subheader("📈 知识点关注度变化趋势")

    # 确保时间戳格式化并按周聚合
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['week'] = df['timestamp'].dt.strftime('%Y-%U')

    # 展开知识点列表并计算每周知识点频率
    weekly_kp_data = []
    for week, group in df.groupby('week'):
        week_kps = []
        for kps in group['knowledge_points']:
            if isinstance(kps, list):
                week_kps.extend(kps)
            else:
                week_kps.extend([kp.strip() for kp in kps.split(',')])

        # 获取本周Top5知识点
        top_kps = Counter(week_kps).most_common(5)
        for kp, count in top_kps:
            weekly_kp_data.append({
                '周': week,
                '知识点': kp,
                '提问次数': count
            })

    if not weekly_kp_data:
        st.warning("没有足够的周数据进行趋势分析")
        return

    weekly_df = pd.DataFrame(weekly_kp_data)

    # 计算所有周的Top10知识点
    all_kps = [kp for kps in df['knowledge_points'] for kp in
               (kps if isinstance(kps, list) else [k.strip() for k in kps.split(',')])]
    top10_kps = [kp for kp, _ in Counter(all_kps).most_common(10)]

    # 过滤只显示Top10知识点的趋势
    top_kp_trends = weekly_df[weekly_df['知识点'].isin(top10_kps)]

    if len(top_kp_trends) > 0:
        # 创建知识点学习趋势图
        fig = px.line(
            top_kp_trends,
            x='周',
            y='提问次数',
            color='知识点',
            markers=True,
            title='主要知识点关注度周趋势',
            labels={'周': '周次', '提问次数': '提问频次'}
        )
        fig.update_layout(xaxis_title='周次', yaxis_title='提问频次', legend_title='知识点')
        st.plotly_chart(fig, use_container_width=True)

        # 热门知识点转移图
        st.subheader("🔄 学习焦点转移分析")

        # 为每周找出最热门知识点
        weekly_hot_kps = []
        for week, group in weekly_df.groupby('周'):
            top_kp = group.loc[group['提问次数'].idxmax()]
            weekly_hot_kps.append({
                '周次': week,
                '热点知识点': top_kp['知识点'],
                '提问次数': top_kp['提问次数']
            })

        hot_kps_df = pd.DataFrame(weekly_hot_kps)

        # 创建热门知识点转移表格
        st.dataframe(
            hot_kps_df,
            column_config={
                "周次": st.column_config.TextColumn("周次"),
                "热点知识点": st.column_config.TextColumn("热点知识点"),
                "提问次数": st.column_config.NumberColumn("提问次数", format="%d")
            },
            use_container_width=True,
            hide_index=True
        )

        # 提取学习行为见解
        if len(hot_kps_df) > 1:
            focus_changes = sum(1 for i in range(len(hot_kps_df) - 1)
                                if hot_kps_df.iloc[i]['热点知识点'] != hot_kps_df.iloc[i + 1]['热点知识点'])

            st.markdown(f"""
            ### 📝 学习行为分析：

            - **关注点稳定性**: {'较低' if focus_changes / len(hot_kps_df) > 0.5 else '较高'}
            - **焦点转移频率**: {focus_changes} 次转移 / {len(hot_kps_df)} 周
            - **学习风格推断**: {'可能偏向于广泛学习多个主题' if focus_changes / len(hot_kps_df) > 0.5 else '可能专注于系统掌握少量主题'}
            """)


def analyze_learning_intensity(df):
    """
    分析学习强度变化和学习规律

    参数:
        df: 包含timestamp和knowledge_points的DataFrame
    """
    st.subheader("🔥 学习强度分析")

    # 确保时间戳格式正确
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 按日期统计提问量
    df['date'] = df['timestamp'].dt.date
    daily_counts = df.groupby('date').size().reset_index(name='提问数量')
    daily_counts['date'] = pd.to_datetime(daily_counts['date'])
    daily_counts['星期'] = daily_counts['date'].dt.day_name()

    # 添加星期几的排序
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_map = {
        'Monday': '周一', 'Tuesday': '周二', 'Wednesday': '周三',
        'Thursday': '周四', 'Friday': '周五', 'Saturday': '周六', 'Sunday': '周日'
    }
    daily_counts['星期'] = daily_counts['星期'].map(weekday_map)
    daily_counts['星期序号'] = daily_counts['date'].dt.dayofweek

    # 创建学习强度趋势图
    fig = px.line(
        daily_counts,
        x='date',
        y='提问数量',
        title='每日学习强度变化趋势',
        labels={'date': '日期', '提问数量': '提问数量'}
    )
    fig.update_layout(xaxis_title='日期', yaxis_title='提问数量')
    st.plotly_chart(fig, use_container_width=True)

    # 按星期几分组查看学习习惯
    weekday_stats = daily_counts.groupby('星期').agg(
        平均提问数=('提问数量', 'mean'),
        最大提问数=('提问数量', 'max'),
        最小提问数=('提问数量', 'min')
    ).reset_index()

    # 确保按星期几排序
    weekday_stats['排序'] = weekday_stats['星期'].map({v: k for k, v in enumerate(map(weekday_map.get, weekday_order))})
    weekday_stats = weekday_stats.sort_values('排序').drop('排序', axis=1)

    # 工作日vs周末对比
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📅 按星期几学习强度")
        fig = px.bar(
            weekday_stats,
            x='星期',
            y='平均提问数',
            color='星期',
            title='星期几平均学习强度',
            text='平均提问数',
            labels={'星期': '星期', '平均提问数': '平均提问数量'}
        )
        fig.update_layout(xaxis={'categoryorder': 'array', 'categoryarray': list(map(weekday_map.get, weekday_order))})
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 工作日vs周末
        daily_counts['是否周末'] = daily_counts['星期'].isin(['周六', '周日']).map({True: '周末', False: '工作日'})
        weekend_vs_weekday = daily_counts.groupby('是否周末')['提问数量'].mean().reset_index()

        fig = px.pie(
            weekend_vs_weekday,
            values='提问数量',
            names='是否周末',
            title='工作日 vs 周末平均学习强度',
            color='是否周末',
            color_discrete_map={'工作日': '#3366CC', '周末': '#DC3912'},
            hole=0.4
        )
        fig.update_traces(textinfo='percent+label+value')
        st.plotly_chart(fig, use_container_width=True)

    # 计算学习规律性指标
    cv = daily_counts['提问数量'].std() / daily_counts['提问数量'].mean() if daily_counts['提问数量'].mean() > 0 else 0
    weekend_ratio = weekend_vs_weekday[weekend_vs_weekday['是否周末'] == '周末']['提问数量'].values[0] / \
                    weekend_vs_weekday[weekend_vs_weekday['是否周末'] == '工作日']['提问数量'].values[0] \
        if '工作日' in weekend_vs_weekday['是否周末'].values and \
           weekend_vs_weekday[weekend_vs_weekday['是否周末'] == '工作日']['提问数量'].values[0] > 0 else 0

    most_active_day = weekday_stats.loc[weekday_stats['平均提问数'].idxmax()]['星期']

    st.markdown(f"""
    ### 💡 学习规律分析：

    - **学习规律性指数**: {(1 - min(cv, 1)) * 100:.1f}% (越高表示学习强度越稳定)
    - **周末/工作日比例**: {weekend_ratio:.2f} (>1表示周末学习更多，<1表示工作日学习更多)
    - **最活跃学习日**: {most_active_day}
    - **建议**: {"建议更均衡分配学习时间，保持稳定的学习节奏" if cv > 0.5 else "当前学习强度分配较为均衡，建议保持"}
    """)


# ---------------------------
# 6. 个性化反馈与预测模块
# ---------------------------

def create_learning_profile(df, data):
    """
    创建学生个性化学习画像

    参数:
        df: 包含timestamp和knowledge_points的DataFrame
        data: knowledge_points数据列表
    """
    """
    生成个性化学习画像

    参数:
        df: 包含时间戳和知识点的原始DataFrame
        data: 二维知识点列表

    功能:
        1. 分析学习强度模式
        2. 评估知识点掌握度
        3. 生成可视化学习报告
        4. 提供个性化建议
    """
    st.header("🧠 学生个性化学习画像")

    with st.expander("📊 什么是学习画像？", expanded=True):
        st.markdown("""
        ### 学习画像帮助您了解：
        - 学生的**学习行为模式**和**知识掌握状况**
        - 学习过程中的**强项**和**薄弱环节**
        - 基于数据分析的**个性化学习建议**

        *学习画像基于学生的提问记录自动生成，可为教学提供参考。*
        """)

    # 确保数据格式正确
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 提取所有唯一知识点
    all_kps = []
    for kps in data:
        if isinstance(kps, list):
            all_kps.extend(kps)
        else:
            all_kps.extend([k.strip() for k in k.split(',')])

    kp_freq = Counter(all_kps)

    # 1. 学习强度概览
    total_questions = len(df)
    active_days = df['timestamp'].dt.date.nunique()
    avg_daily = total_questions / active_days if active_days > 0 else 0

    # 2. 计算学习时段偏好
    df['hour'] = df['timestamp'].dt.hour

    def get_period(hour):
        if 5 <= hour < 12:
            return '上午'
        elif 12 <= hour < 18:
            return '下午'
        else:
            return '晚上'

    df['period'] = df['hour'].apply(get_period)
    period_counts = df['period'].value_counts()
    preferred_period = period_counts.idxmax() if not period_counts.empty else "无明显偏好"

    # 3. 知识点掌握情况评估
    top_kps = kp_freq.most_common(5)
    bottom_kps = kp_freq.most_common()[:-6:-1] if len(kp_freq) > 5 else []

    # 4. 学习连贯性评估
    df = df.sort_values('timestamp')
    df['next_timestamp'] = df['timestamp'].shift(-1)
    df['time_diff_hours'] = (df['next_timestamp'] - df['timestamp']).dt.total_seconds() / 3600
    avg_interval = df['time_diff_hours'].mean()

    # 连续性指标 (间隔小于8小时的占比)
    continuity = (df['time_diff_hours'] < 8).mean() if len(df) > 1 else 0

    # 5. 创建学习画像卡片
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 学习活跃度")
        metric_cols = st.columns(3)
        metric_cols[0].metric("总提问数", f"{total_questions}")
        metric_cols[1].metric("活跃天数", f"{active_days}天")
        metric_cols[2].metric("日均提问", f"{avg_daily:.1f}个")

        # 学习风格雷达图
        # 计算各维度指标 (0-1之间)
        regularity = 1 - min(df['time_diff_hours'].std() / max(df['time_diff_hours'].mean(), 1), 1) if len(
            df) > 1 else 0
        topic_focus = 1 - min(len(kp_freq) / max(len(all_kps), 1), 1)
        persistence = continuity
        time_management = 1 - (df['hour'].isin([23, 0, 1, 2, 3, 4])).mean()
        variety = min(len(kp_freq) / 10, 1) if len(kp_freq) > 0 else 0

        radar_data = pd.DataFrame({
            '维度': ['专注度', '学习持续性', '时间管理', '知识多样性'],
            '得分': [topic_focus, persistence, time_management, variety]
        })

        fig = px.line_polar(
            radar_data,
            r='得分',
            theta='维度',
            line_close=True,
            range_r=[0, 1],
            title="学习风格雷达图"
        )
        fig.update_traces(fill='toself')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🧩 知识关注点")

        # 创建热门知识点条形图
        if top_kps:
            top_kps_df = pd.DataFrame(top_kps, columns=['知识点', '提问次数'])
            fig = px.bar(
                top_kps_df,
                y='知识点',
                x='提问次数',
                orientation='h',
                title='热门关注知识点',
                color='提问次数',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

        # 学习时段偏好
        st.markdown(f"### ⏰ 学习时段偏好: **{preferred_period}**")
        if not period_counts.empty:
            fig = px.pie(
                names=period_counts.index,
                values=period_counts.values,
                title='学习时段分布',
                hole=0.4
            )
            fig.update_traces(textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

    # 6. 学习建议
    st.markdown("### 📝 个性化学习建议")

    # 生成学习建议
    recommendations = []

    # 根据规律性生成建议
    if regularity < 0.4:
        recommendations.append("📆 **提高学习规律性**：建议制定固定的学习计划，保持每日学习习惯。")

    # 根据时段偏好生成建议
    if preferred_period == '晚上':
        recommendations.append("⏰ **优化学习时间**：尝试将部分学习时间调整到白天，可能有助于提高学习效率。")

    # 根据知识点分布生成建议
    if topic_focus < 0.3:
        recommendations.append("🎯 **提高学习专注度**：当前学习主题较为分散，建议阶段性地专注于特定知识领域，深入学习。")

    # 根据持续性生成建议
    if persistence < 0.4:
        recommendations.append("⚡ **增强学习持续性**：您的学习间隔时间较长，建议采用更连贯的学习方式，减少长时间中断。")

    # 根据时间管理生成建议
    if time_management < 0.7:
        recommendations.append("🌙 **改善时间管理**：避免在深夜学习，保持良好的作息习惯有助于提高学习效率。")

    # 根据热门知识点生成建议
    if top_kps:
        difficult_topic = top_kps[0][0]
        recommendations.append(f"📚 **重点知识巩固**：'{difficult_topic}'是您提问最多的知识点，建议进行系统性复习和练习。")

    # 如果建议很少，添加一条通用建议
    if len(recommendations) < 2:
        recommendations.append("🌟 **保持良好习惯**：您的学习模式整体良好，建议保持当前的学习节奏和方法。")

    # 显示建议
    for rec in recommendations:
        st.markdown(rec)

    # 7. 预测未来学习趋势
    if len(df) >= 10:  # 只有数据足够时才进行预测
        st.markdown("### 🔮 学习趋势预测")

        # 简单的线性预测
        df['date'] = df['timestamp'].dt.date
        daily_counts = df.groupby('date').size().reset_index(name='count')
        daily_counts['date'] = pd.to_datetime(daily_counts['date'])
        daily_counts['day_num'] = range(len(daily_counts))

        # 使用过去的数据预测未来7天
        if len(daily_counts) >= 5:
            try:
                from sklearn.linear_model import LinearRegression

                X = daily_counts['day_num'].values.reshape(-1, 1)
                y = daily_counts['count'].values

                model = LinearRegression()
                model.fit(X, y)

                # 预测未来7天
                future_days = np.array(range(len(daily_counts), len(daily_counts) + 7)).reshape(-1, 1)
                predictions = model.predict(future_days)

                # 创建预测数据框
                last_date = daily_counts['date'].max()
                future_dates = [last_date + pd.Timedelta(days=i + 1) for i in range(7)]
                future_df = pd.DataFrame({
                    'date': future_dates,
                    'count': predictions,
                    'type': '预测'
                })

                # 合并历史和预测数据
                daily_counts['type'] = '历史'
                plot_data = pd.concat([
                    daily_counts[['date', 'count', 'type']],
                    future_df
                ])

                # 绘制预测趋势图
                fig = px.line(
                    plot_data,
                    x='date',
                    y='count',
                    color='type',
                    title='学习强度趋势预测 (未来7天)',
                    labels={'date': '日期', 'count': '提问数量', 'type': '数据类型'}
                )
                fig.update_layout(xaxis_title='日期', yaxis_title='预计提问数量')
                st.plotly_chart(fig, use_container_width=True)

                # 趋势解读
                trend = "上升" if model.coef_[0] > 0.1 else "下降" if model.coef_[0] < -0.1 else "稳定"
                st.markdown(f"""
                ### 📈 趋势解读：

                - **学习强度趋势**: {trend}
                - **预计变化率**: {model.coef_[0]:.2f} 提问/天
                - **未来7天平均提问量**: {predictions.mean():.1f} 提问/天
                """)

            except Exception as e:
                st.warning(f"无法生成预测: {str(e)}")
        else:
            st.info("数据点不足，无法进行可靠的趋势预测。建议收集更多数据。")


# ---------------------------
# 6.高级时序分析模块
# ---------------------------
def advanced_time_series_analysis(df):
    """
    针对聊天/提问记录进行更深度的时序分析，包含：
    1. 活动峰值与周期性检测
    2. 对话主题演变与关联性（需存在 'text' 字段）
    3. 知识点时序趋势与行为轨迹分析
    4. 简单知识追踪模型（基于近期与全期提问频次对比）
    要求数据中至少包含 'timestamp'、'knowledge_points'（列表或字符串格式）以及可选的 'text' 字段
    """
    # 1. 数据预处理
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except Exception as e:
        st.error(f"时间戳转换错误: {e}")
        return
    df = df.sort_values('timestamp')
    df['date'] = df['timestamp'].dt.date

    st.header("🔍 深度时序分析")
    # ---------------------------
    # 4. 知识点时序趋势与行为轨迹
    # ---------------------------
    st.subheader("3. 知识点时序趋势与行为轨迹")
    # 将 knowledge_points 转为列表格式（如果非列表，则按逗号分割）
    if isinstance(df['knowledge_points'].iloc[0], list):
        kp_series = df['knowledge_points']
    else:
        kp_series = df['knowledge_points'].apply(lambda x: [kp.strip() for kp in str(x).split(',')])
    # 构建每个知识点的每日出现频次
    daily_kp_records = []
    for idx, row in df.iterrows():
        date_val = row['date']
        kps = row['knowledge_points'] if isinstance(row['knowledge_points'], list) else [kp.strip() for kp in str(
            row['knowledge_points']).split(',')]
        for kp in kps:
            daily_kp_records.append({'date': date_val, 'knowledge_point': kp})
    kp_df = pd.DataFrame(daily_kp_records)
    if kp_df.empty:
        st.info("知识点记录不足，无法进行时序趋势分析。")
    else:
        # 构建数据透视表（日期 x 知识点出现次数）
        kp_pivot = kp_df.groupby(['date', 'knowledge_point']).size().reset_index(name='count')
        kp_table = kp_pivot.pivot(index='date', columns='knowledge_point', values='count').fillna(0)
        st.line_chart(kp_table)  # 简单折线图展示各知识点的日趋势

        # 检测近期是否有知识点频次突然增加
        st.markdown("**近期行为轨迹检测**：")
        # 设定基线期（全期平均）和近期（例如最近 7 天）
        baseline_period = kp_table.index.min(), kp_table.index.max()
        recent_period = kp_table.index.max() - pd.Timedelta(days=7), kp_table.index.max()
        baseline_avg = kp_table.mean()
        recent_avg = kp_table.loc[kp_table.index >= recent_period[0]].mean()
        sudden_increase = (recent_avg - baseline_avg) / (baseline_avg + 1e-6)  # 防止除0
        increase_threshold = st.number_input("检测阈值（%）", min_value=0.0, value=0.5)
        flagged = sudden_increase[sudden_increase > increase_threshold]
        if not flagged.empty:
            st.markdown("下列知识点近期提问频次显著增加：")
            for kp, inc in flagged.items():
                st.markdown(f"- **{kp}**：增加比例 {inc:.2f}")
        else:
            st.info("无明显频次突然增加的知识点。")

    # ---------------------------
    # 5. 简单知识追踪模型
    # ---------------------------
    st.subheader("4. 简单知识追踪模型")
    st.markdown("""
    假设学生对某知识点的反复提问表明掌握不足，利用全期与近期（例如最近 7 天）的提问频次对比，给出一个简单的掌握度估计：
    \n掌握度 = 1 - (近期频次 / (全期平均频次 + 1e-6))
    \n掌握度范围为 0~1，值越低表示掌握可能越差。
    """)
    # 计算全期平均频次和最近7天平均频次
    mastery_df = pd.DataFrame({'知识点': kp_table.columns,
                               '全期均值': kp_table.mean().values})
    recent_data = kp_table.loc[kp_table.index >= recent_period[0]]
    if not recent_data.empty:
        mastery_df['近期均值'] = recent_data.mean().values
    else:
        mastery_df['近期均值'] = mastery_df['全期均值']
    mastery_df['掌握度'] = 1 - (mastery_df['近期均值'] / (mastery_df['全期均值'] + 1e-6))
    mastery_df['掌握度'] = mastery_df['掌握度'].clip(0, 1)
    mastery_df = mastery_df.sort_values('掌握度')
    st.dataframe(mastery_df[['知识点', '全期均值', '近期均值', '掌握度']], use_container_width=True)

    st.markdown("### 综合结论")
    st.markdown(""" 
    - **知识点时序趋势与行为轨迹** 可帮助识别近期频次突然增加的知识点，从而及时干预。  
    - **知识追踪模型** 提供了基于提问行为的掌握度估计，为个性化教学提供数据支撑。
    """)
# ---------------------------
# 7.时间驱动的记忆持久性分析
# ---------------------------
def analyze_memory_persistence(df):
    """
    基于提问时间间隔的遗忘曲线建模
    创新方法：通过提问间隔模式推导记忆强度
    """
    """
        基于时间间隔的记忆强度分析

        参数:
            df: 包含timestamp和knowledge_points的DataFrame

        算法:
            1. 计算相邻提问时间间隔
            2. 使用指数衰减模型估算记忆强度
            3. 生成个性化遗忘曲线
            4. 推荐最佳复习时间点

        公式:
            记忆强度 S = -Δt / ln(R)
            其中R为记忆保留率阈值(默认0.7)
        """
    st.header("🧠 记忆持久性分析（时间驱动版）")

    # ================= 数据预处理 =================
    # 展开知识点并清洗
    df_exp = df.explode('knowledge_points')
    df_exp['knowledge_points'] = df_exp['knowledge_points'].str.strip()
    df_exp = df_exp[df_exp['knowledge_points'] != '']

    # 转换时间戳并按知识点分组
    df_exp['timestamp'] = pd.to_datetime(df_exp['timestamp'])
    df_exp = df_exp.sort_values(['knowledge_points', 'timestamp'])

    # ================= 核心算法 =================
    def calculate_memory_strength(timestamps):
        """
        基于时间间隔计算记忆强度系数S
        创新假设：学生主动提问间隔反映记忆强度
        间隔越长→假设记忆强度越高
        """
        if len(timestamps) < 2:
            return None

        # 计算时间间隔（天）
        intervals = []
        prev_ts = timestamps[0]
        for ts in timestamps[1:]:
            delta = (ts - prev_ts).total_seconds() / 86400
            intervals.append(delta)
            prev_ts = ts

        # 动态权重计算（近期间隔权重更高）
        weights = np.linspace(0.5, 1.5, len(intervals))

        # 创新公式：假设理想保留率R=0.7时学生会提问
        # 推导公式：S = -Δt / ln(R)
        R = 0.7  # 经验阈值
        weighted_S = np.average([-t / np.log(R) for t in intervals], weights=weights)

        return max(weighted_S, 0.1)  # 防止负值

    # ================= 执行分析 =================
    analysis_results = []
    industry_avg = 5.0  # 行业基准值

    for kp, group in df_exp.groupby('knowledge_points'):
        timestamps = group['timestamp'].tolist()
        if len(timestamps) < 2:
            continue

        # 计算记忆强度系数
        S = calculate_memory_strength(timestamps)
        if S is None:
            continue

        # 生成推荐间隔（保留率目标80%）
        optimal_interval = -S * np.log(0.8)

        analysis_results.append({
            "知识点": kp,
            "提问次数": len(timestamps),
            "首次提问": timestamps[0].strftime("%Y-%m-%d"),
            "最后提问": timestamps[-1].strftime("%Y-%m-%d"),
            "记忆强度(S)": S,
            "推荐间隔": round(optimal_interval, 1),
            "行业对比": "高于" if S > industry_avg else "低于"
        })

    # ================= 可视化展示 =================
    if analysis_results:
        df_analysis = pd.DataFrame(analysis_results)

        # 显示核心指标
        st.subheader("📊 记忆强度分析概览")
        col1, col2, col3 = st.columns(3)
        avg_S = df_analysis['记忆强度(S)'].mean()
        col1.metric("平均记忆强度", f"{avg_S:.1f}", f"{avg_S - industry_avg:+.1f} vs 行业")
        col2.metric("最强知识点",
                    df_analysis.loc[df_analysis['记忆强度(S)'].idxmax()]['知识点'],
                    f"S={df_analysis['记忆强度(S)'].max():.1f}")
        col3.metric("需关注知识点",
                    df_analysis.loc[df_analysis['记忆强度(S)'].idxmin()]['知识点'],
                    f"S={df_analysis['记忆强度(S)'].min():.1f}")

        # 交互式分析
        selected_kp = st.selectbox("选择知识点", df_analysis['知识点'].tolist())
        kp_data = df_analysis[df_analysis['知识点'] == selected_kp].iloc[0]

        # 显示详细信息
        st.markdown(f"""
        ### 📝 {selected_kp} 分析结果
        - **记忆强度系数**: {kp_data['记忆强度(S)']:.1f} ({kp_data['行业对比']}行业平均)
        - **历史提问次数**: {kp_data['提问次数']} 次
        - **学习周期**: {kp_data['首次提问']} 至 {kp_data['最后提问']}
        - **推荐复习间隔**: {kp_data['推荐间隔']} 天
        """)

        # 绘制遗忘曲线
        st.subheader("🔮 个性化遗忘曲线")
        days = np.linspace(0, 30, 100)
        retention = np.exp(-days / kp_data['记忆强度(S)'])
        fig = px.area(
            x=days, y=retention,
            labels={'x': '距上次学习天数', 'y': '预计记忆保留率'},
            title=f"'{selected_kp}' 遗忘曲线 (S={kp_data['记忆强度(S)']:.1f})"
        )
        fig.add_vline(x=kp_data['推荐间隔'], line_dash="dot",
                      annotation_text=f"推荐复习时间")
        fig.add_hrect(y0=0.7, y1=0.7, line_width=0,
                      annotation_text="提问阈值", opacity=0.2)
        st.plotly_chart(fig, use_container_width=True)

        # 生成复习计划
        st.subheader("📅 智能复习计划")
        last_study = pd.to_datetime(kp_data['最后提问'])
        next_review = last_study + pd.Timedelta(days=kp_data['推荐间隔'])
        review_plan = [
            ("立即复习", last_study.strftime("%m-%d"), "巩固记忆"),
            ("首次复习", next_review.strftime("%m-%d"), "最佳记忆点"),
            ("二次复习", (next_review + pd.Timedelta(days=kp_data['推荐间隔'] * 1.5)).strftime("%m-%d"), "长期巩固")
        ]

        cols = st.columns(3)
        for i, (title, date, desc) in enumerate(review_plan):
            cols[i].metric(title, date, desc)

    else:
        st.warning("""
        ## 无法生成分析的可能原因：
        1. 没有知识点具有≥2次提问记录
        2. 时间间隔过短（<1小时）
        3. 时间戳格式不正确
        """)

# ---------------------------
# 主程序
# ---------------------------
def main():
    """
    主执行函数

    功能:
        1. 文件上传与解析
        2. 路由到各分析模块
        3. 界面布局管理
    """
    st.set_page_config(page_title="学生知识点分析工具", page_icon="📊", layout="wide")
    st.title("📚 学生提问知识点分析工具")
    st.markdown("---")
    st.header("1️⃣ 数据上传")
    uploaded_file = st.file_uploader("上传 CSV 或 JSON 文件（需包含 knowledge_points 和 timestamp 列，选填 text 列）",
                                     type=["csv", "json"])

    if uploaded_file is not None:
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension == 'json':
                data_json = json.load(uploaded_file)
                df = pd.DataFrame(data_json)
            else:
                st.error("请上传 CSV 或 JSON 格式的数据文件")
                st.stop()

            if 'knowledge_points' not in df.columns:
                st.error("数据中缺少 'knowledge_points' 列！")
                st.stop()
            if 'timestamp' not in df.columns:
                st.warning("数据中缺少 'timestamp' 列，部分时序分析功能将不可用！")

            # 判断 knowledge_points 是否已经为列表格式
            if isinstance(df['knowledge_points'].iloc[0], list):
                data = df['knowledge_points'].tolist()
            else:
                data = df['knowledge_points'].apply(lambda x: [kp.strip() for kp in str(x).split(',')]).tolist()

            freq = knowledge_frequency_analysis(data)
            # 在tab列表中添加新的分析页
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8,tab9= st.tabs([
                "📊 知识点频率分析",
                "🔄 知识点共现分析",
                "⏰ 时序趋势分析",
                "⏱️ 时间偏好分析",
                "📈 每日知识点组成",
                "🧠 因果知识图谱",
                "✨ 个性化反馈",
                "🔍 深度时序分析",
                "🧠 记忆持久性"
            ])

            with tab1:
                st.header("2️⃣ 知识点频率分析")
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📋 知识点频率表格")
                    plot_frequency_table(freq)
                with col2:
                    st.subheader("☁️ 知识点词云图")
                    plot_frequency_wordcloud_streamlit(freq)
                st.markdown("---")
                st.subheader("3️⃣ 知识点频率TOP10图表")
                chart_type = st.radio("选择图表类型：", options=["柱状图", "饼图"], horizontal=True, key="chart_type")
                if chart_type == "柱状图":
                    plot_top_frequency_bar(freq, top_n=10)
                else:
                    plot_top_frequency_pie(freq, top_n=10)

            with tab2:
                st.header("🔄 知识点共现分析")
                with st.expander("📖 分析说明与解读指南", expanded=True):
                    st.markdown("""
                       ### 如何解读共现分析？
                       1. **共现频率表**：显示知识点两两组合的出现次数，高频组合提示教学中的常见知识关联
                       2. **热力图**：颜色越深表示共现频率越高，对角线显示单个知识点出现频次
                       3. **网络图**：
                          - 节点大小反映知识点出现频率
                          - 连线粗细表示共现强度
                          - 紫色节点表示核心枢纽知识点
                          - 紧密连接的群落提示知识模块

                       ### 教学应用价值：
                       ✅ 发现高频组合 → 优化课程设计中的知识点搭配  
                       ✅ 识别核心节点 → 加强重点知识点的教学  
                       ✅ 发现知识群落 → 建立模块化教学体系  
                       ✅ 定位薄弱环节 → 发现应加强关联的教学点

                       *示例：若"三角函数"与"向量"高频共现，建议在教学中强化二者的综合应用训练*
                       """)
                analyze_knowledge_cooccurrence(data)
                st.markdown("---")
                st.subheader("🕸️ 知识点共现网络")
                col1, col2 = st.columns(2)
                with col1:
                    min_cooccurrence = st.slider("最小共现阈值", min_value=1, max_value=10, value=1,
                                                 help="只显示共现次数大于等于此值的知识点对")
                with col2:
                    max_nodes = st.slider("最大节点数量", min_value=5, max_value=30, value=15,
                                          help="限制网络图中显示的知识点数量")
                build_knowledge_network(data, freq, min_cooccurrence, max_nodes)

            with tab3:
                if 'timestamp' in df.columns:
                    st.header("⏰ 时序趋势分析")
                    with st.expander("🔍 分析维度说明", expanded=True):
                        st.markdown("""
                        ### 本模块分析维度包括：
                        1. **学习会话分析**：识别连续学习时段和间隔
                        2. **学习强度分析**：分析每日/每周学习规律
                        3. **知识焦点迁移**：跟踪知识点关注度变化
                        """)

                    df_enriched, session_stats = analyze_learning_sessions(df)
                    analyze_knowledge_learning_curve(df_enriched)
                    analyze_learning_intensity(df_enriched)

                else:
                    st.error("无法进行时序分析，数据中缺少 timestamp 列")

            with tab4:
                if 'timestamp' in df.columns:
                    st.header("⏱️ 学生提问时间偏好分析")
                    analyze_time_preference(df)
                else:
                    st.error("无法进行时间分析，数据中缺少 timestamp 列")

            with tab5:
                if 'timestamp' in df.columns:
                    st.header("📈 每日知识点组成分析")
                    analyze_daily_knowledge_composition(df)
                else:
                    st.error("无法进行每日分析，数据中缺少 timestamp 列")

            with tab6:
                if 'timestamp' in df.columns:
                    analyze_causal_relationships(data)
                else:
                    st.error("需要时间戳数据进行因果时序分析")

            with tab7:
                st.header("✨ 个性化反馈")
                create_learning_profile(df, data)
            with tab8:
                advanced_time_series_analysis(df)
            with tab9:  # 新增分析模块
                if 'timestamp' in df.columns:
                    analyze_memory_persistence(df.copy())
                else:
                    st.error("需要时间戳数据进行记忆持久性分析")
        except Exception as e:
            st.error(f"处理数据时出错: {e}")
            st.exception(e)
    else:
        st.info("""
        ### 👋 使用说明
        1. 请准备包含学生提问知识点的数据文件，支持 CSV 或 JSON 格式。
        2. 数据文件中必须包含以下键/列：
           - `knowledge_points`: 如果是 CSV 文件，请使用逗号分隔的字符串；如果是 JSON，则应为列表格式。
           - `timestamp`: 提问时间，格式如 "2023-01-01 10:30:00"
           - 可选：`text` 字段用于对话主题分析
        3. 上传文件后，系统将自动分析并展示多种可视化结果。
        """)

if __name__ == "__main__":
    main()
