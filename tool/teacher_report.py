import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as np
import streamlit as st
import pandas as pd
import json
from collections import Counter
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder
from langchain_community.llms import  Tongyi
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import time
plt.rcParams['font.sans-serif'] = ['SimHei']  # 适用于 Windows
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

# 自定义CSS样式
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stMetricLabel {font-size: 0.9rem !important;}
    .stMetricValue {font-size: 1.2rem !important;}
    .cluster-box {border-radius:10px; padding:15px; margin:10px 0;}
    .tab-container {margin-top: 20px;}
</style>
""", unsafe_allow_html=True)


def validate_analysis_data(results):
    """数据校验逻辑"""
    required_keys = {
        'top_weak_points': (list, 3),
        'weak_distribution': (dict, 5),
        'study_period_dist': (dict, 3),
        'question_stats': (dict, ['mean', 'std']),
        'cluster_summary': (dict, ['total_clusters', 'cluster_dist']),
        'knowledge_network': (dict, ['top_combinations', 'network_density'])
    }

    for key, (typ, min_len) in required_keys.items():
        if key not in results:
            raise ValueError(f"缺少必要字段: {key}")
        if not isinstance(results[key], typ):
            raise TypeError(f"{key} 类型错误，应为 {typ}")

        # 检查数据完整性
        if isinstance(min_len, list):  # 字典键检查
            for k in min_len:
                if k not in results[key]:
                    raise ValueError(f"{key} 缺少子字段: {k}")
        elif isinstance(min_len, int):  # 最小长度检查
            if len(results[key]) < min_len:
                raise Warning(f"{key} 数据量不足，可能影响分析深度")

# -----------------------------
# 数据加载与预处理
# -----------------------------
@st.cache_data
def load_data(uploaded_file=None):
    """加载数据，支持上传或使用示例数据"""
    if uploaded_file:
        data = json.load(uploaded_file)
    else:
        sample_json = '''[
          {"name":"张三","weak_points":["代数","几何"],"study_period":"早晨","question_count":5},
          {"name":"李四","weak_points":["代数","微积分"],"study_period":"下午","question_count":3},
          {"name":"王五","weak_points":["几何","统计"],"study_period":"晚上","question_count":7},
          {"name":"赵六","weak_points":["微积分","代数"],"study_period":"早晨","question_count":4},
          {"name":"陈七","weak_points":["统计"],"study_period":"下午","question_count":6},
          {"name":"周八","weak_points":["代数","统计"],"study_period":"晚上","question_count":8}
        ]'''
        data = json.loads(sample_json)
    return pd.DataFrame(data)


# -----------------------------
# 薄弱知识点分析模块
# -----------------------------
def analyze_weak_points(df):
    st.header("📚 薄弱知识点分析", divider="rainbow")

    # 数据预处理
    all_weak_points = [point for sublist in df['weak_points'] for point in sublist]
    weak_counter = Counter(all_weak_points)
    df_weak = pd.DataFrame(weak_counter.items(), columns=['知识点', '出现次数']).sort_values('出现次数',
                                                                                             ascending=False)

    # 只保留前五知识点
    df_top5 = df_weak.head(5)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 知识点分布")

        # 添加图表切换选项
        chart_type = st.radio("选择图表类型", ["柱状图", "饼图"], horizontal=True)

        if chart_type == "柱状图":
            fig = px.bar(df_top5,
                         x='知识点',
                         y='出现次数',
                         color='出现次数',
                         color_continuous_scale='Blues',
                         text='出现次数')
            fig.update_layout(height=300)
        else:
            fig = px.pie(df_top5,
                         values='出现次数',
                         names='知识点',
                         hole=0.4,
                         color_discrete_sequence=px.colors.sequential.Blues_r)
            fig.update_traces(textposition='inside', textinfo='percent+label')

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 知识点雷达图")

        # 准备雷达图数据
        categories = df_top5['知识点'].tolist()
        values = df_top5['出现次数'].tolist()

        # 确保雷达图闭合
        if len(categories) > 0:
            categories += categories[:1]
            values += values[:1]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            line=dict(color='royalblue'),
            name='出现次数'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values) * 1.2 if len(values) > 0 else 1],
                    tickfont=dict(size=10)
                )
            ),
            height=350,
            margin=dict(l=50, r=50)
        )
        st.plotly_chart(fig, use_container_width=True)
        # 新增返回值
        return {
            "top_weak_points": df_top5['知识点'].tolist(),
            "weak_distribution": df_weak.set_index('知识点')['出现次数'].to_dict()
        }


# -----------------------------
# 学习行为分析模块
# -----------------------------
def analyze_learning_behavior(df):
    st.header("⏰ 学习行为分析", divider="rainbow")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 学习时段分布")
        period_count = df['study_period'].value_counts()
        fig = px.pie(period_count,
                     values=period_count.values,
                     names=period_count.index,
                     hole=0.4,
                     color_discrete_sequence=px.colors.diverging.Portland)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 提问数量分布")

        # 生成分布直方图
        fig = px.histogram(df,
                           x='question_count',
                           nbins=8,
                           color_discrete_sequence=['#1f77b4'],
                           opacity=0.8)
        fig.update_layout(
            xaxis_title="提问数量",
            yaxis_title="学生人数",
            bargap=0.1,
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
        # 新增返回值
        return {
            "period_dist": df['study_period'].value_counts(normalize=True).to_dict(),
            "question_stats": {
                "mean": df['question_count'].mean(),
                "std": df['question_count'].std()
            }
        }


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import plotly.express as px
import streamlit as st


def feature_engineering(df):
    """生成聚类分析特征，关注提问次数和薄弱知识点，弱化学习时段"""
    # 特征1：提问次数
    df['question_count'] = df['question_count']

    # 特征2：薄弱知识点数量
    df['weak_count'] = df['weak_points'].apply(len)

    # 特征3：知识点嵌入（使用共现矩阵和PCA降维）
    all_points = [point for sublist in df['weak_points'] for point in sublist]
    unique_points = sorted(list(set(all_points)))
    co_matrix = pd.DataFrame(0, columns=unique_points, index=unique_points)

    # 构建知识点共现矩阵
    for points in df['weak_points']:
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                p1, p2 = sorted([points[i], points[j]])
                co_matrix.loc[p1, p2] += 1
                co_matrix.loc[p2, p1] += 1

    # PCA降维到2维
    pca = PCA(n_components=2)
    knowledge_embedding = pca.fit_transform(co_matrix)
    df['knowledge_emb1'] = df['weak_points'].apply(
        lambda x: np.mean([knowledge_embedding[unique_points.index(p)][0] for p in x], axis=0) if x else 0
    )
    df['knowledge_emb2'] = df['weak_points'].apply(
        lambda x: np.mean([knowledge_embedding[unique_points.index(p)][1] for p in x], axis=0) if x else 0
    )

    # 合并特征（不包含学习时段）
    features = df[['question_count', 'weak_count', 'knowledge_emb1', 'knowledge_emb2']]
    return features


def analyze_student_clusters(df):
    st.header("👥 学生群体分析", divider="rainbow")

    # 生成特征矩阵
    features = feature_engineering(df)

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # 聚类参数选择
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("### 聚类参数设置")
        k = st.slider("选择聚类数量", 2, 5, 3)

        # 执行K-Means聚类
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        df['cluster'] = cluster_labels

        # 计算轮廓系数
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        st.metric("轮廓系数", f"{silhouette_avg:.2f}",
                  help="数值越接近1表示聚类效果越好，建议值＞0.5")

    with col2:
        st.markdown("### 聚类可视化（3D）")

        # 使用PCA降维到3维
        pca = PCA(n_components=3)
        reduced = pca.fit_transform(X_scaled)

        # 可视化
        fig = px.scatter_3d(
            df,
            x=reduced[:, 0],
            y=reduced[:, 1],
            z=reduced[:, 2],
            color='cluster',
            hover_name='name',
            size='question_count',
            color_discrete_sequence=px.colors.qualitative.Vivid,
            opacity=0.8
        )
        fig.update_layout(
            scene=dict(
                xaxis_title='主成分1',
                yaxis_title='主成分2',
                zaxis_title='主成分3'
            ),
            height=600,
            margin=dict(l=0, r=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        # 显示分组详情
        st.markdown("### 分组特征分析")
        clusters = sorted(df['cluster'].unique())
        cols = st.columns(len(clusters))

        for idx, col in enumerate(cols):
            with col:
                cluster_df = df[df['cluster'] == idx]
                color = px.colors.qualitative.Vivid[idx]

                # 计算群组特征
                top_weak = pd.Series(
                    [p for sublist in cluster_df['weak_points'] for p in sublist]
                ).value_counts().head(3).index.tolist()

                st.markdown(f"""
                    <div class="cluster-box" style="border-left:4px solid {color}">
                        <h4>群组 {idx} ({len(cluster_df)}人)</h4>
                        {', '.join(cluster_df['name'].tolist())}
                        <hr style="margin:8px 0">
                        <b>核心特征：</b><br>
                        • 平均提问：{cluster_df['question_count'].mean():.1f}次<br>
                        • 平均薄弱点：{cluster_df['weak_count'].mean():.1f}个<br>
                        • 高频薄弱点：{', '.join(top_weak)}
                    </div>
                """, unsafe_allow_html=True)

        # -----------------------------
        # 知识点关联分析模块
        # -----------------------------
        # 新增返回值
                # 新增返回值
        return {
                    "cluster_data": df[['name', 'cluster', 'question_count']],
                    "cluster_stats": {
                        "total_clusters": k,
                        "silhouette_score": round(silhouette_avg, 2),
                        "size_distribution": df['cluster'].value_counts().to_dict(),
                        "features": {
                            "question_mean": df.groupby('cluster')['question_count'].mean().to_dict(),
                            "weak_count_mean": df.groupby('cluster')['weak_count'].mean().to_dict()
                        }
                    }
                }


def analyze_knowledge_network(df):
    st.header("🔗 知识点关联分析", divider="rainbow")

    # 构建共现矩阵
    all_points = [point for sublist in df['weak_points'] for point in sublist]
    unique_points = sorted(list(set(all_points)))
    co_matrix = pd.DataFrame(0, columns=unique_points, index=unique_points)

    for points in df['weak_points']:
        for i in range(len(points)):
            for j in range(i + 1, len(points)):  # 避免重复计算 A-B 和 B-A
                p1, p2 = sorted([points[i], points[j]])  # 排序，确保 A-B 和 B-A 是相同的组合
                co_matrix.loc[p1, p2] += 1
                co_matrix.loc[p2, p1] += 1  # 可以去掉，使用排序保证一次性加1

    # --- 自然语言分析报告 ---
    st.markdown("### 📝 关联分析结论")


    # 最高共现组合查询优化
    max_co = co_matrix.stack().idxmax()
    max_value = co_matrix.loc[max_co[0], max_co[1]]

    # 孤立知识点检测优化
    isolated = [p for p in unique_points if (co_matrix[p].sum() + co_matrix.loc[p].sum()) == 0]

    # 高频知识点统计
    top_3 = Counter(all_points).most_common(3)
    top_3_str = ', '.join([f"{k} ({v}次)" for k, v in top_3])

    analysis_report = f"""
       **关键发现：**
       1. 最强关联组合：**{max_co[0]}** 与 **{max_co[1]}** 共现 {max_value} 次，建议组合教学[4](@ref)
       2. 高频知识点 TOP3：{top_3_str}
       3. 孤立知识点：{', '.join(isolated) if isolated else "无"}（需单独强化）
       4. 网络密度：{(co_matrix.sum().sum() / (len(unique_points) * (len(unique_points) - 1))) * 100:.1f}%
       5. 平均关联强度：{co_matrix.values.mean():.1f} 个组合/知识点
       """
    st.markdown(analysis_report)
    st.divider()

    # --- 交互式可视化 ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 📊 知识点网络")
        # 创建 NetworkX 图
        # 创建 NetworkX 图
        G = nx.Graph()

        # 添加节点
        for point in unique_points:
            G.add_node(point, size=all_points.count(point) * 10)

        # 添加边（共现次数作为权重）
        for (i, j), weight in co_matrix.stack().items():
            if weight > 0:
                G.add_edge(i, j, weight=weight)

        # 生成节点的布局
        pos = nx.spring_layout(G, seed=42)

        # 提取边的坐标信息
        edge_x = []
        edge_y = []
        weights = []

        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            weights.append(edge[2]['weight'])

        # 修正：使用单个数值作为 width（平均宽度）
        edge_width = max(1.0, np.mean(weights) * 0.5)  # 确保 width 是 float
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=edge_width, color='#888'),  # 这里是一个 float，而不是 list
            hoverinfo='none',
            mode='lines')

        node_sizes = [G.nodes[n]['size'] for n in G.nodes]
        node_x = [pos[n][0] for n in G.nodes]
        node_y = [pos[n][1] for n in G.nodes]
        node_text = [f"{n}<br>出现次数: {G.nodes[n]['size'] // 10}" for n in G.nodes]

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            marker=dict(
                size=np.sqrt(node_sizes),
                color=node_sizes,
                colorscale='Blues',
                line_width=2))

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            height=600))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 📈 关联热力图")

        # 使用Plotly交互式热力图
        fig = px.imshow(
            co_matrix,
            labels=dict(x="知识点", y="知识点", color="共现次数"),
            color_continuous_scale='YlGnBu',
            aspect="auto"
        )
        fig.update_xaxes(tickangle=45)
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # --- 组合推荐系统优化 ---
    st.markdown("### 🎯 推荐教学组合")
    co_occurrences = co_matrix.stack().reset_index()
    co_occurrences.columns = ['知识点A', '知识点B', '次数']
    co_occurrences = co_occurrences[
        (co_occurrences['次数'] > 0) &
        (co_occurrences['知识点A'] < co_occurrences['知识点B'])  # 排除重复排列
        ].sort_values('次数', ascending=False).reset_index(drop=True)

    top_combos = co_occurrences.head(10)

    for idx, row in top_combos.iterrows():
        st.markdown(f"""
           - **{row['知识点A']} + {row['知识点B']}**  
             共现强度：{row['次数']} 次（TOP{idx + 1}）  
           """)
# --- 共现组合全量清单 ---
    st.markdown("### 📋 共现组合全量清单")
    st.dataframe(
            co_occurrences,
            column_config={
                "知识点A": st.column_config.TextColumn("起点知识点"),
                "知识点B": st.column_config.TextColumn("关联知识点"),
                "次数": st.column_config.NumberColumn("共现强度", format="%d次")
            },
            height=400,
            use_container_width=True,
            hide_index=True
        )
    # 新增返回值
    return {
        "co_occurrence_matrix": co_matrix,
        "top_combinations": top_combos[['知识点A', '知识点B', '次数']].values.tolist(),
        "network_density": co_matrix.sum().sum() / (len(unique_points) * (len(unique_points) - 1))
    }

def format_student_data(df):
    """将学生数据格式化为自然语言描述"""
    examples = []
    for _, row in df.sample(min(3, len(df))).iterrows():
        desc = (
            f"学生{row['name']}："
            f"薄弱知识点包括{'、'.join(row['weak_points'])}, "
            f"主要在{row['study_period']}学习，"
            f"平均每周提问{row['question_count']}次"
        )
        examples.append(desc)
    return "\n".join(examples)
def generate_analysis_report(df, analysis_results):
    """生成深度分析报告"""
    st.header("🧠 AI 深度分析报告", divider="rainbow")

    # 数据校验
    try:
        validate_analysis_data(analysis_results)
    except Exception as e:
        st.error(f"数据校验失败: {str(e)}")
        return
    # 数据预处理
    raw_data_sample = format_student_data(df)
    # 构建提示模板
    template = """作为教育分析专家，请根据以下数据生成教学分析报告：

## 数据概览
- 学生总数：{student_count}人
- 平均提问：{question_mean}次/人（标准差：{question_std}）
- 主要薄弱点：{top_weak}
- 提问时段分布：{period_dist}
- 班级学习情况分层：{cluster_info}

## 分析要求
1. 现状分析：基于上面的数据分析班级现状，说明班级特征。
2. 问题诊断：基于已有信息：
   -指出该班级的**2-3个教学痛点**， 
   -基于知识点关联的3个教学优化点
   - 高频薄弱点的突破策略
   - 孤立知识点的教学融入方案
3.群体特征解析：
   - 不同学习时段学生的提问模式差异
   - 各群组特征对教学策略的启示
   - 群组间知识点掌握差异分析
4. 教学建议：提供**具体方案**，包含：
   - 针对不同分组，给出适合他们的教学策略，要求详细可实现具体，并给出具体的教学措施和建议。
   - 根据学习时段的资源分配建议。
   - 基于知识关联性，帮助教师优化教学安排。
   -分析目前班级的存在的教学上的问题和缺陷
   -基于群体特征，说明各分组的差异化教学需求
   -指导老师如何利用高频组合设计综合练习题以及加强对孤立知识点的强化方案
4 基于袁术学生数据{raw_data_sample}更深层次挖掘有价值的教学信息，反馈给老师，注意不要与前面的信息重复。
5. 特殊关注：列出需要重点关注的学员及原因

## 附加数据
- 知识点网络密度：{density}（越高表示关联越紧密）
- 前3知识点组合：{top_combinations}
- 孤立知识点：{isolated}

请用专业但易懂的中文撰写,且内容尽可能详尽，丰富，使用Markdown格式，包含分级标题和重点标记。"""

    # 准备模板参数
    params = {
        "student_count": len(df),
        "question_mean": analysis_results['question_stats']['mean'],
        "question_std": analysis_results['question_stats']['std'],
        "top_weak": ", ".join(analysis_results['top_weak_points'][:3]),
        "period_dist": "\n".join([f"- {k}: {v * 100:.1f}%" for k, v in analysis_results['study_period_dist'].items()]),
        "cluster_info": f"{analysis_results['cluster_summary']['total_clusters']}个分组，最大组{max(analysis_results['cluster_summary']['cluster_dist'].values())}人",
        "density": analysis_results['knowledge_network']['network_density'],
        "top_combinations": "\n".join(
            [f"{a}+{b}（{c}次）" for a, b, c in analysis_results['knowledge_network']['top_combinations'][:3]]),
        "isolated": "无" if not analysis_results['knowledge_network'].get('isolated_points') else ", ".join(
            analysis_results['knowledge_network']['isolated_points']),
        "raw_data_sample":raw_data_sample
    }

    # 初始化大模型
    llm = Tongyi(temperature=0.7,top_p=0.9)
    prompt = PromptTemplate(template=template, input_variables=list(params.keys()))
    chain = LLMChain(llm=llm, prompt=prompt)

    # 生成报告
    with st.spinner('AI正在生成深度分析报告...'):
        try:
            report = chain.run(**params)
            st.markdown(report, unsafe_allow_html=True)

            # 添加重新生成按钮
            if st.button("🔄 重新生成报告"):
                report = chain.run(**params)
                st.markdown(report, unsafe_allow_html=True)

        except RuntimeError as e:
            st.error(f"报告生成失败：{str(e)}")
        except Exception as e:
            st.error(f"发生未知错误：{str(e)}")