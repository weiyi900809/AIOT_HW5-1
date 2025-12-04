import streamlit as st
import re
from collections import Counter
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# 設置頁面配置
st.set_page_config(
    page_title="AI vs Human 文章偵測器",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 自訂 CSS 樣式
st.markdown("""
    <style>
    .main {
        max-width: 1000px;
        margin: 0 auto;
    }
    .header-title {
        background: linear-gradient(135deg, #218D8D 0%, #134252 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .metric-card {
        background: #fcfcf9;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #5e5240;
    }
    .verdict-ai {
        background-color: rgba(192, 21, 47, 0.1);
        border: 1px solid #c01547;
        color: #c01547;
    }
    .verdict-human {
        background-color: rgba(34, 197, 94, 0.1);
        border: 1px solid #22c55e;
        color: #22c55e;
    }
    .verdict-uncertain {
        background-color: rgba(168, 75, 47, 0.1);
        border: 1px solid #a84b2f;
        color: #a84b2f;
    }
    </style>
""", unsafe_allow_html=True)

# 初始化 session state
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# ==================== 特徵提取函數 ====================

def extract_features(text):
    """提取文本的多維特徵"""
    
    # 預處理
    words = re.findall(r'[\w\u4e00-\u9fa5]+', text.lower())
    sentences = [s.strip() for s in re.split(r'[。！？\n]+', text) if s.strip()]
    chars = len(text)
    
    # 1. 句長差異 (Burstiness)
    if sentences:
        sentence_lengths = [len(re.findall(r'[\s\u4e00-\u9fa5]+', s)) for s in sentences]
        avg_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
        variance = sum((l - avg_length) ** 2 for l in sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
        burstiness = (variance ** 0.5) / avg_length if avg_length > 0 else 0
    else:
        burstiness = 0
    
    # 2. 詞彙多樣性 (TTR - Type-Token Ratio)
    unique_words = len(set(words))
    ttr = unique_words / len(words) if words else 0
    
    # 3. 符號使用率
    punctuation = re.findall(r'[！？，；：""''（）《》【】…]', text)
    punctuation_rate = len(punctuation) / len(text) if text else 0
    
    # 4. 平均詞長
    avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
    
    # 5. 重複詞率
    word_freq = Counter(words)
    repeated_words = sum(1 for f in word_freq.values() if f > 2)
    top_repeated = repeated_words / unique_words if unique_words > 0 else 0
    
    # 6. 連接詞使用率
    connectors = ['而且', '因為', '所以', '然而', '但是', '雖然', '為了', '由於', '基於', '鑒於']
    connector_count = sum(1 for c in connectors if c in text)
    connector_rate = connector_count / len(sentences) if sentences else 0
    
    # 7. 句子複雜度
    comma_count = len(re.findall(r'[，,]', text))
    complexity = comma_count / len(sentences) if sentences else 0
    
    # 8. Perplexity 簡化估計（基於詞彙分佈）
    total_words = len(words)
    unique_ratio = unique_words / total_words if total_words > 0 else 0
    perplexity_score = 1 - unique_ratio  # AI 文本往往重複度高
    
    # 9. 詞彙頻率分佈（Zipf's Law）
    sorted_freqs = sorted(word_freq.values(), reverse=True)
    if len(sorted_freqs) > 10:
        top_10_ratio = sum(sorted_freqs[:10]) / sum(sorted_freqs)
    else:
        top_10_ratio = 1.0
    
    return {
        'burstiness': max(0, min(1, burstiness / 2)),
        'ttr': ttr,
        'punctuation_rate': punctuation_rate,
        'avg_word_length': max(0, min(1, avg_word_length / 8)),
        'top_repeated': top_repeated,
        'connector_rate': max(0, min(1, connector_rate)),
        'complexity': max(0, min(1, complexity / 0.5)),
        'perplexity_score': perplexity_score,
        'top_10_ratio': top_10_ratio,
        'word_count': len(words),
        'sentence_count': len(sentences),
        'char_count': chars,
        'unique_words': unique_words,
        'word_freq_dist': dict(word_freq.most_common(20))
    }

# ==================== AI 偵測模型 ====================

def detect_ai(features):
    """使用加權特徵計算 AI 生成機率"""
    
    weights = {
        'burstiness': 0.20,           # AI 語句長度較均勻
        'ttr': -0.15,                 # 人工通常更多樣
        'punctuation_rate': 0.15,     # AI 使用標點更規律
        'avg_word_length': -0.10,     # 人工詞彙更複雜
        'top_repeated': 0.15,         # AI 重複使用某些詞彙
        'connector_rate': -0.15,      # 人工更多使用連接詞
        'complexity': -0.10,          # 人工句子結構更複雜
        'perplexity_score': 0.10,     # AI 複雜度較低
        'top_10_ratio': 0.10          # AI 詞彙集中度高
    }
    
    ai_score = 0.5  # 初始中性分數
    
    # 根據特徵調整分數
    ai_score += (1 - features['burstiness']) * weights['burstiness']
    ai_score += features['ttr'] * weights['ttr']
    ai_score += features['punctuation_rate'] * weights['punctuation_rate']
    ai_score += features['avg_word_length'] * weights['avg_word_length']
    ai_score += features['top_repeated'] * weights['top_repeated']
    ai_score += features['connector_rate'] * weights['connector_rate']
    ai_score += features['complexity'] * weights['complexity']
    ai_score += features['perplexity_score'] * weights['perplexity_score']
    ai_score += features['top_10_ratio'] * weights['top_10_ratio']
    
    # 根據文本長度調整信心度
    min_words = 50
    confidence = max(0, min(1, features['word_count'] / min_words))
    ai_score = 0.5 + (ai_score - 0.5) * confidence
    
    return max(0, min(1, ai_score))

# ==================== UI 構建 ====================

# 標題
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<div class="header-title">🔍 AI vs Human 文章偵測器</div>', unsafe_allow_html=True)
    st.markdown("使用多維特徵分析技術，檢測文本是否由 AI 生成")

with col2:
    if st.button("📋 查看歷史", use_container_width=True):
        st.session_state.show_history = not st.session_state.get('show_history', False)

# 輸入區域
st.markdown("---")
st.subheader("📝 輸入文本")

text_input = st.text_area(
    "貼上你的文本內容",
    placeholder="輸入至少 20 個字以獲得準確結果...",
    height=200,
    label_visibility="collapsed"
)

# 顯示字數
if text_input:
    st.caption(f"字數：{len(text_input)} 字 | 詞數：{len(re.findall(r'[\w\u4e00-\u9fa5]+', text_input))} 詞")

# 分析按鈕
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    analyze_btn = st.button("✨ 開始分析", use_container_width=True, type="primary")
with col2:
    clear_btn = st.button("清空", use_container_width=True)

if clear_btn:
    st.session_state.analysis_result = None
    st.rerun()

# ==================== 分析邏輯 ====================

if analyze_btn:
    if len(text_input.strip()) < 20:
        st.error("❌ 請輸入至少 20 個字的文本")
    else:
        with st.spinner("分析中... ⏳"):
            # 提取特徵
            features = extract_features(text_input)
            
            # 計算分數
            ai_score = detect_ai(features)
            human_score = 1 - ai_score
            
            # 保存結果到 session state
            st.session_state.analysis_result = {
                'timestamp': datetime.now(),
                'text_preview': text_input[:100] + "..." if len(text_input) > 100 else text_input,
                'text_full': text_input,
                'ai_score': ai_score,
                'human_score': human_score,
                'features': features
            }
            
            # 添加到歷史
            st.session_state.analysis_history.append(st.session_state.analysis_result)

# ==================== 結果顯示 ====================

if st.session_state.analysis_result:
    result = st.session_state.analysis_result
    ai_percent = int(result['ai_score'] * 100)
    human_percent = int(result['human_score'] * 100)
    features = result['features']
    
    st.markdown("---")
    st.subheader("📊 分析結果")
    
    # 分數卡片
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "🤖 AI 生成機率",
            f"{ai_percent}%",
            delta=None
        )
        # 進度條
        st.progress(result['ai_score'], text="AI 分數")
    
    with col2:
        st.metric(
            "✍️ 人工撰寫機率",
            f"{human_percent}%",
            delta=None
        )
        # 進度條
        st.progress(result['human_score'], text="Human 分數")
    
    # 判決結果
    st.markdown("---")
    if ai_percent > 65:
        verdict = "⚠️ 極可能由 AI 生成"
        verdict_class = "verdict-ai"
    elif ai_percent > 50:
        verdict = "🤔 可能由 AI 生成 (需要進一步確認)"
        verdict_class = "verdict-uncertain"
    elif ai_percent > 35:
        verdict = "📝 可能由人工撰寫 (但也有 AI 成分)"
        verdict_class = "verdict-uncertain"
    else:
        verdict = "✅ 高度可能由人工撰寫"
        verdict_class = "verdict-human"
    
    st.markdown(
        f'<div style="padding: 20px; border-radius: 8px; text-align: center; font-size: 18px; font-weight: bold; {verdict_class}">{verdict}</div>',
        unsafe_allow_html=True
    )
    
    # 詳細分析指標
    st.markdown("---")
    st.subheader("🔬 詳細分析指標")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("句長均勻度", f"{features['burstiness']*100:.1f}%")
    with col2:
        st.metric("詞彙多樣性", f"{features['ttr']*100:.1f}%")
    with col3:
        st.metric("標點規律性", f"{features['punctuation_rate']*100:.1f}%")
    with col4:
        st.metric("平均詞長", f"{features['avg_word_length']:.2f}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("詞彙重複率", f"{features['top_repeated']*100:.1f}%")
    with col2:
        st.metric("連接詞使用率", f"{features['connector_rate']*100:.1f}%")
    with col3:
        st.metric("句子複雜度", f"{features['complexity']*100:.1f}%")
    with col4:
        st.metric("文本統計", f"{features['word_count']} 詞")
    
    # 可視化分析
    st.markdown("---")
    st.subheader("📈 可視化分析")
    
    tab1, tab2, tab3 = st.tabs(["特徵雷達圖", "詞頻分佈", "分數對比"])
    
    with tab1:
        # 雷達圖
        categories = ['句長均勻度', '詞彙多樣性', '標點規律性', '詞彙重複率', '連接詞使用率', '句子複雜度']
        values = [
            features['burstiness'],
            features['ttr'],
            features['punctuation_rate'],
            features['top_repeated'],
            features['connector_rate'],
            features['complexity']
        ]
        
        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='當前文本',
            line_color='#218D8D'
        ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=500,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # 詞頻分佈
        if features['word_freq_dist']:
            df_freq = pd.DataFrame([
                {'詞彙': word, '頻率': freq}
                for word, freq in features['word_freq_dist'].items()
            ])
            
            fig = px.bar(
                df_freq.head(15),
                x='詞彙',
                y='頻率',
                title='頻率最高的 15 個詞彙',
                color='頻率',
                color_continuous_scale='Teal'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # 分數對比
        fig = go.Figure(data=[
            go.Bar(
                x=['AI 生成機率', '人工撰寫機率'],
                y=[result['ai_score'] * 100, result['human_score'] * 100],
                marker_color=['#c01547', '#22c55e'],
                text=[f"{ai_percent}%", f"{human_percent}%"],
                textposition='auto',
                name='分數'
            )
        ])
        
        fig.update_layout(
            height=400,
            showlegend=False,
            yaxis_title="百分比 (%)",
            yaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # 信息提示
    st.info("""
    💡 **提示：**
    - 本工具基於多維度特徵分析（句長差異、詞彙多樣性、符號使用等）
    - 結果為參考性指標，不構成最終裁定
    - 最佳分析文本長度：200+ 字
    - 對中英文混合文本的準確度可能下降
    """)

# 歷史記錄側邊欄
if st.session_state.get('show_history', False) and st.session_state.analysis_history:
    st.markdown("---")
    st.subheader("📋 分析歷史")
    
    for i, record in enumerate(reversed(st.session_state.analysis_history[-5:])):
        with st.expander(f"分析 {i+1} - {record['timestamp'].strftime('%H:%M:%S')}"):
            st.write(f"**預覽：** {record['text_preview']}")
            st.metric("AI 機率", f"{int(record['ai_score']*100)}%")
            st.metric("Human 機率", f"{int(record['human_score']*100)}%")
